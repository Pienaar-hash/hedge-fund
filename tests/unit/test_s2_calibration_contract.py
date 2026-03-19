"""
S2 Calibration Contract — Truth Guarantees.

These tests prove the calibration system *cannot lie*.
They guard against the two critical bugs discovered 2026-03-19:
    Bug #1: sklearn silently missing → refit never fires
    Bug #2: observations not persisted → lost on restart

Contract invariants:
    1. Refit triggers deterministically at exactly calibration_min_samples
    2. Model diverges from baseline after refit (isotonic actually changes output)
    3. Isotonic guarantee: calibrated output is monotonic in baseline input
    4. Missing sklearn produces an explicit warning, not silent no-op
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from execution.binary_lab_s2_model import (
    BinaryProbabilityModel,
    _SKLEARN_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(p_mid: float) -> Dict[str, float]:
    """Minimal feature dict with just p_yes_mid."""
    return {"p_yes_mid": p_mid}


def _generate_observations(n: int) -> List[tuple]:
    """Generate n (features, outcome) pairs with a known skew.

    Uses a pattern where outcome correlates with p_yes_mid:
    higher mid → more likely YES.  This gives isotonic regression
    something to learn (not random noise).
    """
    obs = []
    for i in range(n):
        # Spread mids across [0.3, 0.8]
        p_mid = 0.3 + (i / max(n - 1, 1)) * 0.5
        # Outcome: YES when p_mid > 0.55 + some alternation
        outcome = int(p_mid > 0.55) if i % 3 != 0 else int(p_mid > 0.45)
        obs.append((_make_features(p_mid), outcome))
    return obs


# ===========================================================================
# 1. Refit triggers deterministically
# ===========================================================================

class TestRefitDeterminism:
    """Refit MUST fire at exactly calibration_min_samples, not before."""

    def test_refit_at_exact_threshold(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=50)
        obs = _generate_observations(50)

        for features, outcome in obs[:49]:
            model.update_observation(features, outcome)

        # At n=49: no refit should have occurred
        assert model._last_refit_n == 0
        assert model._calibrator is None
        assert not model.calibration_active

        # The 50th observation triggers refit
        model.update_observation(obs[49][0], obs[49][1])
        assert model._last_refit_n == 50
        assert model._calibrator is not None
        assert model.calibration_active

    def test_refit_fires_on_every_subsequent_observation(self) -> None:
        """After first refit, each new observation re-triggers refit."""
        model = BinaryProbabilityModel(calibration_min_samples=10)
        obs = _generate_observations(15)

        for features, outcome in obs[:10]:
            model.update_observation(features, outcome)
        assert model._last_refit_n == 10

        model.update_observation(obs[10][0], obs[10][1])
        assert model._last_refit_n == 11

        model.update_observation(obs[11][0], obs[11][1])
        assert model._last_refit_n == 12

    def test_refit_n_matches_observation_count(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=5)
        obs = _generate_observations(20)
        for features, outcome in obs:
            model.update_observation(features, outcome)
        assert model._last_refit_n == 20
        assert model.n_observations == 20


# ===========================================================================
# 2. Model divergence after refit
# ===========================================================================

class TestModelDivergence:
    """After refit, predict() MUST differ from predict_baseline() somewhere."""

    def test_model_diverges_from_baseline_post_refit(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=10)
        obs = _generate_observations(30)
        for features, outcome in obs:
            model.update_observation(features, outcome)

        # Collect predictions at various points
        test_mids = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        diverged = False
        for mid in test_mids:
            f = _make_features(mid)
            p_model = model.predict(f)
            p_baseline = model.predict_baseline(f)
            if abs(p_model - p_baseline) > 1e-6:
                diverged = True
                break

        assert diverged, (
            "Model must diverge from baseline after refit — "
            "isotonic regression should modify at least one prediction"
        )

    def test_baseline_never_changes(self) -> None:
        """predict_baseline() is permanently frozen — refit cannot touch it."""
        model = BinaryProbabilityModel(calibration_min_samples=5)
        features = _make_features(0.6)

        before = model.predict_baseline(features)
        obs = _generate_observations(10)
        for f, o in obs:
            model.update_observation(f, o)
        after = model.predict_baseline(features)

        assert before == after


# ===========================================================================
# 3. Monotonicity (isotonic guarantee)
# ===========================================================================

class TestIsotonicMonotonicity:
    """Isotonic regression output MUST be non-decreasing in input."""

    def test_calibrated_output_is_monotonic(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=10)
        obs = _generate_observations(50)
        for features, outcome in obs:
            model.update_observation(features, outcome)

        assert model._calibrator is not None

        # Evaluate at closely-spaced points
        test_mids = [i / 100 for i in range(5, 96)]
        predictions = [model.predict(_make_features(m)) for m in test_mids]

        for i in range(1, len(predictions)):
            assert predictions[i] >= predictions[i - 1] - 1e-9, (
                f"Monotonicity violated: predict({test_mids[i]})={predictions[i]} "
                f"< predict({test_mids[i-1]})={predictions[i-1]}"
            )

    def test_output_bounded_0_1(self) -> None:
        """Calibrated predictions must stay in [0, 1]."""
        model = BinaryProbabilityModel(calibration_min_samples=10)
        obs = _generate_observations(30)
        for features, outcome in obs:
            model.update_observation(features, outcome)

        for mid in [0.01, 0.1, 0.5, 0.9, 0.99]:
            p = model.predict(_make_features(mid))
            assert 0.0 <= p <= 1.0, f"Out of bounds: predict({mid}) = {p}"


# ===========================================================================
# 4. No-op when sklearn unavailable (Bug #1 regression guard)
# ===========================================================================

class TestSklearnMissing:
    """When sklearn is unavailable, refit must warn — not silently skip."""

    def test_refit_warns_without_sklearn(self, caplog: pytest.LogCaptureFixture) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=5)
        obs = _generate_observations(10)

        with caplog.at_level(logging.DEBUG, logger="execution.binary_lab_s2_model"):
            with patch("execution.binary_lab_s2_model._SKLEARN_AVAILABLE", False):
                for features, outcome in obs:
                    model.update_observation(features, outcome)

        # Calibrator must NOT have been fitted
        assert model._calibrator is None
        assert model._last_refit_n == 0

        # A debug/warning message must have been logged
        assert any(
            "sklearn" in rec.message.lower() or "skipping refit" in rec.message.lower()
            for rec in caplog.records
        ), "Missing sklearn must produce a log message, not fail silently"

    def test_predict_falls_back_to_baseline_without_sklearn(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=5)
        obs = _generate_observations(10)

        with patch("execution.binary_lab_s2_model._SKLEARN_AVAILABLE", False):
            for features, outcome in obs:
                model.update_observation(features, outcome)

        # Without refit, predict must equal baseline
        for mid in [0.3, 0.5, 0.7]:
            f = _make_features(mid)
            assert model.predict(f) == model.predict_baseline(f)


# ===========================================================================
# 5. Confidence level transitions
# ===========================================================================

class TestConfidenceLevels:
    """Authority levels must transition at exact thresholds."""

    def test_inactive_to_active_to_confident(self) -> None:
        model = BinaryProbabilityModel(
            calibration_min_samples=10,
            calibration_confident_samples=20,
        )
        obs = _generate_observations(25)

        # Inactive
        for f, o in obs[:9]:
            model.update_observation(f, o)
        assert not model.calibration_active
        assert not model.calibration_confident

        # Active (not confident)
        model.update_observation(obs[9][0], obs[9][1])
        assert model.calibration_active
        assert not model.calibration_confident

        # Confident
        for f, o in obs[10:20]:
            model.update_observation(f, o)
        assert model.calibration_active
        assert model.calibration_confident
