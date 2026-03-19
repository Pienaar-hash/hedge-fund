"""
S2 Metrics Contract — Brier / BSS Truth Guarantees.

Protects against metric corruption.

Contract invariants:
    1. Baseline model (predict = truth) → BSS ≈ 0
    2. Random/noisy predictions → BSS < 0
    3. Perfect predictions → BSS → 1
    4. Brier score is in [0, 1]
    5. BSS decomposition: BSS = 1 - Brier(model) / Brier(climatology)
"""

from __future__ import annotations

import math
from typing import List

import pytest

from execution.binary_lab_s2_model import _brier_score, _log_loss


# ---------------------------------------------------------------------------
# BSS helper (same formula the dashboard / calibration_stats would use)
# ---------------------------------------------------------------------------

def _bss(predictions: List[float], outcomes: List[int]) -> float:
    """Brier Skill Score: 1 - Brier(model) / Brier(climatology)."""
    brier_model = _brier_score(predictions, outcomes)
    if brier_model is None:
        return 0.0
    base_rate = sum(outcomes) / len(outcomes)
    brier_clim = sum((base_rate - o) ** 2 for o in outcomes) / len(outcomes)
    if brier_clim == 0:
        return 0.0
    return 1.0 - brier_model / brier_clim


# ===========================================================================
# 1. Baseline model → BSS ≈ 0
# ===========================================================================

class TestBaselineBSS:
    """When predictions equal the climatological rate, BSS should be ~0."""

    def test_constant_baseline_bss_zero(self) -> None:
        """Predicting the base rate for every outcome → BSS = 0."""
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        base_rate = sum(outcomes) / len(outcomes)  # 0.5
        predictions = [base_rate] * len(outcomes)

        bss = _bss(predictions, outcomes)
        assert abs(bss) < 1e-9, f"Constant baseline should give BSS=0, got {bss}"

    def test_near_baseline_bss_near_zero(self) -> None:
        """Predictions close to base rate → BSS near 0."""
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        base_rate = sum(outcomes) / len(outcomes)
        # Add tiny noise
        predictions = [base_rate + 0.001 * (i % 3 - 1) for i in range(len(outcomes))]

        bss = _bss(predictions, outcomes)
        assert abs(bss) < 0.01, f"Near-baseline BSS should be near 0, got {bss}"


# ===========================================================================
# 2. Random / noisy predictions → BSS < 0
# ===========================================================================

class TestRandomBSS:
    """Bad predictions must produce negative BSS."""

    def test_inverted_predictions_negative_bss(self) -> None:
        """Predicting the opposite of outcomes → strongly negative BSS."""
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        # Invert: predict 0 when outcome is 1, and vice versa
        predictions = [1 - o for o in outcomes]

        bss = _bss(predictions, outcomes)
        assert bss < -0.5, f"Inverted predictions should give BSS << 0, got {bss}"

    def test_random_noise_negative_bss(self) -> None:
        """Random predictions uncorrelated with outcomes → BSS < 0."""
        # Deterministic 'random' sequence
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        predictions = [0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.1, 0.9, 0.4, 0.6,
                       0.2, 0.3, 0.8, 0.7, 0.1, 0.9, 0.6, 0.4, 0.8, 0.2]

        bss = _bss(predictions, outcomes)
        assert bss < 0, f"Random predictions should give BSS < 0, got {bss}"

    def test_extreme_overconfidence_negative_bss(self) -> None:
        """Always predicting 0.99 with 50/50 outcomes → BSS < 0."""
        outcomes = [1, 0] * 20
        predictions = [0.99] * 40

        bss = _bss(predictions, outcomes)
        assert bss < 0, f"Overconfident constant should give BSS < 0, got {bss}"


# ===========================================================================
# 3. Perfect predictions → BSS → 1
# ===========================================================================

class TestPerfectBSS:
    """Perfect probabilistic predictions must yield BSS = 1."""

    def test_perfect_predictions_bss_one(self) -> None:
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        predictions = [float(o) for o in outcomes]

        bss = _bss(predictions, outcomes)
        assert abs(bss - 1.0) < 1e-9, f"Perfect predictions should give BSS=1, got {bss}"

    def test_near_perfect_bss_high(self) -> None:
        """Near-perfect predictions → BSS close to 1."""
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        predictions = [o + 0.01 * (-1 if o == 1 else 1) for o in outcomes]

        bss = _bss(predictions, outcomes)
        assert bss > 0.95, f"Near-perfect should give BSS > 0.95, got {bss}"


# ===========================================================================
# 4. Brier score bounds
# ===========================================================================

class TestBrierBounds:
    """Brier score must be in [0, 1]."""

    def test_brier_perfect(self) -> None:
        outcomes = [1, 0, 1]
        preds = [1.0, 0.0, 1.0]
        assert _brier_score(preds, outcomes) == pytest.approx(0.0)

    def test_brier_worst(self) -> None:
        outcomes = [1, 0, 1]
        preds = [0.0, 1.0, 0.0]
        assert _brier_score(preds, outcomes) == pytest.approx(1.0)

    def test_brier_midpoint(self) -> None:
        outcomes = [1, 0]
        preds = [0.5, 0.5]
        assert _brier_score(preds, outcomes) == pytest.approx(0.25)

    def test_brier_empty_returns_none(self) -> None:
        assert _brier_score([], []) is None

    def test_brier_always_nonnegative(self) -> None:
        outcomes = [1, 0, 1, 1, 0]
        for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
            preds = [p] * len(outcomes)
            score = _brier_score(preds, outcomes)
            assert score >= 0, f"Brier must be >= 0 for p={p}, got {score}"


# ===========================================================================
# 5. Log loss sanity
# ===========================================================================

class TestLogLoss:
    """Basic log loss sanity checks."""

    def test_log_loss_perfect_near_zero(self) -> None:
        outcomes = [1, 0, 1]
        preds = [0.999, 0.001, 0.999]
        ll = _log_loss(preds, outcomes)
        assert ll is not None
        assert ll < 0.01

    def test_log_loss_worst_large(self) -> None:
        outcomes = [1, 0]
        preds = [0.001, 0.999]
        ll = _log_loss(preds, outcomes)
        assert ll is not None
        assert ll > 5.0  # very bad

    def test_log_loss_empty_returns_none(self) -> None:
        assert _log_loss([], []) is None

    def test_log_loss_nonnegative(self) -> None:
        outcomes = [1, 0, 1]
        preds = [0.6, 0.4, 0.7]
        ll = _log_loss(preds, outcomes)
        assert ll is not None
        assert ll >= 0


# ===========================================================================
# 6. BSS decomposition identity
# ===========================================================================

class TestBSSDecomposition:
    """BSS = 1 - Brier(model) / Brier(climatology) exactly."""

    def test_decomposition_identity(self) -> None:
        outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        predictions = [0.6, 0.3, 0.7, 0.8, 0.2, 0.4, 0.9, 0.1, 0.6, 0.5,
                       0.7, 0.6, 0.3, 0.8, 0.2]

        brier_model = _brier_score(predictions, outcomes)
        base_rate = sum(outcomes) / len(outcomes)
        brier_clim = sum((base_rate - o) ** 2 for o in outcomes) / len(outcomes)

        bss = _bss(predictions, outcomes)
        expected = 1.0 - brier_model / brier_clim

        assert abs(bss - expected) < 1e-12, (
            f"BSS decomposition violated: {bss} != {expected}"
        )
