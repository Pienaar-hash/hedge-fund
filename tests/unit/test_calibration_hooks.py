"""Tests for episode-gated calibration hooks (v7.9).

Verifies:
  - Calibration is skipped when episode_count < CALIBRATION_MIN_EPISODES
  - Calibration activates when episode_count >= CALIBRATION_MIN_EPISODES
  - apply_calibration is identity when not calibrated
  - Score schema unchanged; only values evolve when calibrated
  - Sensitivity multipliers adjust factors correctly
  - Weight temperature tuning works as expected
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from execution.intel.symbol_score_v6 import (
    get_calibration_state,
    apply_calibration,
    CalibrationState,
    CALIBRATION_MIN_EPISODES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_expectancy_snapshot(
    symbol: str = "BTCUSDT",
    count: int = 0,
    hit_rate: float | None = None,
    expectancy: float | None = None,
    expectancy_per_risk: float | None = None,
) -> dict:
    """Build a minimal expectancy snapshot for testing."""
    stats: dict = {"count": count}
    if hit_rate is not None:
        stats["hit_rate"] = hit_rate
    if expectancy is not None:
        stats["expectancy"] = expectancy
    if expectancy_per_risk is not None:
        stats["expectancy_per_risk"] = expectancy_per_risk
    return {"symbols": {symbol: stats}}


# ---------------------------------------------------------------------------
# Tests: get_calibration_state
# ---------------------------------------------------------------------------

class TestGetCalibrationState:
    """CalibrationState gating on episode count."""

    def test_zero_episodes_not_calibrated(self):
        snap = _mock_expectancy_snapshot(count=0)
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is False
        assert cal.episode_count == 0

    def test_below_min_not_calibrated(self):
        snap = _mock_expectancy_snapshot(count=CALIBRATION_MIN_EPISODES - 1)
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is False

    def test_at_min_is_calibrated(self):
        snap = _mock_expectancy_snapshot(
            count=CALIBRATION_MIN_EPISODES,
            hit_rate=0.50,
            expectancy_per_risk=0.0,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is True
        assert cal.episode_count == CALIBRATION_MIN_EPISODES

    def test_above_min_is_calibrated(self):
        snap = _mock_expectancy_snapshot(
            count=100,
            hit_rate=0.55,
            expectancy_per_risk=0.3,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is True

    def test_missing_symbol_not_calibrated(self):
        snap = {"symbols": {}}
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is False
        assert cal.episode_count == 0

    def test_high_hit_rate_sharpens_temperature(self):
        snap = _mock_expectancy_snapshot(
            count=60,
            hit_rate=0.65,
            expectancy_per_risk=0.2,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is True
        assert cal.weight_temperature < 1.0, "High hit rate should sharpen weights"

    def test_low_hit_rate_flattens_temperature(self):
        snap = _mock_expectancy_snapshot(
            count=60,
            hit_rate=0.35,
            expectancy_per_risk=-0.1,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.calibrated is True
        assert cal.weight_temperature > 1.0, "Low hit rate should flatten weights"

    def test_positive_epr_boosts_trend_sensitivity(self):
        snap = _mock_expectancy_snapshot(
            count=60,
            hit_rate=0.50,
            expectancy_per_risk=0.4,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.trend_sensitivity > 1.0

    def test_negative_epr_reduces_trend_sensitivity(self):
        snap = _mock_expectancy_snapshot(
            count=60,
            hit_rate=0.50,
            expectancy_per_risk=-0.3,
        )
        cal = get_calibration_state("BTCUSDT", snap)
        assert cal.trend_sensitivity < 1.0


# ---------------------------------------------------------------------------
# Tests: apply_calibration
# ---------------------------------------------------------------------------

class TestApplyCalibration:
    """Calibration overlay application."""

    def test_not_calibrated_identity(self):
        factors = {"trend": 0.7, "carry": 0.6, "expectancy": 0.55, "router": 0.8}
        weights = {"trend": 0.4, "carry": 0.25, "expectancy": 0.2, "router": 0.15}
        cal = CalibrationState(calibrated=False)

        adj_f, adj_w = apply_calibration(factors, weights, cal)
        assert adj_f == factors
        assert adj_w == weights

    def test_calibrated_modifies_factors(self):
        factors = {"trend": 0.7, "carry": 0.6, "expectancy": 0.55, "router": 0.8}
        weights = {"trend": 0.4, "carry": 0.25, "expectancy": 0.2, "router": 0.15}
        cal = CalibrationState(
            calibrated=True,
            trend_sensitivity=1.2,
            carry_sensitivity=0.8,
            expectancy_sensitivity=1.1,
        )

        adj_f, _ = apply_calibration(factors, weights, cal)
        # Trend deviation (0.7 - 0.5 = 0.2) × 1.2 = 0.24 → 0.5 + 0.24 = 0.74
        assert adj_f["trend"] > factors["trend"]
        # Carry deviation (0.6 - 0.5 = 0.1) × 0.8 = 0.08 → 0.5 + 0.08 = 0.58
        assert adj_f["carry"] < factors["carry"]

    def test_weight_temperature_sharpening(self):
        factors = {"trend": 0.7, "carry": 0.6}
        # Unequal weights
        weights = {"trend": 0.7, "carry": 0.3}
        cal = CalibrationState(
            calibrated=True,
            weight_temperature=0.7,  # <1 = sharpen
        )

        _, adj_w = apply_calibration(factors, weights, cal)
        # Sharpening should make the dominant weight even larger
        assert adj_w["trend"] > weights["trend"]

    def test_weight_temperature_flattening(self):
        factors = {"trend": 0.7, "carry": 0.6}
        weights = {"trend": 0.7, "carry": 0.3}
        cal = CalibrationState(
            calibrated=True,
            weight_temperature=1.3,  # >1 = flatten
        )

        _, adj_w = apply_calibration(factors, weights, cal)
        # Flattening should bring weights closer together
        assert adj_w["trend"] < weights["trend"]

    def test_factors_clamped_to_unit(self):
        factors = {"trend": 0.95, "carry": 0.05}
        weights = {"trend": 0.5, "carry": 0.5}
        cal = CalibrationState(
            calibrated=True,
            trend_sensitivity=1.5,
            carry_sensitivity=1.5,
        )

        adj_f, _ = apply_calibration(factors, weights, cal)
        assert 0.0 <= adj_f["trend"] <= 1.0
        assert 0.0 <= adj_f["carry"] <= 1.0

    def test_weights_sum_to_one(self):
        factors = {"trend": 0.7, "carry": 0.6, "expectancy": 0.55, "router": 0.8}
        weights = {"trend": 0.3, "carry": 0.3, "expectancy": 0.2, "router": 0.2}
        cal = CalibrationState(
            calibrated=True,
            weight_temperature=0.8,
        )

        _, adj_w = apply_calibration(factors, weights, cal)
        assert abs(sum(adj_w.values()) - 1.0) < 1e-6

    def test_score_schema_unchanged_by_calibration(self):
        """Calibration only changes values, not the set of keys."""
        factors = {"trend": 0.7, "carry": 0.6, "expectancy": 0.55}
        weights = {"trend": 0.4, "carry": 0.3, "expectancy": 0.3}
        cal = CalibrationState(calibrated=True, trend_sensitivity=1.2)

        adj_f, adj_w = apply_calibration(factors, weights, cal)
        assert set(adj_f.keys()) == set(factors.keys())
        assert set(adj_w.keys()) == set(weights.keys())
