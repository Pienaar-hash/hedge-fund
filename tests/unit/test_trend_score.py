"""Tests for trend_score from price geometry (v7.9).

Verifies:
  - Uptrend series → trend_score > 0.5
  - Downtrend series → trend_score < 0.5
  - Flat/choppy series → trend_score ~ 0.5
  - Regime compression amplifies/dampens appropriately
  - Insufficient data returns neutral 0.5
"""
from __future__ import annotations

import math
import pytest

pytestmark = pytest.mark.unit

from execution.intel.trend_score import (
    compute_trend_score,
    compute_trend_score_from_prices,
    _ols_slope_r2,
    _momentum_zscore,
    _regime_compression,
)


# ---------------------------------------------------------------------------
# Synthetic price series helpers
# ---------------------------------------------------------------------------

def _uptrend(n: int = 100, start: float = 100.0, daily_pct: float = 0.005) -> list[float]:
    """Generate a deterministic uptrend series."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + daily_pct))
    return prices


def _downtrend(n: int = 100, start: float = 100.0, daily_pct: float = 0.005) -> list[float]:
    """Generate a deterministic downtrend series."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 - daily_pct))
    return prices


def _flat(n: int = 100, price: float = 100.0) -> list[float]:
    """Generate a flat price series."""
    return [price] * n


def _choppy(n: int = 100, center: float = 100.0, amplitude: float = 2.0) -> list[float]:
    """Generate a mean-reverting choppy series."""
    prices = []
    for i in range(n):
        prices.append(center + amplitude * math.sin(i * 0.5))
    return prices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeTrendScore:
    """Core trend_score computation."""

    def test_uptrend_above_neutral(self):
        result = compute_trend_score(_uptrend())
        assert result["score"] > 0.5, f"Expected >0.5 for uptrend, got {result['score']}"

    def test_strong_uptrend_well_above_neutral(self):
        result = compute_trend_score(_uptrend(daily_pct=0.01))
        assert result["score"] > 0.6, f"Strong uptrend should score >0.6, got {result['score']}"

    def test_downtrend_below_neutral(self):
        result = compute_trend_score(_downtrend())
        assert result["score"] < 0.5, f"Expected <0.5 for downtrend, got {result['score']}"

    def test_strong_downtrend_well_below_neutral(self):
        result = compute_trend_score(_downtrend(daily_pct=0.01))
        assert result["score"] < 0.4, f"Strong downtrend should score <0.4, got {result['score']}"

    def test_flat_near_neutral(self):
        result = compute_trend_score(_flat())
        assert abs(result["score"] - 0.5) < 0.05, f"Flat series should be ~0.5, got {result['score']}"

    def test_choppy_near_neutral(self):
        result = compute_trend_score(_choppy())
        assert abs(result["score"] - 0.5) < 0.25, f"Choppy series should be ~0.5, got {result['score']}"

    def test_insufficient_data_returns_neutral(self):
        result = compute_trend_score([100.0] * 10)
        assert result["score"] == 0.5

    def test_empty_returns_neutral(self):
        result = compute_trend_score([])
        assert result["score"] == 0.5

    def test_result_contains_components(self):
        result = compute_trend_score(_uptrend())
        assert "components" in result
        assert "slope_score" in result["components"]
        assert "r2" in result["components"]
        assert "momentum_score" in result["components"]

    def test_result_contains_inputs(self):
        result = compute_trend_score(_uptrend())
        assert "inputs" in result
        assert "slope" in result["inputs"]
        assert "n_bars" in result["inputs"]


class TestConvenienceWrapper:
    """compute_trend_score_from_prices returns just the float."""

    def test_returns_float(self):
        score = compute_trend_score_from_prices(_uptrend())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_uptrend_above_neutral(self):
        assert compute_trend_score_from_prices(_uptrend()) > 0.5

    def test_downtrend_below_neutral(self):
        assert compute_trend_score_from_prices(_downtrend()) < 0.5


class TestRegressionHelpers:
    """OLS regression slope and R² computation."""

    def test_perfect_uptrend_r2_one(self):
        ys = [float(i) for i in range(50)]
        slope, r2 = _ols_slope_r2(ys)
        assert slope > 0
        assert abs(r2 - 1.0) < 1e-10

    def test_constant_series_r2_zero(self):
        ys = [5.0] * 50
        slope, r2 = _ols_slope_r2(ys)
        assert abs(slope) < 1e-12
        assert abs(r2) < 1e-12

    def test_too_few_points(self):
        slope, r2 = _ols_slope_r2([1.0, 2.0])
        assert slope == 0.0
        assert r2 == 0.0


class TestMomentumZscore:
    """Momentum z-score computation."""

    def test_rising_prices_positive_z(self):
        # Accelerating uptrend: growth rate increases, so last return > mean
        prices = [100.0 * (1.0 + 0.001 * i) ** i for i in range(1, 52)]
        z = _momentum_zscore(prices, 20)
        assert z > 0

    def test_falling_prices_negative_z(self):
        # Accelerating downtrend: decline rate increases
        prices = [100.0 * (1.0 - 0.001 * i) ** i for i in range(1, 52)]
        z = _momentum_zscore(prices, 20)
        assert z < 0

    def test_insufficient_data_zero(self):
        z = _momentum_zscore([100.0] * 5, 20)
        assert z == 0.0


class TestRegimeCompression:
    """Regime-conditioned compression/expansion."""

    def test_trend_up_amplifies_bullish(self):
        raw = 0.7  # above neutral
        compressed = _regime_compression(raw, "TREND_UP", regime_confidence=0.8)
        assert compressed > raw, "TREND_UP should amplify bullish signal"

    def test_choppy_compresses_toward_neutral(self):
        raw = 0.7
        compressed = _regime_compression(raw, "CHOPPY", regime_confidence=0.8)
        assert compressed < raw, "CHOPPY should compress signal toward neutral"

    def test_no_regime_passthrough(self):
        raw = 0.7
        assert _regime_compression(raw, None) == raw

    def test_zero_confidence_passthrough(self):
        raw = 0.7
        result = _regime_compression(raw, "TREND_UP", regime_confidence=0.0)
        assert abs(result - raw) < 1e-10

    def test_clamped_to_unit(self):
        result = _regime_compression(0.99, "TREND_UP", regime_confidence=1.0)
        assert 0.0 <= result <= 1.0
        result = _regime_compression(0.01, "TREND_DOWN", regime_confidence=1.0)
        assert 0.0 <= result <= 1.0
