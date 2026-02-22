"""Trend scoring from price geometry features (v7.9).

Computes a numeric trend_score ∈ [0, 1] from price-only features:
  - Log-price regression slope over lookback L
  - R² of regression (trend cleanliness)
  - Momentum z-score (return vs rolling mean/std)
  - Regime-conditioned compression using Sentinel-X regime + confidence

Returns 0.5 when data is insufficient or inconclusive.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK = 60          # bars for regression
DEFAULT_MOMENTUM_LOOKBACK = 20  # bars for momentum z-score
MIN_BARS = 30                   # minimum data required


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def _log_returns(prices: Sequence[float]) -> List[float]:
    """Compute log-prices (not returns) for regression."""
    return [math.log(p) for p in prices if p > 0]


def _ols_slope_r2(ys: Sequence[float]) -> tuple[float, float]:
    """
    Ordinary least squares regression of *ys* against sequential index.

    Returns (slope, R²).  slope is in log-price per bar units.
    R² measures trend cleanliness (0 = no fit, 1 = perfect line).
    """
    n = len(ys)
    if n < 3:
        return 0.0, 0.0

    # x = 0, 1, …, n-1
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n

    ss_xy = 0.0
    ss_xx = 0.0
    ss_yy = 0.0
    for i, y in enumerate(ys):
        dx = i - x_mean
        dy = y - y_mean
        ss_xy += dx * dy
        ss_xx += dx * dx
        ss_yy += dy * dy

    if ss_xx == 0:
        return 0.0, 0.0

    slope = ss_xy / ss_xx
    r2 = (ss_xy * ss_xy) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0
    r2 = max(0.0, min(1.0, r2))  # numerical guard
    return slope, r2


def _momentum_zscore(prices: Sequence[float], lookback: int = DEFAULT_MOMENTUM_LOOKBACK) -> float:
    """
    Momentum z-score: current return vs rolling mean/std of returns.

    Uses simple returns over *lookback* bars.
    Returns 0.0 when data is insufficient.
    """
    if len(prices) < lookback + 2:
        return 0.0

    tail = prices[-(lookback + 1):]
    rets = [(tail[i] - tail[i - 1]) / tail[i - 1] for i in range(1, len(tail)) if tail[i - 1] != 0]
    if len(rets) < 5:
        return 0.0

    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 0.0

    if std < 1e-12:
        return 0.0

    return (rets[-1] - mu) / std


# ---------------------------------------------------------------------------
# Regime compression
# ---------------------------------------------------------------------------

_REGIME_TREND_BOOST: Dict[str, float] = {
    "TREND_UP": 0.15,
    "TREND_DOWN": 0.15,
    "BREAKOUT": 0.10,
    "MEAN_REVERT": -0.05,
    "CHOPPY": -0.10,
    "CRISIS": -0.15,
}


def _regime_compression(
    raw_score: float,
    regime: str | None,
    regime_confidence: float = 0.5,
) -> float:
    """
    Compress or expand the raw trend score based on how aligned
    the Sentinel-X regime is with trend-following.

    In trending regimes (TREND_UP, TREND_DOWN, BREAKOUT) the signal
    is pushed further from 0.5 (amplified).  In CHOPPY / CRISIS it is
    compressed toward 0.5.

    *regime_confidence* ∈ [0, 1] scales the adjustment.
    """
    if not regime:
        return raw_score

    boost = _REGIME_TREND_BOOST.get(regime.upper(), 0.0)
    conf = max(0.0, min(1.0, regime_confidence))

    # Distance from neutral
    deviation = raw_score - 0.5
    # Scale the boost by confidence
    adjusted = deviation * (1.0 + boost * conf)
    return max(0.0, min(1.0, 0.5 + adjusted))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_trend_score(
    closes: Sequence[float],
    *,
    lookback: int = DEFAULT_LOOKBACK,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
    regime: str | None = None,
    regime_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute trend_score from price geometry.

    Args:
        closes: Sequence of close prices (oldest → newest).
        lookback: Bars for regression slope (default 60).
        momentum_lookback: Bars for momentum z-score (default 20).
        regime: Sentinel-X primary regime label (optional shaper).
        regime_confidence: Regime confidence ∈ [0, 1] (optional).

    Returns:
        Dict with:
            score: float ∈ [0, 1]  (>0.5 = bullish, <0.5 = bearish)
            components: dict of sub-scores
            inputs: dict of raw feature values
    """
    neutral = {
        "score": 0.5,
        "components": {"slope_score": 0.5, "r2": 0.0, "momentum_score": 0.5},
        "inputs": {"slope": 0.0, "r2": 0.0, "momentum_z": 0.0, "n_bars": len(closes)},
    }

    if len(closes) < MIN_BARS:
        return neutral

    # --- 1. Regression slope + R² ---
    window = closes[-lookback:] if len(closes) >= lookback else closes
    log_prices = _log_returns(window)
    if len(log_prices) < MIN_BARS:
        return neutral

    slope, r2 = _ols_slope_r2(log_prices)

    # Map slope to [0, 1] using tanh.
    # Slope is log-price/bar; typical daily ≈ ±0.005.
    # For 1h bars ×24h, moderate trend ≈ 0.002 per bar.
    # tanh(slope / 0.003) gives ±1 for strong trend → rescale to [0, 1].
    slope_mapped = 0.5 + 0.5 * math.tanh(slope / 0.003)

    # --- 2. Momentum z-score ---
    mom_z = _momentum_zscore(closes, momentum_lookback)
    # Map z-score to [0, 1]: z=±2 → ~0.12/0.88
    momentum_mapped = 0.5 + 0.5 * math.tanh(mom_z / 2.5)

    # --- 3. Blend: slope-weighted-by-R² + momentum ---
    # R² acts as quality gate: high R² → trust slope more
    # Weight: slope contribution = R² × 0.6, momentum = 0.4
    slope_weight = 0.6 * max(0.0, min(1.0, r2))
    momentum_weight = 0.4
    total_weight = slope_weight + momentum_weight
    if total_weight <= 0:
        raw_score = 0.5
    else:
        raw_score = (slope_mapped * slope_weight + momentum_mapped * momentum_weight) / total_weight

    raw_score = max(0.0, min(1.0, raw_score))

    # --- 4. Regime-conditioned compression ---
    final_score = _regime_compression(raw_score, regime, regime_confidence)

    return {
        "score": round(final_score, 6),
        "components": {
            "slope_score": round(slope_mapped, 6),
            "r2": round(r2, 6),
            "momentum_score": round(momentum_mapped, 6),
        },
        "inputs": {
            "slope": round(slope, 8),
            "r2": round(r2, 6),
            "momentum_z": round(mom_z, 4),
            "n_bars": len(window),
            "regime": regime,
            "regime_confidence": round(regime_confidence, 4),
        },
    }


def compute_trend_score_from_prices(
    closes: Sequence[float],
    regime: str | None = None,
    regime_confidence: float = 0.5,
) -> float:
    """
    Convenience wrapper: returns just the trend score float.

    Suitable for injection into intent dicts.
    """
    result = compute_trend_score(
        closes,
        regime=regime,
        regime_confidence=regime_confidence,
    )
    return float(result["score"])
