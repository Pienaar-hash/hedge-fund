from __future__ import annotations

from execution.utils.vol import atr_pct, rolling_sigma

ATR_MED_LOOKBACK = 500


def inverse_vol_size(symbol: str, base_size: float, lookback: int = 50) -> float:
    sigma = rolling_sigma(symbol, lookback=lookback) or 0.0
    sigma = max(float(sigma), 1e-6)
    return float(base_size) * (1.0 / sigma)


def volatility_regime_scale(symbol: str) -> float:
    """
    Simple ATR regime classifier to taper sizing during volatile regimes.
    """
    atr_now = atr_pct(symbol, lookback_bars=50)
    atr_med = atr_pct(symbol, lookback_bars=ATR_MED_LOOKBACK)
    if atr_med <= 0:
        return 1.0
    ratio = atr_now / atr_med
    if ratio < 0.7:
        return 0.75
    if ratio < 1.5:
        return 1.0
    if ratio < 2.5:
        return 0.5
    return 0.25
