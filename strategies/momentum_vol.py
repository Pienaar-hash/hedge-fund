from __future__ import annotations

from typing import Dict, List, Tuple


def ema(arr: List[float], period: int) -> float:
    if not arr or period <= 1 or len(arr) < period:
        return float(arr[-1]) if arr else 0.0
    k = 2.0 / (period + 1)
    e = arr[-period]
    for x in arr[-period + 1 :]:
        e = x * k + e * (1 - k)
    return float(e)


def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
    if not (high and low and close) or len(close) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    # simple moving average of TR
    window = trs[-period:]
    return sum(window) / float(period)


def momentum_vol_signal(
    closes: List[float], highs: List[float], lows: List[float], atr_cap_frac: float = 0.02
) -> Tuple[bool, Dict[str, float]]:
    """Return (emit, info) based on EMA(20/100) trend and ATR cap.
    info = {"ema20":..., "ema100":..., "atr":..., "atr_cap_frac":...}
    """
    e20 = ema(closes, 20)
    e100 = ema(closes, 100)
    a = atr(highs, lows, closes, 14)
    info = {"ema20": e20, "ema100": e100, "atr": a, "atr_cap_frac": atr_cap_frac}
    if e20 <= 0 or e100 <= 0:
        return False, info
    trend_ok = e20 > e100
    # price proxy = last close
    px = float(closes[-1]) if closes else 0.0
    atr_frac = (a / px) if px else 0.0
    vol_ok = atr_frac <= atr_cap_frac
    return bool(trend_ok and vol_ok), info

