from __future__ import annotations

"""
Adaptive maker offset engine (v5.10.3).

Produces a bounded maker offset (bps) informed by router effectiveness,
volatility regime, and time-of-day expectancy buckets.
"""

import datetime as dt
from typing import Dict

from execution.utils.metrics import router_effectiveness_7d
from execution.intel.expectancy_map import hourly_expectancy
from execution.utils.vol import atr_pct

BASELINE_BPS = 2.0
MIN_OFFSET_BPS = 0.5
MAX_OFFSET_BPS = 8.0


def classify_atr_regime(symbol: str) -> str:
    """Return 'quiet', 'normal', 'hot', or 'panic' based on ATR ratio."""
    short = atr_pct(symbol, 50)
    long = atr_pct(symbol, 500)
    if long <= 0:
        return "normal"
    ratio = short / long
    if ratio < 0.5:
        return "quiet"
    if ratio < 1.5:
        return "normal"
    if ratio < 2.5:
        return "hot"
    return "panic"


def _current_hour() -> int:
    return dt.datetime.utcnow().hour


def suggest_maker_offset_bps(symbol: str) -> float:
    """
    Suggest a signed offset in bps relative to mid for maker orders.
    Higher => further from mid (more conservative).
    Lower => closer to mid (more aggressive).
    """

    eff = router_effectiveness_7d(symbol) or {}
    fallback = float(eff.get("fallback_ratio") or 0.0)
    maker_fill = float(eff.get("maker_fill_ratio") or 0.0)
    slip_med = float(eff.get("slip_q50") or 0.0)

    regime = classify_atr_regime(symbol)
    offset = BASELINE_BPS

    if regime == "quiet":
        offset -= 0.5
    elif regime == "hot":
        offset += 0.5
    elif regime == "panic":
        offset += 1.0

    if maker_fill > 0.7:
        offset -= 0.5
    if fallback > 0.5:
        offset += 1.0
    if fallback > 0.8:
        offset += 1.5

    if slip_med > 5.0:
        offset += 0.8
    elif slip_med < 1.0:
        offset -= 0.2

    try:
        hourly = hourly_expectancy(symbol) or {}
    except Exception:
        hourly = {}
    bucket = hourly.get(_current_hour()) or {}
    expect = bucket.get("exp_per_notional")
    try:
        exp_val = float(expect)
    except (TypeError, ValueError):
        exp_val = None
    if exp_val is not None:
        if exp_val > 0:
            offset -= 0.2
        elif exp_val < 0:
            offset += 0.2

    return max(MIN_OFFSET_BPS, min(offset, MAX_OFFSET_BPS))


__all__ = ["classify_atr_regime", "suggest_maker_offset_bps"]
