#!/usr/bin/env python3
"""
Pure SL/TP rules for the hedge bot.

Interfaces:
- compute_sl_tp(entry_px, side, atr, atr_mult, fixed_sl_pct, fixed_tp_pct, trail=None)
- should_exit(prices, entry_px, side, sl_px, tp_px, max_bars, trail=None)

Conventions:
- `side`: "LONG" or "SHORT".
- `atr`: fraction of price (e.g., 0.0025 == 0.25%).
- `fixed_sl_pct` / `fixed_tp_pct` may be given either as percent (e.g., 0.6 => 0.6%)
  **or** as a fraction (e.g., 0.006). We auto-normalize:
    * <= 0.05  -> treated as fraction (<= 5%)
    *  > 0.05  -> treated as percent and divided by 100
"""

from typing import Dict, Iterable, Optional, Tuple


def _normalize_pct(x: float) -> float:
    """
    Interprets values <= 0.05 as already-fractions (<= 5%),
    and larger values as percents that should be divided by 100.
    """
    x = float(x or 0.0)
    if x <= 0.0:
        return 0.0
    if x <= 0.05:
        return x  # already a fraction (e.g., 0.006 => 0.6%)
    return x / 100.0  # percent (e.g., 0.6 => 0.006)


def compute_sl_tp(
    entry_px: float,
    side: str,
    atr: float = 0.0,
    atr_mult: float = 0.0,
    fixed_sl_pct: float = 0.0,
    fixed_tp_pct: float = 0.0,
    trail: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Returns (sl_px, tp_px). No tick-size rounding here.
    """
    side = side.upper()

    sl_pct = _normalize_pct(fixed_sl_pct)
    tp_pct = _normalize_pct(fixed_tp_pct)

    atr = float(atr or 0.0)  # ATR already a fraction
    atr_mult = float(atr_mult or 0.0)
    if atr > 0.0 and atr_mult > 0.0:
        atr_component = atr * atr_mult
        sl_pct = max(sl_pct, atr_component)
        tp_pct = max(tp_pct, atr_component)

    if side == "LONG":
        sl_px = entry_px * (1.0 - sl_pct) if sl_pct > 0 else entry_px
        tp_px = entry_px * (1.0 + tp_pct) if tp_pct > 0 else entry_px
    elif side == "SHORT":
        sl_px = entry_px * (1.0 + sl_pct) if sl_pct > 0 else entry_px
        tp_px = entry_px * (1.0 - tp_pct) if tp_pct > 0 else entry_px
    else:
        raise ValueError("side must be 'LONG' or 'SHORT'")
    return float(sl_px), float(tp_px)


def should_exit(
    prices: Iterable[float],
    entry_px: float,
    side: str,
    sl_px: float,
    tp_px: float,
    max_bars: int,
    trail: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Evaluate exit conditions on a chronological sequence of prices.
    True if TP/SL/TRAIL/TIME condition is met.
    """
    side = side.upper()
    it = [float(x) for x in prices]
    if not it:
        return False
    last = it[-1]

    # Hard TP/SL
    if side == "LONG":
        if last >= tp_px:
            return True  # take profit
        if last <= sl_px:
            return True  # stop loss
    elif side == "SHORT":
        if last <= tp_px:
            return True
        if last >= sl_px:
            return True
    else:
        raise ValueError("side must be 'LONG' or 'SHORT'")

    # Trailing stop (optional)
    if trail and float(trail.get("width_pct", 0)) > 0:
        w = float(trail["width_pct"])
        if side == "LONG":
            peak = max(it)
            trail_px = peak * (1.0 - w)
            if last <= trail_px:
                return True
        else:  # SHORT
            trough = min(it)
            trail_px = trough * (1.0 + w)
            if last >= trail_px:
                return True

    # Time stop
    if max_bars and len(it) >= int(max_bars):
        return True

    return False
