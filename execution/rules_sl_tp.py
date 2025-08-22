"""
Pure SL/TP rules — no I/O. Import and call from screener/executor.
All prices are floats; 'side' is 'LONG' or 'SHORT'.
"""

from __future__ import annotations
from typing import Optional, Dict, Sequence, Tuple

def compute_sl_tp(
    entry_px: float,
    side: str,
    atr: Optional[float] = None,
    atr_mult_sl: Optional[float] = None,
    atr_mult_tp: Optional[float] = None,
    fixed_sl_pct: Optional[float] = None,
    fixed_tp_pct: Optional[float] = None,
) -> Dict[str, float]:
    """
    Returns {'sl_px': float, 'tp_px': float}. If ATR info missing, falls back to fixed pct.
    For LONG: SL below entry, TP above. For SHORT: inverse.
    """
    side = side.upper()
    assert side in ("LONG", "SHORT")
    def _pct(px, pct, up):
        return px * (1 + pct) if up else px * (1 - pct)

    use_atr = (atr is not None) and (atr > 0) and atr_mult_sl and atr_mult_tp
    if use_atr:
        sl_off = atr * float(atr_mult_sl)
        tp_off = atr * float(atr_mult_tp)
        if side == "LONG":
            return {"sl_px": entry_px - sl_off, "tp_px": entry_px + tp_off}
        else:
            return {"sl_px": entry_px + sl_off, "tp_px": entry_px - tp_off}
    else:
        # fallback to fixed pct
        sl_pct = abs(float(fixed_sl_pct or 0.01))
        tp_pct = abs(float(fixed_tp_pct or 0.015))
        if side == "LONG":
            return {"sl_px": _pct(entry_px, sl_pct, up=False), "tp_px": _pct(entry_px, tp_pct, up=True)}
        else:
            return {"sl_px": _pct(entry_px, sl_pct, up=True),  "tp_px": _pct(entry_px, tp_pct, up=False)}

def should_exit(
    closes: Sequence[float],
    entry_px: float,
    side: str,
    sl_px: float,
    tp_px: float,
    max_bars: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Simple bar-close exit logic:
    - Exit if last close breaches SL/TP for the given side.
    - Exit on time stop if max_bars exceeded.
    Returns (exit_bool, reason_str).
    """
    if not closes:
        return (False, "no_data")
    last = float(closes[-1])
    side = side.upper()
    if side == "LONG":
        if last <= sl_px:
            return (True, "hit_sl")
        if last >= tp_px:
            return (True, "hit_tp")
    else:
        if last >= sl_px:
            return (True, "hit_sl")
        if last <= tp_px:
            return (True, "hit_tp")
    if max_bars and len(closes) >= int(max_bars):
        return (True, "time_stop")
    return (False, "hold")
