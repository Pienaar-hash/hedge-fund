"""
fixed_tp_pct: Optional[float] = None,
trail: Optional[float] = None, # trail distance (abs price) or None
) -> SLTP:
side = side.upper()
if side not in ("LONG", "SHORT"):
raise ValueError("side must be LONG or SHORT")


# Base deltas
sl_delta = None
tp_delta = None


if atr and atr_mult:
sl_delta = atr * atr_mult
tp_delta = atr * atr_mult
if fixed_sl_pct:
sl_delta = max(sl_delta or 0, entry_px * (fixed_sl_pct / 100.0))
if fixed_tp_pct:
tp_delta = max(tp_delta or 0, entry_px * (fixed_tp_pct / 100.0))


if sl_delta is None or tp_delta is None:
raise ValueError("Provide atr×mult and/or fixed %s for both SL and TP")


if side == "LONG":
sl_px = entry_px - sl_delta
tp_px = entry_px + tp_delta
else: # SHORT
sl_px = entry_px + sl_delta
tp_px = entry_px - tp_delta


trail_px = None
if trail and trail > 0:
trail_px = trail


return SLTP(sl_px=round(sl_px, 8), tp_px=round(tp_px, 8), trail_px=trail_px)




def should_exit(
prices: Sequence[float],
entry_px: float,
side: str,
sl_px: float,
tp_px: float,
max_bars: int = 0,
trail_px: Optional[float] = None,
) -> Tuple[bool, str]:
"""Return (exit?, reason) given the latest price path.


Reasons: "hit_sl", "hit_tp", "time_stop", "trail_stop", "hold".
"""
side = side.upper()
if not prices:
return False, "hold"
last = prices[-1]


if side == "LONG":
if last <= sl_px:
return True, "hit_sl"
if last >= tp_px:
return True, "hit_tp"
if trail_px:
# trailing from highest close since entry
peak = max(prices)
if last <= peak - trail_px:
return True, "trail_stop"
else: # SHORT
if last >= sl_px:
return True, "hit_sl"
if last <= tp_px:
return True, "hit_tp"
if trail_px:
trough = min(prices)
if last >= trough + trail_px:
return True, "trail_stop"


if max_bars and len(prices) >= max_bars:
return True, "time_stop"


return False, "hold"