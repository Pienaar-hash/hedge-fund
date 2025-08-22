import math
from execution.rules_sl_tp import compute_sl_tp, should_exit

def test_compute_sl_tp_atr_long():
    out = compute_sl_tp(entry_px=100.0, side="LONG", atr=2.0, atr_mult_sl=2.5, atr_mult_tp=3.5)
    assert math.isclose(out["sl_px"], 95.0)
    assert math.isclose(out["tp_px"], 107.0)

def test_compute_sl_tp_fixed_short():
    out = compute_sl_tp(entry_px=200.0, side="SHORT", atr=None, fixed_sl_pct=0.01, fixed_tp_pct=0.02)
    assert math.isclose(out["sl_px"], 202.0)  # short SL above entry
    assert math.isclose(out["tp_px"], 196.0)  # short TP below entry

def test_should_exit_hits():
    sl,tp = 99.0, 105.0
    exit1, why1 = should_exit([100,101,98.9], 100, "LONG", sl, tp, max_bars=100)
    exit2, why2 = should_exit([100,101,105.1], 100, "LONG", sl, tp, max_bars=100)
    assert exit1 and why1=="hit_sl"
    assert exit2 and why2=="hit_tp"

def test_time_stop():
    exit3, why3 = should_exit([1]*96, 1, "LONG", 0.9, 1.1, max_bars=96)
    assert exit3 and why3=="time_stop"
