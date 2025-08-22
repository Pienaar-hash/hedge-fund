import math

# Local copy of the rounding logic to avoid network calls in tests
def round_qty_to_step(qty: float, step: float) -> float:
    return math.floor(qty / step) * step

def ensure_filters(qty: float, px: float, step: float, min_qty: float, min_notional: float):
    # round to step
    q = round_qty_to_step(qty, step)
    if q < min_qty:
        q = min_qty
    if min_notional > 0 and q * px < min_notional:
        need = math.ceil(min_notional / max(px, 1e-12) / step) * step
        q = max(q, need)
    # final round-down to step (safety)
    q = round_qty_to_step(q, step)
    return q

def test_round_down_step():
    assert round_qty_to_step(1.234, 0.001) == 1.234
    assert round_qty_to_step(1.2349, 0.001) == 1.234
    assert round_qty_to_step(5.0, 1.0) == 5.0
    assert round_qty_to_step(5.9, 1.0) == 5.0

def test_min_qty_enforced():
    q = ensure_filters(qty=0.2, px=100, step=0.1, min_qty=0.5, min_notional=0)
    assert q == 0.5

def test_min_notional_enforced():
    # need notional >= 5; px=2 => qty >= 2.5 -> rounded up to step=0.1 => 2.5
    q = ensure_filters(qty=1.0, px=2.0, step=0.1, min_qty=0.1, min_notional=5.0)
    assert math.isclose(q, 2.5)

def test_all_constraints_together():
    # step=1, min_qty=1, min_notional=10, px=3 => need qty>=4 -> step OK
    q = ensure_filters(qty=0.2, px=3.0, step=1.0, min_qty=1.0, min_notional=10.0)
    assert q == 4.0
