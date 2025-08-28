import math
from execution.risk_limits import RiskConfig, RiskState, can_open_position, should_reduce_positions, clamp_order_size

def test_can_open_position_happy_path():
    cfg = RiskConfig(200.0, 1000.0, 5, 5.0, -10.0, 10.0)
    st = RiskState()
    st.open_notional = 100.0
    st.open_positions = 1
    st.portfolio_drawdown_pct = -2.0
    ok, reason = can_open_position("BTCUSDT", 150.0, 3.0, cfg, st)
    assert ok and reason == "ok"

def test_kill_switch_blocks():
    cfg = RiskConfig(200.0, 1000.0, 5, 5.0, -10.0, 10.0)
    st = RiskState()
    st.portfolio_drawdown_pct = -12.0
    ok, reason = can_open_position("BTCUSDT", 50.0, 2.0, cfg, st)
    assert not ok and reason == "kill_switch_triggered"
    assert should_reduce_positions(st, cfg)

def test_per_trade_cap():
    cfg = RiskConfig(100.0, 1000.0, 5, 5.0, -10.0, 10.0)
    st = RiskState()
    st.portfolio_drawdown_pct = -1.0
    ok, reason = can_open_position("BTCUSDT", 150.0, 2.0, cfg, st)
    assert not ok and reason == "exceeds_per_trade_cap"

def test_open_notional_cap():
    cfg = RiskConfig(200.0, 300.0, 5, 5.0, -10.0, 10.0)
    st = RiskState()
    st.open_notional = 200.0
    ok, reason = can_open_position("BTCUSDT", 120.0, 2.0, cfg, st)
    assert not ok and reason == "exceeds_open_notional_cap"

def test_clamp_order_size():
    assert clamp_order_size(0.12349, 0.0001) == 0.1234
    assert clamp_order_size(5.0, 0.01) == 5.0
    assert clamp_order_size(0.009, 0.01) == 0.0
