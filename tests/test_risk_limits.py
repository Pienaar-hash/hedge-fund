from execution.risk_limits import (
    RiskConfig,
    RiskState,
    RiskGate,
    can_open_position,
    check_order,
    clamp_order_size,
    should_reduce_positions,
)


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


# ---- New guardrail tests ----


def _base_cfg():
    return {
        "global": {
            "whitelist": ["BTCUSDT"],
            "min_notional_usdt": 25.0,
            "daily_loss_limit_pct": 3.0,
            "max_trade_nav_pct": 10.0,
        },
        "per_symbol": {
            "BTCUSDT": {"min_notional": 25.0, "max_order_notional": 50.0},
            "ETHUSDT": {"min_notional": 5.0, "max_order_notional": 20.0},
        },
    }


def test_guardrail_not_whitelisted_blocks():
    st = RiskState()
    ok, details = check_order(
        symbol="ETHUSDT",
        side="BUY",
        requested_notional=50.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=_base_cfg(),
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "not_whitelisted" in details.get("reasons", [])


def test_guardrail_below_min_notional_blocks():
    st = RiskState()
    ok, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=10.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=_base_cfg(),
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "below_min_notional" in details.get("reasons", [])


def test_guardrail_symbol_cap_blocks():
    st = RiskState()
    cfg = _base_cfg()
    cfg["per_symbol"]["BTCUSDT"]["max_order_notional"] = 20.0
    ok, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=50.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "symbol_cap" in details.get("reasons", [])


def test_guardrail_day_loss_limit_blocks():
    st = RiskState()
    st.daily_pnl_pct = -3.5
    ok, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=25.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=_base_cfg(),
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "day_loss_limit" in details.get("reasons", [])


def test_guardrail_cooldown_blocks():
    st = RiskState()
    cfg = _base_cfg()
    # add a per-symbol cooldown
    cfg["per_symbol"]["BTCUSDT"]["cooldown_sec"] = 120
    now = 10_000.0
    # last fill 60s ago -> still in cooldown
    st.note_fill("BTCUSDT", now - 60)
    ok, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=25.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=now,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "cooldown" in details.get("reasons", [])
    assert isinstance(details.get("cooldown_until"), float)


def test_guardrail_error_circuit_breaker_blocks():
    st = RiskState()
    cfg = _base_cfg()
    cfg["global"]["error_circuit"] = {"max_errors": 3, "window_sec": 60}
    now = 50_000.0
    # 3 errors within window
    st.note_error(now - 10)
    st.note_error(now - 5)
    st.note_error(now - 1)
    ok, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=30.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=now,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
    )
    assert not ok
    assert "circuit_breaker" in details.get("reasons", [])


def test_risk_gate_symbol_cap_blocks_when_limit_exceeded():
    gate = RiskGate({
        "sizing": {"max_symbol_exposure_pct": 20.0},
        "risk": {},
    })

    class DummySnapshot:
        def __init__(self):
            self.nav = 1000.0
            self.gross_map = {"BTCUSDT": 190.0}

        def refresh(self) -> None:
            return None

        def current_nav_usd(self) -> float:
            return self.nav

        def current_gross_usd(self) -> float:
            return sum(self.gross_map.values())

        def symbol_gross_usd(self):
            return dict(self.gross_map)

    gate.nav_provider = DummySnapshot()

    allowed, reason = gate.allowed_gross_notional("BTCUSDT", 50.0)
    assert not allowed
    assert reason == "symbol_cap"


def test_risk_gate_symbol_cap_allows_within_limit():
    gate = RiskGate({
        "sizing": {"max_symbol_exposure_pct": 25.0},
        "risk": {},
    })

    class DummySnapshot:
        def __init__(self):
            self.nav = 1000.0
            self.gross_map = {"ETHUSDT": 100.0}

        def refresh(self) -> None:
            return None

        def current_nav_usd(self) -> float:
            return self.nav

        def current_gross_usd(self) -> float:
            return sum(self.gross_map.values())

        def symbol_gross_usd(self):
            return dict(self.gross_map)

    gate.nav_provider = DummySnapshot()
    allowed, reason = gate.allowed_gross_notional("ETHUSDT", 120.0)
    assert allowed
    assert reason == ""


def test_micro_notional_lev_included_allows_10_blocks_9():
    st = RiskState()
    cfg = {
        "global": {
            "whitelist": ["BTCUSDT"],
            "min_notional_usdt": 10.0,
            "max_portfolio_gross_nav_pct": 100.0,
            "max_leverage": 20,
        },
        "per_symbol": {"BTCUSDT": {"max_order_notional": 25.0, "max_leverage": 20}},
    }
    # Gross notional is passed (cap), leverage separately
    ok, d = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=10.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=20.0,
    )
    assert ok, d

    ok2, d2 = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=9.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=20.0,
    )
    assert not ok2
    assert "below_min_notional" in d2.get("reasons", [])
