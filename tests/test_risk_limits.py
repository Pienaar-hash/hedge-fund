from typing import Any, Dict

import pytest

from execution import risk_limits as risk_limits_module
from execution.risk_limits import (
    RiskConfig,
    RiskState,
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
            "max_trade_nav_pct": 0.2,
            "trade_equity_nav_pct": 0.15,
            "nav_freshness_seconds": 1_000_000,
        },
        "per_symbol": {
            "BTCUSDT": {
                "min_notional": 25.0,
                "max_order_notional": 50.0,
                "max_nav_pct": 0.25,
                "max_leverage": 4,
            },
            "ETHUSDT": {
                "min_notional": 5.0,
                "max_order_notional": 20.0,
                "max_nav_pct": 0.2,
                "max_leverage": 4,
            },
        },
    }


def test_check_order_returns_detail_on_nav_warning(monkeypatch):
    st = RiskState()
    cfg = _base_cfg()
    cfg["global"]["nav_freshness_seconds"] = 5
    cfg["global"]["fail_closed_on_nav_stale"] = False

    monkeypatch.setattr(
        risk_limits_module,
        "get_nav_freshness_snapshot",
        lambda: (10.0, True),
    )

    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=30.0,
        price=1000.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
    )

    assert veto is False
    assert isinstance(detail, dict)
    assert detail.get("nav_fresh") in {False, True}


def _fresh_nav(monkeypatch, nav_value: float):
    nav_snapshot = {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value}
    monkeypatch.setattr(risk_limits_module, "nav_health_snapshot", lambda threshold_s=None: dict(nav_snapshot))
    monkeypatch.setattr(risk_limits_module, "get_nav_freshness_snapshot", lambda: (0.0, True))


def test_trade_caps_allow_moderate_notional(monkeypatch):
    _fresh_nav(monkeypatch, 4400.0)
    st = RiskState()
    cfg = _base_cfg()
    cfg["per_symbol"]["BTCUSDT"]["max_order_notional"] = 5_000.0
    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=400.0,
        price=0.0,
        nav=4400.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=3.0,
    )
    assert veto is False
    reasons = detail.get("reasons") or []
    assert "max_trade_nav_pct" not in reasons
    assert "trade_gt_equity_cap" not in reasons


def test_trade_equity_clamp_triggers_between_15_and_20_pct(monkeypatch):
    _fresh_nav(monkeypatch, 4400.0)
    st = RiskState()
    cfg = _base_cfg()
    cfg["per_symbol"]["BTCUSDT"]["max_order_notional"] = 10_000.0
    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=800.0,
        price=0.0,
        nav=4400.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=2.0,
    )
    assert veto is True
    reasons = detail.get("reasons") or []
    assert "trade_gt_equity_cap" in reasons
    assert "max_trade_nav_pct" not in reasons
    thresholds = detail.get("thresholds", {})
    observations = detail.get("observations", {})
    expected_pct = (800.0 / 4400.0)
    assert pytest.approx(thresholds.get("trade_equity_nav_pct")) == 0.15
    assert pytest.approx(thresholds.get("max_trade_nav_pct")) == 0.20
    assert pytest.approx(observations.get("trade_equity_nav_obs")) == expected_pct
    assert pytest.approx(observations.get("max_trade_nav_obs")) == expected_pct


def test_trade_max_trade_nav_pct_triggers(monkeypatch):
    _fresh_nav(monkeypatch, 4400.0)
    st = RiskState()
    cfg = _base_cfg()
    cfg["per_symbol"]["BTCUSDT"]["max_order_notional"] = 20_000.0
    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=1_200.0,
        price=0.0,
        nav=4400.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=4.0,
    )
    assert veto is True
    reasons = detail.get("reasons") or []
    assert "max_trade_nav_pct" in reasons
    thresholds = detail.get("thresholds", {})
    observations = detail.get("observations", {})
    expected_pct = (1200.0 / 4400.0)
    assert pytest.approx(thresholds.get("trade_equity_nav_pct")) == 0.15
    assert pytest.approx(thresholds.get("max_trade_nav_pct")) == 0.20
    assert pytest.approx(observations.get("trade_equity_nav_obs")) == expected_pct
    assert pytest.approx(observations.get("max_trade_nav_obs")) == expected_pct


def test_guardrail_not_whitelisted_blocks():
    st = RiskState()
    veto, details = check_order(
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
    assert veto is True
    assert "not_whitelisted" in details.get("reasons", [])


def test_guardrail_below_min_notional_blocks():
    st = RiskState()
    veto, details = check_order(
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
    assert veto is True
    assert "min_notional" in details.get("reasons", [])


def test_guardrail_symbol_cap_blocks():
    st = RiskState()
    cfg = _base_cfg()
    cfg["per_symbol"]["BTCUSDT"]["max_order_notional"] = 20.0
    veto, details = check_order(
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
    assert veto is True
    assert "symbol_cap" in details.get("reasons", [])


def test_guardrail_global_leverage_cap_blocks_without_symbol_override():
    st = RiskState()
    cfg = _base_cfg()
    cfg["global"]["max_leverage"] = 4
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=30.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=5.0,
    )
    assert veto is True
    assert "leverage_exceeded" in details.get("reasons", [])


def test_symbol_notional_guard_respects_cfg(monkeypatch):
    monkeypatch.setattr(risk_limits_module, "is_in_asset_universe", lambda s: True, raising=False)
    monkeypatch.setattr(risk_limits_module, "total_notional_7d", lambda: 100.0, raising=False)
    monkeypatch.setattr(risk_limits_module, "notional_7d_by_symbol", lambda s: 30.0, raising=False)
    cfg = {"global": {"symbol_notional_share_cap_pct": 20.0}}
    assert risk_limits_module.symbol_notional_guard("BTCUSDT", cfg) is False
    cfg["per_symbol"] = {"BTCUSDT": {"symbol_notional_share_cap_pct": 40.0}}
    assert risk_limits_module.symbol_notional_guard("BTCUSDT", cfg) is True


def test_symbol_dd_guard_respects_cfg(monkeypatch):
    monkeypatch.setattr(risk_limits_module, "is_symbol_disabled", lambda s: False, raising=False)
    monkeypatch.setattr(risk_limits_module, "is_in_asset_universe", lambda s: True, raising=False)
    dd_values = iter([-2.0, -5.0])
    monkeypatch.setattr(risk_limits_module, "dd_today_pct", lambda s: next(dd_values), raising=False)
    captured: Dict[str, Any] = {}

    def fake_disable(symbol, ttl_hours=0, reason=""):
        captured["symbol"] = symbol
        captured["reason"] = reason

    monkeypatch.setattr(
        risk_limits_module,
        "disable_symbol_temporarily",
        fake_disable,
        raising=False,
    )
    cfg = {"global": {"symbol_drawdown_cap_pct": 4.0}}
    assert risk_limits_module.symbol_dd_guard("BTCUSDT", cfg) is True
    assert risk_limits_module.symbol_dd_guard("BTCUSDT", cfg) is False
    assert captured.get("symbol") == "BTCUSDT"


def test_guardrail_symbol_leverage_override_takes_precedence():
    st = RiskState()
    cfg = _base_cfg()
    cfg["global"]["max_leverage"] = 5
    cfg["per_symbol"]["BTCUSDT"]["max_leverage"] = 3
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=30.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=4.0,
    )
    assert veto is True
    assert "leverage_exceeded" in details.get("reasons", [])
    veto2, details2 = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=30.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=2.0,
    )
    assert not veto2, details2


def test_guardrail_day_loss_limit_blocks():
    st = RiskState()
    st.daily_pnl_pct = -3.5
    veto, details = check_order(
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
    assert veto is True
    assert "day_loss_limit" in details.get("reasons", [])


def test_guardrail_cooldown_blocks():
    st = RiskState()
    cfg = _base_cfg()
    # add a per-symbol cooldown
    cfg["per_symbol"]["BTCUSDT"]["cooldown_sec"] = 120
    now = 10_000.0
    # last fill 60s ago -> still in cooldown
    st.note_fill("BTCUSDT", now - 60)
    veto, details = check_order(
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
    assert veto is True
    assert "cooldown" in details.get("reasons", [])
    cooldown_until = details.get("observations", {}).get("cooldown_until")
    assert isinstance(cooldown_until, float)


def test_guardrail_error_circuit_breaker_blocks():
    st = RiskState()
    cfg = _base_cfg()
    cfg["global"]["error_circuit"] = {"max_errors": 3, "window_sec": 60}
    now = 50_000.0
    # 3 errors within window
    st.note_error(now - 10)
    st.note_error(now - 5)
    st.note_error(now - 1)
    veto, details = check_order(
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
    assert veto is True
    assert "circuit_breaker" in details.get("reasons", [])


def test_risk_gate_is_noop_shim():
    from execution.risk_limits import RiskGate

    gate = RiskGate({})
    allowed, reason = gate.allowed_gross_notional("BTCUSDT", 50.0)
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
            "nav_freshness_seconds": 1_000_000,
        },
        "per_symbol": {"BTCUSDT": {"max_order_notional": 25.0, "max_leverage": 20}},
    }
    # Gross notional is passed (cap), leverage separately
    veto, d = check_order(
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
    assert veto is False, d

    veto2, d2 = check_order(
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
    assert veto2 is True
    assert "min_notional" in d2.get("reasons", [])
