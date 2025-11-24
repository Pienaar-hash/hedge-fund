import time

import pytest

from execution import risk_limits as rl
from execution.risk_loader import normalize_percentage, load_risk_config


def test_normalize_percentage():
    assert normalize_percentage(0.25) == 0.25
    assert normalize_percentage(5.0) == 0.05


def test_symbol_cap_allows_within_limit(monkeypatch):
    monkeypatch.setattr(rl, "nav_health_snapshot", lambda *args, **kwargs: {"age_s": 0, "sources_ok": True, "fresh": True})
    monkeypatch.setattr(rl, "get_confirmed_nav", lambda: {"nav": 10000.0, "detail": {"breakdown": {"USDT": 10000.0}}})
    monkeypatch.setattr(
        rl,
        "_drawdown_snapshot",
        lambda g_cfg: {
            "drawdown": {"pct": 0.0, "peak_nav": 0.0, "nav": 0.0, "abs": 0.0},
            "daily_loss": {"pct": 0.0},
            "usable": True,
            "stale_flags": {},
            "nav_health": {},
            "peak_state": {},
            "assets": {},
        },
    )
    state = rl.RiskState()
    allowed, detail = rl.check_order(
        symbol="BTCUSDT",
        side="LONG",
        requested_notional=1000.0,
        price=20000.0,
        nav=10000.0,
        open_qty=0.0,
        now=time.time(),
        cfg=load_risk_config(),
        state=state,
        current_gross_notional=0.0,
    )
    assert allowed is True
    assert "symbol_cap" not in detail.get("reasons", [])


def test_check_order_respects_symbol_cap(monkeypatch):
    from execution import risk_limits as rl

    nav_total = 11196.18125646
    cfg = load_risk_config()
    # Force fresh NAV and clean drawdown state
    monkeypatch.setattr(rl, "nav_health_snapshot", lambda *_args, **_kwargs: {"age_s": 0, "sources_ok": True, "fresh": True})
    monkeypatch.setattr(
        rl,
        "get_confirmed_nav",
        lambda: {"nav": nav_total, "detail": {"breakdown": {"USDT": nav_total}}},
    )
    monkeypatch.setattr(
        rl,
        "_drawdown_snapshot",
        lambda _g: {
            "drawdown": {"pct": 0.0, "peak_nav": nav_total, "nav": nav_total, "abs": 0.0},
            "daily_loss": {"pct": 0.0},
            "usable": True,
            "stale_flags": {},
            "nav_health": {},
            "peak_state": {},
            "assets": {},
        },
    )
    g_cfg = cfg.get("global", {})
    trade_equity_pct = g_cfg.get("trade_equity_nav_pct", 0.0)
    requested_notional = nav_total * float(trade_equity_pct)

    allowed, detail = rl.check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=requested_notional,
        price=25000.0,
        nav=nav_total,
        open_qty=0.0,
        now=time.time(),
        cfg=cfg,
        state=rl.RiskState(),
        current_gross_notional=0.0,
        lev=g_cfg.get("max_leverage", 1.0),
        open_positions_count=0,
        tier_name=None,
        current_tier_gross_notional=0.0,
    )
    assert allowed is True  # allowed path -> True
    assert detail.get("reasons") == []


def test_check_order_returns_reasons_on_veto(monkeypatch):
    from execution import risk_limits as rl

    cfg = load_risk_config()
    monkeypatch.setattr(rl, "nav_health_snapshot", lambda *_args, **_kwargs: {"age_s": 9999, "sources_ok": False, "fresh": False})
    allowed, detail = rl.check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=1000.0,
        price=20000.0,
        nav=0.0,
        open_qty=0.0,
        now=time.time(),
        cfg=cfg,
        state=rl.RiskState(),
        current_gross_notional=0.0,
    )
    assert allowed is False
    assert detail.get("reasons")
