import pytest

from execution import risk_limits
from execution.risk_limits import RiskState, check_order


def _fresh_nav(monkeypatch, nav_value: float) -> None:
    monkeypatch.setattr(
        "execution.risk_limits.nav_health_snapshot",
        lambda threshold_s=None: {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value},
    )


def test_trade_equity_cap_applied_once(monkeypatch):
    _fresh_nav(monkeypatch, 1_000.0)
    cfg = {"global": {"trade_equity_nav_pct": 0.01, "max_trade_nav_pct": 0.02, "min_notional_usdt": 5.0}}
    st = RiskState()
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=15.0,  # 1.5% of nav
        price=30_000.0,
        nav=1_000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=1.0,
        open_positions_count=0,
        tier_name=None,
        current_tier_gross_notional=0.0,
    )
    assert veto is True
    # Only one cap reason should be emitted, and it can use either the legacy
    # equity-cap label or the standardized max-trade label.
    assert len(details["reasons"]) == 1
    assert details["reasons"][0] in {"trade_gt_equity_cap", "max_trade_nav_pct", "max_trade_nav"}


def test_nav_stale_veto(monkeypatch):
    monkeypatch.setattr(
        "execution.risk_limits.nav_health_snapshot",
        lambda threshold_s=None: {"age_s": 9999.0, "sources_ok": False, "fresh": False, "nav_total": 0.0},
    )
    cfg = {"global": {"min_notional_usdt": 5.0}}
    st = RiskState()
    veto, details = check_order(
        symbol="ETHUSDT",
        side="BUY",
        requested_notional=10.0,
        price=1800.0,
        nav=0.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        lev=1.0,
        open_positions_count=0,
        tier_name=None,
        current_tier_gross_notional=0.0,
    )
    assert veto is True
    assert "nav_stale" in details.get("reasons", [])
