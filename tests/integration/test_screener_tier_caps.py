import json
import pytest
import execution.risk_limits as risk_limits
from execution.risk_limits import RiskState, check_order
from execution import risk_loader
import os
from pathlib import Path


def _risk_cfg(tmp_path):
    cfg = {
        "global": {
            "min_notional_usdt": 10.0,
            "max_portfolio_gross_nav_pct": 15.0,
            "max_concurrent_positions": 3,
            "trade_equity_nav_pct": 0.15,
            "max_trade_nav_pct": 0.2,
            "tiers": {
                "CORE": {"per_symbol_nav_pct": 8.0},
                "SATELLITE": {"per_symbol_nav_pct": 4.0},
                "TACTICAL": {"per_symbol_nav_pct": 2.0},
                "ALT-EXT": {"per_symbol_nav_pct": 1.0},
            },
        },
        "per_symbol": {"BTCUSDT": {"max_order_notional": 25.0, "max_nav_pct": 100.0}},
    }
    p = tmp_path / "risk_limits.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _tiers_cfg(tmp_path):
    tiers = {"CORE": ["BTCUSDT"], "SATELLITE": ["ETHUSDT"]}
    p = tmp_path / "tiers.json"
    p.write_text(json.dumps(tiers))
    return str(p)


def _stub_nav(monkeypatch, nav_value: float):
    monkeypatch.setattr(
        risk_limits,
        "nav_health_snapshot",
        lambda threshold_s=None: {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value},
    )


def reload_cfg():
    path = os.getenv("RISK_LIMITS_CONFIG")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def test_not_listed_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    # route configs to temp
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 1000.0)

    # Force not listed for FOOUSDT
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: False if s == "FOOUSDT" else True, raising=True)
    ok, reasons, _ = sc.would_emit("FOOUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "not_listed" in reasons


def test_portfolio_cap_veto(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 2000.0)
    st = RiskState()
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=200.0,
        price=0.0,
        nav=2000.0,
        open_qty=0.0,
        now=0.0,
        cfg=reload_cfg(),
        state=st,
        current_gross_notional=299.0,
    )
    assert veto is True
    assert "portfolio_cap" in details.get("reasons", [])


def test_tier_cap_veto(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 2000.0)
    st = RiskState()
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=200.0,
        price=0.0,
        nav=2000.0,
        open_qty=0.0,
        now=0.0,
        cfg=reload_cfg(),
        state=st,
        current_gross_notional=0.0,
        tier_name="CORE",
        current_tier_gross_notional=159.0,
    )
    assert veto is True
    assert "tier_cap" in details.get("reasons", [])


def test_max_concurrent_positions_veto(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 2000.0)
    st = RiskState()
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=200.0,
        price=0.0,
        nav=2000.0,
        open_qty=0.0,
        now=0.0,
        cfg=reload_cfg(),
        state=st,
        open_positions_count=3,
    )
    assert veto is True
    assert "max_concurrent" in details.get("reasons", [])


def test_orderbook_adverse_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    from execution import orderbook_features as ob
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(ob, "topn_imbalance", lambda _s, limit=20: -0.5, raising=True)
    _stub_nav(monkeypatch, 1000.0)
    ok, reasons, _ = sc.would_emit("BTCUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "ob_adverse" in reasons


def test_trade_equity_nav_veto(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 1000.0)
    st = RiskState()
    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=160.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=reload_cfg(),
        state=st,
    )
    assert veto is True
    assert "trade_gt_equity_cap" in detail.get("reasons", [])
    thresholds = detail.get("thresholds", {})
    observations = detail.get("observations", {})
    assert pytest.approx(thresholds.get("trade_equity_nav_pct")) == 0.15
    assert observations.get("trade_equity_nav_obs") > 0.15


def test_trade_max_nav_veto(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    _stub_nav(monkeypatch, 1000.0)
    st = RiskState()
    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=250.0,
        price=0.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=reload_cfg(),
        state=st,
    )
    assert veto is True
    reasons = detail.get("reasons", [])
    assert "max_trade_nav_pct" in reasons
    assert "trade_gt_equity_cap" in reasons
    thresholds = detail.get("thresholds", {})
    observations = detail.get("observations", {})
    assert pytest.approx(thresholds.get("max_trade_nav_pct")) == 0.20
    assert observations.get("max_trade_nav_obs") > 0.20
