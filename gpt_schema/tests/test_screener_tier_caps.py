import json
import pytest


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
        "per_symbol": {"BTCUSDT": {"max_order_notional": 25.0}},
    }
    p = tmp_path / "risk_limits.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _tiers_cfg(tmp_path):
    tiers = {"CORE": ["BTCUSDT"], "SATELLITE": ["ETHUSDT"]}
    p = tmp_path / "tiers.json"
    p.write_text(json.dumps(tiers))
    return str(p)


def test_not_listed_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    # route configs to temp
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))

    # Force not listed for FOOUSDT
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: False if s == "FOOUSDT" else True, raising=True)
    ok, reasons, _ = sc.would_emit("FOOUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "not_listed" in reasons


def test_portfolio_cap_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    # Portfolio cap 15% of 2000 => 300. Current gross=299, request gross=10*20=200 -> block
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=2000.0,
        current_gross_notional=299.0,
    )
    assert ok is False
    assert ("portfolio_cap" in reasons) or ("max_gross_nav_pct" in reasons)


def test_tier_cap_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    # CORE per-symbol cap 8% of 2000 => 160. current tier gross=159, req gross=200 -> block
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=2000.0,
        current_tier_gross_notional=159.0,
    )
    assert ok is False
    assert "tier_cap" in reasons


def test_max_concurrent_positions_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=2000.0,
        open_positions_count=3,
    )
    assert ok is False
    assert "max_concurrent" in reasons


def test_orderbook_adverse_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    from execution import orderbook_features as ob
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(ob, "topn_imbalance", lambda _s, limit=20: -0.5, raising=True)
    ok, reasons, _ = sc.would_emit("BTCUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "ob_adverse" in reasons


def test_trade_equity_nav_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc

    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    ok, reasons, extra = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=160.0,
        lev=1.0,
        nav=1000.0,
    )
    assert ok is False
    assert "trade_gt_10pct_equity" in reasons
    trade_meta = extra.get("trade_nav", {})
    assert pytest.approx(trade_meta.get("trade_equity_nav_pct")) == 15.0
    assert trade_meta.get("trade_equity_nav_obs") > 15.0


def test_trade_max_nav_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc

    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    ok, reasons, extra = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=250.0,
        lev=1.0,
        nav=1000.0,
    )
    assert ok is False
    assert "trade_gt_max_trade_nav_pct" in reasons
    assert "trade_gt_10pct_equity" in reasons
    trade_meta = extra.get("trade_nav", {})
    assert pytest.approx(trade_meta.get("max_trade_nav_pct")) == 20.0
    assert trade_meta.get("max_trade_nav_obs") > 20.0
