import json


def _risk_cfg(tmp_path):
    cfg = {
        "global": {
            "min_notional_usdt": 10.0,
            "max_portfolio_gross_nav_pct": 15.0,
            "max_concurrent_positions": 3,
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
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "0")

    # Force not listed for FOOUSDT
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: False if s == "FOOUSDT" else True, raising=True)
    ok, reasons, _ = sc.would_emit("FOOUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "not_listed" in reasons


def test_portfolio_cap_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "0")
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(sc, "_score_symbol", lambda *a, **k: {"p": 1.0}, raising=True)
    # Portfolio cap 15% of 1000 => 150. Current gross=149, request gross=10*20=200 -> block
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=1000.0,
        current_gross_notional=149.0,
    )
    assert ok is False
    assert ("portfolio_cap" in reasons) or ("max_gross_nav_pct" in reasons)


def test_tier_cap_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "0")
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(sc, "_score_symbol", lambda *a, **k: {"p": 1.0}, raising=True)
    # CORE per-symbol cap 8% of 1000 => 80. current tier gross=79, req gross=200 -> block
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=1000.0,
        current_tier_gross_notional=79.0,
    )
    assert ok is False
    assert "tier_cap" in reasons


def test_max_concurrent_positions_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "0")
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(sc, "_score_symbol", lambda *a, **k: {"p": 1.0}, raising=True)
    ok, reasons, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=1000.0,
        open_positions_count=3,
    )
    assert ok is False
    assert "max_concurrent" in reasons


def test_orderbook_adverse_veto(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    from execution import orderbook_features as ob
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "1")
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(ob, "topn_imbalance", lambda _s, limit=20: -0.5, raising=True)
    ok, reasons, _ = sc.would_emit("BTCUSDT", "BUY", notional=10.0, lev=20.0, nav=1000.0)
    assert ok is False
    assert "ob_adverse" in reasons


def test_would_emit_nav_non_positive(monkeypatch, tmp_path):
    from execution import signal_screener as sc
    monkeypatch.setenv("RISK_LIMITS_CONFIG", _risk_cfg(tmp_path))
    monkeypatch.setenv("SYMBOL_TIERS_CONFIG", _tiers_cfg(tmp_path))
    monkeypatch.setenv("ORDERBOOK_GATE_ENABLED", "0")
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda s: True, raising=True)
    monkeypatch.setattr(sc, "_score_symbol", lambda *a, **k: {"p": 1.0}, raising=True)

    ok_zero, reasons_zero, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=0.0,
    )
    assert not ok_zero
    assert reasons_zero == ["nav_non_positive"]

    ok_none, reasons_none, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=None,
    )
    assert not ok_none
    assert reasons_none == ["nav_non_positive"]
