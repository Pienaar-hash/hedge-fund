from execution.utils.execution_alerts import classify_alerts


def _types(snapshot):
    alerts = classify_alerts(snapshot)
    return {a["type"] for a in alerts}


def test_alerts_router_and_slippage_execution_hardening():
    snap = {
        "symbol": "BTCUSDC",
        "router": {
            "fallback_ratio": 0.7,
            "slip_q50": 6.0,
            "maker_fill_ratio": 0.3,
        },
        "risk": {"dd_today_pct": -0.3, "toggle_active": False},
        "vol": {"atr_regime": "normal"},
        "sizing": {"sharpe_7d": 0.5},
    }

    types = _types(snap)
    assert "router_fallback_high" in types
    assert "slippage_high" in types


def test_alerts_dd_kill_execution_hardening():
    snap = {
        "symbol": "ETHUSDC",
        "router": {"fallback_ratio": 0.1, "slip_q50": 2.0},
        "risk": {"dd_today_pct": -4.5, "toggle_active": True},
        "vol": {"atr_regime": "quiet"},
        "sizing": {"sharpe_7d": -1.2},
    }

    types = _types(snap)
    assert "dd_kill" in types
    assert "symbol_disabled" in types


def test_alerts_atr_panic_execution_hardening():
    snap = {
        "symbol": "SOLUSDC",
        "router": {"fallback_ratio": 0.1, "slip_q50": 1.0},
        "risk": {"dd_today_pct": -0.2, "toggle_active": False},
        "vol": {"atr_regime": "panic"},
        "sizing": {"sharpe_7d": 2.0},
    }
    types = _types(snap)
    assert "atr_panic" in types
