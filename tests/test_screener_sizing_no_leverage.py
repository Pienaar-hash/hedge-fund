import math
import json
from pathlib import Path

import pytest

import execution.signal_screener as sc
from execution.risk_limits import RiskState, check_order


def _stub_nav(monkeypatch, nav_value: float):
    monkeypatch.setattr(
        sc,
        "nav_health_snapshot",
        lambda: {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value},
    )


def _make_strategy(leverage: float):
    return {
        "enabled": True,
        "symbol": "ETHUSDT",
        "timeframe": "15m",
        "params": {
            "per_trade_nav_pct": 0.025,
            "leverage": leverage,
            "entry": {"type": "always_on"},
        },
    }


@pytest.fixture
def screener_stubs(monkeypatch, tmp_path):
    price = 2500.0

    monkeypatch.setattr(sc, "_load_strategy_list", lambda: [_make_strategy(3.0)])
    monkeypatch.setattr(sc, "resolve_allowed_symbols", lambda: (["ETHUSDT"], {"ETHUSDT": "CORE"}))
    monkeypatch.setattr(sc, "symbol_tier", lambda s: "CORE")
    monkeypatch.setattr(sc, "allow_trade", lambda s: True)
    monkeypatch.setattr(
        sc,
        "_entry_gate_result",
        lambda sym, sig, enabled=True: (False, {"metric": 0.0}),
    )
    monkeypatch.setattr(sc, "get_klines", lambda s, tf, limit=150: [[0, 0, 0, 0, price]] * 150)
    monkeypatch.setattr(sc, "get_price", lambda s: price)
    monkeypatch.setattr(
        sc,
        "get_symbol_filters",
        lambda s: {"MIN_NOTIONAL": {"minNotional": 20.0}, "LOT_SIZE": {"stepSize": 0.001, "minQty": 0.001}},
    )
    monkeypatch.setattr(sc, "symbol_min_notional", lambda s: 0.0)
    monkeypatch.setattr(sc, "symbol_min_gross", lambda s: 0.0)
    # Mock positions at the exchange_utils module level to intercept the local import
    from execution import exchange_utils
    monkeypatch.setattr(exchange_utils, "get_positions", lambda: [])
    monkeypatch.setattr(sc.PortfolioSnapshot, "current_gross_usd", lambda self: 0.0)
    # Mock trend filter to NEUTRAL so signals pass through
    monkeypatch.setattr(sc, "_trend_filter", lambda *args, **kwargs: "NEUTRAL")
    return price


def test_screener_sizing_ignores_leverage(monkeypatch, screener_stubs):
    price = screener_stubs
    nav_used = 10_000.0
    _stub_nav(monkeypatch, nav_used)

    intents = sc.generate_signals_from_config()
    assert len(intents) == 1
    intent = intents[0]

    expected = max(nav_used * 0.025, 20.0)
    assert math.isclose(intent["gross_usd"], expected, rel_tol=1e-6)
    assert math.isclose(intent["qty"], expected / price, rel_tol=1e-6)
    assert intent["leverage"] == 3.0

    # Change leverage and ensure size stays the same
    monkeypatch.setattr(sc, "_load_strategy_list", lambda: [_make_strategy(10.0)])
    intents2 = sc.generate_signals_from_config()
    intent2 = intents2[0]
    assert math.isclose(intent2["gross_usd"], expected, rel_tol=1e-6)
    assert math.isclose(intent2["qty"], expected / price, rel_tol=1e-6)
    assert intent2["leverage"] == 10.0
