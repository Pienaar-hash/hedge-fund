import time

import pytest

from execution import nav as navmod


def test_nav_includes_quotes_and_btc(monkeypatch):
    monkeypatch.setattr(navmod, "get_futures_balances", lambda: {"USDT": 5000.0, "USDC": 2000.0, "BTC": 1.0})

    class DummyClient:
        is_stub = False

        def ticker_price(self, symbol):
            if symbol == "BTCUSDT":
                return {"price": "30000"}
            return {"price": "0"}

    monkeypatch.setattr(navmod, "get_um_client", lambda: DummyClient())
    nav_val, detail = navmod.compute_trading_nav({})
    assert pytest.approx(nav_val, rel=1e-6) == 5000.0 + 2000.0 + 30000.0
    assert detail["breakdown"]["BTC"] == pytest.approx(30000.0)
    assert detail["fresh"] is True
    assert detail["source"] == "live"


def test_nav_cache_fallback(monkeypatch):
    monkeypatch.setattr(navmod, "get_futures_balances", lambda: {})

    cached = {"nav": 9000.0, "detail": {"breakdown": {"USDT": 9000.0}}}
    monkeypatch.setattr(navmod, "get_confirmed_nav", lambda: cached)
    monkeypatch.setattr(navmod, "nav_health_snapshot", lambda *_, **__: {"fresh": False, "sources_ok": True})
    nav_val, detail = navmod.compute_trading_nav({})
    assert nav_val == 9000.0
    assert detail["source"] == "cache"
    assert detail["fresh"] is False
