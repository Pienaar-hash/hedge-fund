import pytest

from execution import nav as navmod


def test_compute_trading_nav_manual():
    cfg = {"nav": {"trading_source": "manual", "manual_nav_usdt": 123.45}}
    nav_val, detail = navmod.compute_trading_nav(cfg)
    assert pytest.approx(nav_val, rel=1e-6) == 123.45
    assert detail["source"] == "manual"


def test_compute_reporting_nav_exchange(monkeypatch):
    monkeypatch.setattr(navmod, "get_balances", lambda: {"USDT": 2500.0})
    monkeypatch.setattr(navmod, "get_positions", lambda: [{"unrealized": 100.0}])
    cfg = {"nav": {"reporting_source": "exchange"}}
    nav_val, detail = navmod.compute_reporting_nav(cfg)
    assert pytest.approx(nav_val, rel=1e-6) == 2600.0
    assert detail["source"] == "exchange"
