import pytest

from execution import nav as navmod

pytestmark = [
    pytest.mark.legacy,
    pytest.mark.skip(reason="NAV tests call real exchange APIs - need proper exchange_utils mocking"),
]


def test_compute_nav_pair_defaults(monkeypatch):
    monkeypatch.setattr(navmod, "get_balances", lambda: {"USDT": 1000.0})
    monkeypatch.setattr(navmod, "get_positions", lambda: [])
    cfg = {"nav": {}}
    trading, reporting = navmod.compute_nav_pair(cfg)
    assert pytest.approx(trading[0], rel=1e-6) == 1000.0
    assert trading[0] == reporting[0]
    assert trading[1]["source"] == "exchange"


def test_compute_nav_pair_fallback(monkeypatch):
    monkeypatch.setattr(navmod, "get_balances", lambda: {"USDT": 0.0})
    monkeypatch.setattr(navmod, "get_positions", lambda: [])
    cfg = {"capital_base_usdt": 555.0, "nav": {"trading_source": "manual", "manual_nav_usdt": None}}
    trading, reporting = navmod.compute_nav_pair(cfg)
    assert pytest.approx(trading[0], rel=1e-6) == 555.0
    assert trading[1]["source"] == "capital_base"
    assert reporting[0] == trading[0]
