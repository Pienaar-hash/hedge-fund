import pytest

from execution import nav as navmod


def test_nav_summary_trading_only(monkeypatch):
    monkeypatch.setattr(navmod, "get_balances", lambda: {"USDT": 1000.0})
    monkeypatch.setattr(navmod, "get_positions", lambda: [])

    cfg = {
        "nav": {
            "trading_source": "exchange",
            "reporting_source": "exchange",
        }
    }

    summary = navmod.compute_nav_summary(cfg)
    assert pytest.approx(summary["futures_nav"], rel=1e-6) == 1000.0
    assert pytest.approx(summary["total_nav"], rel=1e-6) == 1000.0
    assert "treasury_nav" not in summary
    assert "reserves_nav" not in summary

    reporting_nav, detail = navmod.compute_reporting_nav(cfg)
    assert pytest.approx(reporting_nav, rel=1e-6) == 1000.0
    assert detail["source"] in {"exchange", "exchange_cache"}

    trading_nav, trading_detail = navmod.compute_trading_nav(cfg)
    assert pytest.approx(trading_nav, rel=1e-6) == 1000.0
    assert trading_detail["source"] == "exchange"
