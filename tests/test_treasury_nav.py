import pytest

from execution import nav as navmod


def test_nav_summary_and_reporting_include_treasury(monkeypatch):
    monkeypatch.setattr(navmod, "get_balances", lambda: {"USDT": 1000.0})
    monkeypatch.setattr(navmod, "get_positions", lambda: [])

    prices = {"BTCUSDT": 40_000.0, "XAUTUSDT": 2_000.0}
    price_calls: list[str] = []

    def fake_price(symbol: str) -> float:
        price_calls.append(symbol)
        return prices[symbol]

    monkeypatch.setattr(navmod, "get_price", fake_price)

    def fake_load(path: str):
        if path.endswith("treasury.json"):
            return {"BTC": 0.025, "XAUT": 0.59, "USDC": 100.0}
        if path.endswith("strategy_config.json"):
            return {"nav": {"include_spot_treasury": True}}
        return {}

    monkeypatch.setattr(navmod, "_load_json", fake_load)

    cfg = {
        "nav": {
            "trading_source": "exchange",
            "reporting_source": "exchange",
            "include_spot_treasury": True,
        }
    }

    expected_treasury = (0.025 * prices["BTCUSDT"]) + (0.59 * prices["XAUTUSDT"]) + 100.0

    summary = navmod.compute_nav_summary(cfg)
    assert pytest.approx(summary["futures_nav"], rel=1e-6) == 1000.0
    assert pytest.approx(summary["treasury_nav"], rel=1e-6) == expected_treasury
    assert pytest.approx(summary["total_nav"], rel=1e-6) == 1000.0 + expected_treasury
    assert summary["details"]["treasury"]["treasury"]["BTC"]["qty"] == 0.025
    assert "USDCUSDT" not in price_calls

    reporting_nav, detail = navmod.compute_reporting_nav(cfg)
    assert pytest.approx(reporting_nav, rel=1e-6) == 1000.0 + expected_treasury
    assert pytest.approx(detail["treasury_total_usdt"], rel=1e-6) == expected_treasury
    assert detail["source"] == "exchange+treasury"

    trading_nav, trading_detail = navmod.compute_trading_nav(cfg)
    assert pytest.approx(trading_nav, rel=1e-6) == 1000.0
    assert trading_detail["source"] == "exchange"
