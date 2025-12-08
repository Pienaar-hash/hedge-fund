import execution.exchange_utils as exchange_utils


def test_xaut_routes_to_coingecko(monkeypatch):
    called = {"count": 0}

    def fake_get_coingecko_prices():
        called["count"] += 1
        return {"XAUT": 4100.0}

    monkeypatch.setattr(
        exchange_utils,
        "get_coingecko_prices",
        fake_get_coingecko_prices,
        raising=True,
    )

    price = exchange_utils.get_price("XAUTUSDT", venue="fapi")

    assert called["count"] == 1, "Expected CoinGecko fallback to be invoked"
    assert 3500.0 < price < 4500.0
