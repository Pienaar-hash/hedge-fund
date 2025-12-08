import pytest

from execution.utils import compute_treasury_pnl


def test_compute_treasury_pnl_returns_symbol_dict_with_pct():
    snapshot = {
        "treasury": {
            "assets": [
                {
                    "asset": "BTC",
                    "balance": 1.0,
                    "price_usdt": 22000.0,
                    "avg_entry_price": 20000.0,
                    "usd_value": 22000.0,
                },
                {
                    "asset": "USDT",
                    "balance": 1000.0,
                    "price_usdt": 1.0,
                    "usd_value": 1000.0,
                },
            ]
        }
    }

    pnl = compute_treasury_pnl(snapshot)

    assert isinstance(pnl, dict)
    assert "BTC" in pnl
    btc_entry = pnl["BTC"]
    assert isinstance(btc_entry, dict)
    assert btc_entry["value_usd"] == 22000.0
    assert isinstance(btc_entry["pnl_pct"], float)
    assert btc_entry["pnl_pct"] == pytest.approx(10.0, rel=1e-3)
