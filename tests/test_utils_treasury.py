"""
Validates get_treasury_snapshot and compute_treasury_pnl helpers.
"""

import json
from execution import utils

def test_treasury_snapshot_and_pnl(tmp_path):
    treasury_file = tmp_path / "treasury.json"
    treasury_file.write_text(json.dumps({
        "BTC": {"value_usd": 10000, "avg_entry_price": 9500},
        "USDT": {"value_usd": 2000, "avg_entry_price": 1.0}
    }))

    snap = utils.get_treasury_snapshot(path=str(treasury_file))
    pnl = utils.compute_treasury_pnl(snap)

    assert "BTC" in pnl and "pnl_pct" in pnl["BTC"]
    assert isinstance(pnl["BTC"]["pnl_pct"], float)
    assert abs(pnl["BTC"]["pnl_pct"]) < 10  # sanity margin
