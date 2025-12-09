from __future__ import annotations

import time

import pytest

from execution.intel.symbol_score_v6 import build_symbol_scores

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def test_symbol_scores_contains_hybrid_fields() -> None:
    expectancy_snapshot = {
        "symbols": {
            "BTCUSDT": {"expectancy": 0.15, "hit_rate": 0.55},
        }
    }
    router_health_snapshot = {
        "updated_ts": time.time(),
        "router_health": {
            "global": {"quality_score": 0.8},
            "per_symbol": {
                "BTCUSDT": {"quality_score": 0.9, "avg_slippage_bps": 1.0},
            },
        },
    }

    scores = build_symbol_scores(expectancy_snapshot, router_health_snapshot)

    assert "updated_ts" in scores
    assert isinstance(scores.get("symbols"), list)
    assert scores["symbols"], "symbols list should not be empty"

    btc_entry = next((s for s in scores["symbols"] if s.get("symbol") == "BTCUSDT"), None)
    assert btc_entry is not None
    assert "hybrid_score" in btc_entry
    assert "hybrid" in btc_entry
    hybrid_block = btc_entry["hybrid"]
    assert "router_quality_score" in hybrid_block
    assert "min_for_emission" in hybrid_block
    assert isinstance(hybrid_block.get("passes_emission"), bool)
