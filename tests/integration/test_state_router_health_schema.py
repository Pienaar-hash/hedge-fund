from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.state_publish import write_router_health_state

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def test_router_health_schema_written_from_stats(tmp_path: Path) -> None:
    router_stats_snapshot = {
        "updated_ts": "2024-01-01T00:00:00Z",
        "window_seconds": 900,
        "min_events": 2,
        "per_symbol": {
            "BTCUSDT": {
                "avg_slippage_bps": 5.0,
                "slippage_drift_bps": 5.0,
                "avg_latency_ms": 150.0,
                "twap_usage_ratio": 0.4,
                "last_order_ts": 1700000000,
                "last_fill_ts": 1700000050,
                "total_notional": 1000.0,
                "twap_notional": 400.0,
                "router_bucket": "A_HIGH",
                "child_orders": {"count": 2, "fill_ratio": 0.9},
            },
            "ETHUSDT": {
                "avg_slippage_bps": 8.0,
                "slippage_drift_bps": 8.0,
                "avg_latency_ms": 420.0,
                "twap_usage_ratio": 0.2,
                "last_order_ts": 1700000100,
                "last_fill_ts": 1700000200,
                "total_notional": 500.0,
                "twap_notional": 100.0,
                "router_bucket": "B_MEDIUM",
                "child_orders": {"count": 1, "fill_ratio": 0.5},
            },
        },
    }

    write_router_health_state({}, tmp_path, router_stats_snapshot=router_stats_snapshot)

    path = tmp_path / "router_health.json"
    assert path.exists()
    payload = json.loads(path.read_text())

    assert "router_health" in payload
    rh = payload["router_health"]
    assert "global" in rh and "per_symbol" in rh
    assert isinstance(rh["per_symbol"], dict)

    btc = rh["per_symbol"]["BTCUSDT"]
    assert btc["slippage_drift_bucket"] in {"GREEN", "YELLOW", "RED"}
    assert btc["latency_bucket"] in {"FAST", "NORMAL", "SLOW"}
    assert "quality_score" in btc
    assert "child_orders" in btc

    global_block = rh["global"]
    assert "quality_score" in global_block
    assert payload.get("router_health_score") == global_block["quality_score"]
