from __future__ import annotations

import pytest

from execution.router_metrics import RouterQualityConfig
from execution.state_publish import _build_router_health_from_stats


def test_quality_scoring_applies_buckets_and_penalties() -> None:
    cfg = RouterQualityConfig()
    stats_map = {
        "BTCUSDT": {
            "avg_slippage_bps": 4.0,
            "slippage_drift_bps": 4.0,
            "avg_latency_ms": 120.0,
            "twap_usage_ratio": 0.5,
            "total_notional": 1000.0,
            "router_bucket": "B_MEDIUM",
            "child_orders": {"count": 1, "fill_ratio": 1.0},
        }
    }

    payload = _build_router_health_from_stats(stats_map, cfg)
    per_symbol = payload["router_health"]["per_symbol"]["BTCUSDT"]

    assert per_symbol["slippage_drift_bucket"] == "YELLOW"
    assert per_symbol["latency_bucket"] == "FAST"
    # base 0.8 + bucket(-0.05) + drift(-0.05) + twap_penalty(0.05) = 0.65
    assert per_symbol["quality_score"] == pytest.approx(0.65)


def test_global_quality_prefers_notional_weighted_bucket() -> None:
    cfg = RouterQualityConfig()
    stats_map = {
        "BIG": {
            "avg_slippage_bps": 1.0,
            "slippage_drift_bps": 1.0,
            "avg_latency_ms": 80.0,
            "twap_usage_ratio": 1.0,
            "total_notional": 5000.0,
            "router_bucket": "A_HIGH",
            "child_orders": {"count": 0, "fill_ratio": 0.0},
        },
        "SMALL": {
            "avg_slippage_bps": 10.0,
            "slippage_drift_bps": 10.0,
            "avg_latency_ms": 800.0,
            "twap_usage_ratio": 0.0,
            "total_notional": 100.0,
            "router_bucket": "C_LOW",
            "child_orders": {"count": 0, "fill_ratio": 0.0},
        },
    }

    payload = _build_router_health_from_stats(stats_map, cfg)
    global_block = payload["router_health"]["global"]

    assert global_block["router_bucket"] == "A_HIGH"
    assert global_block["slippage_drift_bucket"] == "GREEN"
    assert global_block["latency_bucket"] == "FAST"
    assert global_block["quality_score"] > 0.79
