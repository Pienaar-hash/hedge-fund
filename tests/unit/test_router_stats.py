from __future__ import annotations

import time

import pytest

from execution.order_router import RouterStats


def test_router_stats_aggregates_slippage_latency() -> None:
    stats = RouterStats(window_seconds=100, min_events=1, ema_alpha=0.3)
    t0 = time.time()
    stats.update_on_fill(
        symbol="BTCUSDT",
        intended_price=100.0,
        fill_price=101.0,
        ts_sent=t0 - 1.0,
        ts_fill=t0,
        is_twap_child=False,
        notional=1000.0,
    )
    stats.update_on_fill(
        symbol="BTCUSDT",
        intended_price=200.0,
        fill_price=200.5,
        ts_sent=t0,
        ts_fill=t0 + 0.5,
        is_twap_child=True,
        notional=2000.0,
        intended_notional=2100.0,
    )

    snapshot = stats.snapshot(now=t0 + 1.0)
    per_symbol = snapshot["per_symbol"]["BTCUSDT"]

    assert snapshot["updated_ts"]
    assert per_symbol["event_count"] == 2
    assert per_symbol["avg_slippage_bps"] == pytest.approx(50.0)
    expected_latency = ((1000.0 * 1000.0) + (500.0 * 2000.0)) / 3000.0
    assert per_symbol["avg_latency_ms"] == pytest.approx(expected_latency)
    assert per_symbol["twap_usage_ratio"] == pytest.approx(2000.0 / 3000.0)
    assert per_symbol["child_orders"]["fill_ratio"] == pytest.approx(2000.0 / 2100.0)
    # EMA drift should smooth towards most recent slippage value
    assert per_symbol["slippage_drift_bps"] == pytest.approx(77.5)


def test_router_stats_trims_window_but_respects_min_events() -> None:
    stats = RouterStats(window_seconds=10, min_events=2)
    t0 = time.time()
    stats.update_on_fill("ETHUSDT", 10.0, 10.1, t0 - 25.0, t0 - 25.0, False, 100.0)
    stats.update_on_fill("ETHUSDT", 10.0, 10.05, t0 - 5.0, t0 - 5.0, False, 100.0)
    stats.update_on_fill("ETHUSDT", 10.0, 10.02, t0 - 1.0, t0 - 1.0, False, 100.0)

    per_symbol = stats.snapshot(now=t0)["per_symbol"]["ETHUSDT"]

    # Oldest event is trimmed once we exceed min_events within the window
    assert per_symbol["event_count"] == 2
    assert per_symbol["last_fill_ts"]
