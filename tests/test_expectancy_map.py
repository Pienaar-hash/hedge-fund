from execution.intel.expectancy_map import best_hours, hourly_expectancy


def test_hourly_expectancy_buckets_events_execution_intelligence(monkeypatch):
    fake_events = [
        {"symbol": "BTCUSDC", "ts": 1707345600, "realized_pnl": 10.0, "notional": 1000.0, "slippage_bps": 1.0},
        {"symbol": "BTCUSDC", "ts": 1707345600, "realized_pnl": -5.0, "notional": 500.0, "slippage_bps": 2.0},
        {"symbol": "BTCUSDC", "ts": 1707349200, "realized_pnl": 2.0, "notional": 200.0, "slippage_bps": 0.5},
    ]

    def fake_get_recent_router_events(symbol=None, window_days=7):
        assert symbol == "BTCUSDC"
        assert window_days == 7
        return fake_events

    monkeypatch.setattr(
        "execution.intel.expectancy_map.get_recent_router_events",
        fake_get_recent_router_events,
    )

    stats = hourly_expectancy("BTCUSDC")
    assert isinstance(stats, dict)
    assert sum(row["count"] for row in stats.values()) == 3
    assert stats[min(stats.keys())]["slip_bps_avg"] is not None


def test_best_hours_filters_by_trades_and_expectancy_execution_intelligence(monkeypatch):
    def fake_hourly_expectancy(symbol=None):
        assert symbol == "BTCUSDC"
        return {
            9: {"count": 10, "exp_per_notional": -0.001, "slip_bps_avg": 2.0},
            10: {"count": 25, "exp_per_notional": 0.002, "slip_bps_avg": 1.0},
        }

    monkeypatch.setattr(
        "execution.intel.expectancy_map.hourly_expectancy",
        fake_hourly_expectancy,
    )

    hours = best_hours("BTCUSDC", min_trades=20, min_expectancy=0.0)
    assert hours == [10]
