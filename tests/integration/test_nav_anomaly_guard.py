from __future__ import annotations

from execution.drawdown_tracker import NavAnomalyConfig, is_nav_anomalous


def test_nav_anomaly_small_move_not_flagged() -> None:
    cfg = NavAnomalyConfig(enabled=True, max_multiplier_intraday=3.0, max_gap_abs_usd=20000.0)
    assert is_nav_anomalous(10000.0, 10500.0, cfg) is False


def test_nav_anomaly_multiplier_trips_guard() -> None:
    cfg = NavAnomalyConfig(enabled=True, max_multiplier_intraday=3.0, max_gap_abs_usd=20000.0)
    assert is_nav_anomalous(10000.0, 40000.0, cfg) is True


def test_nav_anomaly_gap_trips_guard() -> None:
    cfg = NavAnomalyConfig(enabled=True, max_multiplier_intraday=3.0, max_gap_abs_usd=20000.0)
    assert is_nav_anomalous(10000.0, 32000.0, cfg) is True


def test_nav_anomaly_no_baseline_allows_update() -> None:
    cfg = NavAnomalyConfig(enabled=True, max_multiplier_intraday=3.0, max_gap_abs_usd=20000.0)
    assert is_nav_anomalous(0.0, 37000.0, cfg) is False


def test_nav_peak_update_blocks_anomalies() -> None:
    cfg = NavAnomalyConfig(enabled=True, max_multiplier_intraday=3.0, max_gap_abs_usd=20000.0)
    peak = 0.0
    for nav in (10000.0, 10500.0, 40000.0):
        if not is_nav_anomalous(peak, nav, cfg):
            peak = max(peak, nav)
    assert peak == 10500.0
