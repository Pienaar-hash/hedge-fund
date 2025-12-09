from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from execution import diagnostics_metrics as dm

pytestmark = pytest.mark.unit


def test_veto_and_order_counters_increment():
    dm.reset_diagnostics()
    dm.record_signal_emitted()
    dm.record_signal_emitted()
    dm.record_order_placed()
    dm.record_veto("max_concurrent")

    snap = dm.build_runtime_diagnostics_snapshot()
    vc = snap.veto_counters

    assert vc.total_signals == 2
    assert vc.total_orders == 1
    assert vc.total_vetoes == 1
    assert vc.by_reason.get("max_concurrent") == 1
    assert vc.last_signal_ts is not None
    assert vc.last_order_ts is not None
    assert vc.last_veto_ts is not None


def test_liveness_missing_timestamps_flagged_idle():
    dm.reset_diagnostics()
    cfg = {
        "enabled": True,
        "max_idle_signals_seconds": 5,
        "max_idle_orders_seconds": 5,
        "max_idle_exits_seconds": 5,
        "max_idle_router_events_seconds": 5,
    }
    alerts = dm.compute_liveness_alerts(cfg)

    assert alerts.idle_signals
    assert alerts.idle_orders
    assert alerts.idle_exits
    assert alerts.idle_router
    assert alerts.missing.get("signals_idle_seconds") is True
    assert "signals_idle_seconds" in alerts.details


def test_liveness_with_fresh_timestamps_not_idle(monkeypatch):
    dm.reset_diagnostics()
    now = datetime.now(timezone.utc)
    iso_now = now.isoformat()
    vc = dm.get_veto_counters()
    es = dm.get_exit_status()
    vc.last_signal_ts = iso_now
    vc.last_order_ts = iso_now
    es.last_exit_trigger_ts = iso_now
    es.last_router_event_ts = iso_now

    cfg = {
        "enabled": True,
        "max_idle_signals_seconds": 30,
        "max_idle_orders_seconds": 30,
        "max_idle_exits_seconds": 30,
        "max_idle_router_events_seconds": 30,
    }

    alerts = dm.compute_liveness_alerts(cfg)
    assert not alerts.idle_signals
    assert not alerts.idle_orders
    assert not alerts.idle_exits
    assert not alerts.idle_router
    assert alerts.missing == {}


def test_liveness_over_threshold_sets_idle():
    dm.reset_diagnostics()
    stale = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    vc = dm.get_veto_counters()
    vc.last_signal_ts = stale

    cfg = {
        "enabled": True,
        "max_idle_signals_seconds": 30,
    }
    alerts = dm.compute_liveness_alerts(cfg)
    assert alerts.idle_signals
    assert alerts.details.get("signals_idle_seconds", 0) > 30
