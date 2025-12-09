from __future__ import annotations

from execution.diagnostics_metrics import (
    build_runtime_diagnostics_snapshot,
    compute_liveness_alerts,
    get_exit_status,
    get_veto_counters,
    record_order_placed,
    record_signal_emitted,
    record_veto,
    reset_diagnostics,
)
from datetime import datetime, timedelta, timezone


def test_veto_counters_accumulate() -> None:
    reset_diagnostics()
    record_signal_emitted()
    record_signal_emitted()
    record_order_placed()
    record_veto("max_concurrent")
    record_veto("symbol_cap")

    snap = build_runtime_diagnostics_snapshot()
    vc = snap.veto_counters

    assert vc.total_signals == 2
    assert vc.total_orders == 1
    assert vc.total_vetoes == 2
    assert vc.by_reason.get("max_concurrent") == 1
    assert vc.by_reason.get("symbol_cap") == 1
    assert vc.last_veto_ts is not None


def test_liveness_missing_timestamps_flags_idle():
    reset_diagnostics()
    cfg = {
        "enabled": True,
        "max_idle_signals_seconds": 10,
        "max_idle_orders_seconds": 10,
        "max_idle_exits_seconds": 10,
        "max_idle_router_events_seconds": 10,
    }
    alerts = compute_liveness_alerts(cfg)
    assert alerts.idle_signals
    assert alerts.idle_orders
    assert alerts.idle_exits
    assert alerts.idle_router
    assert alerts.missing.get("signals_idle_seconds") is True
    assert "signals_idle_seconds" in alerts.details


def test_liveness_thresholds_respected():
    reset_diagnostics()
    now_iso = datetime.now(timezone.utc).isoformat()
    vc = get_veto_counters()
    es = get_exit_status()
    vc.last_signal_ts = now_iso
    vc.last_order_ts = now_iso
    es.last_exit_trigger_ts = now_iso
    es.last_router_event_ts = now_iso

    cfg = {
        "enabled": True,
        "max_idle_signals_seconds": 120,
        "max_idle_orders_seconds": 120,
        "max_idle_exits_seconds": 120,
        "max_idle_router_events_seconds": 120,
    }
    alerts = compute_liveness_alerts(cfg)
    assert not alerts.idle_signals
    assert not alerts.idle_orders
    assert not alerts.idle_exits
    assert not alerts.idle_router
    assert alerts.missing == {}

    stale = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
    vc.last_signal_ts = stale
    alerts = compute_liveness_alerts(cfg)
    assert alerts.idle_signals
    assert alerts.details.get("signals_idle_seconds", 0) > 120
