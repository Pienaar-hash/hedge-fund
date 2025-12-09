from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from execution.diagnostics_metrics import (
    compute_liveness_alerts,
    get_exit_status,
    record_router_event,
    reset_diagnostics,
)

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def test_router_event_updates_liveness_details() -> None:
    reset_diagnostics()
    record_router_event()
    cfg = {"enabled": True, "max_idle_router_events_seconds": 30}

    alerts = compute_liveness_alerts(cfg)

    assert "router_idle_seconds" in alerts.details
    assert alerts.idle_router is False


def test_router_idle_flag_trips_after_threshold() -> None:
    reset_diagnostics()
    es = get_exit_status()
    es.last_router_event_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()

    alerts = compute_liveness_alerts({"enabled": True, "max_idle_router_events_seconds": 60})

    assert alerts.idle_router is True
    assert alerts.details.get("router_idle_seconds", 0) >= 120 - 1
