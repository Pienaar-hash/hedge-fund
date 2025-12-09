from __future__ import annotations

import json

import pytest

from execution import state_publish
from execution.diagnostics_metrics import (
    record_order_placed,
    record_router_event,
    record_signal_emitted,
    record_veto,
    reset_diagnostics,
    update_exit_pipeline_status,
)

pytestmark = pytest.mark.integration


def test_write_runtime_diagnostics_state(tmp_path, monkeypatch):
    reset_diagnostics()
    record_signal_emitted()
    record_order_placed()
    record_veto("max_concurrent")
    record_router_event()
    update_exit_pipeline_status(
        open_positions_count=2,
        tp_sl_registered_count=1,
        tp_sl_missing_count=1,
        underwater_without_tp_sl_count=0,
        tp_sl_coverage_pct=0.5,
        ledger_registry_mismatch=True,
        mismatch_breakdown={"missing_tp_sl_entries": 1},
    )

    monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path / "state")

    payload = state_publish.write_runtime_diagnostics_state(liveness_cfg={"enabled": True})
    path = state_publish.STATE_DIR / "diagnostics.json"

    assert path.exists()
    data = json.loads(path.read_text())
    diag = data.get("runtime_diagnostics")

    assert diag is not None
    assert diag["veto_counters"]["total_signals"] == 1
    assert diag["veto_counters"]["total_orders"] == 1
    assert diag["veto_counters"]["total_vetoes"] == 1
    assert "exit_pipeline" in diag
    assert diag["exit_pipeline"]["tp_sl_coverage_pct"] == 0.5
    assert diag["exit_pipeline"]["ledger_registry_mismatch"] is True
    assert diag["exit_pipeline"]["mismatch_breakdown"]["missing_tp_sl_entries"] == 1
    assert diag["exit_pipeline"]["last_router_event_ts"] is not None
    live = diag.get("liveness") or {}
    assert "missing" in live
    # Snapshot returned matches what we wrote
    assert payload["runtime_diagnostics"]["veto_counters"]["total_vetoes"] == 1
