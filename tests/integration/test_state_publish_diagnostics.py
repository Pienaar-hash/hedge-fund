from __future__ import annotations

import json

from execution import state_publish
from execution.diagnostics_metrics import record_order_placed, record_signal_emitted, record_veto, reset_diagnostics


def test_write_runtime_diagnostics_state(tmp_path, monkeypatch):
    reset_diagnostics()
    record_signal_emitted()
    record_order_placed()
    record_veto("max_concurrent")

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
    # Snapshot returned matches what we wrote
    assert payload["runtime_diagnostics"]["veto_counters"]["total_vetoes"] == 1
