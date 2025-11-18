from __future__ import annotations

from typing import Dict

from execution import executor_live


def test_v6_flag_snapshot_reflects_globals(monkeypatch):
    monkeypatch.setattr(executor_live, "INTEL_V6_ENABLED", True)
    monkeypatch.setattr(executor_live, "RISK_ENGINE_V6_ENABLED", True)
    monkeypatch.setattr(executor_live, "PIPELINE_V6_SHADOW_ENABLED", False)
    monkeypatch.setattr(executor_live, "ROUTER_AUTOTUNE_V6_ENABLED", True)
    monkeypatch.setattr(executor_live, "FEEDBACK_ALLOCATOR_V6_ENABLED", False)
    monkeypatch.setattr(executor_live, "ROUTER_AUTOTUNE_V6_APPLY_ENABLED", True)
    snapshot = executor_live.get_v6_flag_snapshot()
    assert snapshot["INTEL_V6_ENABLED"] is True
    assert snapshot["PIPELINE_V6_SHADOW_ENABLED"] is False
    assert snapshot["ROUTER_AUTOTUNE_V6_APPLY_ENABLED"] is True


def test_v6_runtime_probe_writes(monkeypatch):
    records: Dict[str, Dict] = {}

    def fake_writer(payload):
        records["payload"] = payload

    monkeypatch.setattr(executor_live, "write_v6_runtime_probe_state", fake_writer)
    monkeypatch.setattr(executor_live, "_LAST_V6_RUNTIME_PROBE", 0.0)
    monkeypatch.setattr(executor_live, "_V6_RUNTIME_PROBE_INTERVAL", 0.0)
    monkeypatch.setattr(executor_live, "get_v6_flag_snapshot", lambda: {"INTEL_V6_ENABLED": True})
    executor_live._maybe_write_v6_runtime_probe(force=True)
    assert "payload" in records
    assert records["payload"]["INTEL_V6_ENABLED"] is True
