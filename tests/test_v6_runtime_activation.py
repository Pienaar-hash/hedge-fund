from __future__ import annotations

from typing import Dict

from execution import executor_live
from execution.v6_flags import V6Flags


def test_v6_flag_snapshot_reflects_globals(monkeypatch):
    monkeypatch.setattr(
        executor_live,
        "get_flags",
        lambda: V6Flags(
            intel_v6_enabled=True,
            risk_engine_v6_enabled=True,
            pipeline_v6_shadow_enabled=False,
            router_autotune_v6_enabled=True,
            feedback_allocator_v6_enabled=False,
            router_autotune_v6_apply_enabled=True,
        ),
    )
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
    monkeypatch.setattr(
        executor_live,
        "get_flags",
        lambda: V6Flags(
            intel_v6_enabled=True,
            risk_engine_v6_enabled=False,
            pipeline_v6_shadow_enabled=False,
            router_autotune_v6_enabled=False,
            feedback_allocator_v6_enabled=False,
            router_autotune_v6_apply_enabled=False,
        ),
    )
    executor_live._maybe_write_v6_runtime_probe(force=True)
    assert "payload" in records
    assert records["payload"]["INTEL_V6_ENABLED"] is True
