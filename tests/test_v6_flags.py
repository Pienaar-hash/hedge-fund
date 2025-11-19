from __future__ import annotations

import logging

from execution import v6_flags


def test_get_flags_and_snapshot(monkeypatch, caplog):
    monkeypatch.setenv("INTEL_V6_ENABLED", "1")
    monkeypatch.setenv("RISK_ENGINE_V6_ENABLED", "1")
    monkeypatch.setenv("PIPELINE_V6_SHADOW_ENABLED", "0")
    monkeypatch.setenv("ROUTER_AUTOTUNE_V6_ENABLED", "1")
    monkeypatch.setenv("FEEDBACK_ALLOCATOR_V6_ENABLED", "0")
    monkeypatch.setenv("ROUTER_AUTOTUNE_V6_APPLY_ENABLED", "1")
    flags = v6_flags.get_flags(refresh=True)
    snapshot = v6_flags.flags_to_dict(flags)
    assert snapshot["INTEL_V6_ENABLED"] is True
    assert snapshot["PIPELINE_V6_SHADOW_ENABLED"] is False
    logger = logging.getLogger("test_v6_flags")
    caplog.set_level(logging.INFO, logger="test_v6_flags")
    v6_flags.log_v6_flag_snapshot(logger, flags=flags)
    assert "INTEL_V6_ENABLED=1" in caplog.text
    assert "PIPELINE_V6_SHADOW_ENABLED=0" in caplog.text
