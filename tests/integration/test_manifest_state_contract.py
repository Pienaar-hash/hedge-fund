from __future__ import annotations

import json

import pytest


@pytest.mark.integration
def test_manifest_contains_core_surfaces():
    manifest = json.loads(open("v7_manifest.json").read())
    state_files = manifest.get("state_files", {})
    for required in ["nav", "nav_state", "positions_state", "positions_ledger", "risk_snapshot", "runtime_diagnostics", "router_health"]:
        assert required in state_files
        entry = state_files[required]
        assert "path" in entry and "owner" in entry
