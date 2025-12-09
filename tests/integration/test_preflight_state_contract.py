from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.preflight import state_health_report

pytestmark = pytest.mark.integration


def test_preflight_detects_state_issues(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Stale nav_state
    (state_dir / "nav_state.json").write_text(json.dumps({"updated_ts": 0, "total_equity": 0, "drawdown": 0}))
    # Malformed diagnostics (missing runtime_diagnostics key)
    (state_dir / "diagnostics.json").write_text(json.dumps({"ts": 1}))

    health = state_health_report(state_dir=state_dir, allowable_lag_seconds=0.01)

    assert "nav_state" in health["stale_files"] or any("nav_state" in entry for entry in health["schema_violations"])
    assert any("diagnostics" in entry for entry in health["schema_violations"])
    assert "risk_snapshot" in health["missing_files"]
