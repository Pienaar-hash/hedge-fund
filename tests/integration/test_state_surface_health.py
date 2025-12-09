from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.state_v7 import validate_surface_health

pytestmark = pytest.mark.integration


def test_state_surface_health_with_stale_and_missing(tmp_path: Path):
    # Seed a stale nav_state and valid risk_snapshot
    nav_path = tmp_path / "nav_state.json"
    nav_path.write_text(json.dumps({"updated_ts": 0, "total_equity": 0, "drawdown": 0}))
    risk_path = tmp_path / "risk_snapshot.json"
    risk_path.write_text(json.dumps({"updated_ts": 1, "risk_mode": "OK", "dd_state": {"state": "NORMAL"}}))

    health = validate_surface_health(state_dir=tmp_path, allowable_lag_seconds=10)
    assert "nav_state" in health["stale_files"]
    assert "positions_state" in health["missing_files"]
