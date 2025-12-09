from __future__ import annotations

import json
from pathlib import Path

from dashboard.state_v7 import validate_surface_health


def test_surface_health_missing_files(monkeypatch, tmp_path: Path):
    health = validate_surface_health(state_dir=tmp_path, allowable_lag_seconds=1)
    assert "nav_state" in health["missing_files"]


def test_surface_health_detects_stale(monkeypatch, tmp_path: Path):
    nav_path = tmp_path / "nav_state.json"
    nav_path.write_text(json.dumps({"updated_ts": 0, "total_equity": 0, "drawdown": 0}))
    health = validate_surface_health(state_dir=tmp_path, allowable_lag_seconds=1)
    assert "nav_state" in health["stale_files"] or "nav_state" in health["schema_violations"]
