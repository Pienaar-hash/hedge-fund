from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.state_publish import write_nav_state

pytestmark = pytest.mark.integration


def test_nav_state_has_required_fields(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    write_nav_state({"nav": 1000.0, "nav_usd": 1000.0, "updated_ts": "2025-01-01T00:00:00Z"}, state_dir=state_dir)

    path = state_dir / "nav_state.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "updated_ts" in data
    assert "nav" in data or "nav_usd" in data or "nav_total" in data
    assert "aum" in data
