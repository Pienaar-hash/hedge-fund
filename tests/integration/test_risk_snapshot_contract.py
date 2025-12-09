from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.state_publish import write_risk_snapshot_state

pytestmark = pytest.mark.integration


def test_risk_snapshot_contract_fields(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    write_risk_snapshot_state({"nav_total": 5000.0, "positions": []}, state_dir=state_dir)

    data = json.loads((state_dir / "risk_snapshot.json").read_text())
    required = [
        "updated_ts",
        "risk_mode",
        "dd_state",
        "circuit_breaker",
        "anomalies",
    ]
    for key in required:
        assert key in data
