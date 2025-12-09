from __future__ import annotations

import json
from pathlib import Path

from execution.state_publish import write_risk_snapshot_state


def test_risk_snapshot_has_required_fields(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    payload = {"nav_total": 10_000, "positions": []}
    write_risk_snapshot_state(payload, state_dir=state_dir)

    data = json.loads((state_dir / "risk_snapshot.json").read_text())
    assert "updated_ts" in data
    assert "risk_mode" in data
    assert "dd_state" in data
    assert "circuit_breaker" in data
    assert "anomalies" in data
