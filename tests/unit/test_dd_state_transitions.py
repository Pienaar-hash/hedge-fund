from __future__ import annotations

import json
from pathlib import Path

from execution.state_publish import write_risk_snapshot_state


def test_dd_state_transitions_from_normal_to_drawdown(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Initial snapshot with peak nav
    write_risk_snapshot_state({"nav_total": 100_000}, state_dir=state_dir)
    # Next snapshot with lower nav triggers drawdown state
    write_risk_snapshot_state({"nav_total": 80_000}, state_dir=state_dir)

    data = json.loads((state_dir / "risk_snapshot.json").read_text())
    dd_state = data.get("dd_state", {})
    assert str(dd_state.get("state", "")).upper() in {"DRAWDOWN", "RECOVERY"}
