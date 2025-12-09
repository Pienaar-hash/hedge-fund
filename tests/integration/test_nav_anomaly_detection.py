from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.state_publish import write_risk_snapshot_state

pytestmark = pytest.mark.integration


def test_nav_anomaly_detection_across_snapshots(tmp_path: Path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # First snapshot baseline
    write_risk_snapshot_state({"nav_total": 1000.0}, state_dir=state_dir)

    # Tight threshold to force anomaly on big jump
    monkeypatch.setattr(
        "execution.drawdown_tracker.load_nav_anomaly_config",
        lambda cfg=None: type("Cfg", (), {"enabled": True, "max_multiplier_intraday": 1.2, "max_gap_abs_usd": 100.0})(),
    )
    write_risk_snapshot_state({"nav_total": 2000.0}, state_dir=state_dir)

    data = json.loads((state_dir / "risk_snapshot.json").read_text())
    anomalies = data.get("anomalies", {})
    assert "nav_jump" in anomalies
