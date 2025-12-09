from __future__ import annotations

import json
from pathlib import Path

from execution.state_publish import write_risk_snapshot_state


def test_nav_jump_anomaly_flag(tmp_path: Path, monkeypatch):
    # Seed previous snapshot
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "risk_snapshot.json").write_text(json.dumps({"nav_total": 100_000}))

    # Configure a tight anomaly threshold
    monkeypatch.setattr(
        "execution.drawdown_tracker.load_nav_anomaly_config",
        lambda cfg=None: type("Cfg", (), {"enabled": True, "max_multiplier_intraday": 1.1, "max_gap_abs_usd": 1000.0})(),
    )

    write_risk_snapshot_state({"nav_total": 150_000}, state_dir=state_dir)

    data = json.loads((state_dir / "risk_snapshot.json").read_text())
    anomalies = data.get("anomalies", {})
    assert "nav_jump" in anomalies
