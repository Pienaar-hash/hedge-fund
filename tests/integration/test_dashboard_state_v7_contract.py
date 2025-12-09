from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import state_v7

pytestmark = pytest.mark.integration


def test_dashboard_state_loader_and_badges(tmp_path: Path, monkeypatch):
    # Seed minimal state surfaces
    nav_path = tmp_path / "nav_state.json"
    nav_path.write_text(json.dumps({"total_equity": 1500, "drawdown": 0, "updated_at": "2025-01-01T00:00:00Z"}))
    risk_path = tmp_path / "risk_snapshot.json"
    risk_path.write_text(json.dumps({"risk_mode": "OK", "dd_state": {"state": "normal"}, "updated_ts": 1}))
    router_path = tmp_path / "router_health.json"
    router_path.write_text(json.dumps({"updated_ts": 1, "router_health": {"global": {"quality_score": 0.85}}}))
    vol_path = tmp_path / "vol_regimes.json"
    vol_path.write_text(json.dumps({"current_regime": "normal"}))

    monkeypatch.setattr(state_v7, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state_v7, "NAV_STATE_PATH", nav_path)
    monkeypatch.setattr(state_v7, "RISK_STATE_PATH", risk_path)
    monkeypatch.setattr(state_v7, "ROUTER_HEALTH_STATE_PATH", router_path)
    monkeypatch.setattr(state_v7, "VOL_REGIMES_PATH", vol_path)

    state = state_v7.load_all_state()
    assert state["nav"]["nav_usd"] == 1500.0

    badges = state_v7.get_regime_badges(
        vol_state=state_v7.load_vol_regimes(),
        risk_snapshot=state_v7.load_risk_snapshot(),
        router_health=state_v7.load_router_health_state(),
    )
    assert badges["vol"].startswith("VOL_")
    assert badges["router"].startswith("ROUTER_")
    assert badges["risk"].startswith("RISK_")
