from __future__ import annotations

import json
from pathlib import Path

from dashboard import state_v7


def test_load_all_state_with_missing_files(monkeypatch, tmp_path: Path):
    # Point all state paths to empty temp files/dirs
    monkeypatch.setattr(state_v7, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state_v7, "NAV_STATE_PATH", tmp_path / "nav_state.json")
    monkeypatch.setattr(state_v7, "NAV_DETAIL_PATH", tmp_path / "nav.json")
    monkeypatch.setattr(state_v7, "KPI_V7_STATE_PATH", tmp_path / "kpis_v7.json")
    monkeypatch.setattr(state_v7, "POSITIONS_STATE_PATH", tmp_path / "positions_state.json")
    monkeypatch.setattr(state_v7, "POSITIONS_PATH", tmp_path / "positions.json")
    monkeypatch.setattr(state_v7, "POSITIONS_LEDGER_PATH", tmp_path / "positions_ledger.json")
    monkeypatch.setattr(state_v7, "ROUTER_HEALTH_STATE_PATH", tmp_path / "router_health.json")
    monkeypatch.setattr(state_v7, "DIAGNOSTICS_STATE_PATH", tmp_path / "diagnostics.json")
    monkeypatch.setattr(state_v7, "RISK_STATE_PATH", tmp_path / "risk_snapshot.json")

    result = state_v7.load_all_state()
    assert "nav" in result and "kpis" in result
    assert result["positions"] == []


def test_load_all_state_with_minimal_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(state_v7, "STATE_DIR", tmp_path)
    nav_path = tmp_path / "nav_state.json"
    nav_path.write_text(json.dumps({"total_equity": 1000, "drawdown": 0, "updated_at": "2025-01-01T00:00:00Z"}))
    monkeypatch.setattr(state_v7, "NAV_STATE_PATH", nav_path)

    result = state_v7.load_all_state()
    assert result["nav"]["nav_usd"] == 1000.0
