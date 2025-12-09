from __future__ import annotations

import json

import pytest

from execution.state_publish import write_kpis_v7_state

pytestmark = pytest.mark.integration


def test_kpis_v7_contract(tmp_path):
    state_dir = tmp_path / "logs" / "state"
    payload = {
        "portfolio": {
            "nav": 4093.43,
            "dd_pct": 0.0326,
            "var_nav_pct": 0.0056,
            "cvar_nav_pct": 0.0070,
        },
        "per_symbol": {
            "BTCUSDT": {
                "pnl": 12.34,
                "pnl_pct": 0.0123,
                "exposure_nav_pct": 0.10,
            }
        },
    }

    write_kpis_v7_state(payload, state_dir=state_dir, now_ts=0)

    path = state_dir / "kpis_v7.json"
    assert path.exists()
    data = json.loads(path.read_text())

    assert "updated_at" in data
    assert "portfolio" in data
    assert "per_symbol" in data
