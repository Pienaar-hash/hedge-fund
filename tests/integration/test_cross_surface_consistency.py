from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.state_v7 import validate_surface_health

pytestmark = pytest.mark.integration


def test_cross_surface_positions_without_ledger(tmp_path: Path):
    pos_path = tmp_path / "positions_state.json"
    pos_path.write_text(json.dumps({"updated_ts": 1, "positions": [{"symbol": "BTCUSDT", "qty": 1, "positionSide": "LONG"}]}))
    ledger_path = tmp_path / "positions_ledger.json"
    ledger_path.write_text(json.dumps({"updated_ts": 1, "entries": []}))

    health = validate_surface_health(state_dir=tmp_path, allowable_lag_seconds=1000)
    assert any("positions_without_ledger" in item for item in health["cross_surface_violations"])
