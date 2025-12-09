from __future__ import annotations

import json

import pytest

from execution import executor_live

pytestmark = pytest.mark.integration


def _temp_positions_path(tmp_path):
    return tmp_path / "positions_state.json"


def test_write_positions_state_writes_positions(monkeypatch, tmp_path):
    target = _temp_positions_path(tmp_path)
    monkeypatch.setattr(executor_live, "POSITIONS_STATE_PATH", target)
    rows = [
        {"symbol": "BTCUSDT", "qty": 0.5, "entry_price": 10000.0, "mark_price": 10100.0},
        {"symbol": "ETHUSDT", "qty": -1.2, "entry_price": 2000.0, "mark_price": 1990.0},
    ]

    executor_live._write_positions_state(rows, updated_ts="2024-01-01T00:00:00Z")

    data = json.loads(target.read_text())
    assert "positions" in data
    assert len(data["positions"]) == 2
    assert data["updated_at"] == "2024-01-01T00:00:00Z"


def test_non_zero_positions_require_prices(monkeypatch, tmp_path):
    target = _temp_positions_path(tmp_path)
    monkeypatch.setattr(executor_live, "POSITIONS_STATE_PATH", target)
    rows = [
        {"symbol": "BTCUSDT", "qty": 1.0, "entry_price": 10000.0, "mark_price": 10100.0},
    ]

    executor_live._write_positions_state(rows, updated_ts="2024-01-01T00:00:00Z")
    payload = json.loads(target.read_text())

    for row in payload["positions"]:
        if abs(row.get("qty", 0)) > 0:
            assert row["entry_price"] > 0
            assert row["mark_price"] > 0


def test_zero_quantity_rows_allowed(monkeypatch, tmp_path):
    target = _temp_positions_path(tmp_path)
    monkeypatch.setattr(executor_live, "POSITIONS_STATE_PATH", target)
    rows = [
        {"symbol": "XRPUSDT", "qty": 0.0, "entry_price": 0.0, "mark_price": 0.0},
    ]

    executor_live._write_positions_state(rows, updated_ts="2024-01-01T00:00:00Z")
    payload = json.loads(target.read_text())

    assert payload["positions"][0]["qty"] == 0.0
