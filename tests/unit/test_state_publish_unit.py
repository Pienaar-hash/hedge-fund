from __future__ import annotations

import pytest

from execution import state_publish

pytestmark = pytest.mark.unit


def test_write_positions_state_builds_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_atomic(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    monkeypatch.setattr(state_publish, "_atomic_write_state", fake_atomic)
    rows = [{"symbol": "BTCUSDT", "qty": 0.1, "entry_price": 10000.0, "mark_price": 10100.0}]

    target = tmp_path / "positions_state.json"
    state_publish.write_positions_state(rows, path=target, updated_at="2024-01-01T00:00:00Z")

    assert captured["path"] == target
    payload = captured["payload"]
    assert payload["updated_at"] == "2024-01-01T00:00:00Z"
    assert payload["positions"][0]["symbol"] == "BTCUSDT"


def test_write_positions_ledger_state_atomic(monkeypatch, tmp_path):
    called = {}

    def fake_atomic(path, payload):
        called["path"] = path
        called["payload"] = payload

    monkeypatch.setattr(state_publish, "_atomic_write_state", fake_atomic)
    payload = {"entries": [{"symbol": "ETHUSDT", "side": "LONG"}], "updated_at": "2024-01-01T00:00:00Z"}
    target = tmp_path / "positions_ledger.json"

    state_publish.write_positions_ledger_state(payload, path=target)

    assert called["path"] == target
    assert called["payload"]["entries"][0]["symbol"] == "ETHUSDT"


def test_write_kpis_v7_state_sets_updated(monkeypatch, tmp_path):
    called = {}

    def fake_atomic(path, payload):
        called["path"] = path
        called["payload"] = payload

    monkeypatch.setattr(state_publish, "_atomic_write_state", fake_atomic)
    payload = {"portfolio": {"nav": 1.0}, "per_symbol": {}}
    target = tmp_path / "kpis_v7.json"

    state_publish.write_kpis_v7_state(payload, path=target, now_ts=0)

    assert called["path"] == target
    written = called["payload"]
    assert "updated_at" in written
    assert written["portfolio"]["nav"] == 1.0
