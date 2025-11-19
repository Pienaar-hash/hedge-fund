from __future__ import annotations

import json

from execution import sync_state


def test_firestore_enabled_env(monkeypatch):
    monkeypatch.setattr(sync_state, "_ENV", "prod", raising=False)
    monkeypatch.delenv("FIRESTORE_ENABLED", raising=False)
    monkeypatch.setenv("ALLOW_PROD_SYNC", "0")
    assert sync_state._firestore_enabled() is False
    monkeypatch.setenv("ALLOW_PROD_SYNC", "1")
    assert sync_state._firestore_enabled() is True
    monkeypatch.setattr(sync_state, "_ENV", "dev", raising=False)
    monkeypatch.delenv("ALLOW_PROD_SYNC", raising=False)
    assert sync_state._firestore_enabled() is True


def test_synced_state_reader_handles_canonical_schema(tmp_path, monkeypatch):
    payload = {
        "items": [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 1.0, "notional": 100.0},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -0.5, "notional": -50.0},
        ],
        "nav": 1000.0,
        "engine_version": "v6.0-beta-preview",
        "v6_flags": {"INTEL_V6_ENABLED": True},
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    path = tmp_path / "synced_state.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sync_state, "SYNCED_STATE", str(path), raising=False)
    snapshot = sync_state._read_positions_snapshot(str(path))
    assert snapshot["items"] and snapshot["items"][0]["symbol"] == "BTCUSDT"
    exposure = sync_state._exposure_from_positions(snapshot["items"])
    assert isinstance(exposure, dict)
