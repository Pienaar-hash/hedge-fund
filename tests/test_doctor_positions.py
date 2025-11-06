"""
Tests Doctor's live/cached position fetch logic.
Validates fallback to cached position_state.jsonl when Binance client unavailable.
"""

import pytest
from scripts import doctor

class DummyClient:
    def __init__(self, positions):
        self.positions = positions

@pytest.mark.asyncio
async def test_collect_positions_fallback(monkeypatch, tmp_path):
    # Simulate cached file
    cached = tmp_path / "position_state.jsonl"
    cached.write_text('{"symbol":"ETHUSDT","size":0.5}\n')

    def fake_get_cached_positions(*_):
        return [{"symbol": "ETHUSDT", "size": 0.5}]

    monkeypatch.setattr(doctor, "get_cached_positions", fake_get_cached_positions)

    positions = doctor._collect_positions(None)  # Client=None triggers fallback
    assert positions, "Expected cached positions fallback"
    assert positions[0]["symbol"] == "ETHUSDT"
