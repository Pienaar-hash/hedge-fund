"""
Smoke-tests Firestore publishing alignment for v5.8.
Ensures heartbeat + positions writers target the new telemetry/health
and state/positions (items-schema) paths.
"""

import pytest
from execution import firestore_utils

@pytest.mark.asyncio
async def test_publish_health_and_positions_schema(monkeypatch):
    written = {}

    async def fake_set(path, payload):
        written[path] = payload

    # Patch internal Firestore setter
    monkeypatch.setattr(firestore_utils, "_firestore_set_json", fake_set)

    # Exercise helper
    await firestore_utils.publish_health_if_needed(env="test", service="executor", status={"ok": True})
    await firestore_utils.publish_positions(env="test", positions=[{"symbol": "BTCUSDT"}])

    # Validate paths + keys
    assert any("telemetry/health" in p for p in written), "health doc not written"
    assert any("state/positions" in p for p in written), "positions doc not written"
    for payload in written.values():
        assert isinstance(payload, dict)
