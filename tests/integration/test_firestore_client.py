from __future__ import annotations

import types
from typing import List

import pytest

import utils.firestore_client as fc
import execution.sync_state as sync_state


def test_firestore_set_retries(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(fc, "_sleep", lambda _s: None)

    monkeypatch.setattr(fc._LOG, "warning", lambda msg, *a, **k: calls.append(("warn", msg)))
    monkeypatch.setattr(fc._LOG, "error", lambda msg, *a, **k: calls.append(("error", msg)))

    class FakeDoc:
        def __init__(self) -> None:
            self.calls = 0
            self.path = "col/doc"

        def set(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("boom")
            return {"ok": True}

    doc = fc._DocWrapper(FakeDoc())
    result = doc.set({"foo": "bar"})

    assert result == {"ok": True}
    assert any(kind == "warn" for kind, _ in calls)
    assert not any(kind == "error" for kind, _ in calls)
    assert doc._doc.calls == 3


def test_sync_state_health_degrades(monkeypatch) -> None:
    calls: List[str] = []

    def failing_sync(_db):
        raise RuntimeError("boom")

    monkeypatch.setattr(sync_state, "_sync_once_with_db", failing_sync)
    monkeypatch.setattr(sync_state, "_publish_health", lambda *a, **k: calls.append("publish"))

    with pytest.raises(RuntimeError):
        sync_state.sync_once()

    assert not calls, "local sync should not attempt Firestore health writes"
