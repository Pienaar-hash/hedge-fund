from __future__ import annotations

import types

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
    sync_state._FIRESTORE_FAIL_COUNT = 0
    monkeypatch.setenv("ENV", "pytest")

    records = []

    class FakeDoc:
        def __init__(self, path: str) -> None:
            self.path = path

        def set(self, payload, merge=True):
            records.append((self.path, payload, merge))
            return None

        def collection(self, name: str):
            return FakeCollection(f"{self.path}/{name}")

        def get(self):
            snap = types.SimpleNamespace()
            snap.exists = False
            snap.to_dict = lambda: {}
            return snap

    class FakeCollection:
        def __init__(self, path: str) -> None:
            self.path = path

        def document(self, name: str):
            return FakeDoc(f"{self.path}/{name}")

        def add(self, *_a, **_k):
            return (FakeDoc(f"{self.path}/doc"), None)

    class FakeDB:
        def collection(self, name: str):
            return FakeCollection(name)

    class FakeCtx:
        def __enter__(self):
            return FakeDB()

        def __exit__(self, exc_type, exc, tb):
            return False

    def failing_sync(db):
        raise RuntimeError("boom")

    monkeypatch.setattr(sync_state, "_sync_once_with_db", failing_sync)
    monkeypatch.setattr(sync_state, "with_firestore", lambda: FakeCtx())

    try:
        sync_state.sync_once()
    except RuntimeError:
        pass

    health_payloads = [
        payload for path, payload, _ in records if path.endswith("telemetry/health")
    ]
    assert health_payloads, "health payload was not written"
    payload = health_payloads[-1]
    assert payload["firestore_ok"] is False
    assert payload["last_error"]
    assert payload["ts"] > 0
