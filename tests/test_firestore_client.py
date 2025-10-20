from __future__ import annotations

import types

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
    sync_state._FIRESTORE_FAIL_COUNT = 0
    sync_state._LAST_SUCCESS_TS = None
    monkeypatch.setenv("ENV", "pytest")

    telemetry_records = []
    heartbeat_records = []

    monkeypatch.setattr(
        sync_state,
        "write_doc",
        lambda _db, path, payload, require=True: telemetry_records.append(
            (path, payload, require)
        )
        or True,
    )
    monkeypatch.setattr(
        sync_state,
        "publish_heartbeat",
        lambda _db, env, service, status, **kwargs: heartbeat_records.append(
            (env, service, status, kwargs)
        )
        or True,
    )

    class FakeDB:
        def collection(self, name: str):
            return types.SimpleNamespace(name=name)

    def failing_sync(db):
        raise RuntimeError("boom")

    monkeypatch.setattr(sync_state, "_sync_once_with_db", failing_sync)
    monkeypatch.setattr(sync_state, "get_db", lambda strict=True: FakeDB())

    with pytest.raises(RuntimeError):
        sync_state.sync_once()

    assert telemetry_records, "health payload was not written"
    path, payload, require = telemetry_records[-1]
    assert path.endswith("telemetry/health")
    assert payload["firestore_ok"] is False
    assert payload["last_error"]
    assert payload["ts"] > 0
    assert require is False

    assert heartbeat_records, "heartbeat was not published"
    env, service, status, kwargs = heartbeat_records[-1]
    assert env == "pytest"
    assert service == "sync_state"
    assert status == "degraded"
    assert "extra" in kwargs and "last_error" in kwargs["extra"]
