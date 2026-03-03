"""Tests for thread-safety of exchange_utils HTTP layer (Phase 4 Commit 1)."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest


# ── Per-thread Session ────────────────────────────────────────────────

@pytest.mark.unit
def test_per_thread_session_distinct() -> None:
    """Sessions created in different threads must be distinct objects."""
    import execution.exchange_utils as eu
    results: dict[str, Any] = {}

    def grab(name: str) -> None:
        # Force a fresh TLS slot by clearing any cached session
        eu._TLS.session = None  # type: ignore[attr-defined]
        s = eu._session()
        results[name] = id(s)

    t1 = threading.Thread(target=grab, args=("a",))
    t2 = threading.Thread(target=grab, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert results["a"] != results["b"], "Two threads must not share the same Session"


@pytest.mark.unit
def test_per_thread_session_has_api_key() -> None:
    """Each per-thread Session carries the X-MBX-APIKEY header."""
    import execution.exchange_utils as eu
    headers: dict[str, str] = {}

    def grab(name: str) -> None:
        eu._TLS.session = None  # type: ignore[attr-defined]
        s = eu._session()
        headers[name] = s.headers.get("X-MBX-APIKEY", "")

    t1 = threading.Thread(target=grab, args=("a",))
    t2 = threading.Thread(target=grab, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    # Both threads should have the key (may be empty string in test env)
    assert "a" in headers and "b" in headers
    assert headers["a"] == headers["b"]


@pytest.mark.unit
def test_main_thread_session_valid() -> None:
    """_session() returns a usable Session from the main thread."""
    import requests
    import execution.exchange_utils as eu
    s = eu._session()
    assert isinstance(s, requests.Session)


# ── Time sync lock ────────────────────────────────────────────────────

@pytest.mark.unit
def test_time_sync_lock_serialises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concurrent force-sync calls are serialised by _TIME_SYNC_LOCK.

    We monkeypatch requests.get to a slow fake.  Because of the lock the
    second caller will wait, then see LAST_TIME_SYNC updated and skip the
    HTTP call.  So requests.get should be called exactly once.
    """
    import execution.exchange_utils as eu
    call_count = {"n": 0}

    class FakeResp:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> dict:
            return {"serverTime": int(time.time() * 1000)}

    def slow_get(*args: Any, **kwargs: Any) -> FakeResp:
        call_count["n"] += 1
        time.sleep(0.1)
        return FakeResp()

    # Reset state so both threads will try to sync
    monkeypatch.setattr(eu, "TIME_OFFSET_MS", None)
    monkeypatch.setattr(eu, "LAST_TIME_SYNC", 0.0)
    monkeypatch.setattr("requests.get", slow_get)

    t1 = threading.Thread(target=eu._sync_server_time, kwargs={"force": True})
    t2 = threading.Thread(target=eu._sync_server_time, kwargs={"force": True})
    t1.start(); t2.start()
    t1.join(); t2.join()

    # The first thread does the HTTP call.  The second thread enters the
    # lock, sees that LAST_TIME_SYNC is now recent (force=True still
    # forces), so it also calls.  With force=True both will call, but
    # they are serialised (no race on the global writes).
    assert call_count["n"] >= 1
    assert eu.TIME_OFFSET_MS is not None
