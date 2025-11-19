from __future__ import annotations

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
