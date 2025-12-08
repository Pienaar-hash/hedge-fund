import time

import execution.utils.toggle as toggle


def test_bootstrap_from_firestore_populates_cache_execution_hardening(monkeypatch):
    now = time.time()
    docs = [
        {"symbol": "BTCUSDC", "until": now + 3600, "reason": "ops_disable"},
        {"symbol": "OLDUSDC", "until": now - 10, "reason": "expired"},
    ]

    monkeypatch.setattr(toggle, "fetch_symbol_toggles", lambda: docs)

    toggle._SYMBOL_TOGGLES.clear()
    toggle._BOOTSTRAPPED = False

    assert toggle.is_symbol_disabled("BTCUSDC") is True
    assert toggle.is_symbol_disabled("OLDUSDC") is False
