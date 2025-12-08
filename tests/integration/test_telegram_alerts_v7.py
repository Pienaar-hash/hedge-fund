"""
Test telegram_alerts_v7 module.

v7 is a low-noise, state-driven alert system that only sends one JSON
payload per 4h candle close. Per-tick alerts (atr_regime, dd_state, etc.)
are intentionally disabled.
"""
import json
import time

import execution.telegram_alerts_v7 as ta7


def test_alerts_4h_close_only(tmp_path, monkeypatch):
    """v7 only sends 4h-close JSON state summaries (low-noise policy)."""
    sent = []

    def fake_send(message: str, silent: bool = False):
        sent.append({"msg": message, "silent": bool(silent), "ts": time.time()})
        return True

    monkeypatch.setattr(ta7.telegram_utils, "send_telegram", fake_send)
    # Force config on for test
    monkeypatch.setattr(
        ta7,
        "_load_config",
        lambda: {
            "enabled": True,
            "bot_token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id_env": "TELEGRAM_CHAT_ID",
            "min_interval_seconds": 1,
            "alerts": {
                "atr_regime": {"enabled": True, "min_interval_seconds": 1},
                "dd_state": {"enabled": True, "min_interval_seconds": 1},
                "risk_mode": {"enabled": True, "min_interval_seconds": 1},
                "close_4h": {"enabled": True, "min_interval_seconds": 1},
            },
        },
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")

    # Use a temporary state path so tests don't touch repo logs
    monkeypatch.setattr(ta7, "STATE_PATH", tmp_path / "telegram_state.json")

    # Loading should create default state
    state = ta7.load_state()
    assert isinstance(state, dict)

    now_ts = time.time()
    bar_ts = int(now_ts // (4 * 3600)) * (4 * 3600)

    # Run alerts with kpis and nav snapshot
    ta7.run_alerts({
        "now_ts": now_ts,
        "kpis_snapshot": {"atr_regime": "low", "dd_state": "none"},
        "nav_snapshot": {"nav": 11173.87, "ts": now_ts},
    })

    # Should have sent a 4h JSON payload (not per-tick ATR alerts)
    assert len(sent) == 1
    payload = json.loads(sent[0]["msg"])
    assert "atr_regime" in payload
    assert "last_4h_close_ts" in payload
    assert payload["atr_regime"] == "low"
    assert payload["last_4h_close_ts"] == bar_ts

    # State file should be updated
    data = json.loads((tmp_path / "telegram_state.json").read_text())
    assert "last_sent" in data
    assert data["atr_regime"] == "low"
    assert data["last_4h_close_ts"] == bar_ts

    # Same 4h bar with different atr_regime should be suppressed (low-noise)
    sent.clear()
    ta7.run_alerts({
        "now_ts": now_ts + 60,  # 1 minute later, same 4h bar
        "kpis_snapshot": {"atr_regime": "high"},
        "nav_snapshot": {"nav": 11200.00, "ts": now_ts + 60},
    })
    assert len(sent) == 0  # Suppressed because same 4h bar
