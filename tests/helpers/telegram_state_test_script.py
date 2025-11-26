#!/usr/bin/env python3
"""Comprehensive test script moved into tests/helpers for manual runs.

Usage:
  PYTHONPATH=. python tests/helpers/telegram_state_test_script.py
"""
from __future__ import annotations

import json
import os
import time

import execution.telegram_alerts_v7 as ta7


def setup_fake_sender():
    sent = []

    def fake_send(message: str, silent: bool = False):
        print("[fake_send]", message)
        sent.append({"msg": message, "silent": bool(silent), "ts": time.time()})
        return True

    ta7.telegram_utils.send_telegram = fake_send
    return sent


def enable_config():
    ta7._load_config = lambda: {
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
    }


def run_alerts_for(ctx: dict):
    try:
        ta7.run_alerts(ctx)
    except Exception as exc:
        print("[test] run_alerts raised:", exc)


def main():
    print("[test] Starting telegram alerts v7 simulation")
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "fake-token")
    os.environ.setdefault("TELEGRAM_CHAT_ID", "-1")

    sent = setup_fake_sender()
    enable_config()

    state = ta7.load_state()
    print("[test] initial state:")
    print(json.dumps(state, indent=2, sort_keys=True))

    now = time.time()

    print("[test] Simulate ATR low")
    run_alerts_for({"now_ts": now, "kpis_snapshot": {"atr_regime": "low"}, "nav_snapshot": {}})
    time.sleep(0.1)

    print("[test] Simulate ATR high")
    run_alerts_for({"now_ts": now + 2, "kpis_snapshot": {"atr_regime": "high"}, "nav_snapshot": {}})
    time.sleep(0.1)

    print("[test] Simulate Drawdown mild")
    run_alerts_for({"now_ts": now + 4, "kpis_snapshot": {"dd_state": "mild", "drawdown": {"dd_pct": 0.03}}, "nav_snapshot": {}})
    time.sleep(0.1)

    print("[test] Simulate Risk Mode dd_guard")
    run_alerts_for({"now_ts": now + 6, "risk_snapshot": {"summary": {"risk_mode": "dd_guard"}}, "kpis_snapshot": {}, "nav_snapshot": {}})
    time.sleep(0.1)

    future_bar = (int(now) // (4 * 3600) + 1) * (4 * 3600) + 2
    print("[test] Simulate 4h close at ts", future_bar)
    run_alerts_for({"now_ts": future_bar + 1, "kpis_snapshot": {"atr_regime": "high", "dd_state": "mild"}, "nav_snapshot": {"ts": future_bar + 1, "nav": 12345.67}})
    time.sleep(0.1)

    print("[test] Fake-sent messages:")
    for s in sent:
        print(json.dumps(s, indent=2))

    path = ta7.STATE_PATH
    try:
        print("[test] Final state file content:")
        print(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("[test] Failed reading final state:", exc)


if __name__ == "__main__":
    main()
