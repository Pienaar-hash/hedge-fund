#!/usr/bin/env python3
"""Smoke script moved into tests/helpers for manual runs.

Usage:
  PYTHONPATH=. python tests/helpers/smoke_telegram_state.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from execution import telegram_alerts_v7 as ta7


def main() -> None:
    print("[smoke] Loading state...")
    state = ta7.load_state()
    path = ta7.STATE_PATH
    print("[smoke] state_path:", path)
    print("[smoke] current state:")
    print(json.dumps(state, indent=2, sort_keys=True))

    print("[smoke] Updating sample keys and saving...")
    now = time.time()
    state.setdefault("last_sent", {})["smoke_test"] = {"value": "ok", "ts": now}
    state["atr_regime"] = state.get("atr_regime") or "low"
    ta7.save_state(state)

    print("[smoke] Reloading file to verify...")
    try:
        text = path.read_text(encoding="utf-8")
        print(text)
    except Exception as exc:
        print("[smoke] failed reading state file:", exc)


if __name__ == "__main__":
    main()
