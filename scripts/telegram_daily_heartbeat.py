#!/usr/bin/env python3
"""
Telegram Daily Heartbeat — Mechanical Portfolio Summary

Channel discipline:
  - Fixed time, same format every day, no extra commentary.
  - Telegram = heartbeat. Fund-ops = email/PDF only.
  - No reactive messaging. No reply-thread analysis.

Reads live state via ops/daily_summary.py, appends activation-window
status, sends once via execution/telegram_utils.send_telegram().

Logs every attempt (success or failure) to logs/telegram_heartbeat.jsonl.

Usage:
  PYTHONPATH=. python scripts/telegram_daily_heartbeat.py           # send
  PYTHONPATH=. python scripts/telegram_daily_heartbeat.py --dry-run # preview
  PYTHONPATH=. python scripts/telegram_daily_heartbeat.py --test    # send test ping
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ── Paths ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_HEARTBEAT_LOG = _ROOT / "logs" / "telegram_heartbeat.jsonl"
_AW_STATE = _ROOT / "logs" / "state" / "activation_window_state.json"


def _safe_json(path: Path) -> Dict[str, Any]:
    """Load JSON, return empty dict on failure."""
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def _activation_window_block() -> str:
    """Build the activation-window status line(s) for the heartbeat."""
    state = _safe_json(_AW_STATE)
    if not state:
        return "Cert Window:      —"

    active = state.get("active", False)
    halted = state.get("halted", False)
    elapsed = state.get("elapsed_days", 0)
    total = state.get("duration_days", 14)
    remaining = state.get("remaining_days", 0)
    dd = state.get("drawdown_pct", 0)
    manifest_ok = state.get("manifest_intact", True)
    config_ok = state.get("config_intact", True)
    expired = state.get("window_expired", False)

    if halted:
        return f"Cert Window:      HALTED — {state.get('halt_reason', 'unknown')}"
    if expired:
        return "Cert Window:      COMPLETED ✓"
    if not active:
        return "Cert Window:      INACTIVE"

    integrity = "OK" if (manifest_ok and config_ok) else "DRIFT"
    return (
        f"Cert Window:      Day {elapsed:.0f}/{total}  "
        f"({remaining:.0f}d left)  DD:{dd:.2f}%  Integrity:{integrity}"
    )


def build_heartbeat_message(now: Optional[datetime] = None) -> str:
    """Generate complete heartbeat message: daily summary + activation window.

    Returns the plain-text string ready for Telegram (no markdown).
    """
    # Lazy import to avoid import-time side effects
    sys.path.insert(0, str(_ROOT))
    from ops.daily_summary import generate_daily_summary

    summary = generate_daily_summary(now=now)
    aw_block = _activation_window_block()

    # Insert activation-window status before the closing border
    lines = summary.split("\n")
    # Find the last border line and insert before it
    close_idx = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("═"):
            close_idx = i
            break
    lines.insert(close_idx, aw_block)
    lines.insert(close_idx, "")  # blank separator

    return "\n".join(lines)


def _log_attempt(ok: bool, dry_run: bool, message_len: int, error: str = "") -> None:
    """Append heartbeat attempt to JSONL log (append-only)."""
    try:
        _HEARTBEAT_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": int(time.time()),
            "utc": datetime.now(timezone.utc).isoformat(),
            "action": "heartbeat_daily",
            "dry_run": dry_run,
            "sent": ok,
            "message_len": message_len,
        }
        if error:
            entry["error"] = error
        with open(_HEARTBEAT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging failure must never block


def send_heartbeat(dry_run: bool = False) -> bool:
    """Build and send the daily heartbeat.

    Returns True if message was sent (or dry-run previewed) successfully.
    """
    now = datetime.now(timezone.utc)
    message = build_heartbeat_message(now=now)

    if dry_run:
        print("═══ DRY RUN — Message preview ═══")
        print(message)
        print(f"\n[{len(message)} chars, would send to Telegram]")
        _log_attempt(ok=True, dry_run=True, message_len=len(message))
        return True

    # Send via the existing telegram_utils infrastructure
    try:
        from execution.telegram_utils import send_telegram
        ok = send_telegram(message, silent=False)
        if ok:
            print(f"✅ Daily heartbeat sent ({len(message)} chars)")
        else:
            print("❌ Daily heartbeat send returned False (check TELEGRAM_ENABLED)")
        _log_attempt(ok=ok, dry_run=False, message_len=len(message))
        return ok
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"❌ Daily heartbeat failed: {error_msg}")
        _log_attempt(ok=False, dry_run=False, message_len=len(message), error=error_msg)
        return False


def send_test_ping() -> bool:
    """Send a minimal test message to confirm Telegram delivery."""
    try:
        from execution.telegram_utils import send_telegram
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = f"🟢 Heartbeat test ping — {now}"
        ok = send_telegram(msg, silent=False)
        if ok:
            print(f"✅ Test ping sent: {msg}")
        else:
            print("❌ Test ping failed (check TELEGRAM_ENABLED)")
        return ok
    except Exception as exc:
        print(f"❌ Test ping exception: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram Daily Heartbeat — mechanical portfolio summary"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview message without sending",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a minimal test ping to confirm delivery",
    )
    args = parser.parse_args()

    if args.test:
        ok = send_test_ping()
        sys.exit(0 if ok else 1)

    ok = send_heartbeat(dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
