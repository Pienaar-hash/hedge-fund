#!/usr/bin/env python3
"""
Post–Testnet Reset Recovery — restores fund continuity after a Binance reset.

Run this AFTER a testnet reset is detected (manually or by reset_guard).
It reads the most recent daily snapshot and:

  1. Restores NAV log (equity curve) from snapshot → preserves chart history
  2. Writes a cycle watermark (environment_meta.json)
  3. Resets peak_state.json (prevents phantom drawdown from pre-reset peak)
  4. Logs the recovery event to environment_events.jsonl
  5. Appends a cycle_transition entry

Does NOT modify:
  - Episode ledger (survives reset intact — local data)
  - Execution JSONL logs (append-only, exchange-independent)
  - Doctrine events (local data, unaffected)

Usage:
    python scripts/post_reset_recovery.py                    # auto-detect latest snapshot
    python scripts/post_reset_recovery.py --snapshot-date 2026-02-22
    python scripts/post_reset_recovery.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SNAPSHOT_BASE = Path("logs/snapshots")
NAV_LOG = Path("logs/nav_log.json")
PEAK_STATE = Path("logs/cache/peak_state.json")
ENV_META = Path("logs/state/environment_meta.json")
ENV_EVENTS = Path("logs/execution/environment_events.jsonl")
CYCLE_TRANSITIONS = Path("logs/cycle_transitions.jsonl")
NAV_STATE = Path("logs/state/nav_state.json")


def _find_latest_snapshot() -> Optional[Path]:
    """Find the most recent snapshot directory."""
    if not SNAPSHOT_BASE.exists():
        return None
    dirs = sorted(
        [d for d in SNAPSHOT_BASE.iterdir() if d.is_dir()],
        reverse=True,
    )
    return dirs[0] if dirs else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _get_exchange_balance() -> float:
    """Read current exchange balance from nav_state if available."""
    try:
        data = _read_json(NAV_STATE)
        for key in ("nav_usd", "total_equity", "nav"):
            val = data.get(key)
            if val:
                return float(val)
    except Exception:
        pass
    return 10_000.0  # Binance testnet default


def _current_cycle_id() -> str:
    meta = _read_json(ENV_META)
    return meta.get("cycle_id", "CYCLE_TEST_000")


def _next_cycle_id() -> str:
    current = _current_cycle_id()
    if current.startswith("CYCLE_TEST_"):
        try:
            num = int(current.split("_")[-1])
            return f"CYCLE_TEST_{num + 1:03d}"
        except (ValueError, IndexError):
            pass
    # Also check cycle_transitions for the latest
    try:
        if CYCLE_TRANSITIONS.exists():
            with open(CYCLE_TRANSITIONS) as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                to_cycle = last.get("to_cycle", "")
                if to_cycle.startswith("CYCLE_"):
                    parts = to_cycle.replace("CYCLE_", "").replace("TEST_", "")
                    num = int(parts)
                    return f"CYCLE_TEST_{num + 1:03d}"
    except Exception:
        pass
    return "CYCLE_TEST_001"


def recover(
    snapshot_date: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Execute post-reset recovery.

    Returns summary dict.
    """
    now = time.time()
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()[:19] + "Z"

    # 1. Locate snapshot
    if snapshot_date:
        snap_dir = SNAPSHOT_BASE / snapshot_date
        if not snap_dir.exists():
            print(f"[recovery] ERROR: Snapshot not found: {snap_dir}")
            return {"status": "error", "reason": f"snapshot {snapshot_date} not found"}
    else:
        snap_dir = _find_latest_snapshot()
        if not snap_dir:
            print("[recovery] ERROR: No snapshots found in logs/snapshots/")
            return {"status": "error", "reason": "no snapshots"}

    print(f"[recovery] Using snapshot: {snap_dir.name}")
    manifest = _read_json(snap_dir / "snapshot_manifest.json")
    print(f"  Snapshot taken: {manifest.get('snapshot_ts', 'unknown')}")
    print(f"  Files in snapshot: {manifest.get('files_copied', '?')}")

    new_cycle = _next_cycle_id()
    old_cycle = _current_cycle_id()
    balance = _get_exchange_balance()

    print(f"  Cycle transition: {old_cycle} → {new_cycle}")
    print(f"  Current balance: ${balance:,.2f}")

    if dry_run:
        print("[recovery] DRY RUN — no changes made")
        return {"status": "dry_run", "snapshot": snap_dir.name, "new_cycle": new_cycle}

    actions = []

    # 2. Restore NAV log from snapshot (preserves equity curve)
    snap_nav = snap_dir / "nav_log.json"
    if snap_nav.exists():
        if NAV_LOG.exists():
            backup = NAV_LOG.with_suffix(".pre_recovery.json")
            shutil.copy2(str(NAV_LOG), str(backup))
            print(f"  Backed up current nav_log → {backup.name}")

        # Merge: snapshot nav entries + any new entries from current log
        snap_entries = json.loads(snap_nav.read_text()) if snap_nav.exists() else []
        current_entries = json.loads(NAV_LOG.read_text()) if NAV_LOG.exists() else []

        if isinstance(snap_entries, list) and snap_entries:
            snap_last_t = snap_entries[-1].get("t", 0)
            # Keep only current entries newer than snapshot
            new_entries = [e for e in current_entries if e.get("t", 0) > snap_last_t]

            # Add a discontinuity marker (NAV resets to exchange default)
            # This prevents the chart from showing a misleading spike/drop
            if snap_entries and balance > 0:
                marker_entry = {
                    "nav": balance,
                    "t": now - 1,  # Just before "now"
                    "unrealized_pnl": 0,
                    "_reset_marker": True,
                }
                merged = snap_entries + [marker_entry] + new_entries
            else:
                merged = snap_entries + new_entries

            NAV_LOG.write_text(json.dumps(merged))
            print(f"  NAV log restored: {len(snap_entries)} snapshot + {len(new_entries)} new = {len(merged)} total")
            actions.append("nav_log_restored")
    else:
        print("  WARN: No nav_log.json in snapshot — equity curve will restart from scratch")

    # 3. Reset peak state (prevent phantom drawdown)
    PEAK_STATE.parent.mkdir(parents=True, exist_ok=True)
    peak_data = {
        "peak_nav": balance,
        "peak_ts": now,
        "peak_date": now_iso[:10],
        "reset_reason": f"testnet_reset_{new_cycle}",
    }
    PEAK_STATE.write_text(json.dumps(peak_data, indent=2))
    print(f"  Peak state reset to ${balance:,.2f}")
    actions.append("peak_reset")

    # 4. Write environment meta (cycle watermark)
    ENV_META.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "cycle_id": new_cycle,
        "last_testnet_reset_ts": now,
        "last_testnet_reset_iso": now_iso,
        "pre_reset_snapshot": snap_dir.name,
        "pre_reset_balance": balance,
        "updated_ts": now,
    }
    ENV_META.write_text(json.dumps(meta, indent=2))
    print(f"  Environment meta: cycle={new_cycle}")
    actions.append("env_meta_written")

    # 5. Log recovery event
    recovery_event = {
        "ts": now_iso,
        "event": "POST_RESET_RECOVERY",
        "source": "post_reset_recovery.py",
        "cycle_id": new_cycle,
        "previous_cycle": old_cycle,
        "snapshot_used": snap_dir.name,
        "exchange_balance": balance,
        "actions": actions,
    }
    _append_jsonl(ENV_EVENTS, recovery_event)
    print(f"  Recovery event logged to environment_events.jsonl")

    # 6. Log cycle transition
    transition = {
        "ts": now_iso,
        "from_cycle": old_cycle,
        "to_cycle": new_cycle,
        "trigger": "testnet_reset",
        "snapshot": snap_dir.name,
    }
    _append_jsonl(CYCLE_TRANSITIONS, transition)
    print(f"  Cycle transition: {old_cycle} → {new_cycle}")

    print(f"\n[recovery] ✓ Recovery complete — {new_cycle} active")
    print(f"  Next steps:")
    print(f"    1. Verify dashboard loads: curl -s http://localhost:8501")
    print(f"    2. Restart executor if needed: sudo supervisorctl restart hedge:hedge-executor")
    print(f"    3. Verify NAV reads from exchange: cat logs/state/nav.json | python3 -m json.tool | head")

    return {
        "status": "ok",
        "cycle": new_cycle,
        "snapshot": snap_dir.name,
        "actions": actions,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-reset recovery")
    parser.add_argument("--snapshot-date", help="YYYY-MM-DD of snapshot to use (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    recover(snapshot_date=args.snapshot_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
