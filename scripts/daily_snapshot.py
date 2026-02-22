#!/usr/bin/env python3
"""
Daily State Snapshot — archives critical fund state for disaster recovery.

Designed to run via cron every day. Creates a timestamped archive of:
  - All logs/state/ JSON files (point-in-time snapshots)
  - NAV log (equity curve history)
  - Cache files (peak_state, nav_confirmed)
  - Episode ledger
  - Execution log stats (line counts, not full copies — those are huge)

Archives are stored in logs/snapshots/YYYY-MM-DD/ and pruned after 30 days.

Usage:
    python scripts/daily_snapshot.py          # normal run
    python scripts/daily_snapshot.py --full   # also copies execution JSONL logs

Cron entry (add to crontab):
    0 0 * * * cd /root/hedge-fund && PYTHONPATH=. ./venv/bin/python scripts/daily_snapshot.py >> /var/log/hedge-snapshot.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


SNAPSHOT_BASE = Path("logs/snapshots")
STATE_DIR = Path("logs/state")
CACHE_DIR = Path("logs/cache")
EXEC_LOG_DIR = Path("logs/execution")
NAV_LOG = Path("logs/nav_log.json")
DOCTRINE_LOG = Path("logs/doctrine_events.jsonl")
RETENTION_DAYS = 30


def _today_label() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _snapshot_dir() -> Path:
    d = SNAPSHOT_BASE / _today_label()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _copy_safe(src: Path, dest_dir: Path) -> bool:
    """Copy a file to dest_dir, returning True on success."""
    if not src.exists():
        return False
    try:
        shutil.copy2(str(src), str(dest_dir / src.name))
        return True
    except Exception as e:
        print(f"  WARN: Failed to copy {src}: {e}")
        return False


def _count_lines(path: Path) -> int:
    """Count lines in a file without loading it all into memory."""
    try:
        count = 0
        with open(path, "rb") as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return -1


def snapshot_state(full: bool = False) -> dict:
    """Take a daily state snapshot.

    Args:
        full: If True, also copy all execution JSONL files (can be 300+ MB).

    Returns:
        Summary dict with file counts and sizes.
    """
    snap_dir = _snapshot_dir()
    ts = time.time()
    copied = 0
    skipped = 0
    total_bytes = 0

    print(f"[snapshot] {_today_label()} → {snap_dir}")

    # 1. State files (logs/state/*.json)
    state_sub = snap_dir / "state"
    state_sub.mkdir(exist_ok=True)
    if STATE_DIR.exists():
        for f in STATE_DIR.glob("*.json"):
            if _copy_safe(f, state_sub):
                total_bytes += f.stat().st_size
                copied += 1
            else:
                skipped += 1
    print(f"  State files: {copied} copied")

    # 2. Cache files
    cache_sub = snap_dir / "cache"
    cache_sub.mkdir(exist_ok=True)
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            if _copy_safe(f, cache_sub):
                total_bytes += f.stat().st_size
                copied += 1

    # 3. NAV log (equity curve — critical)
    if _copy_safe(NAV_LOG, snap_dir):
        total_bytes += NAV_LOG.stat().st_size
        copied += 1
        print(f"  NAV log: {NAV_LOG.stat().st_size / 1024:.1f} KB")

    # 4. Execution log inventory (always — lightweight metadata)
    if EXEC_LOG_DIR.exists():
        inventory = {}
        for f in sorted(EXEC_LOG_DIR.glob("*.jsonl")):
            inventory[f.name] = {
                "lines": _count_lines(f),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc
                ).isoformat()[:19] + "Z",
            }
        inv_path = snap_dir / "execution_log_inventory.json"
        inv_path.write_text(json.dumps(inventory, indent=2))
        print(f"  Execution log inventory: {len(inventory)} files tracked")

        # 5. Full execution log copy (optional, large)
        if full:
            exec_sub = snap_dir / "execution"
            exec_sub.mkdir(exist_ok=True)
            for f in EXEC_LOG_DIR.glob("*.jsonl"):
                if _copy_safe(f, exec_sub):
                    total_bytes += f.stat().st_size
                    copied += 1
            print(f"  Execution logs: full copy ({total_bytes / 1024 / 1024:.1f} MB)")

    # 6. Doctrine events (large but irreplaceable audit trail)
    if DOCTRINE_LOG.exists():
        doc_lines = _count_lines(DOCTRINE_LOG)
        doc_size = DOCTRINE_LOG.stat().st_size
        # Only full-copy if < 50 MB, otherwise just record metadata
        if full or doc_size < 50 * 1024 * 1024:
            if _copy_safe(DOCTRINE_LOG, snap_dir):
                total_bytes += doc_size
                copied += 1
                print(f"  Doctrine events: {doc_lines} lines ({doc_size / 1024 / 1024:.1f} MB)")
        else:
            meta = {"lines": doc_lines, "size_bytes": doc_size}
            (snap_dir / "doctrine_events_meta.json").write_text(json.dumps(meta))
            print(f"  Doctrine events: metadata only ({doc_size / 1024 / 1024:.1f} MB too large)")

    # 7. Write snapshot manifest
    manifest = {
        "snapshot_ts": datetime.now(tz=timezone.utc).isoformat()[:19] + "Z",
        "snapshot_unix": ts,
        "date": _today_label(),
        "files_copied": copied,
        "files_skipped": skipped,
        "total_bytes": total_bytes,
        "full_mode": full,
    }
    (snap_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - ts
    print(f"  ✓ {copied} files, {total_bytes / 1024:.0f} KB in {elapsed:.1f}s")

    return manifest


def prune_old_snapshots(retention_days: int = RETENTION_DAYS) -> int:
    """Remove snapshots older than retention_days. Returns count removed."""
    if not SNAPSHOT_BASE.exists():
        return 0

    cutoff = datetime.now(tz=timezone.utc).timestamp() - (retention_days * 86400)
    removed = 0

    for d in sorted(SNAPSHOT_BASE.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "snapshot_manifest.json"
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text())
                if m.get("snapshot_unix", 0) < cutoff:
                    shutil.rmtree(str(d))
                    removed += 1
                    continue
            except Exception:
                pass
        # Fallback: parse directory name as date
        try:
            dir_date = datetime.strptime(d.name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if dir_date.timestamp() < cutoff:
                shutil.rmtree(str(d))
                removed += 1
        except ValueError:
            pass

    if removed:
        print(f"[snapshot] Pruned {removed} snapshots older than {retention_days}d")
    return removed


def main():
    parser = argparse.ArgumentParser(description="Daily state snapshot")
    parser.add_argument("--full", action="store_true",
                        help="Include full execution JSONL logs (large)")
    args = parser.parse_args()

    snapshot_state(full=args.full)
    prune_old_snapshots()


if __name__ == "__main__":
    main()
