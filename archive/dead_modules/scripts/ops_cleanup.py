#!/usr/bin/env python3
from pathlib import Path
import argparse
import gzip
import shutil
from datetime import datetime

LOG_DIR = Path("logs")

PURGE_FILES = [
    "veto_exec_ETHUSDT.json",
    "veto_exec_BTCUSDT.json",
    "veto_exec_LINKUSDT.json",
    "veto_exec_SOLUSDT.json",
    "veto_exec_SUIUSDT.json",
]

PURGE_GLOBS = [
    "audit_orders_*.jsonl",
    "positions.jsonl",
    "nav.jsonl.bak",
]

PURGE_COLLECTIONS = [
    "orders_replay",
    "positions_replay",
    "nav_replay",
    "veto_exec",        # if you materialize veto docs here
    # add any extra temp collections used by dashboard
]

def clear_local_logs():
    removed = 0
    for name in PURGE_FILES:
        p = LOG_DIR / name
        if p.exists():
            p.unlink()
            removed += 1
    for pattern in PURGE_GLOBS:
        for p in LOG_DIR.glob(pattern):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    print(f"[cleanup] removed {removed} local log files")

def clear_firestore():
    from utils.firestore_client import get_db

    db = get_db()
    total = 0
    for coll in PURGE_COLLECTIONS:
        try:
            stream = list(db.collection(coll).limit(1000).stream())
        except Exception as exc:
            print(f"[cleanup] skip {coll}: {exc}")
            continue
        while stream:
            for doc in stream:
                try:
                    doc.reference.delete()
                    total += 1
                except Exception as exc:
                    print(f"[cleanup] delete failed {coll}/{doc.id}: {exc}")
            stream = list(db.collection(coll).limit(1000).stream())
        print(f"[cleanup] purged collection {coll}")
    print(f"[cleanup] total firestore docs deleted: {total}")


def rotate_execution_logs(max_bytes: int = 5_000_000, keep: int = 10) -> None:
    base = LOG_DIR / "execution"
    if not base.exists():
        print(f"[cleanup] execution log dir missing: {base}")
        return
    archive_dir = base / "archive_manual"
    archive_dir.mkdir(parents=True, exist_ok=True)
    rotated = 0
    for path in sorted(base.glob("*.jsonl")):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size <= max_bytes:
            continue
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{path.name}.{ts}.gz"
        try:
            with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.write_text("", encoding="utf-8")
            rotated += 1
            _prune_archives(archive_dir, path.name, keep)
            print(f"[cleanup] rotated {path} -> {archive_path}")
        except Exception as exc:
            print(f"[cleanup] rotation failed for {path}: {exc}")
    if rotated == 0:
        print("[cleanup] no execution logs required rotation")
    else:
        print(f"[cleanup] rotated {rotated} execution logs")


def _prune_archives(archive_dir: Path, base_name: str, keep: int) -> None:
    if keep <= 0:
        return
    pattern = f"{base_name}.*.gz"
    try:
        archives = sorted(
            archive_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except FileNotFoundError:
        return
    for stale in archives[keep:]:
        try:
            stale.unlink()
        except Exception:
            continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--firestore", action="store_true", help="also purge firestore")
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="rotate large execution logs into compressed archives",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=5_000_000,
        help="size threshold (bytes) before rotation triggers",
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=10,
        help="number of archived files to retain per log",
    )
    args = ap.parse_args()

    clear_local_logs()
    if args.firestore:
        clear_firestore()
    if args.rotate:
        rotate_execution_logs(max_bytes=args.max_bytes, keep=args.keep)
    print("✅ cleanup complete")
