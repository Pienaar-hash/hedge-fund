#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time

from utils.firestore_client import get_db

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--firestore", action="store_true", help="also purge firestore")
    args = ap.parse_args()

    clear_local_logs()
    if args.firestore:
        clear_firestore()
    print("✅ cleanup complete")
