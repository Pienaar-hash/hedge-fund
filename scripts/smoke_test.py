#!/usr/bin/env python3
"""
End-to-end Firestore freshness smoke test.

Checks that executor and sync heartbeats are recent and that `scripts.doctor`
reports Firestore freshness OK.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

try:
    from utils.firestore_client import get_db
except Exception:  # pragma: no cover - optional dependency not installed
    get_db = None


class SmokeFailure(RuntimeError):
    """Raised when a smoke check fails."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Firestore freshness smoke test")
    parser.add_argument(
        "--env",
        default=os.getenv("ENV", "dev"),
        help="Firestore environment namespace (default ENV or dev)",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=int(os.getenv("SMOKE_MAX_AGE_SECONDS", "120")),
        help="Maximum allowed heartbeat age in seconds (default 120)",
    )
    return parser.parse_args()


def _to_epoch_seconds(value: Any) -> float | None:
    """Convert Firestore timestamp representations to seconds since epoch."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val / 1000.0 if val > 1e12 else val
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            return None
    return None


def _fetch_heartbeat(db: Any, env: str, process: str) -> Tuple[float, Dict[str, Any]]:
    path = f"hedge/{env}/health/{process}"
    doc_ref = db.collection("hedge").document(env).collection("health").document(process)
    if getattr(doc_ref, "_is_noop", False):
        raise SmokeFailure(f"Firestore client unavailable (no-op) for {path}")
    doc = doc_ref.get()
    if not getattr(doc, "exists", False):
        raise SmokeFailure(f"Missing heartbeat document at {path}")
    data = doc.to_dict() or {}
    ts_value = data.get("ts") or data.get("timestamp") or data.get("time")
    ts_epoch = _to_epoch_seconds(ts_value)
    if ts_epoch is None:
        raise SmokeFailure(f"Heartbeat missing valid timestamp at {path}")
    return ts_epoch, data


def _check_age(process: str, ts_epoch: float, threshold: int) -> float:
    age = time.time() - ts_epoch
    if age < 0:
        age = 0.0
    if age > threshold:
        raise SmokeFailure(
            f"{process} heartbeat stale ({age:.1f}s > {threshold}s threshold)"
        )
    return age


def _run_doctor(env_name: str) -> str:
    if sys.executable is None:
        raise SmokeFailure("Cannot determine Python executable for doctor run")
    doctor_env = os.environ.copy()
    doctor_env.setdefault("ENV", env_name)
    repo_root = os.getcwd()
    current_path = doctor_env.get("PYTHONPATH", "")
    doctor_env["PYTHONPATH"] = (
        repo_root if not current_path else f"{repo_root}:{current_path}"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.doctor"],
        capture_output=True,
        text=True,
        env=doctor_env,
        check=False,
    )
    output = proc.stdout.strip().splitlines()
    if proc.returncode != 0:
        raise SmokeFailure(f"scripts.doctor exited with {proc.returncode}")
    summary_line = next((line for line in reversed(output) if "[doctor] summary" in line), None)
    if not summary_line:
        raise SmokeFailure("scripts.doctor summary line not found")
    summary_plain = _strip_ansi(summary_line)
    if "FIRESTORE: OK" not in summary_plain.upper():
        raise SmokeFailure(f"Doctor freshness not OK: {summary_plain}")
    return summary_plain


def _strip_ansi(text: str) -> str:
    ansi_pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_pattern.sub("", text)


def main() -> int:
    args = _parse_args()
    if get_db is None:
        raise SmokeFailure("Firestore client unavailable: install google-cloud-firestore")
    db = get_db()
    executor_ts, _ = _fetch_heartbeat(db, args.env, "executor_live")
    sync_ts, _ = _fetch_heartbeat(db, args.env, "sync_state")
    executor_age = _check_age("executor_live", executor_ts, args.max_age_seconds)
    sync_age = _check_age("sync_state", sync_ts, args.max_age_seconds)
    summary = _run_doctor(args.env)
    verdict = (
        f"SMOKE PASS env={args.env} executor={executor_age:.1f}s "
        f"sync_state={sync_age:.1f}s doctor=\"{summary}\""
    )
    print(verdict)
    return 0


def _entry() -> int:
    try:
        return main()
    except SmokeFailure as exc:
        print(f"SMOKE FAIL {exc}", file=sys.stdout)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"SMOKE FAIL unexpected error: {exc}", file=sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entry())
