import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from execution.log_utils import get_logger, log_event
from execution.exchange_utils import get_positions  # fallback if no local positions
from execution.sync_state import sync_leaderboard, sync_nav, sync_positions
from utils.firestore_client import get_db

LEADERBOARD_FILE = "leaderboard.json"
NAV_FILE = "nav_log.json"
PEAK_FILE = "peak_state.json"
STATE_FILE = "synced_state.json"  # adjust if your project uses a different name

INTERVAL = int(os.getenv("SYNC_INTERVAL_SEC", "15"))
HEARTBEAT_INTERVAL = 60.0
LOG_HEART = get_logger("logs/execution/sync_heartbeats.jsonl")

_LAST_HEARTBEAT = 0.0
_LATENCY_MA_MS: float | None = None


def load_json_safe(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_data_payload():
    # Leaderboard: list of rows or empty list
    leaderboard = load_json_safe(LEADERBOARD_FILE, [])

    # NAV: series (list of {t, equity}) and peak (float)
    series = load_json_safe(NAV_FILE, [])
    peak_state = load_json_safe(PEAK_FILE, {})
    peak = peak_state.get("peak")
    if peak is None:
        try:
            peak = max((pt.get("equity", 0.0) for pt in series), default=0.0)
        except Exception:
            peak = 0.0

    nav = {
        "series": series,
        "peak": peak,
        "updated_at": now_iso(),
    }

    # Positions: prefer local state file; fallback to live exchange query
    state = load_json_safe(STATE_FILE, {})
    positions = state.get("positions")
    if not isinstance(positions, list):
        try:
            positions = get_positions()
        except Exception:
            positions = []

    positions_payload = {
        "rows": positions,
        "updated_at": now_iso(),
    }

    return {
        "leaderboard": leaderboard,
        "nav": nav,
        "positions": positions_payload,
    }


def run_once(db, env):
    data = build_data_payload()
    # sync_* expect (db, data, env)
    sync_leaderboard(db, data, env)
    sync_nav(db, data, env)
    sync_positions(db, data, env)
    rows = data.get("positions", {}).get("rows") or []
    leaderboard_rows = data.get("leaderboard") or []
    nav_series = data.get("nav", {}).get("series") or []
    work_items = len(rows) + len(leaderboard_rows) + len(nav_series)
    return work_items


def _maybe_emit_heartbeat(work_items: int) -> None:
    global _LAST_HEARTBEAT, _LATENCY_MA_MS
    now = time.time()
    if (now - _LAST_HEARTBEAT) < HEARTBEAT_INTERVAL:
        return
    _LAST_HEARTBEAT = now
    payload = {
        "service": "sync_daemon",
        "ts": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "work_items": int(work_items),
        "latency_ms": _LATENCY_MA_MS,
    }
    try:
        log_event(LOG_HEART, "heartbeat", payload)
    except Exception:
        pass


if __name__ == "__main__":
    db = get_db()
    env = os.getenv("ENV", "prod")
    while True:
        start = time.perf_counter()
        try:
            work_items = run_once(db, env)
            sys.stdout.write("✔ sync ok\n")
            sys.stdout.flush()
        except Exception:
            traceback.print_exc()
            time.sleep(5)
            work_items = 0
        latency_ms = (time.perf_counter() - start) * 1000.0
        if _LATENCY_MA_MS is None:
            _LATENCY_MA_MS = latency_ms
        else:
            _LATENCY_MA_MS = (_LATENCY_MA_MS * 0.8) + (latency_ms * 0.2)
        _maybe_emit_heartbeat(work_items)
        time.sleep(INTERVAL)
