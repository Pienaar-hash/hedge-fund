import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from execution.log_utils import get_logger, log_event
from execution.exchange_utils import get_positions  # fallback if no local positions
from execution.sync_state import sync_leaderboard, sync_nav, sync_positions
from utils.firestore_client import get_db, publish_heartbeat

LOGGER = logging.getLogger("sync_daemon")
_ENV_DEFAULT = "prod"
_ENV = (os.environ.get("HEDGE_ENV") or os.environ.get("ENV") or _ENV_DEFAULT).strip() or _ENV_DEFAULT
os.environ["ENV"] = _ENV

LEADERBOARD_FILE = "leaderboard.json"
NAV_FILE = "nav_log.json"
PEAK_FILE = "logs/cache/peak_state.json"
STATE_FILE = "synced_state.json"  # adjust if your project uses a different name

SOURCE_MAX_AGE_SEC = int(os.getenv("SYNC_SOURCE_MAX_AGE_SEC", "600"))
NAV_MAX_AGE_SEC = int(os.getenv("SYNC_NAV_MAX_AGE_SEC", str(SOURCE_MAX_AGE_SEC)))
PEAK_MAX_AGE_SEC = int(os.getenv("SYNC_PEAK_MAX_AGE_SEC", str(SOURCE_MAX_AGE_SEC)))
POSITIONS_MAX_AGE_SEC = int(os.getenv("SYNC_POSITIONS_MAX_AGE_SEC", str(SOURCE_MAX_AGE_SEC)))

INTERVAL = int(os.getenv("SYNC_INTERVAL_SEC", "15"))
HEARTBEAT_INTERVAL = 60.0
LOG_HEART = get_logger("logs/execution/sync_heartbeats.jsonl")

_LAST_HEARTBEAT = 0.0
_LATENCY_MA_MS: float | None = None
_ERROR_COUNT = 0
_LAST_SUCCESS_TS: float | None = None


def load_json_safe(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _current_env() -> str:
    env = (os.environ.get("HEDGE_ENV") or os.environ.get("ENV") or _ENV).strip() or _ENV_DEFAULT
    os.environ["ENV"] = env
    return env


def _is_fresh(path: str, max_age_s: int) -> bool:
    try:
        return (time.time() - os.path.getmtime(path)) <= max_age_s
    except Exception:
        return False


def _require_fresh(path: str, max_age_s: int, label: str) -> None:
    if not _is_fresh(path, max_age_s):
        raise RuntimeError(f"{label} stale (> {max_age_s}s) or missing at {path}")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_data_payload():
    # Leaderboard: list of rows or empty list
    if _is_fresh(LEADERBOARD_FILE, SOURCE_MAX_AGE_SEC):
        leaderboard = load_json_safe(LEADERBOARD_FILE, [])
    else:
        LOGGER.warning("[syncd] stale leaderboard cache: %s", LEADERBOARD_FILE)
        leaderboard = []

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
    if _is_fresh(STATE_FILE, POSITIONS_MAX_AGE_SEC):
        state = load_json_safe(STATE_FILE, {})
    else:
        LOGGER.warning("[syncd] stale or missing positions cache: %s", STATE_FILE)
        state = {}
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
    env = env or _current_env()
    _require_fresh(NAV_FILE, NAV_MAX_AGE_SEC, "nav_log")
    _require_fresh(PEAK_FILE, PEAK_MAX_AGE_SEC, "peak_state")
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


def _publish_daemon_heartbeat(
    db,
    env: str,
    status: str,
    work_items: int,
    error: Exception | None,
) -> None:
    extra: Dict[str, Any] = {
        "work_items": int(work_items),
        "latency_ms": _LATENCY_MA_MS,
    }
    if error is not None:
        extra["last_error"] = str(error)
    try:
        publish_heartbeat(
            db,
            env,
            "sync_daemon",
            status,
            error_count=_ERROR_COUNT,
            last_success_ts=_LAST_SUCCESS_TS,
            extra=extra,
        )
    except Exception as exc:
        LOGGER.warning("[syncd] heartbeat_publish_failed: %s", exc)


def main() -> None:
    global _ERROR_COUNT, _LAST_SUCCESS_TS, _LATENCY_MA_MS
    db = get_db(strict=True)
    env = _current_env()
    while True:
        start = time.perf_counter()
        status = "ok"
        error: Exception | None = None
        try:
            work_items = run_once(db, env)
            sys.stdout.write("✔ sync ok\n")
            sys.stdout.flush()
            _ERROR_COUNT = 0
            _LAST_SUCCESS_TS = time.time()
        except Exception as exc:
            _ERROR_COUNT += 1
            status = "degraded"
            error = exc
            traceback.print_exc()
            time.sleep(5)
            work_items = 0
        latency_ms = (time.perf_counter() - start) * 1000.0
        if _LATENCY_MA_MS is None:
            _LATENCY_MA_MS = latency_ms
        else:
            _LATENCY_MA_MS = (_LATENCY_MA_MS * 0.8) + (latency_ms * 0.2)
        _maybe_emit_heartbeat(work_items)
        _publish_daemon_heartbeat(db, env, status, work_items, error)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
