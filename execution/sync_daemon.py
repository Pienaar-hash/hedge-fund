import os, sys, time, json, traceback
from datetime import datetime, timezone

from utils.firestore_client import get_db
from execution.sync_state import sync_leaderboard, sync_nav, sync_positions
from execution.exchange_utils import get_positions  # fallback if no local positions

LEADERBOARD_FILE = "leaderboard.json"
NAV_FILE        = "nav_log.json"
PEAK_FILE       = "peak_state.json"
STATE_FILE      = "synced_state.json"   # adjust if your project uses a different name

INTERVAL = int(os.getenv("SYNC_INTERVAL_SEC", "15"))

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

if __name__ == "__main__":
    db = get_db()
    env = os.getenv("ENV", "prod")
    while True:
        try:
            run_once(db, env)
            sys.stdout.write("✔ sync ok\n"); sys.stdout.flush()
        except Exception:
            traceback.print_exc()
            time.sleep(5)
        time.sleep(INTERVAL)
