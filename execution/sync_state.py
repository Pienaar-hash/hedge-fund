# execution/sync_state.py — Firestore-safe sync daemon

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# Import get_db with a guard for system python
try:
    from utils.firestore_client import get_db  # type: ignore
except Exception:
    import sys
    ROOT = "/root/hedge-fund"
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils.firestore_client import get_db  # type: ignore

# Config
NAV_LOG = os.getenv("NAV_LOG", "nav_log.json")            # local NAV (JSON array or JSON-Lines)
PEAK_STATE = os.getenv("PEAK_STATE", "peak_state.json")   # {"peak": float, "updated_at": iso}
STATE_FILE = os.getenv("STATE_FILE", "synced_state.json") # {"items":[...]} or list
SYNC_INTERVAL_SEC = int(os.getenv("SYNC_INTERVAL_SEC", "20"))
MAX_POINTS = int(os.getenv("NAV_MAX_POINTS", "1000"))

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _norm_t(rec: Dict[str, Any]) -> str | None:
    """Normalize time key: accept 't', 'ts', or 'timestamp'."""
    for k in ("t", "ts", "timestamp"):
        if k in rec and rec[k]:
            return str(rec[k])
    return None

def _read_nav_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    try:
        # First try whole-file JSON (array)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                t = _norm_t(rec)
                eq = rec.get("equity")
                if t is None or eq is None:
                    continue
                rows.append({"t": t, "equity": float(eq)})
            return rows
    except Exception:
        # Fall through to JSONL parse
        pass

    # JSON-Lines fallback
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                t = _norm_t(rec)
                eq = rec.get("equity")
                if t is None or eq is None:
                    continue
                rows.append({"t": t, "equity": float(eq)})
            except Exception:
                continue
    return rows

def _read_peak(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
            return float(
                d.get("peak")
                or d.get("peak_equity")
                or (d.get("portfolio") or {}).get("peak_equity")
                or 0.0
            )
    except Exception:
        return 0.0

def _read_positions(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"items": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
            if isinstance(d, dict) and "items" in d:
                return {"items": d.get("items") or []}
            if isinstance(d, list):
                return {"items": d}
            return {"items": []}
    except Exception:
        return {"items": []}

def commit_nav(db, env: str, rows: List[Dict[str, Any]], peak: float) -> Dict[str, Any]:
    if not rows:
        payload = {"series": [], "peak_equity": float(peak), "updated_at": _now_iso()}
    else:
        slim = rows[-MAX_POINTS:]
        payload = {"series": slim, "peak_equity": float(peak), "updated_at": slim[-1]["t"]}
    (db.collection("hedge").document(env).collection("state")
       .document("nav").set(payload, merge=True))
    return payload

def commit_positions(db, env: str, positions: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"items": positions.get("items") or [], "updated_at": _now_iso()}
    (db.collection("hedge").document(env).collection("state")
       .document("positions").set(payload, merge=True))
    return payload

def commit_leaderboard(db, env: str, positions: Dict[str, Any]) -> Dict[str, Any]:
    count = len(positions.get("items") or [])
    payload = {"items": [{"name": "portfolio", "count": count}], "updated_at": _now_iso()}
    (db.collection("hedge").document(env).collection("state")
       .document("leaderboard").set(payload, merge=True))
    return payload

def commit_live(db, env: str, nav: Dict[str, Any], positions: Dict[str, Any], leaderboard: Dict[str, Any]) -> None:
    live = {"nav": nav, "positions": positions, "leaderboard": leaderboard, "updated_at": _now_iso()}
    (db.collection("hedge").document(env).collection("state")
       .document("live").set(live, merge=True))

def sync_once(db=None, env: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if db is None:
        db = get_db()
    env = env or os.getenv("ENV", "prod")
    rows = _read_nav_rows(NAV_LOG)
    peak = _read_peak(PEAK_STATE)
    positions = _read_positions(STATE_FILE)
    nav_payload = commit_nav(db, env, rows, peak)
    pos_payload = commit_positions(db, env, positions)
    lb_payload  = commit_leaderboard(db, env, positions)
    commit_live(db, env, nav_payload, pos_payload, lb_payload)
    return nav_payload, pos_payload, lb_payload

def run_daemon() -> None:
    db = get_db()
    env = os.getenv("ENV", "prod")
    interval = max(5, SYNC_INTERVAL_SEC)
    print(f"[sync] starting: ENV={env} interval={interval}s files=({NAV_LOG}, {PEAK_STATE}, {STATE_FILE})")
    while True:
        try:
            nav, pos, lb = sync_once(db=db, env=env)
            n = len(nav.get("series") or [])
            print(f"[sync] upsert ok: points={n} peak={nav.get('peak_equity')} at={nav.get('updated_at')}")
        except Exception as e:
            print(f"[sync] error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    run_daemon()

# ---- Back-compat shims for executor ----
def _norm_point(rec):
    # accept {"t":..} or {"ts":..} or {"timestamp":..}
    t = rec.get("t") or rec.get("ts") or rec.get("timestamp")
    return {"t": t, "equity": float(rec.get("equity", 0.0))} if t else None

def sync_nav(db, payload: dict, env: str):
    """
    Expected payload (executor): {
        "series": [{"ts" or "t" or "timestamp": iso, "equity": float}],
        "total_equity": float,
        "realized_pnl": float,
        "unrealized_pnl": float,
        "peak_equity": float,
        "drawdown": float
    }
    Writes Firestore-safe shape:
      nav: {
        series: [{"t": iso, "equity": float}, ...],
        peak_equity, total_equity, realized_pnl, unrealized_pnl, drawdown, updated_at
      }
    """
    series_in = payload.get("series") or []
    series = []
    for rec in series_in:
        p = _norm_point(rec if isinstance(rec, dict) else {})
        if p:
            series.append(p)
    updated_at = series[-1]["t"] if series else _now_iso()
    body = {
        "series": series[-MAX_POINTS:],
        "peak_equity": float(payload.get("peak_equity") or 0.0),
        "total_equity": float(payload.get("total_equity") or 0.0),
        "realized_pnl": float(payload.get("realized_pnl") or 0.0),
        "unrealized_pnl": float(payload.get("unrealized_pnl") or 0.0),
        "drawdown": float(payload.get("drawdown") or 0.0),
        "updated_at": updated_at,
    }
    (db.collection("hedge").document(env).collection("state")
       .document("nav").set(body, merge=True))

def sync_positions(db, payload: dict, env: str):
    items = payload.get("items") or []
    body = {"items": items, "updated_at": _now_iso()}
    (db.collection("hedge").document(env).collection("state")
       .document("positions").set(body, merge=True))

def sync_leaderboard(db, payload: dict, env: str):
    items = payload.get("items") or []
    body = {"items": items, "updated_at": _now_iso()}
    (db.collection("hedge").document(env).collection("state")
       .document("leaderboard").set(body, merge=True))
