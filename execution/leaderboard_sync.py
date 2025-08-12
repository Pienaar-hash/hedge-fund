# execution/leaderboard_sync.py
import json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from firebase_admin import credentials, firestore, initialize_app
from execution.config_loader import get, load

STATE_FILE = "synced_state.json"
PEAK_FILE  = "peak_state.json"

def _load_json(path:str) -> Any:
    if not Path(path).exists(): return {} if path.endswith(".json") else []
    with open(path,"r") as f: return json.load(f)

def _pct(n,d): 
    try: return float(n)/float(d) if float(d)!=0 else 0.0
    except: return 0.0

def compute_assets(state:dict) -> List[dict]:
    rows=[]
    for sym, pos in state.items():
        qty   = float(pos.get("qty") or 0.0)
        entry = float(pos.get("entry") or 0.0)
        last  = float(pos.get("latest_price") or 0.0)
        pnl   = (last - entry)*qty
        ret   = _pct(last-entry, entry)
        rows.append({"symbol": sym, "qty": qty, "entry": entry, "last": last, "unrealized": pnl, "return_pct": ret})
    rows.sort(key=lambda r: r["unrealized"], reverse=True)
    return rows

def compute_strategies(state:dict, peak:dict) -> List[dict]:
    # current value per strategy (sum MTM of owned symbols)
    values={}
    for sym, pos in state.items():
        strat = pos.get("strategy") or "unknown"
        qty   = float(pos.get("qty") or 0.0)
        last  = float(pos.get("latest_price") or 0.0)
        values[strat] = values.get(strat,0.0) + qty*last

    peaks = peak.get("strategies",{}) if isinstance(peak, dict) else {}
    rows=[]
    for skey, val in values.items():
        pk = float(peaks.get(skey,0.0))
        dd = 0.0 if pk==0 else (val - pk)/pk
        rows.append({"strategy": skey, "current_value": val, "peak_value": pk, "drawdown_pct": dd})
    rows.sort(key=lambda r: r["current_value"], reverse=True)
    return rows

def init_db():
    load()
    cred_path = get("runtime.FIREBASE_CREDS_PATH", "config/firebase_creds.json")
    if not Path(cred_path).exists(): return None
    try:
        cred = credentials.Certificate(cred_path)
        initialize_app(cred)
        return firestore.client()
    except Exception:
        return None

def sync_once():
    if not get("execution.leaderboard_enabled", True):
        print("ℹ️ Leaderboard disabled.")
        return False

    state = _load_json(STATE_FILE)
    peak  = _load_json(PEAK_FILE)
    assets = compute_assets(state)
    strats = compute_strategies(state, peak)
    updated = datetime.now(timezone.utc).isoformat()

    db = init_db()
    if db is None:
        print("ℹ️ No Firestore creds — skipping remote write (local only).")
        return False

    try:
        db.collection("hedge_leaderboard").document("assets").set({"rows": assets, "updated_at": updated})
        db.collection("hedge_leaderboard").document("strategies").set({"rows": strats, "updated_at": updated})
        print(f"✅ Leaderboard synced at {updated}")
        return True
    except Exception as e:
        print(f"❌ Leaderboard sync failed: {e}")
        return False

if __name__ == "__main__":
    while True:
        sync_once()
        time.sleep(300)
