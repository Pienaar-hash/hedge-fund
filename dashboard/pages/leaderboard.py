# dashboard/pages/leaderboard.py
import streamlit as st, json, os
import pandas as pd
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="🏆 Leaderboard", layout="wide")
st.title("🏆 Leaderboard")

STATE_FILE = "synced_state.json"
PEAK_FILE  = "peak_state.json"

def _load_json(path):
    if not Path(path).exists(): return {}
    try: return json.load(open(path,"r"))
    except: return {}

def compute_local_assets(state:dict):
    rows=[]
    for sym, pos in state.items():
        qty   = float(pos.get("qty") or 0.0)
        entry = float(pos.get("entry") or 0.0)
        last  = float(pos.get("latest_price") or 0.0)
        pnl   = (last-entry)*qty
        ret   = ((last-entry)/entry) if entry else 0.0
        rows.append({"Symbol": sym, "Qty": qty, "Entry": entry, "Last": last, "Unrealized": pnl, "Return %": ret*100})
    return pd.DataFrame(rows).sort_values("Unrealized", ascending=False)

def compute_local_strats(state:dict, peak:dict):
    values={}
    for sym, pos in state.items():
        strat = pos.get("strategy") or "unknown"
        values[strat] = values.get(strat,0.0) + float(pos.get("qty") or 0.0) * float(pos.get("latest_price") or 0.0)
    peaks = (peak or {}).get("strategies", {})
    rows=[]
    for k,v in values.items():
        pk = float(peaks.get(k,0.0))
        dd = ((v-pk)/pk*100) if pk else 0.0
        rows.append({"Strategy": k, "Current Value": v, "Peak": pk, "DD %": dd})
    return pd.DataFrame(rows).sort_values("Current Value", ascending=False)

@st.cache_data(ttl=60)
def fetch_firestore():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        cred_path = os.getenv("FIREBASE_CREDS_PATH","config/firebase_creds.json")
        if not Path(cred_path).exists():
            return None, None
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        db = firestore.client()
        a = db.collection("hedge_leaderboard").document("assets").get().to_dict()
        s = db.collection("hedge_leaderboard").document("strategies").get().to_dict()
        return a, s
    except Exception:
        return None, None

assets_remote, strats_remote = fetch_firestore()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top Assets")
    if assets_remote and "rows" in assets_remote:
        df = pd.DataFrame(assets_remote["rows"])
        df["Return %"] = df.get("return_pct",0.0)*100
        df = df.rename(columns={"symbol":"Symbol","unrealized":"Unrealized","qty":"Qty","entry":"Entry","last":"Last"})
        st.dataframe(df[["Symbol","Qty","Entry","Last","Unrealized","Return %"]], use_container_width=True)
        st.caption(f"Updated: {assets_remote.get('updated_at','—')}")
    else:
        df = compute_local_assets(_load_json(STATE_FILE))
        st.dataframe(df, use_container_width=True)
        st.caption("Local fallback (Firestore unavailable)")

with c2:
    st.subheader("Top Strategies")
    if strats_remote and "rows" in strats_remote:
        df = pd.DataFrame(strats_remote["rows"])
        df["DD %"] = df.get("drawdown_pct",0.0)*100
        df = df.rename(columns={"strategy":"Strategy","current_value":"Current Value","peak_value":"Peak"})
        st.dataframe(df[["Strategy","Current Value","Peak","DD %"]], use_container_width=True)
        st.caption(f"Updated: {strats_remote.get('updated_at','—')}")
    else:
        df = compute_local_strats(_load_json(STATE_FILE), _load_json(PEAK_FILE))
        st.dataframe(df, use_container_width=True)
        st.caption("Local fallback (Firestore unavailable)")
