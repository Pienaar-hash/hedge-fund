import sys
import os
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Tuple

# Force absolute project root in sys.path for imports
ROOT = "/root/hedge-fund"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from utils.firestore_client import get_db
except ModuleNotFoundError as e:
    raise ImportError(f"Cannot import get_db from utils.firestore_client; sys.path={sys.path}") from e

# --------------------
# Firestore helper (supports live doc fallback)
# --------------------

def fetch_state_document(doc_name: str, env: str = None) -> Dict[str, Any]:
    db = get_db()
    env = env or os.getenv("ENV", "prod")
    # Try direct doc
    doc_ref = db.collection("hedge").document(env).collection("state").document(doc_name)
    snapshot = doc_ref.get()
    if snapshot.exists:
        return snapshot.to_dict() or {}
    # Fallback to live doc shape
    live_ref = db.collection("hedge").document(env).collection("state").document("live")
    live_snap = live_ref.get()
    if live_snap.exists:
        live_data = live_snap.to_dict() or {}
        return live_data.get(doc_name, {}) if isinstance(live_data, dict) else {}
    return {}

# --------------------
# Data parsing helpers
# --------------------

def coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def parse_nav(nav: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize NAV payload to df+kpis.
    Accepts any of:
      - list of dicts: [{"t": iso, "equity": float}, ...]  ✅ Firestore-safe
      - list of lists: [[iso, equity], ...]                 ❌ Firestore rejects nested arrays; parse anyway
      - dict mapping iso->equity: {iso: equity, ...}
    """
    series = nav.get("series")

    # list[dict]
    if isinstance(series, list) and series and isinstance(series[0], dict):
        rows = [[row.get("t"), row.get("equity")] for row in series]
    # list[list]
    elif isinstance(series, list):
        rows = series
    # dict
    elif isinstance(series, dict):
        rows = [[t, v] for t, v in sorted(series.items())]
    else:
        rows = []

    df = pd.DataFrame(rows, columns=["t", "equity"]) if rows else pd.DataFrame(columns=["t", "equity"])  # type: ignore[call-arg]
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t").set_index("t")

    kpis = {
        "points": len(df),
        "peak_equity": coalesce(nav.get("peak_equity"), nav.get("peak"), 0.0),
        "total_equity": float(nav.get("total_equity")) if nav.get("total_equity") is not None
                        else (float(df["equity"].iloc[-1]) if len(df) else 0.0),
        "realized_pnl": float(nav.get("realized_pnl")) if nav.get("realized_pnl") is not None else 0.0,
        "unrealized_pnl": float(nav.get("unrealized_pnl")) if nav.get("unrealized_pnl") is not None else 0.0,
        "drawdown": float(nav.get("drawdown")) if nav.get("drawdown") is not None else 0.0,
        "updated_at": coalesce(nav.get("updated_at"), "—"),
    }
    return df, kpis

# --------------------
# UI rendering
# --------------------

st.set_page_config(page_title="Hedge Dashboard", layout="wide")
ENV = os.getenv("ENV", "prod")
REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "60"))

st.title("📊 Hedge Dashboard")

status = st.empty()
status.info("Loading Firestore…")

try:
    nav = fetch_state_document("nav", env=ENV)
    lb = fetch_state_document("leaderboard", env=ENV)
    pos = fetch_state_document("positions", env=ENV)
except Exception as e:
    st.error(f"Firestore read failed: {e}")
    st.stop()

leaderboard_rows = (lb.get("items") or lb.get("rows") or [])
positions_rows   = (pos.get("items") or pos.get("rows") or [])
nav_df, nav_kpis = parse_nav(nav)
updated_at = coalesce(nav_kpis.get("updated_at"), lb.get("updated_at"), pos.get("updated_at"), default="—")

status.success(f"Loaded · updated_at={updated_at}")

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity", f"{nav_kpis['total_equity']:.2f}")
c2.metric("Peak", f"{nav_kpis['peak_equity']:.2f}")
c3.metric("Realized PnL", f"{nav_kpis['realized_pnl']:.2f}")
c4.metric("Unrealized PnL", f"{nav_kpis['unrealized_pnl']:.2f}")
c5.metric("Drawdown", f"{nav_kpis['drawdown']:.2f}")

# NAV chart
st.subheader("NAV Equity Curve")
if nav_df.empty:
    st.info("No NAV points yet.")
else:
    st.line_chart(nav_df["equity"], use_container_width=True)

# Leaderboard
st.subheader("Leaderboard")
if leaderboard_rows:
    st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)
else:
    st.write("No leaderboard items yet.")

# Positions
st.subheader("Open Positions")
if positions_rows:
    st.dataframe(pd.DataFrame(positions_rows), use_container_width=True, hide_index=True)
else:
    st.write("No open positions.")

st.caption(f"Data source: Firestore · hedge/{ENV}/state/* (or live)")
