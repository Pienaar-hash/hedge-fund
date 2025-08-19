
import os
import sys
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

# Ensure the project root is importable (adjust if your project root differs)
PROJECT_ROOT = "/root/hedge-fund"
if PROJECT_ROOT not in sys.path and os.path.isdir(PROJECT_ROOT):
    sys.path.insert(0, PROJECT_ROOT)

# Local package import (our helpers live alongside this file in dashboard/)
from dashboard.dashboard_utils import (
    get_firestore_connection,
    fetch_state_document,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    read_trade_log_tail,
    fmt_ccy,
    fmt_pct,
)

# --------------------------- Page config -------------------------------------
st.set_page_config(page_title="Hedge — Portfolio Dashboard", layout="wide")
ENV = os.getenv("ENV", "prod")
REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "60"))
TRADE_LOG = os.getenv("TRADE_LOG", "trade_log.json")

# Optional auto-refresh (works if streamlit-extras installed)
try:
    from streamlit_extras.st_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    pass  # clean fallback: user can manually refresh

st.title("📊 Hedge — Portfolio Dashboard")
st.caption(f"ENV = {ENV} · refresh ≈ {REFRESH_SEC}s")

# --------------------------- Load Firestore ----------------------------------
status = st.empty()
status.info("Loading data from Firestore…")

try:
    db = get_firestore_connection()
    nav_doc = fetch_state_document("nav", env=ENV)
    pos_doc = fetch_state_document("positions", env=ENV)
    lb_doc  = fetch_state_document("leaderboard", env=ENV)
except Exception as e:
    st.error(f"Firestore read failed: {e}")
    st.stop()

nav_df, kpis = parse_nav_to_df_and_kpis(nav_doc)
positions = pos_doc.get("items") or []
positions = positions_sorted(positions)
leaderboard = lb_doc.get("items") or []

status.success(f"Loaded · updated_at={nav_doc.get('updated_at','—')}")

# --------------------------- KPI header --------------------------------------
with st.container():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", fmt_ccy(kpis["total_equity"]), delta=None)
    c2.metric("Peak", fmt_ccy(kpis["peak_equity"]))
    c3.metric("DD", fmt_pct(kpis["drawdown"]))
    c4.metric("U‑PnL", fmt_ccy(kpis["unrealized_pnl"]))
    c5.metric("R‑PnL", fmt_ccy(kpis["realized_pnl"]))

# Exposure row (if present on nav doc)
exp = {
    "gross_exposure": nav_doc.get("gross_exposure", 0.0),
    "net_exposure": nav_doc.get("net_exposure", 0.0),
    "largest_position_value": nav_doc.get("largest_position_value", 0.0),
}
st.subheader("Exposure")
e1, e2, e3 = st.columns(3)
e1.metric("Gross", fmt_ccy(exp["gross_exposure"]))
e2.metric("Net", fmt_ccy(exp["net_exposure"]))
e3.metric("Largest Pos", fmt_ccy(exp["largest_position_value"]))

# --------------------------- NAV chart ---------------------------------------
st.subheader("NAV Equity Curve")
if nav_df.empty:
    st.info("No NAV points yet. Run executor + sync_state to populate.")
else:
    st.line_chart(nav_df["equity"], use_container_width=True)

# --------------------------- Positions / Leaderboard -------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Open Positions")
    if positions:
        df_pos = pd.DataFrame(positions)
        # Order columns for readability if present
        cols = [c for c in ["symbol","side","qty","entry_price","mark_price","pnl","notional","leverage","ts"] if c in df_pos.columns]
        st.dataframe(df_pos[cols], use_container_width=True, hide_index=True)
    else:
        st.write("No open positions.")

with right:
    st.subheader("Leaderboard")
    if leaderboard:
        df_lb = pd.DataFrame(leaderboard)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)
    else:
        st.write("No leaderboard items.")

# --------------------------- Recent Trades (local file) ----------------------
st.subheader("Recent Trades (tail of trade_log.json)")
trades = read_trade_log_tail(TRADE_LOG, tail=10)
if trades:
    df_tr = pd.DataFrame(trades)
    # Friendly column ordering if keys exist
    order = [c for c in ["ts","symbol","side","qty","price","notional","realized_pnl","comment","strategy"] if c in df_tr.columns]
    if order:
        df_tr = df_tr[order]
    st.dataframe(df_tr, use_container_width=True, hide_index=True)
else:
    st.caption(f"No trades found at {TRADE_LOG} (dry‑run or not yet emitted).")

st.caption(f"Data source: Firestore hedge/{ENV}/state/* · Optional local trade log: {TRADE_LOG}")
