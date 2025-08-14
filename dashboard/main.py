import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Firestore readers (single source of truth)
from dashboard.firestore_helpers import read_leaderboard, read_nav, read_positions

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hedge — Investor Dashboard", layout="wide")
st.title("Hedge — Investor Dashboard")
st.caption("Live Leaderboard • Portfolio NAV • Positions (Firestore)")

# --- AUTO-REFRESH (60s) ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="ref60")
except Exception:
    # If the optional helper isn't installed, we keep the page manual-refresh.
    pass

# --- DATA LOAD ---
lb = read_leaderboard() or {}
nav = read_nav() or {}
pos = read_positions() or {}

updated_at = lb.get("updated_at") or nav.get("updated_at") or pos.get("updated_at")
if updated_at:
    st.caption(f"Updated: {updated_at}")

# --- LAYOUT: 3 core views ---
tab_lb, tab_nav, tab_pos = st.tabs(["Leaderboard", "Portfolio NAV", "Positions"]) 

# ===== Leaderboard =====
with tab_lb:
    lb_df = pd.DataFrame(lb.get("items", []))
    if lb_df.empty:
        st.info("No leaderboard data yet.")
    else:
        # Column order & clean display
        cols = [
            "rank", "strategy", "equity", "pnl", "cagr", "sharpe", "mdd", "win_rate", "trades"
        ]
        show_cols = [c for c in cols if c in lb_df.columns]
        lb_df = lb_df[show_cols].sort_values("rank") if "rank" in lb_df.columns else lb_df
        st.dataframe(lb_df, use_container_width=True)

# ===== Portfolio NAV =====
with tab_nav:
    series = pd.DataFrame(nav.get("series", []))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Equity", f"{float(nav.get('total_equity', 0.0)) :,.2f}")
    k2.metric("Drawdown", f"{float(nav.get('drawdown', 0.0))*100:.2f}%")
    k3.metric("Realized PnL", f"{float(nav.get('realized_pnl', 0.0)) :,.2f}")
    k4.metric("Unrealized PnL", f"{float(nav.get('unrealized_pnl', 0.0)) :,.2f}")

    if series.empty:
        st.info("No NAV series yet.")
    else:
        series = series.copy()
        if "ts" in series.columns:
            series["ts"] = pd.to_datetime(series["ts"], errors="coerce")
            series = series.dropna(subset=["ts"]).sort_values("ts")
            series = series.set_index("ts")
        if "equity" in series.columns:
            st.line_chart(series["equity"], use_container_width=True)
        else:
            st.info("NAV series missing 'equity'.")

# ===== Positions =====
with tab_pos:
    pos_df = pd.DataFrame(pos.get("items", []))
    if pos_df.empty:
        st.info("No open positions.")
    else:
        keep = [
            "symbol", "side", "qty", "entry_price", "mark_price", "pnl", "leverage", "notional", "ts"
        ]
        show_cols = [c for c in keep if c in pos_df.columns]
        st.dataframe(pos_df[show_cols], use_container_width=True)

# --- Guardrails visibly enforced ---
st.caption("Firestore is the single source of truth • No local file reads • Auto-refresh 60s")
