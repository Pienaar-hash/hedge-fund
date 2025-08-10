# dashboard/pages/execution_monitor.py
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="🧪 Execution Monitor", layout="wide")
st.title("🧪 Execution Monitor")

# === File Paths ===
NAV_LOG_PATH = "nav_log.json"
STATE_PATH = "synced_state.json"

# === Load NAV Log ===
def load_nav_log():
    if not os.path.exists(NAV_LOG_PATH):
        return pd.DataFrame()
    with open(NAV_LOG_PATH, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# === Load Synced State ===
def load_synced_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r") as f:
        return json.load(f)

# === Display NAV Log ===
st.subheader("📈 NAV History")
nav_df = load_nav_log()
if not nav_df.empty:
    nav_df["timestamp"] = pd.to_datetime(nav_df["timestamp"])
    nav_df = nav_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    st.line_chart(nav_df.set_index("timestamp")["equity"], height=250)
    st.dataframe(nav_df, use_container_width=True)

    # Show latest balance and metrics
    latest = nav_df.iloc[0]
    st.metric(label="💰 USDT Balance", value=f"${latest['balance']:,.2f}")
    st.metric(label="🔺 Realized PnL", value=f"${latest['realized']:,.2f}")
    st.metric(label="🔸 Unrealized PnL", value=f"${latest['unrealized']:,.2f}")
    st.metric(label="🪙 Total Equity", value=f"${latest['equity']:,.2f}")
else:
    st.warning("No NAV log data found.")

# === Display Synced State ===
st.subheader("📦 Live Portfolio State")
state = load_synced_state()
if state:
    rows = []
    for symbol, data in state.items():
        pnl = data.get("pnl", 0)
        emoji = "🔼" if pnl > 0 else ("🔽" if pnl < 0 else "➖")
        rows.append({
            "Symbol": symbol,
            "Qty": data.get("qty", 0),
            "Entry": data.get("entry", 0),
            "Last Price": data.get("latest_price", 0),
            "PnL": pnl,
            "Status": emoji
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("PnL", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No synced state data found.")

# === Footer ===
st.markdown("---")
st.caption("Refreshed automatically every 5 minutes via executor. Telegram summary sent at 00:00 and 12:00 UTC. Breakout alerts on major NAV moves.")
