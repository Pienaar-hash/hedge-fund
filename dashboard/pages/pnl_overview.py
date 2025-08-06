import streamlit as st
import json
import os
import pandas as pd
import altair as alt
from datetime import datetime
from firebase_admin import credentials, initialize_app, db
import firebase_admin

st.set_page_config(page_title="PnL Overview", layout="wide")

st.title("💼 PnL Tracker — Hedge Fund")

# Firebase setup
FIREBASE_NAV = "/nav_log"
FIREBASE_TRADES = "/trades"
FIREBASE_PNL = "/realized"
FIREBASE_BALANCES = "/balances"

if not firebase_admin._apps:
    with open("firebase_creds.json") as f:
        cred_dict = json.load(f)
        database_url = "https://hedge-fund-sync-default-rtdb.europe-west1.firebasedatabase.app"

    cred = credentials.Certificate(cred_dict)
    initialize_app(cred, {
        "databaseURL": database_url
    })

# Load logs
@st.cache_data

def load_logs():
    trade_log = "data/state/trade_log.json"
    pnl_log = "data/state/pnl_log.json"
    unrealized_file = "data/state/unrealized.json"
    balance_file = "data/state/balances.json"
    nav_file = "data/state/nav_log.json"

    trades, realized, unrealized, balances, nav_log = [], [], {}, {}, []
    if os.path.exists(trade_log):
        with open(trade_log, "r") as f:
            trades = json.load(f)
    if os.path.exists(pnl_log):
        with open(pnl_log, "r") as f:
            realized = json.load(f)
    if os.path.exists(unrealized_file):
        with open(unrealized_file, "r") as f:
            unrealized = json.load(f)
    if os.path.exists(balance_file):
        with open(balance_file, "r") as f:
            balances = json.load(f)
    if os.path.exists(nav_file):
        with open(nav_file, "r") as f:
            nav_log = json.load(f)

    return trades, realized, unrealized, balances, nav_log

trades, realized, unrealized, balances, nav_log = load_logs()

# Net totals
realized_total = sum(x.get("realized_pnl", 0) for x in realized) if realized else 0
unrealized_total = sum(unrealized.values()) if unrealized else 0
usdt_free = float(balances.get("USDT", {}).get("free", 0))
usdt_locked = float(balances.get("USDT", {}).get("locked", 0))
total_equity = realized_total + unrealized_total + usdt_free + usdt_locked

# Ensure directory exists
os.makedirs("data/state", exist_ok=True)

# Log NAV snapshot
nav_snapshot = {
    "timestamp": datetime.utcnow().isoformat(),
    "realized": realized_total,
    "unrealized": unrealized_total,
    "balance": usdt_free + usdt_locked,
    "equity": total_equity,
}
if not os.path.exists("data/state/nav_log.json"):
    with open("data/state/nav_log.json", "w") as f:
        json.dump([nav_snapshot], f, indent=2)
else:
    with open("data/state/nav_log.json", "r") as f:
        nav_data = json.load(f)
    if not nav_data or nav_data[-1]["timestamp"] != nav_snapshot["timestamp"]:
        nav_data.append(nav_snapshot)
        with open("data/state/nav_log.json", "w") as f:
            json.dump(nav_data, f, indent=2)
        db.reference(FIREBASE_NAV).push(nav_snapshot)

# Push other logs to Firebase
if trades:
    db.reference(FIREBASE_TRADES).set(trades)
if realized:
    db.reference(FIREBASE_PNL).set(realized)
if balances:
    db.reference(FIREBASE_BALANCES).set(balances)

# Process PnL tables
if trades:
    df_trades = pd.DataFrame(trades)
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades = df_trades.sort_values("timestamp")
else:
    df_trades = pd.DataFrame()

if realized:
    df_realized = pd.DataFrame(realized)
    df_realized["timestamp"] = pd.to_datetime(df_realized["timestamp"])
    df_realized = df_realized.sort_values("timestamp")
else:
    df_realized = pd.DataFrame()

# Group by symbol for realized
if "symbol" in df_realized.columns and not df_realized.empty:
    symbol_pnl = (
        df_realized.groupby("symbol")["realized_pnl"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
else:
    symbol_pnl = pd.DataFrame()

# Section — Totals
st.subheader("📊 Net Performance")
col1, col2, col3 = st.columns(3)
col1.metric("💰 Realized PnL", f"{realized_total:.2f} USDT")
col2.metric("📈 Unrealized PnL", f"{unrealized_total:.2f} USDT")
col3.metric("🗖 USDT Balance", f"{usdt_free:.2f} free / {usdt_locked:.2f} locked")

# Section — Equity Curve
if not df_realized.empty:
    df_realized["cumulative"] = df_realized["realized_pnl"].cumsum()
    chart = (
        alt.Chart(df_realized)
        .mark_line(point=True)
        .encode(x="timestamp:T", y="cumulative:Q")
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

# Section — NAV Equity Tracker
if nav_log:
    df_nav = pd.DataFrame(nav_log)
    df_nav["timestamp"] = pd.to_datetime(df_nav["timestamp"])
    st.subheader("📈 NAV Equity Curve")
    chart_nav = (
        alt.Chart(df_nav)
        .mark_line(point=True)
        .encode(x="timestamp:T", y="equity:Q")
        .properties(height=300)
    )
    st.altair_chart(chart_nav, use_container_width=True)

# Section — PnL by Symbol
if not symbol_pnl.empty:
    st.subheader("🔍 Realized PnL by Symbol")
    st.dataframe(symbol_pnl, use_container_width=True)
else:
    st.info("No realized PnL entries yet.")

# Optional debug view
with st.expander("📋 Raw Logs"):
    st.code(json.dumps(trades[-3:], indent=2), language="json")
    st.code(json.dumps(realized[-3:], indent=2), language="json")
    st.code(json.dumps(unrealized, indent=2), language="json")
    st.code(json.dumps(balances, indent=2), language="json")
    st.code(json.dumps(nav_snapshot, indent=2), language="json")
