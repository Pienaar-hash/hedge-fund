import streamlit as st
import pandas as pd
import os
from dashboard.utils.dashboard_helpers import list_trade_logs, load_trade_log, compute_trade_summary

st.set_page_config(page_title="Strategy Insights", layout="wide")
st.title("📊 Trades")
st.caption("ℹ️ Portfolio CAGR may exceed individual CAGRs due to capital weighting, trade timing, and compounding effects — not simple averaging.")

# === Load Metrics ===
equity_path = "logs/equity_metrics_summary.csv"
trade_path = "logs/trade_metrics_summary.csv"

equity_metrics = pd.read_csv(equity_path) if os.path.exists(equity_path) else pd.DataFrame()
trade_metrics = pd.read_csv(trade_path) if os.path.exists(trade_path) else pd.DataFrame()

col1, col2 = st.columns(2)

# === Equity Table ===
col1.subheader("📊 Equity Metrics")
if not equity_metrics.empty:
    equity_metrics_sorted = equity_metrics.sort_values("Sharpe", ascending=False)
    col1.dataframe(equity_metrics_sorted.style.format({
        "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "Total Return": "{:.2%}",
        "MaxDrawdown": "{:.2%}", "Volatility": "{:.2%}"
    }), use_container_width=True)

# === Trade Table ===
col2.subheader("📋 Trade Metrics")
if not trade_metrics.empty:
    trade_metrics_sorted = trade_metrics.sort_values("Expectancy", ascending=False)
    col2.dataframe(trade_metrics_sorted.style.format({
        "WinRate": "{:.2%}", "AvgWin": "{:.4f}", "AvgLoss": "{:.4f}",
        "Expectancy": "{:.4f}", "ProfitFactor": "{:.2f}", "PayoffRatio": "{:.2f}"
    }), use_container_width=True)

st.markdown("---")

# === Capital Allocation Over Time ===
st.subheader("🗂️ Capital Allocation Over Time")
st.caption("🔍 This chart visualizes how capital is sequentially deployed across strategies — even with fixed trade sizes, sequencing drives portfolio-level compounding.")
from datetime import datetime
import altair as alt

allocations = []
for file in list_trade_logs():
    df = load_trade_log(file)
    label = file.replace("trades_", "").replace(".csv", "")
    if "exit_time" in df.columns and "capital_used" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        daily_alloc = df.groupby(df["exit_time"].dt.date)["capital_used"].sum().reset_index()
        daily_alloc["strategy"] = label
        daily_alloc.rename(columns={"exit_time": "date"}, inplace=True)
        allocations.append(daily_alloc)

if allocations:
    all_alloc = pd.concat(allocations)
    all_alloc["date"] = pd.to_datetime(all_alloc["date"])
    chart = alt.Chart(all_alloc).mark_area().encode(
        x="date:T",
        y=alt.Y("capital_used:Q", stack="zero", title="Capital Used"),
        color=alt.Color("strategy:N", title="Strategy")
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No capital allocation data available from trade logs.")

# === Strategy Trade Log Viewer ===
st.subheader("📦 Trade Logs by Strategy")
trade_files = list_trade_logs()
if trade_files:
    selected_file = st.selectbox("Choose a strategy log:", trade_files)
    df = load_trade_log(selected_file)
    if not df.empty:
        st.caption(f"Loaded {len(df)} trades from {selected_file}")

        # Optional: Symbol Filter
        symbols = sorted(df["symbol"].unique()) if "symbol" in df.columns else []
        if symbols:
            selected_symbols = st.multiselect("Filter by symbol:", symbols, default=symbols)
            df = df[df["symbol"].isin(selected_symbols)]

        # Optional: Date Filter
        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(df["exit_time"])
            min_date = df["exit_time"].min().date()
            max_date = df["exit_time"].max().date()
            start, end = st.date_input("Date Range:", value=(min_date, max_date))
            df = df[(df["exit_time"].dt.date >= start) & (df["exit_time"].dt.date <= end)]

        if "capital_used" in df.columns:
            st.caption("💰 Capital used column detected.")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Summary")
        stats = compute_trade_summary(df)
        for k, v in stats.items():
            st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
else:
    st.info("No trade logs found in /logs directory.")
