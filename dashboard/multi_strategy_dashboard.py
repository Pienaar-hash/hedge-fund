# === dashboard/multi_strategy_dashboard.py ===
import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "logs"

st.set_page_config(layout="wide")
st.title("📊 Multi-Strategy Backtest Dashboard")

# Helper to load equity curves
def load_equity_curves():
    curves = {}
    for file in glob.glob(f"{LOG_DIR}/equity_curve_*.csv"):
        name = file.replace(f"{LOG_DIR}/equity_curve_", "").replace(".csv", "")
        try:
            df = pd.read_csv(file)
            if 'timestamp' not in df.columns or 'equity' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp', 'equity'], inplace=True)
            df = df[df['equity'].apply(np.isfinite)]
            if df.empty or df['equity'].min() <= 0:
                continue
            df.set_index('timestamp', inplace=True)
            curves[name] = df['equity'] / df['equity'].iloc[0]  # normalize
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}")
    return pd.DataFrame(curves)

# Helper to load trade logs
def load_trade_logs():
    logs = {}
    for file in glob.glob(f"{LOG_DIR}/*_trades_*.csv"):
        name = file.replace(f"{LOG_DIR}/", "").replace(".csv", "")
        try:
            df = pd.read_csv(file)
            if 'entry_time' in df.columns and 'exit_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                logs[name] = df.dropna(subset=['entry_time', 'exit_time'])
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}")
    return logs

# Load equity and trades
equity_df = load_equity_curves()
trade_logs = load_trade_logs()

if equity_df.empty:
    st.error("No valid equity curves found in logs/")
else:
    st.subheader("📈 Equity Curve Comparison")
    strategies = st.multiselect("Select strategies to compare:", equity_df.columns.tolist(), default=equity_df.columns.tolist())
    st.line_chart(equity_df[strategies])

    returns = equity_df[strategies].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252**0.5)
    total_return = equity_df[strategies].iloc[-1] / equity_df[strategies].iloc[0] - 1
    drawdown = (equity_df[strategies] / equity_df[strategies].cummax() - 1).min()

    st.subheader("📊 Performance Summary")
    summary = pd.DataFrame({
        "Sharpe": sharpe,
        "Total Return": total_return,
        "Max Drawdown": drawdown
    })
    st.dataframe(summary.style.format("{:.2%}"))

# Trade logs viewer
if trade_logs:
    st.subheader("📋 Trade Logs Viewer")
    selected_log = st.selectbox("Select strategy-trade log:", list(trade_logs.keys()))
    st.dataframe(trade_logs[selected_log].sort_values("entry_time", ascending=False).reset_index(drop=True))

# Legacy vs vectorbt comparison
legacy_keys = [k for k in equity_df.columns if "vectorbt" not in k]
vbt_keys = [k for k in equity_df.columns if "vectorbt" in k]

common_keys = [k for k in legacy_keys if any(k in v for v in vbt_keys)]
if common_keys:
    st.subheader("🧪 Legacy vs Vectorbt Overlay")
    for base in common_keys:
        base_vbt = next((v for v in vbt_keys if base in v), None)
        if base_vbt and base in equity_df.columns and base_vbt in equity_df.columns:
            st.line_chart(equity_df[[base, base_vbt]].rename(columns={base: "Legacy", base_vbt: "Vectorbt"}))
