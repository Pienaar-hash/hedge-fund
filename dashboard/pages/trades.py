import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure root-level import for dashboard.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.utils.dashboard_helpers import list_trade_logs, load_trade_log, compute_trade_summary

st.set_page_config(page_title="Trade Logs", layout="wide")
st.title("📋 Trade Logs & Statistics")

try:
    trade_files = list_trade_logs()
    if not trade_files:
        st.info("No trade logs found in /logs directory.")
    else:
        selected_file = st.selectbox("Select a trade log:", trade_files)
        df = load_trade_log(selected_file)

        original_count = len(df)
        filtered_count = len(df)
        st.caption(f"🔍 Showing {filtered_count} of {original_count} trades")

        if df.empty:
            st.warning("⚠️ No trades to display.")
            st.stop()

        st.dataframe(df.style.applymap(
            lambda v: 'background-color: #ffe6e6' if isinstance(v, (int, float)) and v < 0 else 'background-color: #e6ffe6',
            subset=[col for col in ["net_ret", "pnl_pct"] if col in df.columns]
        ), use_container_width=True)

        st.subheader("📊 Summary Statistics")
        stats = compute_trade_summary(df)
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Win Rate", f"{stats.get('Win Rate', 0):.2%}")
                st.metric("Avg Win", f"{stats.get('Avg Win', 0):.4f}")
                st.metric("Avg Loss", f"{stats.get('Avg Loss', 0):.4f}")
            with col2:
                st.metric("Expectancy", f"{stats.get('Expectancy', 0):.4f}")
                st.metric("Profit Factor", f"{stats.get('Profit Factor', 0):.2f}")
                st.metric("Payoff Ratio", f"{stats.get('Payoff Ratio', 0):.2f}")

        col = next((c for c in ["pnl_log_return", "net_ret", "pnl_pct"] if c in df.columns), None)
        if col:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].hist(bins=30, color="steelblue", edgecolor="white", ax=ax)
            ax.set_title("Trade Return Distribution")
            st.pyplot(fig)

except Exception as e:
    st.warning(f"Error loading trade logs: {e}")
