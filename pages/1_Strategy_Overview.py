# --- 1_Strategy_Overview.py ---

import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load_data_all import load_all_summary_metrics
from utils.plot_helpers_overview import plot_equity_curve
from utils.alias_strategy_overview_columns import alias_strategy_overview_columns

st.title("📈 Strategy Overview")

st.markdown("""
This section compares performance across all three strategy types:

📊 Momentum  
📉 Volatility Targeting  
➶ Relative Value  

Review each strategy's top configs and cumulative equity growth.
""")

def safe_format_df(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

summary_df = load_all_summary_metrics()
summary_df = alias_strategy_overview_columns(summary_df)

# Validate required columns exist
required_cols = ["sharpe_ratio", "total_return", "max_drawdown"]
if not all(col in summary_df.columns for col in required_cols):
    st.error(f"Missing expected columns in summary: {required_cols}")
    st.dataframe(summary_df)
    st.stop()

# Safely format the key metrics before applying the style
summary_df = safe_format_df(summary_df, required_cols)

st.subheader("📊 Summary Metrics by Strategy")
styled_df = summary_df.style.format("{:.2f}", subset=required_cols).background_gradient(
    subset=required_cols, cmap="RdYlGn"
)
st.dataframe(styled_df, use_container_width=True)

st.subheader("🏅 Top Config KPIs")
selected_strategy = st.selectbox("🎯 Select Strategy", summary_df['strategy'].unique())
strategy_subset = summary_df[summary_df['strategy'] == selected_strategy]
best = strategy_subset.sort_values("sharpe_ratio", ascending=False).iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("📈 Sharpe Ratio", f"{best['sharpe_ratio']:.2f}")
col2.metric("💰 Total Return", f"{best['total_return']:.2%}")
col3.metric("📉 Max Drawdown", f"{best['max_drawdown']:.2%}")

st.subheader("📈 Equity Curves")
for asset in strategy_subset['asset'].unique():
    st.markdown(f"#### {asset}")
    chart = plot_equity_curve(strategy=selected_strategy, asset=asset)
    st.altair_chart(chart, use_container_width=True)