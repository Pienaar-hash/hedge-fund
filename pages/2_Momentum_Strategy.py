# 2_Momentum_Strategy.py
import streamlit as st
import pandas as pd
import altair as alt
from utils.load_data_momentum import load_momentum_results, load_equity_curve_momentum, load_momentum_trades
from utils.plot_helpers_momentum import plot_equity_curve_momentum, plot_trade_distribution_momentum, plot_momentum_heatmap

st.title("⚡ Momentum Strategy")

st.markdown("""
Momentum strategies enter positions based on recent price trends with take-profit (TP) and stop-loss (SL) levels.
This page shows top-performing TP/SL configurations and lets you explore Sharpe ratios and equity curves interactively.
""")

momentum_df = load_momentum_results()
assets = momentum_df['asset'].unique()
timeframes = momentum_df['timeframe'].unique()

col1, col2 = st.columns(2)
selected_asset = col1.selectbox("Select Asset", assets)
selected_tf = col2.selectbox("Select Timeframe", timeframes)

filtered = momentum_df[(momentum_df['asset'] == selected_asset) & (momentum_df['timeframe'] == selected_tf)]

# KPIs
best = filtered.sort_values("Sharpe Ratio", ascending=False).iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Sharpe", f"{best['Sharpe Ratio']:.2f}", f"TP={best['TP']} / SL={best['SL']}")
k2.metric("Return", f"{best['Total Return']:.2%}")
k3.metric("Win Rate", f"{best['Win Rate']:.0%}")
k4.metric("Expectancy", f"{best['Expectancy']:.4f}")

st.markdown("---")

# Heatmap
st.subheader("📊 Sharpe Ratio Heatmap")
heatmap = plot_momentum_heatmap(filtered, metric='Sharpe Ratio')
st.altair_chart(heatmap, use_container_width=True)

# Equity Curve
st.subheader("📈 Equity Curve of Best Config")
equity_chart = load_equity_curve_momentum(asset=selected_asset, timeframe=selected_tf)
st.altair_chart(equity_chart, use_container_width=True)

# Trade Distribution
st.subheader("📂 Trade Return Distribution")
trades_df = load_momentum_trades(asset=selected_asset, timeframe=selected_tf)
if trades_df is not None and not trades_df.empty:
    dist_chart = plot_trade_distribution_momentum(trades_df)
    if dist_chart:
        st.altair_chart(dist_chart, use_container_width=True)
    else:
        st.info("Could not render trade distribution chart due to empty or malformed data.")
else:
    st.info("No trade data available for this configuration.")
