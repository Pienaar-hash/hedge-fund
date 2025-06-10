# multi_page.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Hedge Fund Performance", layout="wide")

# Fix vertical padding cutoff issue
st.markdown("""
<style>
    .main > div:first-child { padding-top: 2rem !important; }
    .block-container { padding-top: 2rem !important; }
    .stMarkdown { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Hedge Fund Multi-Strategy Dashboard")

st.markdown("""
Welcome to the hedge fund dashboard. Use the sidebar to navigate between:
- Strategy Overview
- Momentum Strategy
- Volatility Targeting
- Relative Value

---

This dashboard visualizes backtest results across three core strategy pillars:

1. **Momentum**: Timeframe- and pair-based take-profit/stop-loss grids with expectancy filtering.
2. **Volatility Targeting**: Return consistency via volatility-normalized sizing.
3. **Relative Value**: Pairwise spread trading based on Z-score thresholds.

All strategies are backtested across multiple crypto assets and timeframes, with metrics like:
- Sharpe Ratio
- Total Return
- Win Rate
- Max Drawdown
- Expectancy

Navigate from the sidebar to dive into individual strategies or compare them side-by-side.
""")

# Summary KPIs (dummy example, replace with real aggregated data later)
st.subheader("📌 High-Level Performance Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Sharpe", "3.25", "ETH/DOGE")
col2.metric("Top Return", "+395%", "DOGEUSDT")
col3.metric("Max Win Rate", "65%", "AVAXUSDT")
col4.metric("Top Expectancy", "0.015", "BTCUSDT")

st.markdown("---")

# Placeholder for equity curves or strategy-level comparison
st.caption("Equity curves and detailed strat comparisons will appear in each respective page.")
