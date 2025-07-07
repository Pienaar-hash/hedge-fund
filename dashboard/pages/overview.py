import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(page_title="Strategy Overview", layout="wide")

st.title("📊 Project Overview & System Design")
st.markdown("""
Welcome to the control center of our multi-strategy quant hedge fund. Below you'll find a full breakdown of the project's structure, core components, key metrics, and strategy logic.
""")

# === Project Pitch & System Design ===
st.header("🚀 Pitch & Architecture")
st.markdown("""
**Our Mission**: Build an autonomous crypto hedge fund driven by modular alpha-generating strategies and adaptive allocation.

**System Flow**:
1. **Data Layer**: Historical + real-time from Binance/KUCOIN (processed OHLCV)
2. **Signal Layer**: Momentum, volatility-targeting, and relative value models generate trade signals
3. **Execution Layer**: Paper trading or testnet deployment, risk-controlled
4. **Logging & Evaluation**: Trade logs, equity curves, summary metrics
5. **Visualization**: Streamlit dashboard

Backtests simulate live conditions: fees, max leverage, asymmetric stop-outs, rebalancing.
""")

# === Portfolio vs BTC + SPY Benchmarks ===
st.header("📈 Portfolio Equity vs Benchmarks")
try:
    eq_curve = pd.read_csv("logs/portfolio_simulated_equity.csv")
    eq_curve["timestamp"] = pd.to_datetime(eq_curve["timestamp"])
    eq_curve = eq_curve.sort_values("timestamp")
    eq_curve["norm_equity"] = eq_curve["equity"] / eq_curve["equity"].iloc[0]
    eq_curve["drawdown"] = eq_curve["norm_equity"] / eq_curve["norm_equity"].cummax() - 1

    btc = pd.read_csv("data/processed/btcusdt_1d.csv")
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc[btc["timestamp"] >= eq_curve["timestamp"].min()]
    btc = btc.sort_values("timestamp")
    btc["btc_equity"] = btc["close"] / btc["close"].iloc[0]

    spy_path = "data/processed/sp500_1d.csv"
    if os.path.exists(spy_path):
        spy = pd.read_csv(spy_path)
        spy["timestamp"] = pd.to_datetime(spy["timestamp"])
        spy = spy[spy["timestamp"] >= eq_curve["timestamp"].min()]
        spy = spy.sort_values("timestamp")
        spy["spy_equity"] = spy["close"] / spy["close"].iloc[0]
    else:
        spy = pd.DataFrame()

    st.subheader("Equity Curve")
    base = alt.Chart(eq_curve).encode(x="timestamp:T")

    strat_line = base.mark_line(color="#7FDBFF").encode(
        y=alt.Y("norm_equity:Q", title="Portfolio")
    )
    btc_line = alt.Chart(btc).mark_line(color="orange").encode(
        x="timestamp:T", y=alt.Y("btc_equity:Q", title="BTC")
    )
    if not spy.empty:
        spy_line = alt.Chart(spy).mark_line(color="green").encode(
            x="timestamp:T", y=alt.Y("spy_equity:Q", title="SPY")
        )
        equity_chart = strat_line + btc_line + spy_line
    else:
        equity_chart = strat_line + btc_line

    st.altair_chart(equity_chart.properties(height=300), use_container_width=True)

    st.subheader("📉 Drawdown")
    dd_chart = alt.Chart(eq_curve).mark_area(color="darkred", opacity=0.6).encode(
        x="timestamp:T",
        y=alt.Y("drawdown:Q", title="Drawdown")
    ).properties(height=120)
    st.altair_chart(dd_chart, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load equity curve or benchmark comparison: {e}")

# === Downloads Section ===
st.header("📥 Downloadable Materials")
with st.expander("📁 Investor Downloads"):
    if os.path.exists("docs/one_page_summary.pdf"):
        with open("docs/one_page_summary.pdf", "rb") as f:
            st.download_button("📄 One-Page Summary", f.read(), file_name="one_page_summary.pdf")
    if os.path.exists("docs/investor_agreement.pdf"):
        with open("docs/investor_agreement.pdf", "rb") as f:
            st.download_button("📃 Investor Agreement", f.read(), file_name="investor_agreement.pdf")
    if os.path.exists("docs/investor_pitch_deck.pdf"):
        with open("docs/investor_pitch_deck.pdf", "rb") as f:
            st.download_button("📊 Pitch Deck", f.read(), file_name="investor_pitch_deck.pdf")

st.subheader("🧾 One-Page Strategy Summary (Coming Soon)")
st.button("📤 Export PDF Summary", disabled=True)

# === Glossary of Terms ===
st.header("📘 Glossary")
st.markdown("""
- **CAGR**: Compounded Annual Growth Rate
- **Sharpe Ratio**: Annualized return per unit volatility
- **Max Drawdown**: Worst peak-to-trough equity drop
- **Expectancy**: Avg profit per trade: `win_rate * avg_win + (1 - win_rate) * avg_loss`
- **Profit Factor**: Total profit / total loss
- **Z-score**: Standard deviation of signal or spread from mean
- **ATR**: Average True Range, proxy for volatility
- **Rebalance Period**: Interval to refresh positions or signals
- **Capital Weight**: % of capital allocated per position
""")

# === Summary of Signal Logic ===
st.header("📡 Strategy Signal Logic")
st.markdown("""
**1. Momentum Strategy**:
- Signals from momentum Z-score with filters:
    - Volatility threshold (20-bar stddev)
    - EMA trend check (10 vs 50 EMA)
    - ATR-based stop
    - Reward-risk filter

**2. Volatility Targeting**:
- Targets daily/weekly volatility per asset
- Adjusts leverage based on realized vol vs target
- Optional trend filter (EMA-based)

**3. Relative Value**:
- Pairs trading with rolling beta regression (ETH as base)
- Entry on Z-score of spread divergence, exit on mean reversion or drawdown
- Dynamic capital weight = `min(0.02 + |z| * 0.01, 0.1)`

**Allocator**:
- Future: static or Monte Carlo weight sweeps (e.g. factor Sharpe rank)
- Current: simulate strategies independently or with equal-weight blend
""")

st.image("assets/signal_flow_diagram.png", caption="Visual Signal Flow - Strategy Modules")

st.info("✅ This page summarizes the strategy design, signals, and fund mechanics.")
