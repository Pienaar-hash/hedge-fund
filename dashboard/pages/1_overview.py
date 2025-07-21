import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import zipfile
import tempfile
import glob
import datetime

st.set_page_config(page_title="Strategy Overview", layout="wide")

st.title("📈 Project Overview & System Design")
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
st.caption("ℹ️ Trades are sized with fixed capital to isolate strategy performance. Portfolio equity reflects timing-driven compounding across independent strategies.")
try:
    eq_curve = pd.read_csv("logs/portfolio_simulated_equity_blended.csv")
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

    # === Rolling Metrics Overlay ===
# (Temporarily disabled — file not available)
# st.subheader("📉 Rolling Metrics: Sharpe + Drawdown")
# try:
#     roll_df = pd.read_csv("logs/rolling_metrics_portfolio.csv", parse_dates=["timestamp"])
#     base = alt.Chart(roll_df).encode(x="timestamp:T")
#
#     sharpe_line = base.mark_line(color="green").encode(
#         y=alt.Y("rolling_sharpe:Q", title="Rolling Sharpe")
#     )
#     dd_line = base.mark_area(opacity=0.3, color="darkred").encode(
#         y=alt.Y("drawdown:Q", title="Drawdown")
#     )
#     st.altair_chart((sharpe_line + dd_line).properties(height=150), use_container_width=True)
# except Exception as e:
#     st.warning(f"Could not load rolling metrics: {e}")

    # === Capital Efficiency Chart ===
    st.subheader("⚖️ Capital Efficiency (Equity / Capital Used)")
    try:
        summary_files = [f for f in os.listdir("logs") if f.startswith("summary_") and f.endswith(".csv")]
        efficiency_rows = []
        for f in summary_files:
            df = pd.read_csv(os.path.join("logs", f))
            if "Label" in df.columns and "Trades" in df.columns:
                label = df["Label"].iloc[0]
                eq_path = f"logs/portfolio_simulated_equity_{label}.csv"
                if os.path.exists(eq_path):
                    eq_df = pd.read_csv(eq_path)
                    final_equity = eq_df["equity"].iloc[-1]
                    capital_used = 100000  # assumed baseline per strategy
                    ratio = final_equity / capital_used
                    efficiency_rows.append({"Strategy": label, "Efficiency": ratio})
        if efficiency_rows:
            df_eff = pd.DataFrame(efficiency_rows)
            chart = alt.Chart(df_eff).mark_bar().encode(
                x=alt.X("Strategy:N", sort="-y"),
                y=alt.Y("Efficiency:Q"),
                color=alt.Color("Strategy:N", legend=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No strategy summaries found for efficiency chart.")
    except Exception as ex:
        st.warning(f"Could not render efficiency chart: {ex}")

except Exception as e:
    st.warning(f"Could not load equity curve or benchmark comparison: {e}")

# === Downloads Section ===
st.header("📅 Downloadable Materials")
with st.expander("📁 Investor Downloads"):
    if os.path.exists("docs/one_page_summary.pdf"):
        with open("docs/one_page_summary.pdf", "rb") as f:
            st.download_button("📄 One-Page Summary", f.read(), file_name="one_page_summary.pdf")
    if os.path.exists("docs/investor_agreement.pdf"):
        with open("docs/investor_agreement.pdf", "rb") as f:
            st.download_button("📃 Investor Agreement", f.read(), file_name="investor_agreement.pdf")
    if os.path.exists("docs/investor_pitch_deck.pdf"):
        with open("docs/investor_pitch_deck.pdf", "rb") as f:
            st.download_button("📈 Pitch Deck", f.read(), file_name="investor_pitch_deck.pdf")

# === Export All Logs as ZIP ===
st.subheader("📁 Export All Logs")
today_str = datetime.datetime.today().strftime("%Y%m%d")
zip_filename = f"hedge_logs_{today_str}.zip"

if st.button("📦 Download Logs as ZIP"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
            for filepath in glob.glob("logs/*.csv") + glob.glob("logs/*.png") + glob.glob("docs/*.pdf"):
                zipf.write(filepath, arcname=os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath)))
        with open(tmp_zip.name, "rb") as f:
            st.download_button("📅 Click to Download", f.read(), file_name=zip_filename)

st.subheader("🩾 One-Page Strategy Summary (Coming Soon)")
st.button("📄 Export PDF Summary", disabled=True)

st.info("✅ This page summarizes the strategy design, signals, and fund mechanics.")
