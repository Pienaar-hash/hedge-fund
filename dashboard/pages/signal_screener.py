# dashboard/pages/signal_screener.py
import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../execution")))
from exchange_utils import fetch_ohlcv, client

st.set_page_config(page_title="📡 Signal Screener", layout="wide")
st.title("📡 Signal Screener")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LTCUSDT"]
INTERVAL = "4h"

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signal_row(symbol):
    try:
        df = fetch_ohlcv(client, symbol, interval=INTERVAL, limit=100)
        latest = df.iloc[-1]
        zscore = latest["zscore"]
        trend = "UP" if latest["ema_fast"] > latest["ema_slow"] else "DOWN"
        momentum = latest["momentum"]
        volume = latest["volume"]
        rsi = compute_rsi(df["close"], window=14).iloc[-1]

        vol_ok = volume > df["volume"].rolling(20).mean().iloc[-1]
        momentum_ok = momentum > 0 if trend == "UP" else momentum < 0

        signal = "BUY" if zscore > 1 and trend == "UP" and vol_ok and momentum_ok else \
                 "SELL" if zscore < -1 and trend == "DOWN" and vol_ok and momentum_ok else "HOLD"

        emoji = "✅ BUY" if signal == "BUY" else "❌ SELL" if signal == "SELL" else "— HOLD"

        return {
            "Symbol": symbol,
            "Close": latest["close"],
            "Z-score": round(zscore, 2),
            "Momentum": round(momentum, 4),
            "Vol": round(volume),
            "RSI": round(rsi, 1),
            "Trend": trend,
            "Signal": emoji
        }
    except Exception as e:
        return {"Symbol": symbol, "Signal": f"❌ Error: {e}"}

# === Generate and Display Table ===
st.subheader("🧠 Latest Signals Across Core Assets")
rows = [generate_signal_row(sym) for sym in SYMBOLS]
df = pd.DataFrame(rows)

# === Sidebar filters ===
st.sidebar.header("🔍 Filter Signals")
rsi_min = st.sidebar.slider("Min RSI", 0, 100, 0)
rsi_max = st.sidebar.slider("Max RSI", 0, 100, 100)
trend_filter = st.sidebar.selectbox("Trend", ["All", "UP", "DOWN"])
signal_filter = st.sidebar.selectbox("Signal", ["All", "✅ BUY", "❌ SELL", "— HOLD"])

# === Apply filters ===
if "RSI" in df:
    df = df[(df["RSI"] >= rsi_min) & (df["RSI"] <= rsi_max)]
if trend_filter != "All":
    df = df[df["Trend"] == trend_filter]
if signal_filter != "All":
    df = df[df["Signal"] == signal_filter]

st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Signals update every 4 hours based on z-score, trend, momentum, volume, and RSI filters.")
