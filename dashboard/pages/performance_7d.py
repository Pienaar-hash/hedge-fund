import streamlit as st
import pandas as pd
import json, os
from datetime import datetime, timedelta

st.set_page_config(page_title="📅 7-Day Performance", layout="wide")
st.title("📅 Rolling 7-Day Performance")

NAV_PATH = "nav_log.json"

def load_nav():
    if not os.path.exists(NAV_PATH):
        return pd.DataFrame()
    try:
        arr = json.load(open(NAV_PATH, "r"))
        df = pd.DataFrame(arr)
        if df.empty: return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    except Exception:
        return pd.DataFrame()

def max_drawdown(series: pd.Series) -> float:
    if series.empty: return 0.0
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min() or 0.0)

df = load_nav()
if df.empty or "equity" not in df.columns:
    st.warning("No NAV data yet. Run the executor to populate `nav_log.json`.")
    st.stop()

now = pd.Timestamp.utcnow()
start = now - pd.Timedelta(days=7)
win = df[df["timestamp"] >= start].copy()
if win.empty:
    st.warning("No entries in the last 7 days.")
    st.stop()

eq = win.set_index("timestamp")["equity"].astype(float).dropna()
ret7 = (eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
mdd7 = max_drawdown(eq)

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("7-Day Return", f"{ret7*100:.2f}%")
c2.metric("Max Drawdown (7d)", f"{mdd7*100:.2f}%")
c3.metric("# Observations", f"{len(eq)}")
c4.metric("Last Equity", f"${eq.iloc[-1]:,.2f}")

st.subheader("Equity (last 7 days)")
st.line_chart(eq, height=260)

st.caption("Return = last/first − 1 over the 7-day window. Max DD is computed from the rolling peak within the same window.")
