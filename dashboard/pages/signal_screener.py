import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import time
import altair as alt

# Firebase init
if not firebase_admin._apps:
    cred = credentials.Certificate("/root/hedge-fund/firebase_creds.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

st.set_page_config(page_title="📡 Signal Screener", layout="wide")
st.title("📡 Multi-Asset Signal Screener")

view_mode = st.radio("Select view mode:", ["📋 Compact Table", "📈 Expanded Charts"], horizontal=True)

with st.spinner("Fetching live signals..."):
    docs = db.collection("signals").stream()

    signals = []
    for doc in docs:
        d = doc.to_dict()
        d["symbol"] = doc.id
        d["timestamp_fmt"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(d["timestamp"]))
        signals.append(d)

if not signals:
    st.warning("No signals found.")
    st.stop()

df = pd.DataFrame(signals)
df = df.sort_values("timestamp_fmt", ascending=False)

if view_mode == "📋 Compact Table":
    display_cols = ["symbol", "timestamp_fmt", "signal", "confidence", "rsi", "momentum", "z_score", "source"]
    st.dataframe(df[display_cols], use_container_width=True)

else:
    selected_symbol = st.selectbox("Select asset for signal chart:", sorted(df["symbol"].unique()))
    df_asset = df[df["symbol"] == selected_symbol].sort_values("timestamp_fmt")

    base = alt.Chart(df_asset).encode(
        x=alt.X("timestamp_fmt:T", title="Timestamp"),
    ).properties(width=700, height=300)

    chart1 = base.mark_line(color="green").encode(
        y=alt.Y("confidence:Q", title="Confidence"),
        tooltip=["timestamp_fmt", "confidence"]
    ).interactive()

    chart2 = base.mark_line(color="orange").encode(
        y=alt.Y("rsi:Q", title="RSI"),
        tooltip=["timestamp_fmt", "rsi"]
    ).interactive()

    chart3 = base.mark_line(color="blue").encode(
        y=alt.Y("momentum:Q", title="Momentum"),
        tooltip=["timestamp_fmt", "momentum"]
    ).interactive()

    st.altair_chart(chart1 & chart2 & chart3, use_container_width=True)
