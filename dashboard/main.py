import os
import time
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
import pandas as pd
import altair as alt

# --- FIRESTORE INIT ---
FIREBASE_CREDS_PATH = os.getenv("FIREBASE_CREDS_PATH", "config/firebase_creds.json")
if not os.path.exists(FIREBASE_CREDS_PATH):
    st.error(f"Firebase credentials not found at {FIREBASE_CREDS_PATH}")
    st.stop()

creds = service_account.Credentials.from_service_account_file(FIREBASE_CREDS_PATH)
db = firestore.Client(credentials=creds)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hedge Fund Live Dashboard", layout="wide")
st.title("📊 Hedge Fund — Live Leaderboard & NAV")

# --- FUNCTIONS ---
def fetch_leaderboard():
    """Fetch latest leaderboard from Firestore."""
    doc_ref = db.collection("hedge-fund").document("leaderboard")
    doc = doc_ref.get()
    if doc.exists:
        return pd.DataFrame(doc.to_dict().get("data", []))
    return pd.DataFrame(columns=["symbol", "pnl", "pct_return"])

def fetch_nav_log():
    """Fetch NAV log from Firestore."""
    doc_ref = db.collection("hedge-fund").document("nav_log")
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict().get("data", [])
        return pd.DataFrame(data)
    return pd.DataFrame(columns=["timestamp", "total_equity"])

def format_leaderboard(df):
    """Format leaderboard table for display."""
    if df.empty:
        return df
    df = df.copy()
    df["pnl"] = df["pnl"].astype(float).round(2)
    df["pct_return"] = (df["pct_return"].astype(float) * 100).round(2)
    df.rename(columns={"symbol": "Symbol", "pnl": "PnL (USDT)", "pct_return": "% Return"}, inplace=True)
    df.sort_values("% Return", ascending=False, inplace=True)
    return df.reset_index(drop=True)

def format_nav(df):
    """Format NAV dataframe for chart."""
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["total_equity"] = df["total_equity"].astype(float)
    return df

def nav_chart(df):
    """Create Altair NAV chart."""
    if df.empty:
        return st.warning("No NAV data found in Firestore.")
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("total_equity:Q", title="Total Equity (USDT)"),
            tooltip=["timestamp:T", "total_equity:Q"]
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# --- LIVE REFRESH ---
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 60)

placeholder = st.empty()
while True:
    lb_df = format_leaderboard(fetch_leaderboard())
    nav_df = format_nav(fetch_nav_log())

    with placeholder.container():
        st.subheader(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Leaderboard
        if not lb_df.empty:
            st.dataframe(lb_df, use_container_width=True, height=400)
        else:
            st.warning("No leaderboard data found in Firestore.")
        
        st.markdown("---")
        
        # NAV Chart
        st.subheader("📈 Portfolio Equity Over Time")
        nav_chart(nav_df)

        # Investor bundle download
        st.markdown("---")
        bundle_path = "docs/investor_bundle.zip"
        if os.path.exists(bundle_path):
            st.download_button(
                "📦 Download Investor Bundle",
                data=open(bundle_path, "rb").read(),
                file_name="investor_bundle.zip",
                mime="application/zip"
            )

    time.sleep(refresh_interval)
