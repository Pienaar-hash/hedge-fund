import streamlit as st

st.set_page_config(page_title="Hedge Fund Dashboard", layout="wide")

st.title("📊 Hedge Fund Control Center")

st.markdown("""
Welcome to the multi-strategy hedge fund dashboard.

Use the sidebar to navigate between pages:
- 📈 Portfolio Equity
- 🏆 Strategy Leaderboard
- 📋 Trade Logs
- 📄 Strategy Tear Sheet
- 📘 Overview & Pitch
- 🧪 Execution Monitor (coming soon)

Each page provides insights into performance, risk, strategy logic, and downloadable investor materials.
""")

st.info("Select a page from the sidebar to begin.")
