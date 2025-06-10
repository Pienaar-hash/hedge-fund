import streamlit as st
import os

st.title("📊 Hedge Fund Dashboard Launcher")

dashboards = {
    "Multi-Strategy Overview": "dashboards/multi_strategy_dashboard.py",
    # "Volatility Dashboard": "dashboards/volatility_dashboard.py",
    # Add more here
}

choice = st.selectbox("Choose a dashboard to launch", list(dashboards.keys()))
if st.button("Launch"):
    path = dashboards[choice]
    st.code(f"Run this in terminal:\n\nstreamlit run {path}")
