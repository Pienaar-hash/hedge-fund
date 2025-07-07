import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Execution Panel", layout="wide")

st.title("⚙️ Execution Control Panel")
st.markdown("""
This panel allows switching between paper trading and testnet execution modes, managing strategy deployment, and monitoring position state.
""")

# === Mode Toggle ===
st.header("🔁 Trading Mode")
mode = st.radio("Select execution mode:", ["Backtest Only", "Paper Trading", "Testnet (Live)"])

if mode == "Testnet (Live)":
    st.warning("⚠️ Ensure testnet API keys are loaded and permissions are set correctly.")

# === API Key Input ===
with st.expander("🔐 API Credentials"):
    st.text_input("API Key", type="password")
    st.text_input("Secret Key", type="password")
    st.text_input("Passphrase (if required)", type="password")
    st.button("🔄 Refresh Connection")

# === Strategy Toggles ===
st.header("✅ Active Strategies")
active_strategies = st.multiselect(
    "Enable strategies for execution:",
    ["momentum", "vol_target", "relative_value"],
    default=["momentum"]
)

# === Position Monitor ===
st.header("📈 Position Monitor")
st.info("Live position tracking will appear here once connected to exchange.")

# Placeholder for future real-time portfolio state
dummy_positions = pd.DataFrame({
    "Symbol": ["BTCUSDT", "ETHUSDT"],
    "Side": ["LONG", "SHORT"],
    "Size": [0.01, 0.02],
    "Entry Price": [30000, 1900],
    "Current Price": [31000, 1850],
    "PnL (%)": [3.33, -2.63]
})
st.dataframe(dummy_positions.style.format({"Entry Price": "${:,.2f}", "Current Price": "${:,.2f}", "PnL (%)": "{:.2f}%"}))

# === Execution Logs ===
st.header("📋 Recent Executions")
log_path = "logs/execution_log.csv"
if os.path.exists(log_path):
    df_log = pd.read_csv(log_path)
    st.dataframe(df_log.tail(10))
else:
    st.info("No execution log found.")
