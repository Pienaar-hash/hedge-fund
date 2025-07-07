import streamlit as st
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

st.set_page_config(page_title="🔬 Research Panel", layout="wide")

st.title("🔬 Quant Research & Prototyping")
st.markdown("""
This section aggregates experimental research tools and pipeline prototypes:

- 🧮 Monte Carlo Weight Allocator
- 🔁 Agent-Based Learner (TensorTrade)
- 🤖 LLM Research Copilot
- ⚙️ Kelly / Probabilistic Allocation Models
- ⚡ BTC Intraday Gridsearch (RSI/ATR)
""")

st.header("🧮 Monte Carlo Allocator")
st.code("python\nfrom research.factor_monte_carlo_sweep import simulate_weights")
st.info("Simulates portfolio outcomes using randomized weight distributions and factor scores")

st.header("⚡ BTC Intraday Gridsearch")
try:
    from research.btc_intraday_gridsearch import run_sweep
except ModuleNotFoundError:
    run_sweep = None
    st.warning("⚠️ Could not import 'btc_intraday_gridsearch'. Make sure research/ contains __init__.py and the module exists.")

with st.expander("📊 Run Gridsearch"):
    rsi_range = st.slider("RSI Window Range", 5, 40, (10, 20))
    atr_range = st.slider("ATR Window Range", 5, 40, (10, 20))
    if st.button("🚀 Run BTC Gridsearch"):
        st.info("Running RSI/ATR sweep...")
        try:
            result_path = run_sweep(rsi_range, atr_range)
            st.success(f"✅ Sweep completed: results saved to {result_path}")
        except Exception as e:
            st.error(f"Sweep failed: {e}")

st.header("🧠 LLM Copilot")
st.code("python\nfrom research.llm_research_agent import ask_research_bot")
st.info("LangChain + Vectorstore agent to query research logs and whitepapers")

st.header("🧬 TensorTrade Agent")
st.code("python\nfrom research.tensor_trade_agent import train_policy")
st.info("RL agent trained on momentum signals using TensorTrade gym wrapper")

st.header("📐 MCP Model")
st.code("python\nfrom research.mcp_model import generate_allocation")
st.info("Kelly/Multi-Cap/Probabilistic model for signal-based capital allocation")

st.success("✅ All modules available under /research/. Triggered via CLI, notebooks or future dashboard tabs.")
