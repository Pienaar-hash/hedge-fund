import streamlit as st
import zipfile
import tempfile
import glob
import datetime
import os

st.set_page_config(page_title="Hedge Fund Dashboard", layout="wide")

st.title("📊 Multi-Strategy Dashboard")

st.markdown("""
This dashboard provides a structured view of trading strategy performance, capital use, and research outputs across a multi-strategy portfolio.

Use the sidebar to navigate between pages:
- 📈 Portfolio Equity
- 🏆 Strategy Leaderboard
- 📋 Trade Logs
- 📄 Strategy Tear Sheet
- 📘 Overview & Pitch
- 📚 Research
- 🧪 Execution Monitor (coming soon)

Each page provides insights into performance, risk, strategy logic, and downloadable investor materials.

---

### 🎯 Mission Statement
To build, evaluate, and showcase a systematic multi-strategy trading fund using robust, transparent, and data-driven methods.

### 🧭 Project Summary
This dashboard consolidates strategy backtests, portfolio simulation, and trade-level analytics. It supports research workflows, pitch readiness, and strategic optimization.

### 💸 Funding Objective
This platform supports early-stage investor conversations by surfacing alpha signals, capital efficiency, and risk-adjusted performance in a clear, auditable format.
""")

st.info("Select a page from the sidebar to begin.")

# === Research Resources ===
st.header("🔬 Research Resources")
st.markdown("""
**Available Modules:**
- `factor_monte_carlo_sweep.py`: Simulate weight distributions
- `btc_intraday_gridsearch.py`: RSI/ATR grid optimization
- `llm_research_agent.py`: LangChain research assistant
- `tensor_trade_agent.py`: RL environment for signal learning
- `mcp_model.py`: Kelly + probabilistic allocators

All modules live in `/research/` and are accessible via CLI, notebook, or dashboard tab.
""")

# === Glossary ===
st.header("📘 Glossary of Terms")
st.markdown("""
- **CAGR**: Compounded Annual Growth Rate  
- **Sharpe Ratio**: Annualized return per unit volatility  
- **Max Drawdown**: Worst peak-to-trough equity drop  
- **Expectancy**: Avg profit per trade  
- **Profit Factor**: Total profit / total loss  
- **Z-score**: Deviation of signal/spread from mean  
- **ATR**: Average True Range (volatility proxy)  
- **Rebalance Period**: Strategy refresh interval  
- **Capital Weight**: Capital % allocated per trade  
""")

# === Strategy Logic ===
st.header("📡 Strategy Signal Logic")
st.markdown("""
**1. Momentum Strategy**:
- Z-score from momentum + EMA trend + volatility filter  
- Entry filter: reward-risk estimate  
- Exit: ATR trailing stop

**2. Volatility Targeting**:
- Vol target per asset  
- Leverage = target vol / realized vol  
- Optional trend filter

**3. Relative Value**:
- Rolling beta regression spread  
- Z-score triggers mean reversion trades  
- Stop-out on spread overshoot or drawdown

**Allocator**:
- Simulated as equal weight  
- Future: Kelly, Monte Carlo, Sharpe-ranked blends  
""")

st.markdown("---")
st.subheader("📁 Investor Download Bundle")

today_str = datetime.datetime.today().strftime("%Y%m%d")
zip_filename = f"hedge_logs_{today_str}.zip"

if st.button("📦 Download Investor Bundle"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
            for filepath in glob.glob("logs/*.csv") + glob.glob("logs/*.png") + glob.glob("docs/*.pdf"):
                zipf.write(filepath, arcname=os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath)))
        with open(tmp_zip.name, "rb") as f:
            st.download_button("📅 Click to Download", f.read(), file_name=zip_filename, mime='application/zip')

    st.caption("Includes trade logs, performance charts, and PDF documents for investor review.")
