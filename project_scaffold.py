# === Project Folder Structure ===
# hedge-fund/
├── core/
│   ├── backtest_runner.py
│   ├── duckdb_query.py
│   ├── strategy_base.py
│   ├── portfolio_simulator.py
│   ├── execution_engine.py         # 🔁 NEW: live/paper execution logic
│   ├── signal_generator.py         # 🔁 NEW: unified signal bus
├── strategies/
│   ├── momentum.py
│   ├── volatility_target.py
│   ├── relative_value.py
│   ├── factor_allocation.py
│   ├── ict.py
│   └── btc_intraday_gridsearch.py  # 🔁 NEW: RSI/ATR/EMA gridsearch script
├── dashboard/
│   ├── multi_strategy_dashboard.py
│   ├── utils.py
│   ├── overview_page.py            # 🔁 NEW: system overview, pitch, glossary
│   ├── strategy_pages/             # 🔁 NEW: modular tabs
│   │   ├── momentum_tab.py
│   │   ├── volatility_tab.py
│   │   ├── relative_value_tab.py
│   │   ├── factor_allocation_tab.py
│   │   └── btc_gridsearch_tab.py
│   ├── execution_panel.py         # 🔁 NEW: real-time execution monitor
│   └── research_outputs.py        # 🔁 NEW: AI/LLM research insights
├── research/
│   ├── factor_monte_carlo_sweep.py
│   ├── tensor_trade_agent.py       # 🔁 NEW: agent-based learning
│   ├── mcp_model.py                # 🔁 NEW: Kelly/Multi-Cap/Probabilistic models
│   ├── llm_research_agent.py       # 🔁 NEW: research copilot w/ langchain/RAG
├── logs/
│   ├── *.csv (trades, equity curves, metrics)
│   ├── *.png (equity plots, sweep visualizations)
├── data/
│   ├── raw/
│   └── processed/*.csv
├── config/
│   └── strategy_config.json
├── notebooks/                      # 🔁 NEW: explorative Jupyter ideas
├── requirements.txt
├── README.md

# === README.md ===
"""
# Hedge Fund Strategy Simulator

This project implements a **production-grade crypto hedge fund platform** featuring multi-strategy backtesting, portfolio simulation, AI agents, and real-time readiness.

## 🚀 Strategies
- **Momentum (Trend-following)**: Ranks assets by past returns.
- **Volatility Targeting (Risk Parity)**: Targets fixed daily volatility per asset.
- **Relative Value (Stat Arb)**: Mean-reverting z-score strategy.
- **Factor Allocation**: Monte Carlo blend of strategy returns.
- **BTC Intraday Gridsearch**: RSI, ATR trailing stop, EMA filter, volatility filters.

## 📂 Key Folders
- `core/` - Strategy orchestration, signal generation, execution.
- `strategies/` - Modular alpha engines.
- `dashboard/` - Streamlit app with overview, strategy panels, execution tab.
- `research/` - Agents, RAG copilots, modeling experiments.
- `logs/` - Results for trades, equity, metrics.
- `data/` - Historical price data.
- `notebooks/` - Research notebooks.

## 📊 Metrics Tracked
For each strategy and portfolio:
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- CAGR
- Calmar Ratio
- Win Rate
- Monthly Return Stats
- Rolling Sharpe (e.g. 90D)

## 🖥 Dashboard Layout
- **Overview Page**: Pitch, glossary, portfolio curve.
- **Strategy Tabs**: Momentum, Vol Target, Value, Gridsearch.
- **Execution Panel**: Live/testnet trade feed.
- **Research Panel**: LLM/tensor/monte carlo diagnostics.

## 🛠 How to Run
1. Edit `config/strategy_config.json`
2. Backtest all strategies:
   ```bash
   python -m core.backtest_runner
   ```
3. Launch dashboard:
   ```bash
   streamlit run dashboard/multi_strategy_dashboard.py
   ```

## 🧪 Example Output
```
Blended Portfolio Metrics:
Sharpe: 0.59 | Max Drawdown: -21.1% | CAGR: 6.5%
```

---

## 📌 Roadmap
- [x] Modular strategy engine
- [x] Monte Carlo factor weights
- [x] Portfolio simulator
- [x] Streamlit dashboard
- [ ] Live testnet execution
- [ ] LLM + tensor trade agents
- [ ] Deploy investor web app
"""
