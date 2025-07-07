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
Each strategy must log metrics that support both investor presentation and internal diagnostics. Metrics include:
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Rolling Max Drawdown
- CAGR, Cumulative Return
- Win Rate, Avg Win / Avg Loss, Expectancy
- Monthly Return Stats
- Rolling Sharpe / Volatility / Correlation (e.g., 30D, 90D)
- Trade Duration, Turnover, Exposure
- Per-symbol attribution (if multi-asset)

## 🔍 Logs and Output Expectations
- Trade logs: `logs/{strategy}_trades_{label}.csv`
- Equity curves: `logs/equity_curve_{label}.csv` (normalized to 1.0)
- Summary metrics: `logs/summary_{label}.csv`
- Optional per-symbol logs for leaderboard and heatmaps

All logs must be structured for ingestion by dashboard, simulator, and execution modules.

## 🖥 Dashboard Layout
- **Overview Page**: Pitch, glossary, portfolio curve.
- **Strategy Tabs**: Momentum, Vol Target, Value, Gridsearch.
- **Execution Panel**: Live/testnet trade feed.
- **Research Panel**: LLM/tensor/monte carlo diagnostics.

## 📦 Strategy Requirements
Each strategy must support the following interfaces:
- `configure(params: dict)` – for strategy hyperparameters
- `run()` – to generate trade signals and backtest logic
- `log_results(label: str)` – to save trade logs, equity, and summary metrics
- `get_signals(timestamp)` (optional) – for live signal generation

Additionally, each strategy should log a `metadata_{label}.json` file containing:
- Strategy name and label
- Parameter set used
- Signal type (cross-sectional, time-series, etc.)
- Risk model (e.g. vol targeting, fixed size)
- Features used

These requirements ensure full compatibility with the simulator, dashboard, and execution engine.

## 🖥️ CLI & Scheduling Tools
A lightweight CLI interface will manage strategy execution, scheduling, and report generation.

### ⚙️ Command Templates
```bash
# Run single strategy
hedge run momentum --label default

# Run full backtest suite
hedge run all

# Generate summary leaderboard
hedge report all

# Export dashboard-ready logs
hedge export logs

# Schedule recurring testnet signals
hedge schedule live --every 1h --strategy momentum
```

### 📦 Under the Hood
These commands will interface with:
- `core/backtest_runner.py` for orchestrated runs
- `core/portfolio_simulator.py` for capital-weighted backtests
- `core/execution_engine.py` for live sync
- `logs/` as the universal output sink for all scheduled modules

---

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

## 🧩 System Architecture
- Modular strategy engines extend `strategy_base.py`
- Signal routing through `signal_generator.py`
- Portfolio-level simulation via `portfolio_simulator.py`
- Execution logic (paper/live) routed through `execution_engine.py`
- Dashboard tabs render from `logs/` files, no recomputation
- Agents and LLM tooling live in `research/`

## 🧪 Strategy Debugging Pipeline
We will improve each strategy systematically using this process:

- Tune parameters or filters based on historical performance
- Tighten entry/exit logic with conditional overlays
- Add diagnostic overlays (volatility, trend, spread z-score)
- Confirm if edge is consistent across timeframes, symbols, and market regimes
- Log per-asset performance to identify standout symbols or failure modes

This process will be applied to: `momentum`, `volatility_target`, `relative_value`, `factor_allocation`, `ict`, and all gridsearch variants.

## 🧠 Development Roadmap
The roadmap below defines major engineering and research milestones required to make the platform investor-grade, execution-ready, and quant-scalable.
- [ ] Modular strategy engine with unified I/O contract and metadata logs
- [ ] Monte Carlo factor weights with walkforward validation and correlation diagnostics
- [ ] Portfolio simulator with capital constraints, rebalancing schedules, and rolling backtest capability
- [ ] Streamlit dashboard with leaderboard, tab-level tear sheets, rolling risk analytics, and equity overlays
- [ ] Live testnet execution with fill tracking, latency logging, and strategy sync
- [ ] LLM + tensor trade agents capable of signal inspection, anomaly flagging, and strategy adaptation
- [ ] Deploy investor web app with shareable strategy insights, investor dashboards, and PDF export pipeline
"""
