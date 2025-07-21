# 🧠 Crypto Hedge Fund Dashboard

This repo contains a full simulation pipeline, multi-strategy backtests, and dashboard-ready log exports for a quantitative crypto hedge fund.

---

## ✅ Included Strategies

- **Momentum** — Z-score, trend & vol filtered, ATR exits
- **Volatility Targeting** — Realized vol-based risk parity
- **Relative Value** — Cointegration mean reversion (ETH pairs)
- **Factor Allocation** — Static or optimized portfolio blending

---

## 🔁 Regenerate Logs

Use this script to reset all logs and produce clean outputs:

```bash
python generate_dashboard_logs.py
```

It will:
- Archive old logs to `logs_archive/YYYYMMDD_HHMM/`
- Rerun all strategies with optimized parameters
- Simulate portfolios (equal & capital-weighted)
- Output equity curves, rolling Sharpe/DD, heatmaps, and ranked trades

---

## 📂 Output Structure (logs/)

| File | Purpose |
|------|---------|
| `momentum_trades_<symbol>.csv` | Per-asset trade logs |
| `relative_value_trades_<pair>.csv` | RelVal strategy logs |
| `*_trades_ranked.csv` | Dashboard leaderboard logs |
| `portfolio_simulated_equity.csv` | Equal-weight portfolio |
| `portfolio_simulated_equity_weighted.csv` | Capital-weighted portfolio |
| `equity_curve_<strategy>_<symbol>.csv` | Strategy equity overlays |
| `leaderboard_<strategy>.csv` | Sharpe/CAGR/Drawdown rankings |
| `rolling_metrics_portfolio.csv` | Rolling Sharpe & drawdown |
| `pnl_clusters.csv` | Heatmap of returns by hour/day |

---

## ⚙️ Strategy Configuration

All strategy logic and parameters are defined in:

📄 [`strategy_config.json`](./strategy_config.json)

Modify this file to adjust lookbacks, thresholds, symbols, or rebalancing logic.

---

## 📊 Dashboard

Once logs are generated, you can load them into the [Streamlit dashboard](dashboard/) to view:

- Strategy equity curves
- Trade logs
- Portfolio overlays
- Performance leaderboards
- Diagnostic heatmaps

---

## 🧠 Final Notes

Built with:
- `pandas`, `matplotlib`, `seaborn`
- Custom backtest engine
- Modular strategy classes in `core/` and `strategies/`