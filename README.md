# Hedge Fund Env Overview

## Treasury Reporting vs Trading NAV

- `config/treasury.json` lists off-exchange holdings for investor/dashboards only; executor sizing ignores it.
- `execution/nav.py` keeps `compute_trading_nav` scoped to the Binance futures wallet so order sizing/risk checks use live futures equity only.
- `dashboard/app.py` surfaces three KPIs (Futures NAV, Treasury NAV, Total NAV) where only Total NAV includes the treasury valuation.
- Executors and `execution/risk_limits.py` work off the futures NAV exclusively; treasury balances never feed risk calculations.

Refer to `AGENTS.md` for development workflows and sandbox rules.
