# Codex 5.1 — Authoritative Entry Points (v5.9.5)

## Execution stack (LIVE)

- execution/executor_live.py
- execution/order_router.py
- execution/risk_limits.py
- execution/signal_screener.py
- execution/utils/__init__.py
- execution/utils/metrics.py
- execution/utils/vol.py
- execution/utils/expectancy.py
- execution/utils/toggle.py
- execution/utils/execution_health.py
- execution/utils/execution_alerts.py
- execution/router_metrics.py
- execution/firestore_utils.py
- execution/telegram_utils.py
- execution/drawdown_tracker.py
- execution/sync_state.py

## Dashboard stack (LIVE)

- dashboard/app.py
- dashboard/live_helpers.py
- dashboard/dashboard_utils.py
- dashboard/router_health.py

## Tests to guide Codex

- tests/test_execution_hardening_*.py
- tests/test_execution_health.py
- tests/test_execution_alerts.py
- tests/test_router_metrics_*.py
- tests/test_symbol_toggle_bootstrap.py
- tests/test_execution_alerts.py

## Supervisor entrypoints

- hedge:executor   → execution/executor_live.py
- hedge:dashboard  → dashboard/app.py
- hedge:sync_state → execution/sync_state.py

> When using Codex 5.1 in the side panel, restrict multi-file edits to these
> entrypoints and their direct dependencies unless explicitly doing archive work.
