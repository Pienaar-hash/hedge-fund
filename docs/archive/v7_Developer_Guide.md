## v7 Developer Guide

### Runtime entrypoints
- Executor: `bin/run-executor.sh` (uses `execution/executor_live.py`), `scripts/run_executor_once.sh` for one-shot.
- Dashboard: `bin/run-dashboard.sh` (Streamlit app in `dashboard/app.py`).
- Sync/state mirror: `bin/run-sync.sh` (runs `execution/sync_state.py`).
- Probes/daemons: `scripts/strategy_probe.py`, `scripts/pipeline_shadow_heartbeat.py`, `scripts/pipeline_compare_service.py`, `scripts/route_debug.py`, `scripts/exec_debug.py`.

### Configuration
- Strategies: `config/strategy_config.json` (universe, per_trade_nav_pct, leverage metadata, signal params).
- Risk: `config/risk_limits.json` (fractional caps; testnet overrides).
- Runtime/router: `config/runtime.yaml` (post-only defaults, offsets, child sizes, fees).
- Universe/tiers: `config/pairs_universe.json`, `config/symbol_tiers.json`.
- Registry/settings/assets: `config/strategy_registry.json`, `config/offexchange_holdings.json`, `config/settings.json`.

### Adding/editing logic
- Risk tuning: `execution/risk_limits.py` (caps, veto reasons), `risk_engine_v6.py` (orchestration). Use `risk_loader.normalize_percentage` for new pct fields.
- Routing: `execution/order_router.py`, offsets/policy in `execution/intel/*` (maker_offset, router_policy, router_autotune*).
- Telemetry: add fields in `execution/state_publish.py` and mirror handling in `execution/sync_state.py`; emit events via `execution/events.py`.
- Dashboard KPIs: extend panels in `dashboard/*.py` and ensure corresponding state fields are published.
- Screener sizing/signals: `execution/signal_screener.py`, `signal_generator.py`; sizing is unlevered (gross_usd not multiplied by leverage).

### Testing
- Core risk/contract tests: `tests/test_risk_limits.py`, `tests/test_risk_caps_v7.py`, `tests/test_tier_caps.py`, `tests/test_screener_tier_caps.py`, `tests/test_screener_sizing_no_leverage.py`.
- Router/risk integration probes: `scripts/strategy_probe.py`, `scripts/pipeline_compare_service.py`.
- Use `pytest` with `BINANCE_TESTNET=1 DRY_RUN=1` for dry runs.

### Telemetry/state expectations
- Ensure `state_publish.py` writes JSON under `logs/state/` (nav, positions, risk, router).
- Risk vetoes should appear in `logs/execution/risk_vetoes.jsonl`.
- Router/health errors recorded via `execution.utils.execution_health`.

### Patching tips
- Keep caps as fractions (0–1). Leverage is metadata; do not scale gross_usd in screener/executor.
- Avoid touching NAV sourcing unless necessary; nav_total used for risk.
- When adding fields, update both publisher and dashboard consumer if visible.
