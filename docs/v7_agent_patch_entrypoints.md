## v7 Agent Patch Entrypoints

- `execution/executor_live.py`: main loop wiring intents → risk → router; apply patches for execution flow, retries, telemetry hooks.
- `execution/risk_engine_v6.py`: orchestrates risk evaluation; place new caps/toggles here; respects DRY_RUN/testnet flags.
- `execution/risk_limits.py`: cap math, veto reasons, nav freshness; add/adjust thresholds or new guards here.
- `execution/order_router.py`: maker/taker flow, post-only behaviour, offsets; adjust routing logic or metrics.
- `execution/state_publish.py`: state files published to `logs/state/*`; extend for new telemetry fields or KPIs.
- `execution/sync_state.py`: mirror/cleanup for dashboard; add handling for new state files.
- `execution/signal_screener.py`: intent sizing/signals; ensure unlevered sizing contract is preserved; add new filters or signals.
- `execution/intel/*` (maker_offset, router_policy, autotune, symbol_score): tuning hooks for offsets/policies/intel scores.
- `dashboard/*.py` (panels/helpers): render new telemetry/KPIs; add new cards/tables; update router/intel views.
- `config/*`:
  - `risk_limits.json`: caps and overrides.
  - `strategy_config.json`: sizing params, per_trade_nav_pct, leverage metadata.
  - `runtime.yaml`: router/runtime knobs (offsets, fees, child size).
  - `pairs_universe.json`, `symbol_tiers.json`: universe/tiers for gating.
