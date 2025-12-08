## v7 Topology (current repo)

### Directory tree (active)
- execution/: risk/intent/routing/nav/telemetry core.
- execution/intel/: offsets, router policy/autotune, pipeline compare/shadow, expectancy.
- execution/utils/: metrics, vol, toggles, health helpers.
- dashboard/: panels/helpers for nav/router/pipeline/intel.
- scripts/: probes, heartbeat/compare daemons, executor runners, debug tools.
- config/: strategy/risk/runtime/universe/tiers/assets/settings.
- ops/: nginx/supervisor configs, patterns.
- bin/: run scripts for executor/dashboard/sync.
- VERSION: release marker.

### Module groups & responsibilities
- Intent/sizing: `signal_generator.py`, `signal_screener.py`, `position_sizing.py`, `pnl_tracker.py`.
- Risk: `risk_limits.py`, `risk_engine_v6.py`, `risk_loader.py`, `universe_resolver.py`, `drawdown_tracker.py`.
- Routing: `order_router.py`, `router_metrics.py`, `intel/router_policy.py`, `intel/maker_offset.py`, `intel/router_autotune_v6.py`, `intel/router_autotune_apply_v6.py`.
- Exchange + NAV: `exchange_utils.py`, `nav.py`, `pnl_tracker.py`, `fills.py`.
- Telemetry/state: `state_publish.py`, `sync_state.py`, `events.py`, `trade_logs.py`, `metrics_normalizer.py`, `runtime_config.py`.
- Intel/scoring: `intel/symbol_score.py`, `intel/symbol_score_v6.py`, `intel/expectancy_map.py`, `intel/expectancy_v6.py`.
- Dashboard: `app.py`, `nav_helpers.py`, `pipeline_panel.py`, `intel_panel.py`, `router_health.py`, `router_policy.py`, `live_helpers.py`, `dashboard_utils.py`.
- Scripts/services: `strategy_probe.py`, `pipeline_shadow_heartbeat.py`, `pipeline_compare_service.py`, `route_debug.py`, `exec_debug.py`, `run_executor_once.sh`, `run-sync.sh`, `run-dashboard.sh`.

### Execution→Telemetry→Dashboard pipeline
- Screener builds intents → executor_live pulls intents → risk_limits/risk_engine_v6 vet → order_router routes → events/trade_logs/state_publish write JSON/JSONL → sync_state mirrors → dashboard panels read `logs/state/*` and risk/router telemetry.

### Runtime surfaces
- Logs/state under `logs/state/*` (published by state_publish).
- Veto logs `logs/execution/risk_vetoes.jsonl`.
- Router metrics/events via `events.py`, `router_metrics.py`.

### Notable inactive/legacy
- size_model.py remains for compatibility but screener sizing is unlevered and active path in `signal_screener.py`.
