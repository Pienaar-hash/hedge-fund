## v7 Telemetry Contract

### Publishers
- `execution/state_publish.py`: writes every canonical JSON surface under `logs/state/` (nav, positions, risk snapshots, router health, KPIs, execution health, pipeline summaries, intel hints).
- `execution/sync_state.py`: mirrors the local JSON files for dashboards/Firestore while guarding schema hygiene and nav freshness.
- `execution/events.py`: JSONL order/fill/error stream (`logs/execution/orders_executed.jsonl` + `order_metrics.jsonl`).
- `execution/router_metrics.py`: builds router-level aggregates and enriches the router health payload with maker/taker/fallback stats.
- `execution/trade_logs.py` / `pnl_tracker.py`: record trade/PnL context and feed downstream KPI builders.
- `risk_limits.py`: emits `logs/execution/risk_vetoes.jsonl` with structured gate diagnostics.

### Canonical state files (`logs/state/`)
- `nav.json` / `nav_state.json`: live NAV totals, NAV age, equity breakdown, sources_ok, `aum` slices, and `nav_health_snapshot` metadata.
- `positions.json` / `positions_state.json`: filtered positions with symbol, qty, entry/mark prices, leverage, and PnL, normalized via `_normalize_positions`.
- `risk_snapshot.json`: `dd_state`, `daily_loss_frac`, `risk_mode`, `router_stats`, per-symbol risk/vol sizing, and enriched drawdown/daily-loss fractions added by `write_risk_snapshot_state`.
- `router_health.json`: per-symbol maker/taker counts, fill ratios, fallback ratios, slippage quantiles, the computed `router_health_score`, and policy metadata.
- `kpis_v7.json`: aggregated KPIs built from nav/risk/router/expectancy snapshots (`build_kpis_v7`), including nav age, risk dd_state, ATR regime, fee/PnL ratio, router quality, and per-symbol ATR/DD mirrors.
- `expectancy_v6.json` & `symbol_scores_v6.json`: expectancy statistics and symbol-wise scoring inputs powering the intel + feedback allocator loops.
- `router_policy_suggestions_v6.json` & `risk_allocation_suggestions_v6.json`: policy change recommendations and allocator cues computed by the intel loops.
- `pipeline_v6_shadow_head.json` & `pipeline_v6_compare_summary.json`: pipeline heartbeat + compare metadata required before flipping pipeline flags.
- `execution_health.json` & `universe.json`: execution health snapshots (via `compute_execution_health`) and universe metadata used for diagnostics.
- `synced_state.json`: consolidated payload (`items`, `nav`, `engine_version`, `v6_flags`) consumed by `sync_state.py`.
- `v6_runtime_probe.json`: runtime flag snapshot (engine version + enabled flags) written whenever any v6 flag is enabled.

### JSONL telemetry (`logs/execution/` & `logs/`)
- `orders_executed.jsonl`: `log_event()` writes ACK/FILL/CLOSE events with required fields, used by expectancy and pipeline compare.
- `order_metrics.jsonl`: `_append_order_metrics()` records router latency, fallback/slippage, and applied policy offsets.
- `risk_vetoes.jsonl`: `_emit_veto()` streams structured veto diagnostics, thresholds, nav_total, and dd_state/ATR context.
- `router_health.jsonl` & `pipeline*.jsonl`: append-only mirrors of state snapshots for forensic tailing.

### Consumers & invariants
- Dashboard panels (`nav_helpers`, `pipeline_panel`, `router_health`, `intel_panel`, `kpi_panel`) read these canonical files and expect `spin` invariants: fractions rather than percentages, `nav_total` derived from `nav_health_snapshot`, and stable per-symbol ATR/DD data.
- External monitoring can tail the JSONL feeds for risk/router alerts; the telemetry contract forbids telemetry from mutating strategy/risk decisions.
- Tests such as `tests/test_executor_state_files.py`, `tests/test_v6_runtime_activation.py`, `tests/test_pipeline_v6_shadow.py`, and `tests/test_router_autotune_v6.py` assert the required files exist with their schemas and flag snapshots.
