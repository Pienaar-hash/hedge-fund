# v6 Runtime Telemetry Contract

## Producers & cadence
- **Executor tick** – `_pub_tick()` (`execution/executor_live.py:3291-3358`) runs every loop (~once per second) and writes nav, positions, risk, symbol scores, router health, and `synced_state.json` via the `execution/state_publish.py:77-190` helpers.
- **Risk/Intel refresh** – `_maybe_emit_risk_snapshot()` and `_maybe_publish_execution_intel()` (`execution/executor_live.py:280-405`) run every 2–30 minutes, updating risk snapshots, router health, expectancy, symbol scores, router suggestions, risk allocation suggestions, and v6 runtime probes.
- **Pipeline daemons** – `scripts/pipeline_shadow_heartbeat.py:1-38` (default 10 minutes) and `scripts/pipeline_compare_service.py:1-33` (default 15 minutes) refresh the shadow head + compare summary states.
- **Sync process** – `execution/sync_state.py:21-1258` mirrors local state to Firestore every `SYNC_INTERVAL_SEC` (default 20 seconds) and persists drawdown/nav caches for dashboards.

## Canonical state files (`logs/state/`)
| File | Writer | Schema | Consumer |
| --- | --- | --- | --- |
| `nav.json` | `write_nav_state()` (`execution/state_publish.py:98-110`) | `{series: [ {t,equity,...} ], peak_equity, total_equity, gross_exposure_usd, net_exposure_usd, largest_position_usd, updated_at}` – exposures appended via `_exposure_from_positions()` (`execution/sync_state.py:1003-1015`). | Dashboard cards, sync_state, doctor.
| `positions.json` | `write_positions_state()` (`execution/state_publish.py:94-110`) | `{items: [{symbol, side, qty, entry_price, mark_price, pnl, leverage, notional, ts}], gross_exposure_usd, net_exposure_usd, largest_position_usd, updated_at}` – normalization logic lives in `_normalize_positions_items()` (`execution/sync_state.py:970-997`). | Dashboard tables, `execution.screenshot` tooling.
| `risk_snapshot.json` | `_maybe_emit_risk_snapshot()` (`execution/executor_live.py:280-309`) calling `write_risk_snapshot_state()` | `{updated_ts, symbols: [{symbol, router: {maker_fill_ratio, fallback_ratio, slip_q50,q95,warnings}, risk: {dd_today_pct, risk_flags, sharpe_state}, vol, sizing}]}` per `execution/utils/execution_health.py:1-110`. | Dashboard drilldowns, feedback allocator.
| `router_health.json` | `_build_router_health_snapshot()` + `write_router_health_state()` (`execution/executor_live.py:152-272`) | `{updated_ts, symbols: [{symbol, maker_fill_rate, fallback_rate, slippage_p50/p95, policy:{maker_first, taker_bias, quality, reason}}]}`. | Symbol score + router auto-tune inputs.
| `expectancy_v6.json` | `execution/intel/expectancy_v6.py:260-333` via `write_expectancy_state()`; fields include `symbols` dict (per-symbol expectancy stats), `hours`, `regimes`, `lookback_hours`, `sample_count`, `updated_ts`. | Symbol scoring, router auto-tune, feedback allocator.
| `symbol_scores_v6.json` | `execution/intel/symbol_score_v6.py:1-137` via `write_symbol_scores_state()`; array of `{symbol, score, components, inputs}` plus `updated_ts`. | Feedback allocator, dashboard intel tab.
| `router_policy_suggestions_v6.json` | `execution/intel/router_autotune_v6.py:1-200` via `write_router_policy_suggestions_state()`; contains `symbols` array with `current_policy`, `proposed_policy`, `regime`, `quality`, `rationale`, `lookback_days`. | Router auto-apply (`execution/intel/router_autotune_apply_v6.py:1-155`), CLI probe.
| `risk_allocation_suggestions_v6.json` | `execution/intel/feedback_allocator_v6.py:260-402` via `write_risk_allocation_suggestions_state()`; includes `{global:{current_equity_usd,current_drawdown_pct,risk_mode}, symbols:[{symbol,score,expectancy,router_policy,caps,suggested_caps,suggested_weight,rationale}]}`. | Manual ops review, future auto-cap tuning.
| `pipeline_v6_shadow_head.json` | `scripts/pipeline_shadow_heartbeat.py:1-38` via `write_pipeline_v6_shadow_state()`; `{total, allowed, vetoed, generated_ts, last_decision}` per `execution/pipeline_v6_shadow.py:87-116`. | Ops dashboards, migration gating.
| `pipeline_v6_compare_summary.json` | `execution/intel/pipeline_v6_compare.py:1-98` via `write_pipeline_v6_compare_summary()`; `{generated_ts, sample_size, veto_mismatch_pct, size_diff_stats:{mean,p50,p95}, slippage_diff_bps:{...}, is_warmup, warmup_reason, min_sample_size}`. | Release checks before enabling v6 flags; treat `is_warmup=true` as informational until sample_size exceeds the steady-state minimum.
| `v6_runtime_probe.json` | `_maybe_write_v6_runtime_probe()` (`execution/executor_live.py:237-251`) using `write_v6_runtime_probe_state()`; captures `{INTEL_V6_ENABLED,..., engine_version, ts}` from `execution/v6_flags.py:18-92`. | Runtime health monitor, tests (`tests/test_v6_runtime_activation.py:1-30`).
| `synced_state.json` | `write_synced_state()` (`execution/state_publish.py:171-190`) called from `_pub_tick()`; `{items, nav, engine_version, v6_flags, updated_at}` per `build_synced_state_payload()` (`execution/state_publish.py:153-190`). `engine_version` must be non-empty and `v6_flags` must carry the full flag snapshot. | `execution/sync_state.py:1185-1238`, downstream data brokers.

## JSONL telemetry (`logs/execution/` & `logs/`)
- `logs/execution/orders_executed.jsonl` – `execution/events.write_event()` (`execution/events.py:1-70`) receives ACK/FILL/CLOSE events with required fields enforced by `_REQUIRED_FIELDS`. Feeds expectancy builder and pipeline compare.
- `logs/execution/order_metrics.jsonl` – `_append_order_metrics()` (`execution/executor_live.py:138-142`) writes router latency, fallback, slippage, and policy data. Read by `execution/utils/metrics.py:73-145`, router auto-tune, and risk autotune.
- `logs/execution/risk_vetoes.jsonl` – `execution/risk_limits.LOG_VETOES` stores structured veto reasons for audit.
- `logs/router_health.jsonl` – `_maybe_emit_router_health_snapshot()` writes the same payload mirrored to `logs/state/router_health.json` so ops can tail JSONL history.
- `logs/pipeline_v6_shadow.jsonl` – `execution/pipeline_v6_shadow.append_shadow_decision()` writes every shadow result for forensic review.
- `logs/pipeline_v6_compare.jsonl` – `execution/intel/pipeline_v6_compare.compare_pipeline_v6()` records each diff plus periodic heartbeats.

## Contract tests
- `tests/test_executor_state_files.py:1-38` ensures `_pub_tick()` populates nav/positions/risk/synced payloads and includes `v6_flags`.
- `tests/test_v6_runtime_activation.py:1-30` verifies runtime probes include all flag keys.
- `tests/test_pipeline_v6_compare_runtime.py:1-33` asserts the compare scheduler writes JSONL entries and respects intervals.

All telemetry artifacts listed here exist at runtime and are regenerated continuously; no legacy files (multi_strategy dashboards, v5 caches) are referenced anywhere in the v6 contract.
