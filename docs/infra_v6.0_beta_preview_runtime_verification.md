# v6.0-beta-preview Runtime Verification

## Component Status

component | status | notes / proof
--------- | ------ | -------------
Executor core (`execution/executor_live.py`) | Pass | `_pub_tick()` runs every loop, logs `[v6-runtime] state write complete …`, and writes `nav.json`, `positions.json`, `risk_snapshot.json`, `symbol_scores_v6.json`, and `synced_state.json` via `execution.state_publish`.
Telemetries v2 (`execution/state_publish.py`, logs/state/*) | Pass | `logs/state/` on the VPS now contains `nav.json`, `positions.json`, `risk_snapshot.json`, `synced_state.json`, and all v6 intel files. Tests: `tests/test_executor_state_files.py`, `tests/test_state_publish_stats.py`, `tests/test_sync_state_firestore_enabled.py`.
Screener (`execution/signal_screener.py`) | Pass | Imports `log_v6_flag_snapshot` at startup, uses shared v6 flags, and remains flag-gated. Covered by `tests/test_signal_screener.py`.
RiskEngine v6 (`execution/risk_engine_v6.py`) | Pass | Executor and screener call `RiskEngineV6` when `RISK_ENGINE_V6_ENABLED=1`. Telemetry risk snapshot comes from `engine.build_risk_snapshot()`. Tests: `tests/test_risk_engine_v6.py`, `tests/test_risk_limits.py`.
Router policy + autotune (`execution/order_router.py`, `execution/intel/router_policy.py`, `execution/intel/router_autotune_v6.py`) | Pass | Router policy snapshot feeds telemetry and intel; autotune suggestions emitted to `logs/state/router_policy_suggestions_v6.json`; apply gate controlled by `ROUTER_AUTOTUNE_V6_APPLY_ENABLED`. Tests: `tests/test_router_policy.py`, `tests/test_router_autotune_v6.py`.
Intel surfaces (expectancy, symbol score, feedback allocator, pipeline shadow/compare) | Pass | Executor intel loop writes `expectancy_v6.json`, `symbol_scores_v6.json`, `risk_allocation_suggestions_v6.json`; pipeline shadow and compare produce both JSONL and state summaries. Tests: `tests/test_expectancy_v6.py`, `tests/test_symbol_score_v6.py`, `tests/test_feedback_allocator_v6.py`, `tests/test_pipeline_v6_shadow.py`, `tests/test_pipeline_v6_compare.py`, `tests/test_pipeline_v6_compare_runtime.py`.
sync_state + Firestore (`execution/sync_state.py`, `execution/firestore_utils.py`) | Pass* | Firestore writes remain guarded; `_firestore_enabled()` + new import guards prevent crashes when Firestore is unavailable. `sync_state` now consumes `logs/state/synced_state.json` schema. Tests: `tests/test_sync_state_firestore_enabled.py`.
Supervisor stack (`ops/hedge.conf`) | Pass | Every program exports `PYTHONPATH="/root/hedge-fund"` with consistent ENV flags. Executor, sync_state, pipeline shadow/compare are RUNNING. Dashboard remains pending due to Streamlit spawn limitations in this environment.
Dashboard (`dashboard/app.py`) | Pending | Streamlit entry verified via tests, but Supervisor spawn is blocked by environment socket restrictions; dashboard refactor is scheduled for a future batch.

## Telemetry Contract Highlights

- **Nav & Positions:** `_pub_tick()` writes `logs/state/nav.json` and `logs/state/positions.json` via `write_nav_state()` / `write_positions_state()`. Dashboard helpers (`dashboard/live_helpers.py`) and intel builders (`execution/intel/expectancy_v6.py`) consume these files.
- **Risk Snapshot:** `_maybe_emit_risk_snapshot()` refreshes `_LAST_RISK_SNAPSHOT`, which `_pub_tick()` flushes to `logs/state/risk_snapshot.json`. Tests ensure the writer produces output even when risk data is empty.
- **Synced State Schema:** Built with `execution.state_publish.build_synced_state_payload()` producing `{items, nav, engine_version, v6_flags, updated_at}`. sync_state reads this via `_read_positions_snapshot()` and degrades gracefully when missing.
- **Symbol Scores & Intel:** `execution.intel.symbol_score_v6` generates `symbol_scores_v6.json`; executor keeps the latest snapshot for telemetry even when the intel loop is idle.
- **Router Health & Pipeline:** Router health snapshots write both JSON state and a rotating JSONL log; pipeline shadow + compare maintain `pipeline_v6_shadow_head.json`, `pipeline_v6_compare_summary.json`, and `logs/pipeline_v6_compare.jsonl`.

## v5 vs v6 Regression Guard

To verify legacy behavior with v6 flags **off**:

```bash
INTEL_V6_ENABLED=0 \
RISK_ENGINE_V6_ENABLED=0 \
PIPELINE_V6_SHADOW_ENABLED=0 \
ROUTER_AUTOTUNE_V6_ENABLED=0 \
FEEDBACK_ALLOCATOR_V6_ENABLED=0 \
ROUTER_AUTOTUNE_V6_APPLY_ENABLED=0 \
pytest tests/test_risk_limits.py \
       tests/test_screener_tier_caps.py \
       tests/test_router_smoke.py \
       tests/test_order_router_ack.py \
       tests/test_exchange_dry_run.py \
       tests/test_order_metrics.py -q
```

These suites continue to exercise the hardened v5.10 pathways and must remain green before toggling v6 features on.

## Operator Runbook (v6 flags ON)

1. **Check Flags:** Tail executor stdout for `[v6] flags …` line during startup. All v6 flags should be `1` except `ROUTER_AUTOTUNE_V6_APPLY_ENABLED=0`.
2. **Telemetry Files:** `ls logs/state` should show nav/positions/risk/synced/symbol scores plus all existing intel files. `_pub_tick()` logs `[v6-runtime] state write complete …` each loop.
3. **Intel & Router:** `tail logs/router_health.jsonl` for router snapshots; `tail logs/pipeline_v6_compare.jsonl` for shadow/compare diffs + heartbeats.
4. **sync_state:** `supervisorctl status hedge-sync_state` should be RUNNING; its stdout should log `[sync] Firestore heartbeat skipped (disabled)` unless Firestore is enabled.
5. **v5 Parity Run:** Temporarily disable v6 flags and run the pytest bundle above before making production changes.

## Next Steps toward v6.0-rc

- Resolve the dashboard Supervisor spawn issue (Streamlit service hardening + new telemetry dashboards).
- Optional: enable `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` for production once the allowlist and monitoring are finalized.
- Optional: enable Firestore (`FIRESTORE_ENABLED=1`) and validate remote exposure dashboards using the synced_state schema.

With executor, intel, telemetry, and sync_state all verified, the v6.0-beta-preview runtime meets the core architectural contract under Supervisor.
