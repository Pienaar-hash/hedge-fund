# infra_v6.0 Telemetry Audit

## Objectives
Confirm that every state/log artifact required by the dashboard, sync_state, and migration probes is produced inside the repo with no stale v5 references.

## Findings
1. **State writers centralized** – `execution/state_publish.py:77-190` contains every writer used by the runtime. Functions cover nav, positions, router health, risk snapshot, expectancy, symbol scores, router suggestions, risk allocation suggestions, pipeline shadow head, pipeline compare summary, and v6 runtime probe states. No other module writes into `logs/state/`.
2. **Executor tick coverage** – `_pub_tick()` (`execution/executor_live.py:3291-3358`) writes nav, positions, risk snapshot, symbol scores, and synced state on every loop. Failures are logged per file, and the final log line `[v6-runtime] state write complete ...` is visible in `logs/execution/*`.
3. **Intel cadence** – `_maybe_publish_execution_intel()` (`execution/executor_live.py:341-405`) guards expectancy, symbol scores, router suggestions, and risk allocation suggestions via `EXEC_INTEL_PUBLISH_INTERVAL_S`. Each component writes to its respective file under `logs/state/`. Tests `tests/test_router_autotune_v6.py`, `tests/test_feedback_allocator_v6.py`, and `tests/test_symbol_score_v6.py` confirm the files are populated with the expected schema.
4. **Router metrics & health** – `_maybe_emit_router_health_snapshot()` (`execution/executor_live.py:254-272`) mirrors router health to both `logs/router_health.jsonl` and `logs/state/router_health.json`. This data is re-used by symbol scoring and router auto-tune, ensuring telemetry loops internally.
5. **Pipeline artifacts** – `execution/pipeline_v6_shadow.py:1-116` logs to `logs/pipeline_v6_shadow.jsonl`. `scripts/pipeline_shadow_heartbeat.py:1-38` and `scripts/pipeline_compare_service.py:1-33` keep `logs/state/pipeline_v6_shadow_head.json` and `logs/state/pipeline_v6_compare_summary.json` fresh. Tests confirm these files exist and are written (`tests/test_pipeline_v6_shadow.py:1-64`, `tests/test_pipeline_v6_compare_runtime.py:1-33`).
6. **Sync + dashboard** – `execution/sync_state.py:21-1258` consumes only the local files listed above, wraps them in drawdown/NAV payloads, and mirrors to Firestore when allowed. `dashboard/app.py:1-130` reads from `logs/state/` rather than legacy dashboards. There are no references to "multi_strategy_dashboard" or other v5 sensors.
7. **Runtime flag log** – `write_v6_runtime_probe_state()` is invoked via `_maybe_write_v6_runtime_probe()` (`execution/executor_live.py:237-251`) whenever any v6 flag is enabled, leaving an auditable record in `logs/state/v6_runtime_probe.json`.

## Conclusion
Telemetry coverage is complete and self-contained. Every consumer (sync_state, dashboard, CLI probes) reads from the canonical `logs/state/` directory, and every writer traces back to `execution/state_publish.py`. There are no v5 or legacy telemetry references left in the tree.
