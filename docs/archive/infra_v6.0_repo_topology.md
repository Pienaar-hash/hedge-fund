# infra_v6.0 Repo Topology

## execution/
- `execution/executor_live.py:1-1745` – runtime brain (signals → risk → router → telemetry) with embedded intel/heartbeat hooks.
- `execution/v6_flags.py:18-92` – centralized env flag loader for INTEL/RISK/PIPELINE/ROUTER toggles.
- Risk modules: `execution/risk_limits.py:1-210`, `execution/risk_engine_v6.py:1-114`, `execution/risk_autotune.py:1-210`.
- Intel + sizing: `execution/intel/expectancy_v6.py:1-333`, `execution/intel/symbol_score_v6.py:1-137`, `execution/intel/router_autotune_v6.py:1-200`, `execution/intel/feedback_allocator_v6.py:260-402`, `execution/size_model.py`, `execution/position_sizing.py:1-70`.
- Pipeline + telemetry: `execution/pipeline_v6_shadow.py:1-116`, `execution/intel/pipeline_v6_compare.py:1-98`, `execution/state_publish.py:77-190`, `execution/sync_state.py:21-1258`.
- Utilities: `execution/log_utils.py:1-150`, `execution/events.py:1-70`, `execution/utils/*` (metrics, vol, toggles), `execution/universe_resolver.py:1-160`.

## dashboard/
- `dashboard/app.py:1-130` – Streamlit UI that reads `logs/state/*.json`, doctor caches, and router health snapshots.
- Helpers in `dashboard/dashboard_utils.py`, `dashboard/router_health.py`, `dashboard/live_helpers.py` align UI metrics with the same state writers in execution.

## scripts/
- Ops CLIs (smoke tests, registry tools) plus three v6-critical daemons: `scripts/pipeline_shadow_heartbeat.py:1-38`, `scripts/pipeline_compare_service.py:1-33`, `scripts/expectancy_v6_builder.py:1-39`.
- Probes for intel contracts: `scripts/router_autotune_v6_probe.py:1-78`, `scripts/feedback_allocator_probe_v6.py:1-70`, `scripts/expectancy_v6_builder.py` reuse execution modules so CLI runs match Supervisor output.

## utils/
- Shared clients (`utils/firestore_client.py:1-155`) and general helpers (`utils/__init__.py:1-120`). Firestore wrappers return `_NoopDB` instances locally so the runtime stays read-only unless creds exist.

## tests/
- Extensive pytest suite verifying every contract. Key coverage: `tests/test_executor_state_files.py:1-38`, `tests/test_pipeline_v6_shadow.py:1-64`, `tests/test_pipeline_v6_compare_runtime.py:1-33`, `tests/test_router_autotune_v6.py:1-60`, `tests/test_router_autotune_apply_v6.py:1-80`, `tests/test_feedback_allocator_v6.py:1-120`, `tests/test_v6_runtime_activation.py:1-30`.

## config/
- Risk + universe definitions (`config/risk_limits.json:1-74`, `config/pairs_universe.json:1-80`), runtime router tuning (`config/runtime.yaml:1-25`), strategy registry, dashboard settings. Everything is JSON/YAML under version control—no generated configs.

## ops/
- `ops/hedge.conf:1-52` – Supervisor manifest for prod/test clusters. Sets `PYTHONPATH=/root/hedge-fund`, exports `ALLOW_PROD_WRITE/SYNC`, and manages stdout logfiles per program.

The directories above are the only ones referenced by the live runtime. No legacy dashboards or v5 monoliths remain wired into Supervisor; every dependency is visible under these paths.
