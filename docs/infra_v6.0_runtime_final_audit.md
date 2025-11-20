# infra_v6.0 Runtime Final Audit

## Scope
Executor entrypoint, Supervisor wiring, and runtime flag surfaces as of this repo snapshot.

## Checks performed
1. **Entry process parity** – Verified that `execution/executor_live.py:1-1745` is the only trading process. Supervisor config (`ops/hedge.conf:1-52`) references only `executor_live`, the Streamlit dashboard, `execution/sync_state.py`, and the two pipeline daemons. No legacy multi-strategy dashboards or v5 executors are referenced.
2. **Flag propagation** – `execution/v6_flags.py:18-92` defines the canonical flag set. `_maybe_write_v6_runtime_probe()` (`execution/executor_live.py:237-251`) records the snapshot to `logs/state/v6_runtime_probe.json`; `tests/test_v6_runtime_activation.py:1-30` verifies this path. Every telemetry write uses `get_v6_flag_snapshot()` so the `synced_state` payload is fully annotated (`execution/state_publish.py:153-190`).
3. **Telemetry completeness** – `_pub_tick()` (`execution/executor_live.py:3291-3358`) writes nav, positions, risk, symbol scores, router health, and synced_state via `execution/state_publish`. `tests/test_executor_state_files.py:1-38` ensures this occurs even when telemetry writes are mocked. No other writers touch these files, so the runtime meets the state contract documented elsewhere.
4. **NAV & drawdown guards** – `execution/sync_state.py:1085-1258` enforces NAV cutoffs, drawdown snapshots, and Firestore gating. `_firestore_enabled()` defaults to OFF unless `ALLOW_PROD_WRITE`/`ALLOW_PROD_SYNC` are set, preventing accidental prod writes.
5. **Version tagging** – `_ENGINE_VERSION` in `execution/executor_live.py:116-126` reads the repo `VERSION` file (now set to `v6.0-rc ...`) and falls back to `"v6.0-beta-preview"` only when missing. This value is emitted in `synced_state.json` and runtime probes, and tests assert it is non-empty.
6. **Testing** – The pytest suite contains dedicated runtime coverage (`tests/test_executor_state_files.py`, `tests/test_v6_runtime_activation.py`, `tests/test_pipeline_v6_compare_runtime.py`, `tests/test_router_autotune_apply_v6.py`). No skipped/stubbed tests mention deprecated v5 runtimes.
7. **Resilience & warmup signals** – Router auto-apply logs missing/stale allocator state and defaults to cautious instead of defensive; pipeline compare summaries expose `is_warmup`/`warmup_reason` to prevent over-interpreting early samples.

## Verdict
All runtime surfaces are v6-compliant, flag logging is active, and no Supervisor entries reference legacy dashboards. The executor is safe to run under the documented `ops/hedge.conf` manifest, provided the `VERSION` file is updated before tagging the release build.
