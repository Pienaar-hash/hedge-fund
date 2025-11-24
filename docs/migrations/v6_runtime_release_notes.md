# v6 Runtime Release Notes (rc)

## Highlights
- **Typed risk engine** – `execution/risk_engine_v6.py:1-114` wraps `risk_limits.check_order()` so both the executor and the pipeline shadow report structured `RiskDecision` diagnostics.
- **Telemetry-first executor** – `_pub_tick()` (`execution/executor_live.py:3291-3358`) now writes nav, positions, risk, router health, symbol scores, and `synced_state.json` every loop via the shared writers in `execution/state_publish.py:77-190`.
- **Router auto-tune v6** – `execution/intel/router_autotune_v6.py:1-200` + `execution/intel/router_autotune_apply_v6.py:1-155` add bounded maker/taker and offset adjustments, with state stored at `logs/state/router_policy_suggestions_v6.json` and applied only when flags/allowlists permit.
- **Feedback allocator v6** – `execution/intel/feedback_allocator_v6.py:260-402` publishes nav/drawdown-aware cap recommendations to `logs/state/risk_allocation_suggestions_v6.json`.
- **Pipeline parity** – `execution/pipeline_v6_shadow.py:1-116` and `execution/intel/pipeline_v6_compare.py:1-98` keep a rolling shadow log plus compare summary under `logs/pipeline_v6_*.json[l]`.
- **Runtime probe** – `_maybe_write_v6_runtime_probe()` (`execution/executor_live.py:237-251`) records INTEL/RISK/PIPELINE/ROUTER flags in `logs/state/v6_runtime_probe.json` for audits.
- **Engine labeling** – `_ENGINE_VERSION` now reads the repo `VERSION` (set to `v6.0-rc ...`) and is emitted in `synced_state.json`/runtime probes, avoiding fallback mislabels.

## Operational updates
- Supervisor manifest simplified to `ops/hedge.conf:1-52` (only five programs, consistent PYTHONPATH/ENV across services).
- Dashboard (`dashboard/app.py:1-130`) reads only `logs/state/*.json`, eliminating the need for legacy Firestore dashboards.
- Sync service (`execution/sync_state.py:21-1258`) is hardened with NAV cutoff options (`NAV_CUTOFF_ISO`, `NAV_CUTOFF_SECAGO`) and refuses to write Firestore unless `ALLOW_PROD_WRITE/ALLOW_PROD_SYNC` are set.
- Router auto-apply tolerates missing/stale allocator state while still blocking explicit defensive mode; warnings surface when allocator state is absent or stale.
- Pipeline compare summary marks warmup via `is_warmup`/`warmup_reason` so early sample windows are clearly flagged.

## Tests & validation
- Added/updated coverage for runtime telemetry and intel: `tests/test_executor_state_files.py`, `tests/test_v6_runtime_activation.py`, `tests/test_pipeline_v6_shadow.py`, `tests/test_pipeline_v6_compare_runtime.py`, `tests/test_router_autotune_v6.py`, `tests/test_router_autotune_apply_v6.py`, `tests/test_feedback_allocator_v6.py`.

## Upgrade notes
- Update `config/risk_limits.json` and `config/pairs_universe.json` from git before rolling so `RiskEngineV6` sees the latest tiers/caps.
- Populate `logs/state/expectancy_v6.json` and `logs/state/symbol_scores_v6.json` (via `scripts/expectancy_v6_builder.py` and `scripts/router_autotune_v6_probe.py`) before enabling router auto-apply.
- Refresh `VERSION` so `_ENGINE_VERSION` logs show `v6.0-rc` instead of the default fallback `v6.0-beta-preview`.
