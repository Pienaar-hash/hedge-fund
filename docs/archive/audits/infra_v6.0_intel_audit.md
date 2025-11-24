# infra_v6.0 Intel Audit

## Scope
Expectancy, symbol scores, router auto-tune, feedback allocator, and router auto-apply.

## Review
1. **Expectancy builder** – `execution/intel/expectancy_v6.py:1-333` loads trades + router metrics from `logs/execution/` and writes `logs/state/expectancy_v6.json`. CLI `scripts/expectancy_v6_builder.py:1-39` uses the same module, so manual refreshes can be compared to background jobs. Tests in `tests/test_expectancy_v6.py` verify output fields.
2. **Symbol scoring** – `execution/intel/symbol_score_v6.py:1-137` merges expectancy rows with router health snapshots to produce normalized 0–1 scores. `_LAST_SYMBOL_SCORES_STATE` is captured inside the executor (`execution/executor_live.py:371-375`) for use in `synced_state` and telemetry. Tests `tests/test_symbol_score_v6.py` and `tests/test_execution_intel_sizing.py` cover weighting.
3. **Router auto-tune** – `execution/intel/router_autotune_v6.py:1-200` produces proposals stored in `logs/state/router_policy_suggestions_v6.json`. The module inspects `router_autotune_v6` settings in `config/risk_limits.json` to enforce min/max bias and offset steps. Tests `tests/test_router_autotune_v6.py:1-60` guarantee maker/taker shifts respect those bounds.
4. **Router apply** – `execution/intel/router_autotune_apply_v6.py:1-155` enforces environment toggles, symbol allowlists, max delta caps, and risk-mode gating. Tests `tests/test_router_autotune_apply_v6.py:1-80` exercise the guardrails (flag off, defensive mode, quality filter). Live routing hits this module through `execution/order_router.py` when computing per-symbol policies.
5. **Feedback allocator** – `execution/intel/feedback_allocator_v6.py:260-402` emits `logs/state/risk_allocation_suggestions_v6.json`. The CLI probe (`scripts/feedback_allocator_probe_v6.py:1-70`) demonstrates parity and renders suggested caps. Tests `tests/test_feedback_allocator_v6.py:1-120` prove risk-mode scaling works.
6. **Executor orchestration** – `_maybe_publish_execution_intel()` (`execution/executor_live.py:341-405`) refreshes the entire intel chain when `INTEL_V6_ENABLED` is true. Router suggestions and allocator outputs only refresh at their own intervals (configurable via env), and the executor logs each write.
7. **No legacy dependencies** – There are no references to legacy "intel dashboards" or v5 aggregator jobs. Everything flows through the state files under `logs/state/` with the writers described above.

## Conclusion
All intel surfaces are generated from repo-local code, tested, and exported to stable JSON files. The contract is suitable for v6.0-rc launch.
