# Infra v6.0 Runtime Activation & Telemetry Sanity — Batch 11

## Module Presence Check

Verified the v6 stack exists in this checkout:

- `execution/risk_engine_v6.py` (v6 risk wrapper)  
- `execution/pipeline_v6_shadow.py` (shadow orchestrator)  
- Intel modules:  
  - `execution/intel/expectancy_v6.py`  
  - `execution/intel/symbol_score_v6.py`  
  - `execution/intel/router_policy.py` (v6 helpers)  
  - `execution/intel/router_autotune_v6.py` + `execution/intel/router_autotune_apply_v6.py`  
  - `execution/intel/feedback_allocator_v6.py`  
  - `execution/intel/pipeline_v6_compare.py`
- Telemetry writers in `execution/state_publish.py` cover nav/positions/router_health/risk_snapshot plus v6 heads (expectancy, scores, router suggestions, risk allocator, pipeline shadow, runtime probe, compare summary).
- Tests present under `tests/test_*v6*.py`.

## Flag Wiring & Runtime Hooks

- Environment flags are parsed in `execution/executor_live.py` (`INTEL_V6_ENABLED`, `RISK_ENGINE_V6_ENABLED`, `PIPELINE_V6_SHADOW_ENABLED`, `ROUTER_AUTOTUNE_V6_ENABLED`, `FEEDBACK_ALLOCATOR_V6_ENABLED`, `ROUTER_AUTOTUNE_V6_APPLY_ENABLED`) as well as in `execution/signal_screener.py`, `execution/intel/router_autotune_apply_v6.py`, etc.
- Screener & executor pathways:  
  - Screener calls RiskEngineV6 when `RISK_ENGINE_V6_ENABLED` is true.  
  - Executor uses RiskEngineV6 in `_evaluate_order_risk` when the flag is set.  
  - Pipeline shadow orchestrator (`pipeline_v6_shadow.run_pipeline_v6_shadow`) is invoked via `_maybe_run_pipeline_v6_shadow` when `PIPELINE_V6_SHADOW_ENABLED` is true.  
  - Intel refresh loop (`_maybe_publish_execution_intel`) respects `INTEL_V6_ENABLED` and writes expectancy/scores/router suggestions/risk allocations.  
  - Router auto-tune suggestions write through `write_router_policy_suggestions_state`; application layer is gated by `ROUTER_AUTOTUNE_V6_APPLY_ENABLED`.

## Telemetry / State Outputs

- `execution/state_publish.py` writes canonical v2 state (nav, positions, router_health, risk_snapshot, universe) even without trades.  
- v6 intel states: `expectancy_v6.json`, `symbol_scores_v6.json`, `router_policy_suggestions_v6.json`, `risk_allocation_suggestions_v6.json`, `pipeline_v6_shadow_head.json`, `pipeline_v6_compare_summary.json`, and the new `v6_runtime_probe.json`.
- Router metrics (`logs/execution/order_metrics.jsonl`) now include auto-tune metadata.

## New Runtime Logging & Heartbeat (Batch 11)

- Executor now logs consolidated v6 flags on startup:  
  `"[v6] flags INTEL_V6_ENABLED=1 ... ROUTER_AUTOTUNE_V6_APPLY_ENABLED=0"`
- Runtime probe writer (`v6_runtime_probe.json`) records the flag snapshot periodically (default every 300s) even when there is no trading activity, ensuring at least one v6 state write per run.

## Tests Added / Updated

- `tests/test_v6_runtime_activation.py` validates flag snapshot parsing and the runtime probe writer.
- Existing router and telemetry tests updated to cover new metadata/writers.

## Outstanding Notes

- With the new logging/probe plus existing intel refresh loops, ops should now see explicit evidence in logs/state whenever the v6 stack is enabled, even in idle markets.  
- Further activation (e.g., enabling router auto-tune apply) still requires populating the allowlist env var and observing `router_metrics` telemetry for `autotune_applied=true`.
