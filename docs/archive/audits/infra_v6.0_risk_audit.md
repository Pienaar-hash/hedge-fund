# infra_v6.0 Risk Audit

## Scope
Risk limits, RiskEngineV6, RiskAutotuner, nav freshness guards, and allocator outputs.

## Observations
1. **Config hygiene** – `config/risk_limits.json:1-74` lists every global and per-symbol cap used in prod. Fields match the expectations inside `execution/risk_limits.py:1-210` (daily loss, drawdown, nav share caps, min notional, tier buckets). No stale v5 keys remain.
2. **Engine implementation** – `execution/risk_engine_v6.py:1-114` wraps legacy `check_order()` but exposes typed `OrderIntent`/`RiskDecision` objects used by the executor, pipeline shadow, and telemetry writers. Tests (`tests/test_risk_engine_v6.py`, `tests/test_risk_limits.py`) assert allow/veto parity.
3. **Autotune** – `execution/risk_autotune.py:1-210` reads cached doctor and order metrics to tune `RiskGate` fields. It persists state to `logs/cache/risk_state.json` and logs each adjustment via python logging. There are no silent config rewrites.
4. **Allocator telemetry** – `execution/intel/feedback_allocator_v6.py:260-402` generates a complete `risk_allocation_suggestions_v6.json` (global risk mode, per-symbol caps/weights). Tests `tests/test_feedback_allocator_v6.py:1-120` cover normal vs cautious modes and ensure suggestions sum to ≤ 1.0.
5. **Nav freshness enforcement** – `execution/risk_limits.is_nav_fresh()` + `enforce_nav_freshness_or_veto()` gate orders when nav snapshots are stale. The code points to `logs/cache/nav_confirmed.json` and `logs/nav_log.json`, which are produced by `execution/nav.py:1-100` and `_persist_nav_log()` inside the executor.
6. **Telemetry** – Risk snapshots are flushed via `_maybe_emit_risk_snapshot()` (`execution/executor_live.py:280-309`) using `execution/utils/execution_health.compute_execution_health()` and stored in `logs/state/risk_snapshot.json`. Dashboard and allocator consume this file; no remote-only telemetry exists.
7. **Flags/Audit trail** – `RISK_ENGINE_V6_ENABLED` is part of the runtime probe (`execution/v6_flags.py:18-59`, `_maybe_write_v6_runtime_probe()`), so enabling/disabling the new engine is auditable.

## Conclusion
Risk enforcement is entirely contained within the current repo. Config, runtime logic, and telemetry agree on schema, and the automated tests cover all major scenarios. No adjustment is needed before v6.0-rc cut besides updating `VERSION` once the release is tagged.
