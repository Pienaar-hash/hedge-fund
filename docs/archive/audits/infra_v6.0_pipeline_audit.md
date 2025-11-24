# infra_v6.0 Pipeline Audit

## Components reviewed
- Shadow runner: `execution/pipeline_v6_shadow.py:1-116`.
- Executor hooks: `_maybe_run_pipeline_v6_shadow_heartbeat()` and `_maybe_run_pipeline_v6_compare()` (`execution/executor_live.py:1660-1718`).
- Daemons: `scripts/pipeline_shadow_heartbeat.py:1-38`, `scripts/pipeline_compare_service.py:1-33`.
- Comparison engine: `execution/intel/pipeline_v6_compare.py:1-98`.
- State/log artifacts: `logs/pipeline_v6_shadow.jsonl`, `logs/state/pipeline_v6_shadow_head.json`, `logs/pipeline_v6_compare.jsonl`, `logs/state/pipeline_v6_compare_summary.json`.

## Audit highlights
1. **Shadow parity** – `run_pipeline_v6_shadow()` builds `OrderIntent` objects with nav + exposure context, runs `RiskEngineV6.check_order()`, and only computes size/router decisions when risk allows. Inputs reference current configs (`_RISK_CFG` and `_PAIRS_CFG` from `execution/executor_live.py:224-235`), guaranteeing parity with live routing.
2. **Logging** – Shadow decisions are appended via `PIPELINE_SHADOW_LOGGER` (JsonlLogger). Heartbeat summary writes to `logs/state/pipeline_v6_shadow_head.json` with atomic replace semantics because it reuses `write_pipeline_v6_shadow_state()` from `execution/state_publish.py:130-136`.
3. **Compare alignment** – The compare engine aligns events by symbol and records mismatch metrics + size/slippage diffs. It writes per-diff JSONL entries and a summary JSON; `tests/test_pipeline_v6_compare_runtime.py:1-33` ensures the scheduler runs once per interval and writes to disk even when there are no diffs (heartbeat events set `heartbeat=True`).
4. **Supervisor readiness** – `ops/hedge.conf:25-48` includes both pipeline daemons, so the heartbeat/compare files remain hot even if the executor is paused. Interval env vars (`PIPELINE_SHADOW_HEARTBEAT_INTERVAL`, `PIPELINE_COMPARE_INTERVAL`) can be tuned via Supervisor environment overrides.
5. **No legacy dependencies** – The current repo contains no v5 pipeline runners or references to `pipeline_v5`. All pipeline state lives under `logs/` and is consumed directly by tests + docs.
6. **Warmup guardrails** – The compare summary now includes `is_warmup`/`warmup_reason`/`min_sample_size` to flag small sample windows. Ops should treat `is_warmup=True` snapshots as informational only and defer flag flips until steady-state (`is_warmup=False`).

## Verdict
Shadow and compare telemetry is complete, parity metrics are persisted locally, and Supervisor keeps the services running. With the documented tests in place, the pipeline contract is ready for v6.0-rc launch.

## Sizing parity telemetry
`logs/state/pipeline_v6_compare_summary.json` now includes a `sizing_diff_stats` block (p50/p95 deltas, upsize_count, sample_size) to track divergence between expected v6 sizing and live executor sizing. Any `upsize_count > 0` should be treated as a red flag during audits.
