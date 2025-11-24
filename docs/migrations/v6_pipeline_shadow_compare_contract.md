# v6 Pipeline Shadow & Compare Contract

## Shadow generation
- Implementation: `execution/pipeline_v6_shadow.py:1-116`.
- Inputs:
  - `symbol` chosen from the executor's last positions snapshot (`execution/executor_live.py:1660-1675`).
  - `signal` dict with `side`, `notional`, `price`, `leverage`, tier + exposure context.
  - `nav_state` derived from `_LAST_NAV_STATE` plus portfolio gross fields.
  - `positions_state` (raw rows from `_LAST_POSITIONS_STATE`).
  - `risk_limits_cfg` (`config/risk_limits.json`), `pairs_universe_cfg` (`config/pairs_universe.json`), `sizing_cfg` (risk sizing section).
- Flow: `_order_intent_from_signal()` builds an `OrderIntent`, `RiskEngineV6.check_order()` returns a `RiskDecision`, `_size_order()` calls `execution.size_model.suggest_gross_usd()`, `_router_decision()` snapshots policy + offsets.
- JSONL log: each run appends to `logs/pipeline_v6_shadow.jsonl` via `append_shadow_decision()`. Files rotate automatically because `execution/log_utils.JsonlLogger` handles rotation in the logger returned by `get_logger()`.

## Heartbeats
- `_maybe_run_pipeline_v6_shadow_heartbeat()` (see `execution/executor_live.py:1660-1705`) calls `run_pipeline_v6_shadow()` opportunistically every `_PIPELINE_V6_HEARTBEAT_INTERVAL_S` (default 600s) when the executor has valid nav/positions.
- `scripts/pipeline_shadow_heartbeat.py:1-38` wraps `pipeline_v6_shadow.load_shadow_decisions(limit=100)` + `build_shadow_summary()` to push aggregated `{total, allowed, vetoed, generated_ts}` into `logs/state/pipeline_v6_shadow_head.json`. This daemon is managed by Supervisor (`ops/hedge.conf:25-48`).

## Compare service
- `execution/intel/pipeline_v6_compare.py:1-98` reads the shadow JSONL tail and live `logs/execution/orders_executed.jsonl` + `logs/execution/order_metrics.jsonl`. `_align_events()` matches shadow rows to the most recent live order per symbol.
- For each pair it records `veto_mismatch` (shadow allowed but no live order), `size_pct_diff`, and `slippage_diff_bps`. `_summarize_diffs()` ships aggregate stats.
- Outputs:
  - JSONL diffs at `logs/pipeline_v6_compare.jsonl` for forensics.
  - Summary JSON at `logs/state/pipeline_v6_compare_summary.json` via `write_pipeline_v6_compare_summary()` (`execution/state_publish.py:134-140`).
- Scheduler options:
  - Executor background job `_maybe_run_pipeline_v6_compare()` (`execution/executor_live.py:1707-1718`) runs when `PIPELINE_V6_SHADOW_ENABLED` and `_PIPELINE_V6_COMPARE_INTERVAL_S` allow.
  - Dedicated daemon `scripts/pipeline_compare_service.py:1-33` executes every `PIPELINE_COMPARE_INTERVAL` (default 900s) to guarantee out-of-band comparisons even if the executor is paused.

## Contract expectations
- Shadow log entries include: `symbol`, `intent` (serialized `OrderIntent`), `risk_decision`, optional `size_decision`, `router_decision`, `timestamp`.
- Head snapshot exposes `total`, `allowed`, `vetoed`, `generated_ts`, `last_decision` for monitoring.
- Compare summary exposes: `generated_ts`, `sample_size`, `veto_mismatch_pct`, `size_diff_stats:{mean,p50,p95}`, `slippage_diff_bps:{mean,p50,p95}`, plus warmup metadata `is_warmup`, `warmup_reason`, and `min_sample_size`. `is_warmup=True` (with `warmup_reason="sample_size_below_min"`) indicates the tail window is too small to trust parity stats; ops should wait until `is_warmup=False` before acting on compare metrics.

## Validation
- `tests/test_pipeline_v6_shadow.py:1-64` proves allowed/veto branches, JSONL logging, and summary math.
- `tests/test_pipeline_v6_compare_runtime.py:1-33` validates the scheduler gating logic and JSONL emission, ensuring no duplicate runs when `_LAST_PIPELINE_V6_COMPARE` is fresh.

The components above are the complete contract—no other files or dashboards are involved. If `logs/pipeline_v6_shadow.jsonl` or `logs/state/pipeline_v6_compare_summary.json` disappear, it means the executor or Supervisor daemons have stopped and v6 launch readiness should be considered blocked.
