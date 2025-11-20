# v6 Intel Contract

## Data sources
- Trade and router telemetry: `logs/execution/orders_executed.jsonl` + `logs/execution/order_metrics.jsonl` via `execution/events.py:1-70` and `_append_order_metrics()` (`execution/executor_live.py:138-142`).
- Router health snapshot: `logs/state/router_health.json` from `_build_router_health_snapshot()` (`execution/executor_live.py:152-193`).
- Risk snapshot: `logs/state/risk_snapshot.json` from `_maybe_emit_risk_snapshot()` (`execution/executor_live.py:280-309`).
- Nav snapshot: `logs/state/nav.json` for equity context.

## Expectancy builder
- Implementation: `execution/intel/expectancy_v6.py:1-333`.
- CLI: `scripts/expectancy_v6_builder.py:1-39` for ad-hoc refreshes.
- Output: `logs/state/expectancy_v6.json` containing `{symbols:{SYMBOL:{count, hit_rate, expectancy, expectancy_per_risk, max_drawdown}}, hours:{hour:int -> expectancy}, regimes:{regime:str -> expectancy}, lookback_hours, sample_count, metadata.nav_snapshot}`.
- Contract tests: `tests/test_expectancy_v6.py`, `tests/test_expectancy_map.py` (verify numeric stability).

## Symbol scores
- Implementation: `execution/intel/symbol_score_v6.py:1-137`.
- Inputs: expectancy snapshot + router health snapshot.
- Output: `logs/state/symbol_scores_v6.json` with `symbols:[{symbol, score, components, inputs}]` and `updated_ts`.
- The executor caches `_LAST_SYMBOL_SCORES_STATE` and rewrites the file every `_maybe_publish_execution_intel()` call (`execution/executor_live.py:341-375`).
- Tests: `tests/test_symbol_score_v6.py`, `tests/test_execution_intel_sizing.py` ensure scaling and deterministic ordering.

## Router auto-tune
- Suggestion builder: `execution/intel/router_autotune_v6.py:1-200`.
  - Reads expectancy + symbol scores + router health, clamps proposals via `AutotuneBounds`, and records rationales plus lookback metadata.
  - Writes `logs/state/router_policy_suggestions_v6.json` via `write_router_policy_suggestions_state()` (`execution/state_publish.py:122-125`).
- Application: `execution/intel/router_autotune_apply_v6.py:1-155`.
  - Reads suggestions lazily, enforces env toggles, per-symbol allowlist, max bias delta, offset step/absolute bounds, quality filters, and risk-mode bans.
  - Called by `execution/order_router.py` whenever policy metadata is fetched to set maker/taker preferences.
- Tests: `tests/test_router_autotune_v6.py:1-60` (suggestion math) and `tests/test_router_autotune_apply_v6.py:1-80` (application safety) cover the full cycle.

## Feedback allocator v6
- Implementation: `execution/intel/feedback_allocator_v6.py:260-402`.
- Inputs: expectancy snapshot, symbol scores, router policy suggestions, nav snapshot, risk snapshot, `config/risk_limits.json`, `config/pairs_universe.json`.
- Output: `logs/state/risk_allocation_suggestions_v6.json` with:
  - `global` block containing `current_equity_usd`, `current_drawdown_pct`, `risk_mode` (normal/cautious/defensive) derived from drawdown thresholds.
  - `symbols` entries that bundle expectancy/router score context plus `caps` and `suggested_caps` for NAV/trade/concurrent limits and `suggested_weight` contributions.
- CLI: `scripts/feedback_allocator_probe_v6.py:1-70` prints the top rows and writes the same state file, ensuring manual probes match automation.
- Tests: `tests/test_feedback_allocator_v6.py:1-120` assert weighting, risk-mode scaling, and graceful handling of missing intel.

## Executor integration
- `_maybe_publish_execution_intel()` (`execution/executor_live.py:341-405`) orchestrates the entire intel refresh. It writes expectancy → router health → symbol scores → router suggestions → risk allocation suggestions in that order and stores `_LAST_SYMBOL_SCORES_STATE` for `synced_state` writes.
- Flags: `execution/v6_flags.py:18-59` exposes `INTEL_V6_ENABLED`, `ROUTER_AUTOTUNE_V6_ENABLED`, `FEEDBACK_ALLOCATOR_V6_ENABLED`, and `ROUTER_AUTOTUNE_V6_APPLY_ENABLED`. `_maybe_publish_execution_intel()` checks them before running heavy analysis, while `_maybe_write_v6_runtime_probe()` logs the snapshot for audits (`execution/executor_live.py:237-251`).

## State consumers
- Router policy apply (`execution/order_router.py`) pulls the latest `router_policy_suggestions_v6.json` via `execution/intel/router_autotune_apply_v6.py`.
- Risk allocator outputs are referenced manually today but kept under version control to validate future auto-cap tuning.
- Dashboard intel panels read `expectancy_v6.json`, `symbol_scores_v6.json`, and router suggestions directly via helper functions in `dashboard/live_helpers.py`.

This document covers the entire intel contract—no legacy v5 intel paths or dashboards remain. Any client that wants intel data should read the files enumerated above and expect the schemas described here.
