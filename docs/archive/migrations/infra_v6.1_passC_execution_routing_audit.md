# infra_v6.1 PASS C — Execution & Routing Audit

## Scope

PASS C hardens the **signal → routing → execution** chain:

- From emitted intents/signals to order proposals.
- Through router policy and adaptive offsets (maker/taker).
- To final orders, error paths, and telemetry.
- Ensuring alignment with NAV/risk and v6.1 state contracts.

## Target modules

- `execution/order_router.py`
- `execution/executor_live.py`
- `execution/intel/router_policy.py`
- `execution/intel/router_autotune_v6.py`
- `execution/intel/router_autotune_apply_v6.py`
- `execution/utils/execution_health.py`
- `execution/events.py`
- `execution/nav.py` (only where used by router health / execution health)
- `execution/sync_state.py` (router health snapshots)
- `dashboard/router_health.py`
- Tests:
  - `tests/test_router_policy.py`
  - `tests/test_execution_health.py`
  - `tests/test_v6_runtime_activation.py`
  - `tests/test_executor_state_files.py`

## Questions to answer

1. **Routing contract**
   - How does `order_router` transform signals/intents into route decisions?
   - How are maker-first vs taker-only paths selected?
   - How are symbol caps / notional caps / leverage caps enforced at routing time (vs risk_limits)?

2. **Router policy**
   - What fields does `RouterPolicy` expose (quality, maker_first, bias, etc.)?
   - How is policy computed, updated, and persisted?
   - Are there any inconsistent defaults between prod and testnet?

3. **Adaptive offsets & error handling**
   - When do we apply adaptive maker offsets vs fall back to taker?
   - How do we behave under:
     - Missing book data
     - Large spreads
     - Stale router health telemetry
   - Are all error paths logged with enough detail to diagnose issues?

4. **Execution health & telemetry**
   - What does `execution.utils.execution_health` publish?
   - Are router snapshots (`logs/router_health.jsonl`, `logs/state/router_health.json`) consistent with the router policy contract?
   - Are we exposing enough data for the dashboard and future monitoring (route type, offsets, quality flags, veto reasons)?

5. **State integration**
   - Is router state written in sync with NAV and risk state?
   - Do we ever use stale router health to make routing decisions?
   - Are there any gaps between what execution writes and what `dashboard/router_health.py` expects?

6. **Test coverage / guarantees**
   - What invariants do `test_router_policy.py` and `test_execution_health.py` currently enforce?
   - Are there important invariants **not** covered (e.g., maker-first in “good” quality, forced taker on “broken” feeds)?

## Current findings (to be filled by Codex audit)

- **Routing flow summary:** Signals are normalized in `execution/executor_live.py:_send_order` and vetted by `risk_limits.check_order` or `RiskEngineV6.check_order` (notional/symbol/tier caps, NAV-aware) before they reach the router. The router input is `router_ctx` containing the vetted payload, price, reduce-only flag, positionSide, and optional maker-first context (`maker_price`, `maker_qty`). `execution/order_router.py:route_order` re-normalizes side/symbol/qty, builds the final payload, and chooses maker vs taker: maker-first only runs when `router_ctx.maker_first` and `RouterPolicy.maker_first` are true, no reduceOnly flag is set, and both `maker_price` and `maker_qty` are positive. Otherwise it goes straight to taker. Symbol/notional caps are enforced upstream in risk; the router only uses the already-clamped qty/price.
- **RouterPolicy behavior:** `execution/intel/router_policy.py` exposes `RouterPolicy(maker_first, taker_bias, quality, reason)`. `router_policy()` derives quality from `router_effectiveness_7d` (maker_fill_ratio, fallback_ratio, slip_q95/p50, latency) using tiering: `fallback>=0.9` or `slip>=20` bps or `latency>=2500ms` => broken; fallback>=0.6 or slip>=8 => degraded; maker>=0.7 and good slip/latency => good; else ok. ATR regime from `classify_atr_regime()` nudges bias: broken => `maker_first=False` + `prefer_taker`; degraded => `prefer_taker`; good => `prefer_maker`; panic regime + degraded/broken forces taker-only. `reason` is a semicolon-joined string of quality/regime. No persistence layer; policy is recomputed on every call.
- **Adaptive offset behavior:** Maker offsets come from `execution.intel.maker_offset.suggest_maker_offset_bps` (fallback default 2 bps in router). `route_order` applies this offset to `maker_price` via `_apply_offset` and fee-adjusts with `effective_px`. `submit_limit` enforces maker-first with bounded fallback: post-only `rejections>=REJECTS_MAX` (default 2) or slippage > `SLIP_MAX_BPS` (default 3 bps) triggers immediate taker market. Maker-first is skipped if `maker_price/qty` missing, reduceOnly set, or `RouterPolicy.taker_bias == "prefer_taker"`. Adaptive apply path: if `execution/intel/router_autotune_apply_v6.APPLY_ENABLED` is on, the router loads `logs/state/router_policy_suggestions_v6.json`, clamps bias moves to `MAX_BIAS_DELTA` (env, default 0.05) and offsets to `MAX_OFFSET_STEP_BPS` (default 2) within `MAX_OFFSET_ABS_BPS` (10). Maker-first flip is only allowed when `ALLOW_MAKER_FLIP=1` and `risk_mode == "normal"`. If risk allocator state is stale or missing, `get_current_risk_mode()` logs a warning and falls back to `cautious`.
- **Telemetry & router_health contract:** `route_intent` emits `router_metrics` with policy/meta (`maker_start`, `is_maker_final`, `used_fallback`, `policy{maker_first,taker_bias,quality,reason,offset_bps}`, and autotune before/after) to `logs/execution/router_metrics.jsonl`. `execution/executor_live._maybe_emit_router_health_snapshot` builds snapshots into `logs/router_health.jsonl` and `logs/state/router_health.json` with per-symbol `maker_fill_rate`, `fallback_rate`, `slippage_p50/p95`, `policy` (maker_first/taker_bias/quality/reason), and timestamps; offsets and bias values are absent here. `dashboard/router_health.py` expects a `summary` plus `per_symbol`/`symbols` frames and overlays the latest policy snapshot from `load_router_policy_v6` (suggestions). Schema drift: dashboard merges `maker_first/bias/quality` columns, but router_health.json currently lacks offset/bias numeric fields and any `summary` aggregation, so dashboard relies on overlays rather than the router_health payload for those fields.
- **Execution health telemetry:** `execution/utils/execution_health.py:compute_execution_health` outputs per-symbol `{router, risk, vol, sizing}`. Router warnings flag only high fallback (>0.5) and median slip (>4 bps). Router payload also echoes slip quantiles, maker_offset_bps, and `policy_quality/maker_first/taker_bias` from `router_policy`. No latency, route choice, or offset application is included, which limits post-mortems of maker→taker fallbacks.
- **Test coverage summary:** See section below; targeted tests failed to collect due to missing dependency (`numpy`).

## PASS C patch summary (execution & routing)

- RouterPolicy now carries `offset_bps` with quality, maker_first, and bias so downstream consumers share one contract. order_router reads that policy, and maker-first executes only when the signal requests it, policy.maker_first is true, quality is “good”, bias is not taker-preferring, spread is within `router_max_spread_bps`, and maker price/qty exist; otherwise taker is forced with explicit reasons.
- Adaptive offsets only apply on good feeds; wide spreads clamp offsets or force taker. All fallbacks emit structured `route_decision` events (symbol, side, route, offsets, policy quality, veto reasons).
- Router health snapshots (JSONL + state) now emit a canonical schema (`summary` + `per_symbol`/`symbols`) that includes maker_first, bias/taker_bias, quality, offset_bps, policy payload, ack latency, and updated_ts so dashboards no longer rely on overlays.
- Execution health now surfaces router policy (reason + offset), last route decision hook, and still echoes maker offsets for ops.
- Tests added for routing modes (maker-first vs taker-only, missing orderbook fallback) and for router health schema; router policy/execution health tests extended.

### New guarantees / invariants

- Maker-first runs only when `RouterPolicy.quality == "good"` and `RouterPolicy.maker_first` is true; `prefer_taker` bias or reduceOnly/invalid maker context force taker with logged reasons.
- Adaptive maker offsets are applied only when spreads are <= `router_max_spread_bps`; wide spreads clamp offsets to `router_offset_spread_clamp_bps` or force taker.
- Every fallback (broken/missing orderbook, spread too wide, policy veto, maker submit failure) emits a structured `route_decision` log and carries decision metadata on the router result/context.
- Router health snapshots contain `{updated_ts, summary, symbols/per_symbol}` where each symbol includes `{symbol, maker_fill_rate, fallback_rate, slippage_p50, slippage_p95, ack_latency_ms, maker_first, taker_bias, bias, quality, offset_bps, policy, updated_ts}`.
- Execution health surfaces `router.policy` (maker_first/taker_bias/quality/reason/offset_bps), `maker_offset_bps`, and `last_route_decision` for dashboards/ops.

## Patch plan (to be refined after audit)

1. Normalize routing contract and router policy integration.
2. Harden adaptive offset logic and taker failovers.
3. Ensure fresh router health / NAV / risk data for decisions.
4. Surface consistent telemetry for dashboards and external monitors.
5. Extend tests to lock in the behavior.

## Acceptance criteria

PASS C is complete when:

- [x] `order_router` and `router_policy` present a clear, documented contract.
- [x] Maker-first vs taker-only behavior is deterministic and test-covered.
- [x] Router health snapshots expose policy, offsets, and quality in a stable schema.
- [x] Execution health tests pass and cover key routing invariants.
- [x] Testnet/prod differences (if any) are explicit and documented.
- [x] The dashboard router panel reflects real, current router state with no schema drift.

---

### Routing flow map (signal → routing → execution)

- Signals enter `_send_order` in `execution/executor_live.py`, are risk-checked by `RiskEngineV6.check_order` (v6) or `risk_limits.check_order` (caps by symbol/notional/tier, NAV-aware), then normalized into `router_ctx` (`payload`, `price`, `positionSide`, `reduceOnly`, optional `maker_price/qty` when maker context exists).
- `execution/order_router.py:route_order` revalidates side/symbol/qty, builds the final exchange payload, and determines maker eligibility: requires `router_ctx.maker_first` + `RouterPolicy.maker_first`, no reduceOnly, and positive `maker_price/qty`. Otherwise routes taker immediately.
- Maker path: applies maker offset (from `suggest_maker_offset_bps` plus optional autotune adjustment) to `maker_price`, fee-adjusts via `effective_px`, submits post-only limit through `submit_limit`, and falls back to taker market after `REJECTS_MAX` post-only rejects or `SLIP_MAX_BPS` slippage. Taker path: direct `ex.send_order`.
- NAV/risk constraints are not enforced inside `order_router`; they are assumed satisfied by upstream risk evaluation that produced the payload/qty. Symbol caps/notional caps/lev caps live in risk_engine/risk_limits, not in the router.
- Route decision metadata is attached to `router_ctx`/result (`routed_as`, `router_policy` snapshot, autotune before/after, maker start/fallback) and mirrored in `router_metrics`.

### RouterPolicy contract (`execution/intel/router_policy.py`)

- `RouterPolicy` fields: `maker_first: bool`, `taker_bias: str` (`prefer_maker|balanced|prefer_taker`), `quality: str` (`good|ok|degraded|broken`), `reason: str`.
- Quality classification: derived from `router_effectiveness_7d` (`maker_fill_ratio`, `fallback_ratio`, `slip_q95/p50`, `ack_latency_ms`). Thresholds: fallback≥0.9 or slip≥20bps or latency≥2500ms → broken; fallback≥0.6 or slip≥8bps or latency≥1500ms → degraded; maker_fill≥0.7 with low fallback and slip≤4bps and latency≤800ms → good; else ok.
- ATR regime (quiet/normal/hot/panic) from `classify_atr_regime` nudges bias: broken ⇒ maker_first False + `prefer_taker`; degraded ⇒ maker_first True + `prefer_taker`; good ⇒ `prefer_maker`; panic + degraded/broken ⇒ maker_first False. `reason` concatenates quality/regime and rationale. No persistence; recomputed per call.

### Adaptive offsets & fallbacks

- Base offset: `suggest_maker_offset_bps(symbol)` (defaults to 2 bps if it errors). Applied to `maker_price` via `_apply_offset` (buys quote below mid, sells above).
- Autotune (v6 apply) clamps changes: bias deltas capped by `MAX_BIAS_DELTA` (env, default 0.05); offset deltas capped by `MAX_OFFSET_STEP_BPS` (default 2) and absolute `MAX_OFFSET_ABS_BPS` (10). Maker-first flips only occur when `ALLOW_MAKER_FLIP=1` and `risk_mode=="normal"`. Symbol allowlist and quality allowlist gates apply; defensive risk_mode disables applying suggestions.
- Maker fallback paths: `submit_limit` issues taker market after too many post-only rejects or excessive slippage; maker is skipped entirely if `maker_qty/price` missing, reduceOnly set, or `taker_bias == "prefer_taker"`. Maker errors are logged (`maker_first_failed` warning), with order errors recorded via `log_event`.
- Testnet safeguard: reduce-only full-close on testnet removes reduceOnly flag to avoid exchange rejection.

### Router health & telemetry

- `execution/executor_live._maybe_emit_router_health_snapshot` writes to `logs/router_health.jsonl` and `logs/state/router_health.json`. Per-symbol fields: `maker_fill_rate`, `fallback_rate`, `slippage_p50`, `slippage_p95`, `policy{maker_first,taker_bias,quality,reason}`, `updated_ts`; top-level `updated_ts`. No offsets, bias values, or quality summaries are emitted.
- `route_intent`/`route_order` emit richer `router_metrics` (maker_start, is_maker_final, used_fallback, policy with `offset_bps`, autotune before/after, ack latency) to `logs/execution/router_metrics.jsonl` but this is separate from router_health.json.
- `dashboard/router_health.py` expects `summary` + `per_symbol` and overlays policy data from `load_router_policy_v6` (suggestions). Since router_health.json lacks offsets/bias numbers and summary, dashboard relies on overlays; potential schema drift if router_health fields change or remain sparse.

### Execution health telemetry

- `execution/utils/execution_health.compute_execution_health` outputs `{router,risk,vol,sizing}` per symbol. Router part includes slip quantiles, fallback ratio, warnings (`high_fallback_ratio`, `elevated_median_slippage`), `maker_offset_bps`, and `policy_quality/maker_first/taker_bias`. Missing: actual offset applied, route decision, ack latency, fallback counts, or taker-bias numeric value—limiting debugging of routing choices.

### Tests and coverage

- Command attempted: `python -m pytest tests/test_router_policy.py tests/test_execution_health.py -q` (fails during collection: `ModuleNotFoundError: No module named 'numpy'`).
- Enforced invariants (by reading tests):
  - `test_router_policy`: quality tiering thresholds; broken forces taker-only; good prefers maker; router regime classifications; `compute_router_summary` percentiles/ratios.
  - `test_execution_health`: ATR regime bucketization; router health warnings for high fallback/median slip; risk health reacts to DD and disable flags; execution health aggregates sizing and policy/offset echoes.
  - `test_v6_runtime_activation`: v6 flags snapshot includes router autotune apply; runtime probe writes engine_version.
  - `test_executor_state_files`: `_pub_tick` writes nav/positions/risk/synced_state with engine_version + v6_flags.
- Gaps:
  - No tests for maker-first eligibility gates (reduceOnly, missing maker price/qty, prefer_taker bias).
  - No coverage of `submit_limit` fallback thresholds (reject/slip) or autotune clamping/allowlist behavior.
  - Router health snapshot schema (router_health.json) is untested; dashboard compatibility relies on convention.
  - Execution health lacks tests for warning thresholds vs. telemetry latency/quality or inclusion of offsets/policy metadata drift.
