# v6.0-beta Preview — Runtime, Telemetry, and Service Wiring Audit

Audit date: _latest repo state in `/root/hedge-fund`_.

This pass followed the scope in `infra_v6.0_beta_preview_full_runtime_audit.prompt.txt`, tracing the running executor, supporting services, telemetry writers, dashboards, and legacy leftovers. Findings are organized by objective with file+line references and patch recommendations.

---

## Runtime Wiring Map

| Component | Entry point(s) | Notes |
| --- | --- | --- |
| Executor | `execution/executor_live.py:3288-3336` | `_pub_tick()` runs once per loop and only updates `logs/positions.json`, `logs/nav_log.json`, and `logs/spot_state.json` (no `logs/state/*`). The runtime still stubs `publish_nav_value()` and other Firestore writers to `pass` (`execution/executor_live.py:456-477`). |
| Screener | `execution/signal_screener.py:1-520` | Duplicates `RISK_ENGINE_V6_ENABLED` env parsing at import (`execution/signal_screener.py:202-205`), so flag drift is possible whenever env vars change after interpreter start. |
| V6 flags | `execution/executor_live.py:85-111` | Spec references `_log_v6_flag_snapshot()`, but no such helper exists. Startup simply logs once inline (`execution/executor_live.py:3288-3294`), so other services cannot reuse the logging contract. |
| Intel refresh | `_maybe_publish_execution_intel()` is only called for symbols that currently have positions (`execution/executor_live.py:3195-3201`). Flat books skip the intel refresh entirely, so telemetry never emits during zero-trade windows. |
| Pipeline shadow | `_maybe_run_pipeline_v6_shadow()` is invoked only inside `_send_order()` when an order is about to be transmitted (`execution/executor_live.py:2195-2211`). With no intents emitted (weekends, risk blocks, or dry markets) the shadow pipeline never runs. |

**Patch goals**

1. Add a shared `_log_v6_flag_snapshot()` (or module such as `execution/v6_flags.py`) and wire every service (executor, screener, sync daemon) to call it during startup so logs+telemetry consistently prove flag states.
2. Centralize flag parsing so `RISK_ENGINE_V6_ENABLED`, `ROUTER_AUTOTUNE_V6_APPLY_ENABLED`, etc., are read exactly once and imported elsewhere, instead of re-parsed globals in modules like `execution/signal_screener.py:202-205` and `execution/intel/router_autotune_apply_v6.py:11-51`.
3. Trigger `_maybe_publish_execution_intel()` per-loop regardless of open positions, with its own interval guard, so telemetry runs during idle markets.

---

## Telemetry Schema Map (current vs. required)

| State / Log | Expected behavior | Observed source | Gap |
| --- | --- | --- | --- |
| `logs/state/nav.json`, `logs/state/positions.json` | Continuously written even when empty. | Not written at all: executor redefines `publish_nav_value()` to `pass` (`execution/executor_live.py:456-477`), and no caller invokes `execution/state_publish.publish_positions()`. | Dashboards and syncers see stale/missing nav + position data during the entire run. |
| `logs/state/router_health.json` | Periodic snapshots even with zero orders. | Only written inside `_mirror_router_metrics()` (`execution/executor_live.py:242-257`), which is only called when an order ACK arrives (`execution/executor_live.py:2780-2833`). | Router health never exists on idle days, so dashboards render empty tables. |
| `logs/state/risk_snapshot.json` | Rolling execution health snapshot. | `_maybe_publish_execution_health()` is only called from the same order ACK path (`execution/executor_live.py:2828-2829`). | Risk telemetry is missing until the first live order. |
| `logs/state/expectancy_v6.json` | Should exist even with zero trades. | Writer is gated by `if expectancy_snapshot.get("symbols")` (`execution/executor_live.py:352-355`), so empty snapshots skip the file entirely. | No proof that INTEL_v6 is alive in flat markets. |
| `logs/state/router_policy_suggestions_v6.json`, `risk_allocation_suggestions_v6.json` | Interval-based, even if intel data is empty. | Same intel loop; because it never runs when `active_symbols` is empty (`execution/executor_live.py:3195-3201`), both files stay stale. |
| `logs/state/pipeline_v6_shadow_head.json` | Mirror of latest shadow run plus counters. | Populated only when `_maybe_run_pipeline_v6_shadow()` executes, and the payload contains only aggregated counts (`execution/pipeline_v6_shadow.py:172-181`). | No head data + no writes when there are no live orders. |
| `logs/state/pipeline_v6_compare_summary.json` | Aggregated diff of v5 vs v6. | Only produced by the manual CLI probe (`scripts/pipeline_v6_compare_probe.py`) and tests; nothing calls `compare_pipeline_v6()` in production (`execution/intel/pipeline_v6_compare.py:1-118`). | Dashboard tables in `dashboard/live_helpers.py:37-125` have no backing state. |
| `logs/state/router_health.json` → dashboard | `dashboard/router_health.py:17-59` expects the file and silently returns `None` when missing. | Because the file never appears without orders, the router dashboard remains blank. |
| JSONL logs (`orders_executed.jsonl`, `router_metrics.jsonl`, `pipeline_v6_shadow.jsonl`) | Rotation / bounded size. | Most writers use `append_jsonl()` (`execution/log_utils.py:64-88`), which has no rotation; only `get_logger()` rotates. | JSONL files grow unbounded in long-running sessions. |

**Patch goals**

1. Have `_pub_tick()` call `write_nav_state()`, `write_positions_state()`, and emit `logs/synced_state.json` every loop. Keep the local caches for backwards compatibility, but ensure `logs/state/*` always exists.
2. Add background timers (e.g., `_maybe_emit_router_health_snapshot()` and `_maybe_emit_risk_snapshot()`) that run from `_loop_once()` so idle periods still produce files.
3. Always persist intel snapshots, even when empty: remove the truthiness gate around `expectancy_snapshot.get("symbols")` and write placeholder payloads with `sample_count=0`.
4. Schedule `execution.intel.pipeline_v6_compare.compare_pipeline_v6()` off the executor loop (e.g., reuse the intel timer) so dashboards receive `pipeline_v6_compare_summary.json`.
5. Swap JSONL writers that must rotate (router metrics, order attempts, pipeline shadow log) to use `execution.log_utils.get_logger()` instead of `append_jsonl()`, or add a rotation helper for those files.

---

## Intel v6 Flow Map

Flow: `_loop_once()` → `_maybe_publish_execution_intel(symbol)` per active symbol → expectancy/scores/autotune/allocator writers.

Observations:

* `active_symbols` derives solely from current open positions (`execution/executor_live.py:3188-3201`). If the account is flat, `_maybe_publish_execution_intel` is never invoked, so `expectancy_v6.json`, `symbol_scores_v6.json`, router autotune suggestions, feedback allocator suggestions, and `universe.json` never refresh.
* Autotune suggestions are only written when `ROUTER_AUTOTUNE_V6_ENABLED` is on and the intel loop executes. `order_router` then reads `get_symbol_suggestion()` from `execution/intel/router_autotune_apply_v6.py`, which re-parses `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` at import (`execution/intel/router_autotune_apply_v6.py:11-51`). If env vars are flipped at runtime, executor and order router can disagree.

**Patch goals**

1. Move `_maybe_publish_execution_intel()` outside the `active_symbols` loop and guard it with its own timestamp, ensuring idle runs still refresh intel.
2. Persist intel snapshots even when they are empty (see telemetry table).
3. Extract a single `execution/v6_flags.py` that exposes both parsed booleans and a helper to log+snapshot them, so screener, executor, and router autotune apply run off the same data.

---

## Shadow Pipeline & Pipeline Comparison

* `_maybe_run_pipeline_v6_shadow()` is only entered when `_send_order()` would otherwise submit a live order (`execution/executor_live.py:2195-2211`). That means the shadow pipeline never runs if:
  * The screener queue is empty (weekends).
  * Risk vetoes occur earlier in `_send_order()` and the live order is never attempted.
  * Operators run diagnostics with no market flow.
* Even when it runs, the state writer stores only counts (`execution/pipeline_v6_shadow.py:172-181`). No “head” sample or metadata is persisted, so dashboards cannot show the latest proposal vs. live order.
* `pipeline_v6_compare` is never scheduled. The only runtime entrypoints are the CLI probe (`scripts/pipeline_v6_compare_probe.py:1-33`) and tests.

**Patch goals**

1. Add a periodic “shadow heartbeat” that samples a placeholder signal (e.g., from the current universe) so `pipeline_v6_shadow_head.json` and `pipeline_v6_shadow.jsonl` are refreshed even if no orders were sent.
2. Extend `pipeline_v6_shadow_head.json` to include the most recent decision payload (intent, risk result, size, router info), not just counts.
3. Trigger `compare_pipeline_v6()` from the executor or a cron task so `logs/state/pipeline_v6_compare_summary.json` always exists.

---

## Dashboard Ingestion Compatibility

* `dashboard/router_health.py:17-59` and `dashboard/live_helpers.py:37-125` expect the nav/router-health/pipeline state files to be present. Because neither `router_health.json` nor `pipeline_v6_compare_summary.json` is produced automatically, those dashboards always render empty frames in the current runtime.
* Router autotune application relies on `logs/state/router_policy_suggestions_v6.json`. As noted above, that file only updates when there are open positions and intel writes succeed. The dashboard pages that visualize router intel therefore stay stale for hours.

**Patch goals**

1. Produce the missing state files (telemetry section).
2. Add simple guards in `dashboard/live_helpers.py` to surface explicit “state missing” warnings so operators notice when telemetry is absent.

---

## Sync-State, Telemetry Pushers, and Supervisor Wiring

* `execution/sync_state.py` is hard-disabled for Firestore (`_firestore_enabled()` returns `False` and every publisher short-circuits at `execution/sync_state.py:37-76`). Even if the daemon is launched, it will never push nav/positions remotely.
* The daemon expects `logs/synced_state.json` to exist (“executor populates synced_state.json” comment at `execution/sync_state.py:1170-1178`), but the executor only writes `logs/positions.json` (`execution/executor_live.py:2990-3080`). `sync_state` therefore reads an empty payload and cannot compute exposures.
* Supervisor config (`ops/hedge.conf:4-29`) launches only `hedge-executor` and `hedge-dashboard`. There is no `sync_state` or `sync_daemon` program, and stdout/stderr logs have no rotation settings.
* `ops/hedge.conf:15` sets `ENV="prod"` without `ALLOW_PROD_WRITE`, so `execution/executor_live` raises at startup (`execution/executor_live.py:456-464`) and the Supervisor service will crash-loop until operators modify the env by hand.

**Patch goals**

1. Make the executor write `logs/synced_state.json` on each `_pub_tick()` (include `{"positions": rows, "updated_at": ts}`) so both `sync_state` and `sync_daemon` have canonical inputs.
2. Enable Firestore publishing in `execution/sync_state.py` (respect `ALLOW_PROD_WRITE` / `ALLOW_PROD_SYNC`) instead of hard-returning `False`.
3. Add Supervisor configs for `sync_state` or `sync_daemon`, include `stdout_logfile_maxbytes`/`stderr_logfile_maxbytes`, and pipe `ALLOW_PROD_WRITE=1` (or relax the guard) so the executor can actually run under Supervisor.

---

## Legacy / Stale Modules (v5.10 leftovers)

* `execution/hedge_sync.py` is an empty file that documents its own deprecation but still ships in the repo.
* `execution/firestore_mirror.py` references the old direct Google Cloud client even though all call sites were removed; the executor now uses `execution/firestore_utils` (which itself is stubbed to return `None`). Keeping both modules confuses new contributors.
* `scripts/fetch_synced_state.py` (and several files under `archive/`) refer to Firestore collections that no longer exist (`hedge_fund/synced_state`), contributing to operator confusion.

**Patch goals**

1. Remove or clearly quarantine the v5.10-only modules, and document the new sync stack (state_publish → synced_state → sync_daemon).
2. Re-enable `execution/firestore_utils` with real Firestore writes (or wrap them behind a feature flag) so dashboards can read remote telemetry again.

---

## Recommended Patch Set (Batch 12)

1. **Flag plumbing & logging**
   * Introduce a shared `execution/v6_flags.py` (or similar), ensure `_log_v6_flag_snapshot()` exists, and refactor executor, screener, router autotune apply, and future services to import from it. Emit flag snapshots to both logs and the runtime probe on startup.
2. **Telemetry heartbeat**
   * Update `_pub_tick()` to write nav/positions/synced-state via `execution.state_publish` and ensure state files exist within the first loop.
   * Add periodic router-health and risk-snapshot writers that do not depend on live orders, and switch JSONL writers to rotating loggers.
   * Adjust `_maybe_publish_execution_intel()` to run once per interval regardless of open positions, and always materialize the intel JSON files even when empty.
3. **Shadow pipeline resiliency**
   * Add a heartbeat job that runs `pipeline_v6_shadow` even when there are no live orders, persist the latest decision payload to `pipeline_v6_shadow_head.json`, and schedule `pipeline_v6_compare` so comparison telemetry exists.
4. **sync_state / supervisor fixes**
   * Have the executor maintain `logs/synced_state.json`.
   * Re-enable Firestore publishing in `execution/sync_state.py`, and either wire `sync_state` or `sync_daemon` into Supervisor with rotated logs.
   * Feed `ALLOW_PROD_WRITE` (or relax the guard) via `ops/hedge.conf` so `ENV=prod` does not crash the executor.
5. **Cleanup**
   * Remove/import-quarantine `execution/hedge_sync.py`, `execution/firestore_mirror.py`, and stale scripts referencing deprecated telemetry paths, replacing them with documentation for the v6 telemetry pipeline.

These items are carried forward to `Batch_12_RuntimeBetaFixes.prompt.txt`.
