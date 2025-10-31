# Quant Infrastructure Audit — Hedge Fund Stack

## Executive Summary
- The NAV pipeline still depends on ad-hoc writers: `_futures_nav_usdt` only refreshes the confirmed snapshot when both balances and positions return cleanly, yet the only periodic writer is an optional helper (`run_nav_writer`) that nothing schedules by default, while `sync_daemon` refuses to run if `nav_log.json` ages past its freshness threshold.【F:execution/nav.py†L29-L83】【F:execution/nav.py†L390-L448】【F:execution/sync_daemon.py†L221-L288】
- Daily loss and drawdown guardrails draw exclusively from `logs/cache/peak_state.json`; the loader just logs staleness and keeps using whatever is on disk, so the veto logic can compare against days-old peaks or realized PnL without any hard fail-safe.【F:execution/drawdown_tracker.py†L50-L127】【F:execution/risk_limits.py†L492-L538】【F:execution/risk_limits.py†L799-L857】
- Firestore publishing lacks strong environment safety: the CLI tools default to the production collection path and spin up fresh clients per call, while `execution/hedge_sync.py` can continuously overwrite NAV/position documents with placeholder zeroes if someone runs it locally.【F:execution/state_publish.py†L27-L35】【F:execution/state_publish.py†L300-L343】【F:execution/hedge_sync.py†L13-L27】

## Detailed Findings

### 1. NAV freshness hinges on manual caretaking
**Observation.** `_futures_nav_usdt` only calls `_persist_confirmed_nav` when both balance and position RPCs succeed, otherwise it logs a warning and relies on `_mark_nav_unhealthy` to flag the cache stale.【F:execution/nav.py†L29-L83】 Keeping `logs/nav_log.json` current is delegated to `run_nav_writer`, a helper loop that must be invoked separately, but nothing in the codebase actually spawns it; the only consumer is the executor’s opportunistic `write_nav_snapshots_pair` call. If that writer isn’t running, `sync_daemon.run_once` raises as soon as the nav log exceeds `SYNC_NAV_MAX_AGE_SEC`, stalling every Firestore sync cycle.【F:execution/nav.py†L390-L448】【F:execution/sync_daemon.py†L221-L288】

**Recommendation.** Promote `run_nav_writer` (or an equivalent scheduler inside the executor) into a supervised process so the nav cache stays fresh even when trading pauses. Additionally, persist the last good NAV alongside explicit source health and make the sync daemon degrade gracefully (e.g., emit heartbeats with `stale=true`) instead of throwing and halting writes outright.

### 2. Drawdown and daily loss stops trust a stale file
**Observation.** Both `load_peak_state` and `compute_intraday_drawdown` rebuild portfolio peaks from a single JSON file, merely defaulting to previous values when inputs are missing.【F:execution/drawdown_tracker.py†L50-L127】 `_drawdown_snapshot` surfaces `peak_state_age` but only logs when it exceeds `PEAK_STATE_MAX_AGE_SEC`; the function still returns the stale numbers, and the veto path compares that drawdown against `daily_loss_limit_pct` and `max_nav_drawdown_pct` without additional freshness checks.【F:execution/risk_limits.py†L492-L538】【F:execution/risk_limits.py†L799-L857】 If the sync process stops updating `peak_state.json`, the risk gate will happily enforce limits against a stale baseline or, worse, a zeroed file.

**Recommendation.** Treat an out-of-date `peak_state` as a hard failure: stop trading, surface an alert, and fall back to exchange-reported realized PnL or NAV history to rebuild the peak on the fly. Persist the computed drawdown state to a durable store (e.g., Firestore) and have the risk layer demand freshness before evaluating loss thresholds.

### 3. Firestore tooling can clobber production telemetry
**Observation.** `execution/state_publish.py` derives `FS_ROOT` from `ENV` at import time with a default of `prod`, and every publish call instantiates a new Firestore client pointed at that path.【F:execution/state_publish.py†L27-L35】【F:execution/state_publish.py†L300-L343】 Running the script locally without overriding `ENV` or `FIRESTORE_ENABLED` will push whatever the terminal sees straight into production collections. Separately, `execution/hedge_sync.py` loops forever writing empty nav series and empty position lists via the shared sync helpers, so a misplaced run can wipe telemetry in whichever environment the credentials target.【F:execution/hedge_sync.py†L13-L27】

**Recommendation.** Centralise Firestore access through `utils.firestore_client.get_db()` so publishers reuse clients, enforce retries, and honour an explicit environment parameter. Ship the CLI tools with `ENV=dev` defaults (and guard rails that refuse to hit prod unless an allow-list flag is set), and either delete `hedge_sync.py` or make it a dry-run utility that cannot reach production collections without explicit confirmation.

