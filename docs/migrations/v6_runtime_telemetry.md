# Migration: v6 Runtime Telemetry Contract

## Overview

The v6 beta runtime now writes a guaranteed set of state files on every executor loop and exposes Firestore-friendly sync services. Operators upgrading from v5.10 must ensure the following directories are writable (`repo/logs/*`).

## State Writers

| File | Producer | Notes |
| --- | --- | --- |
| `logs/state/nav.json` | `execution.executor_live._pub_tick()` | Contains `{"nav": float, "nav_usd": float, "updated_ts": epoch}` every loop, even when flat. |
| `logs/state/positions.json` | executor | `{"rows": [...], "updated": epoch}` derived from the latest exchange snapshot. |
| `logs/synced_state.json` | executor | Superset of positions used by `execution/sync_state.py` to compute exposures. |
| `logs/state/router_health.json` | executor | Emitted on its own timer; no longer tied to live orders. |
| `logs/state/risk_snapshot.json` | executor | Includes diagnostics from risk-engine v6 or the fallback risk health collector. |
| `logs/state/pipeline_v6_shadow_head.json` | executor | Always includes `last_decision` (the most recent shadow result or heartbeat). |
| `logs/state/pipeline_v6_compare_summary.json` | executor + `scripts/pipeline_compare_service.py` | Refreshed every interval so dashboards can render comparison deltas. |

All telemetry JSONL files (`logs/execution/*`) now use rotating `JsonlLogger`s, preventing unbounded growth.

## Services

1. **Executor** – runs `execution.executor_live` with v6 flag logging + runtime probe writers.
2. **Dashboard** – unchanged Streamlit entrypoint.
3. **Sync State** – `execution/sync_state.py` now respects `ALLOW_PROD_WRITE`/`ALLOW_PROD_SYNC` and pushes Firestore snapshots using the new `execution.firestore_utils` client.
4. **Pipeline Shadow Heartbeat** – `scripts/pipeline_shadow_heartbeat.py` keeps the pipeline head summary fresh even when no trades occur.
5. **Pipeline Compare Service** – `scripts/pipeline_compare_service.py` continuously runs the v6 compare routine.

`ops/hedge.conf` ships all five supervisor programs with log rotation and prod-safe env guards (`ALLOW_PROD_WRITE=1`, `ALLOW_PROD_SYNC=1`).

## Firestore Flags

Set `ALLOW_PROD_WRITE=1` (and `ALLOW_PROD_SYNC=1` for `sync_state`) in `.env` to permit remote publishing. Leaving these unset keeps Firestore writes disabled while local JSON state continues to update.
