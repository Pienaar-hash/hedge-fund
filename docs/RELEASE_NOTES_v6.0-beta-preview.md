# v6.0-beta Preview Release Notes

## Runtime & Telemetry

- Centralized all v6 env flags inside `execution/v6_flags.py` with shared logging helpers.
- Executor now writes nav/positions/synced-state snapshots every loop, emits router health + risk snapshots on timers, and runs intel/shadow/compare flows even when no orders are emitted.
- Added pipeline v6 heartbeat + comparison schedulers as well as rotating JSONL loggers for signal/order/router metrics and pipeline shadow logs.

## Services & Sync

- Re-enabled `execution/sync_state.py` Firestore guards, reading `logs/synced_state.json` and respecting `ALLOW_PROD_SYNC`.
- Added supervisor entries for executor, dashboard, sync_state, pipeline shadow heartbeat, and pipeline comparison, each with log rotation.
- Dashboard supervisor entry now runs `streamlit` through `/root/hedge-fund/venv/bin/python3 -m streamlit run dashboard/app.py` so the UI boots cleanly under Supervisor.
- Removed legacy v5 modules (`execution/hedge_sync.py`, `execution/firestore_mirror.py`, `scripts/fetch_synced_state.py`, and the repo-level `archive/` directory).

## Tooling & Docs

- New CLI services under `scripts/` keep pipeline shadow head summaries and pipeline comparisons fresh when running under Supervisor.
- Added migration notes covering the new telemetry contract and state file schema.
