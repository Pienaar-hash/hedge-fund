# Sprint 5.8 Audit — Portfolio Equity Analytics + Realtime Alerts

## dashboard/app.py
- **Problem:** When Firestore NAV data is absent the dashboard calls `_to_optional_float(nav_value)` before `nav_value` is defined, triggering an `UnboundLocalError` and breaking the page in low-data environments (`dashboard/app.py:1110`, `dashboard/app.py:1117`, `dashboard/app.py:1224`).  
  **Recommendation:** Seed `nav_value` (e.g., to `None`) before the fallback branch or guard the fallback with a flag so the dashboard can render even when NAV docs are missing.
- **Problem:** `total_equity_zar` is calculated before `zar_rate` is resolved, so it always remains `None` and the ZAR delta never appears in the UI, even when a fresh rate exists (`dashboard/app.py:1127`, `dashboard/app.py:1273`, `dashboard/app.py:1535`).  
  **Recommendation:** Move the USD→ZAR conversion after the rate resolution (or recompute once the rate is known) so the Total Equity metric can surface the ZAR view.

## dashboard/dashboard_utils.py
- **Problem:** Telemetry readers only look at `telemetry/health` and `health/executor_live`, so any heartbeat written to `telemetry/heartbeats` is invisible to the dashboard (`dashboard/dashboard_utils.py:225`, `dashboard/dashboard_utils.py:233`).  
  **Recommendation:** Either add a read from `telemetry/heartbeats` here or standardise publishers on the `telemetry/health` document so readership stays aligned.

## scripts/doctor.py
- **Problem:** `_collect_positions` invokes `get_live_positions(None)`, but the helper expects a Binance client and returns an empty list when `None` is supplied; the Doctor tab therefore always shows zero open positions regardless of reality (`scripts/doctor.py:312`, `execution/utils.py:360`).  
  **Recommendation:** Instantiate the futures client (or fall back to cached position_state logs) before counting positions so Doctor reflects live exposure.
- **Problem:** Reserves are derived solely from `RESERVE_BTC`/`RESERVE_XAUT` env vars and CoinGecko prices, ignoring the canonical `logs/treasury.json` / Firestore treasury snapshot (`scripts/doctor.py:321`). This causes the Doctor reserves total to drift from the dashboard/Nav totals.  
  **Recommendation:** Source reserves from the same treasury snapshot (or fall back to it when env overrides are unset) to keep NAV + reserves views consistent.
- **Problem:** The ZAR freshness badge reports `"fresh"` whenever `get_usd_to_zar(force=True)` returns a number, but that helper silently falls back to the cached rate on network failure, so stale data is labelled fresh (`scripts/doctor.py:714`, `execution/utils.py:502`).  
  **Recommendation:** Detect when the forced fetch hit the network vs the cache (e.g., return metadata or compare `_USD_ZAR_TS`) and downgrade the source label when serving cached data.
- **Problem:** Heartbeat aggregation reads Firestore `telemetry/health` and local logs but never inspects `telemetry/heartbeats`, so any publisher writing there is ignored (`scripts/doctor.py:540`).  
  **Recommendation:** Extend the heartbeat reader to ingest that document (or consolidate publishers onto the existing `telemetry/health` path) so Doctor surface matches what sync_state emits.

## execution/firestore_utils.py
- **Problem:** `publish_heartbeat` writes to `hedge/{env}/telemetry/heartbeats`, yet every consumer reviewed (dashboard + Doctor) only watches `telemetry/health`, effectively discarding those writes (`execution/firestore_utils.py:511`, `dashboard/dashboard_utils.py:225`).  
  **Recommendation:** Retarget heartbeats to `telemetry/health` (or update consumers to read `telemetry/heartbeats`) to restore heartbeat visibility.
- **Problem:** `publish_positions` stores data at `hedge/{env}/positions/latest` with a `positions` array, but the dashboard looks for `hedge/{env}/state/positions` and an `items` payload, so cloud viewers miss published positions (`execution/firestore_utils.py:594`, `dashboard/app.py:1017`).  
  **Recommendation:** Align the writer and reader—either mirror into `state/positions` with the expected schema or teach the dashboard to read the `positions/latest` document.

## execution/utils.py
- **Problem:** `get_live_positions` assumes a real client and prints/logs errors when handed `None`, which is exactly how Doctor calls it, leading to empty telemetry (`execution/utils.py:360`).  
  **Recommendation:** Accept `None` by falling back to cached position_state logs or require the caller to supply a client so helper usage becomes explicit.

## execution/executor_live.py
- **Problem:** The runtime only emits a Firestore heartbeat during startup; the steady-state `_maybe_emit_heartbeat` writes to the local JSONL but never calls a Firestore publisher, so dashboard/Doctor telemetry ages out quickly (`execution/executor_live.py:286`, `execution/executor_live.py:831`).  
  **Recommendation:** Relay the periodic heartbeat through `publish_health` or `publish_heartbeat` (once paths are aligned) so Firestore reflects live executor health.
- **Problem:** The inline Firestore fallback writes `{"rows": [...]}` to `hedge/{env}/state/positions`, yet the dashboard expects an `items` list, causing empty tables after fallback publishes (`execution/executor_live.py:2496`, `dashboard/app.py:1017`).  
  **Recommendation:** Write the same schema (`items`) that the dashboard consumes (or update the reader to accept `rows`) to keep position counts accurate.

AUDIT COMPLETE — READY FOR SPRINT 5.8 PLANNING
