# 🧭 Quant Infrastructure Patch — v5.7.5

**Focus:** Restore dashboard rendering and live data freshness for sprint v5.7.5.

## 🔧 Key Changes
- `dashboard/dashboard_utils.py`: stop caching noop Firestore clients so subsequent requests retry initialization instead of locking the dashboard onto empty local fallbacks.
- `dashboard/app.py`: harden async cache bootstrap by threading the event-loop fallback when `asyncio.run` is blocked, preserving doctor/router/telemetry cards even inside already-running loops.
- `execution/sync_state.py`: kick off the structured heartbeat thread immediately on bootstrap and throttle heartbeats to a steady 60 s cadence while publishing environment-aware metrics.
- `execution/firestore_utils.py`: add router/positions Firestore publishers using the shared client helpers and cache fallbacks.

## ✅ Validation
- `pytest -q tests/test_async_cache.py tests/test_dashboard_metrics.py tests/test_router_health_v2.py`
- `pytest -q tests/test_firestore_client.py`
- inspected local supervisor sources (`logs/` + `logs/archive/`) — no crash traces for dashboard services.

## 📌 Notes
- Firestore remains optional; failures degrade gracefully to cached local mirrors, but clients now re-attempt connections on every render.
- Async cache warnings are emitted once when the threaded fallback is used, aiding future telemetry correlation.

### 🔄 Post-Release Health Addendum
- `scripts/doctor.py`: patched heartbeat reader to support structured Firestore schema under `heartbeat.{executor_live,sync_state}`.
- Verified sync-state thread publishes every 60 s and doctor CLI now reports both heartbeats as fresh.
