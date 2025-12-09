# v7.6 Agent Patch Entrypoints

Use this as the single authoritative starting point for patchsets.

## Pre-flight Checklist
- Read `v7_manifest.json` for canonical state surfaces, owners, paths.
- Confirm single-writer rule holds for the surface you will touch.
- Plan tests: keep `make test-fast` green; run `make test-runtime` when touching state/telemetry; legacy is opt-in.
- Avoid trading semantic changes unless explicitly requested.

## Core Modules
- Executor loop: `execution/executor_live.py`
- Risk gates: `execution/risk_limits.py`, `execution/risk_engine_v6.py`
- Router: `execution/order_router.py`, intel under `execution/intel/*`
- Exits/ledger: `execution/exit_scanner.py`, `execution/position_ledger.py`, `execution/position_tp_sl_registry.py`
- State publish: `execution/state_publish.py` (atomic writers), `execution/diagnostics_metrics.py`
- Dashboard readers: `dashboard/state_v7.py`, panels under `dashboard/*`
- Mirror: `execution/sync_state.py` (only owns `nav_state.json`)

## State & Diagnostics Surfaces
- Positions: `positions_state.json` (executor), `positions_ledger.json` (executor)
- KPIs: `kpis_v7.json` (executor)
- Diagnostics: `diagnostics.json` (executor; veto/exit/liveness with missing-ts semantics)
- Risk/Router: `risk_snapshot.json`, `router_health.json`
- NAV: `nav.json` (executor), `nav_state.json` (sync_state)

## Patchset Rules
- Use `state_publish` helpers for any new fields; update dashboard loaders + schema tests.
- Diagnostics changes must remain side-effect-free (metrics only).
- Runtime docs: refresh Architecture/State Contract/Diagnostics when adding surfaces.
- Commit hygiene: respect repo layout (execution/, dashboard/, config/, tests/, docs/).
