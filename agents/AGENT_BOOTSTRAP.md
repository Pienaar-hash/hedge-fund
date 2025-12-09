# Agent Bootstrap (v7.6)

This is the first file every GPT/Copilot agent must read before touching the repo.

## Required Context (read before coding)
- `docs/v7.6_Architecture.md`
- `docs/v7.6_State_Contract.md`
- `docs/v7.6_Runtime_Diagnostics.md`
- `docs/v7.6_Testing.md`
- `docs/v7.6_Developer_Guide.md`
- `v7_manifest.json`

## Pre-Patch Checklist
1. Load architecture to understand module boundaries and data flow.
2. Load state contract to know canonical writers and schemas.
3. Load manifest to confirm paths/owners/update frequency.
4. Load diagnostics semantics (liveness, coverage, router activity).
5. Verify test lanes and decide which to run (`make test-fast` required unless doc-only).
6. Confirm you will not modify trading/risk semantics unless the patchset explicitly demands it.

## Repo Invariants
- Single-writer per state surface; atomic JSON writes only.
- State files live under `logs/state/` with canonical names (positions_state, positions_ledger, kpis_v7, diagnostics, nav, nav_state, risk_snapshot, router_health, symbol_scores, rv_momentum, factor_diagnostics).
- Dashboard is read-only; only executor/state_publish/sync_state write state.
- Tests are lane-based: `unit`, `integration`, `runtime`, `legacy`.

## Naming Rules
- Use canonical state file names; do not introduce alternates without manifest + contract updates.
- Keep module names consistent with architecture (execution/*, dashboard/*, config/*).

## Patchset Safety Checklist
- Scope understood and documented.
- Touch only files listed in patchset scope.
- Update manifest and docs if adding/changing state surfaces.
- Add/adjust tests and mark with correct lane markers.
- Run `make test-fast` (unless doc-only); run `make test-runtime` if state/telemetry touched.
- Do not write to real `logs/state` in tests (use `tmp_path`).
