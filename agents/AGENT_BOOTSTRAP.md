# Agent Bootstrap (v7.9)

This is the first file every GPT/Copilot agent must read before touching the repo.

## Required Context (read before coding)
- `agents/GPT_BEHAVIORAL_CONTRACT.md` — division of labor + doctrine library
- `docs/SYSTEM_BASELINE_v7.9.md` — architectural invariants + CI gates
- `v7_manifest.json` — machine-readable state registry (authoritative)
- `VERSION` — canonical version (v7.9)
- `.github/copilot-instructions.md` — component map + coding conventions
- `ops/PHASE_C_DAILY_CHECKPOINT.md` — active experiment SOP
- `ops/C1_OPS_PROTOCOL.md` — enforcement ladder

## Pre-Patch Checklist
1. Load architecture to understand module boundaries and data flow.
2. Load state contract to know canonical writers and schemas.
3. Load manifest to confirm paths/owners/update frequency.
4. Load diagnostics semantics (liveness, coverage, router activity).
5. Verify test lanes and decide which to run (`make test-fast` required unless doc-only).
6. Confirm you will not modify trading/risk semantics unless the patchset explicitly demands it.

## Repo Invariants
- Single-writer per state surface; atomic JSON writes only.
- State files live under `logs/state/` (44 surfaces registered in `v7_manifest.json`).
- Execution logs live under `logs/execution/` (16 JSONL streams, append-only).
- Dashboard is read-only via `dashboard/state_client.py`; only executor/state_publish/sync_state write state.
- Tests are lane-based: `unit`, `integration`, `runtime`, `legacy` (3052 collected).
- Doctrine Kernel (`execution/doctrine_kernel.py`) has NO config — it IS the law.
- DLE shadow observes but does not gate (Phase A/B); C.1 enforcement is entry-only when active.

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
