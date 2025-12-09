# Agent Patchset Protocol (v7.6)

## Lifecycle
1. **Plan** — define scope, files, risks, tests.
2. **Apply** — implement changes respecting repo invariants.
3. **Validate** — run required test lanes (`make test-fast`; add `make test-runtime` when state/telemetry affected).
4. **Document** — update manifest/docs/templates as needed.
5. **Commit/Hand-off** — summarize changes, tests, and remaining risks.

## Patchset Contents
- Objective and scope.
- Files in scope.
- Before/after behavior (no semantic change unless specified).
- Risks and mitigations.
- Tests to run (fast lane required unless doc-only).
- Acceptance criteria.
- Manifest/doc changes if state surfaces are touched.

## Test Rules
- Always run `make test-fast` unless patch is documentation-only.
- Run `make test-runtime` when touching state surfaces, diagnostics, or telemetry contracts.
- Legacy tests are opt-in; do not block fast lane.

## Safety Rules
- Do **not** rename or add state surfaces without updating `v7_manifest.json` and `docs/v7.6_State_Contract.md`.
- Do **not** change diagnostics semantics casually; update runtime diagnostics doc if required.
- Do **not** modify trading/risk logic unless the patchset explicitly authorizes it.
- Do **not** write to `logs/state` in tests; use `tmp_path` and monkeypatch paths.
- Keep single-writer and atomic-write invariants intact.

## Patchset Tracks
- S-tracks (S0–S*): state/diagnostics/infra — no trading logic changes.
- A/B/C tracks: strategy/risk/router features — trading semantics allowed only if explicitly scoped.

## Handoff Checklist
- Summary of changes and files touched.
- Tests executed with results.
- Any skipped runtime/legacy tests noted.
- Open questions/risks for next agent.
