# Patchset Template (v7.6)

## Title
`PATCHSET_v7.6_SX_<name>`

## Objective
- What problem this patchset solves (no trading/risk changes unless scoped).

## Files in Scope
- List files/directories expected to change.

## Before / After
- Briefly describe expected behavior before and after (state/diagnostics/docs only unless scoped).

## Risk Classification
- Low / Medium / High
- Note if touching state surfaces, diagnostics, or (authorized) trading paths.

## Tests
- Required: `make test-fast`
- Add if applicable: `make test-runtime`
- Legacy tests? (opt-in)
- If docs-only, note “tests not run (docs-only)”.

## Manifest / Docs
- Any updates to `v7_manifest.json`?
- Docs to update (architecture, state contract, diagnostics, testing, agent guide)?

## Acceptance Criteria
- Bulleted, concrete criteria for completion.
