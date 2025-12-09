# Agent Safety Rails (v7.6)

## You MUST NOT
- Modify trading logic, risk gates, or veto rules unless patchset explicitly authorizes it.
- Write to `logs/state/*.json` from tests (use `tmp_path` and monkeypatch paths).
- Add new state writers or rename surfaces without updating manifest + state contract docs.
- Change diagnostics semantics casually or without updating runtime diagnostics doc.
- Bypass test lanes or skip required tests without stating why.

## You MUST
- Respect single-writer and atomic-write rules for all state files.
- Keep repo hygiene (no stray files, consistent naming).
- Run `make test-fast` unless patch is documentation-only; add `make test-runtime` when state/telemetry is touched.
- Update docs and manifest when adding/changing state surfaces or diagnostics.
- Use patchset templates/protocol; document scope, risks, tests, and acceptance criteria.
- Keep changes minimal and scoped; avoid touching unrelated files.
