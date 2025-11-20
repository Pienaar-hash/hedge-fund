# Module Resolution Report

## Supervisor Environment
- Every program in `ops/hedge.conf` now exports `PYTHONPATH="/root/hedge-fund"` plus `PYTHONUNBUFFERED=1`, `ENV=prod`, and the existing ALLOW_* flags.
- This guarantees the interpreter sees repo-local packages even before entrypoints tweak `sys.path`.

## Key Module Checks
- `utils.firestore_client` exists (`utils/firestore_client.py`) and is imported lazily by `execution.firestore_utils`. When it is missing, the new guards fall back to a no-op Firestore client so `sync_state` no longer crashes.
- `dashboard.app` imports cleanly under the venv interpreter and logs its ENV hook, confirming Streamlit loads the module without path errors.
- `execution.state_publish` and `execution.sync_state` both target `logs/state/synced_state.json` with the same schema (`items/nav/engine_version/v6_flags/updated_at`). `_read_positions_snapshot()` tolerates missing fields, so telemetry contracts stay aligned.

## Duplicate/Shadowed Trees
- `gpt_schema/` contains a frozen copy of execution modules for schema generation. Because PYTHONPATH points to `/root/hedge-fund`, the live interpreter resolves `execution.*` from the repo root, not from `gpt_schema/`. There are no other repo-level directories named `execution`, so no active shadowing occurs.
- No `tmp/repo_audit` or similar clones exist; `ls tmp` shows only runtime scratch files.

## Outstanding Risks
- The leading empty-string entry in sys.path causes the interpreter to search the working directory first. This matches Python defaults but means stray files in `/root/hedge-fund` could shadow standard modules; keep the repo clean to avoid collisions.
- Streamlit imports emit “No runtime found” warnings when run via `python - <<'PY'`; under Supervisor it runs through `streamlit run ...`, so these warnings do not affect production.
