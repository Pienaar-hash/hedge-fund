# Runtime Path Diff

| Entry Point | PYTHONPATH (effective) | sys.path (first entries) |
| --- | --- | --- |
| executor (execution/executor_live.py) | `/root/hedge-fund` | `["/root/hedge-fund", "", "/root/hedge-fund", ...]` |
| dashboard (dashboard/app.py via streamlit) | `/root/hedge-fund` | `["", "/root/hedge-fund", ...]` |
| sync_state (execution/sync_state.py) | `/root/hedge-fund` | `["", "/root/hedge-fund", ...]` |
| pipeline_shadow_heartbeat | `/root/hedge-fund` | `["", "/root/hedge-fund", ...]` |

Notes:
- Executor preprends the repo root explicitly, so sys.path includes it twice (`"/root/hedge-fund"` plus the leading empty-string entry from the interpreter).
- Dashboard/login helpers rely on Streamlit; when run outside the Streamlit runtime we get “No runtime found” warnings, but imports succeed with repo root in sys.path.
- sync_state bootstraps PATH at module import, inserting `/root/hedge-fund` explicitly; combined with the global PYTHONPATH this results in duplicates but ensures repo modules resolve even if supervisor forgets PYTHONPATH.
- pipeline_shadow_heartbeat (scripts namespace) also sees the repo root plus the empty current-directory entry; no additional bootstrap occurs beyond the global env.
- All entry points pick up site-packages from the virtualenv (`/root/hedge-fund/venv/lib/python3.10/site-packages`), so third-party libs resolve consistently across services.
