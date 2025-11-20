# Runtime v6 PYTHONPATH Verification

1. **Check Supervisor Environment**
   - `sudo supervisorctl status hedge:*` to confirm all programs are RUNNING.
   - `sudo grep -R "PYTHONPATH" /etc/supervisor/conf.d/ops/hedge.conf` should show `PYTHONPATH="/root/hedge-fund"` for each program.

2. **Confirm sys.path at Runtime**
   - For executor: `PYTHONPATH="/root/hedge-fund" ./venv/bin/python -c "import execution.executor_live, sys, json; print(json.dumps(sys.path[:5]))"`.
   - For dashboard: `PYTHONPATH="/root/hedge-fund" ./venv/bin/python -c "import dashboard.app"` (ignore Streamlit cache warnings).
   - For sync_state: `PYTHONPATH="/root/hedge-fund" ./venv/bin/python -c "import execution.sync_state"`.

3. **Verify Telemetry Schema Alignment**
   - Ensure `_pub_tick()` logs `[v6-runtime] pub_tick wrote state: ...` inside `logs/executor.out`.
   - Tail `logs/state/synced_state.json` and confirm it includes `items`, `nav`, `engine_version`, `v6_flags`, `updated_at`.
   - `python - <<'PY'` with `from execution.state_publish import build_synced_state_payload` to assert the same schema as read by `execution.sync_state`.

4. **Watch for Shadowed Modules**
   - `python - <<'PY'` `import pkgutil; print([m.name for m in pkgutil.iter_modules(['gpt_schema']) if m.name=='execution'])` should list only the schema copy; ensure PYTHONPATH does not include `gpt_schema` unless intentional.

5. **Dashboard Streamlit Entrypoint**
   - `sudo tail -f /var/log/hedge-dashboard.out.log` to confirm `streamlit run dashboard/app.py` launched and bound to port 8501.

6. **Firestore Optionality**
   - With `FIRESTORE_ENABLED=0`, run `./venv/bin/python execution/sync_state.py --once` (or similar) and confirm it logs “Firestore heartbeat skipped (disabled)” but continues without stack traces.
