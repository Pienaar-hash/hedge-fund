## Artifact Requests
- `.env` (with secrets redacted) plus env dumps for `BINANCE_*`, `FIREBASE_*`, `TELEGRAM_*`, `OPENAI_API_KEY`, `NAV_CUTOFF_*`, and executor-specific toggles (`DRY_RUN`, `KILL_SWITCH`, `FS_PUB_INTERVAL`).
- Strategy + risk config snapshots: `config/strategy_config.json`, `config/risk_limits.json`, `config/settings.json`, `config/symbol_tiers.json`, `config/pairs_universe.json`, `config/treasury.json`, `config/dashboard.yml`.
- Credential provenance: `config/firebase_creds.json` (if present) or the external path referenced by `FIREBASE_CREDS_PATH`, plus Supervisor environment blocks under `deploy/supervisor/*.conf`.
- Runtime state/logs: `logs/nav_trading.json`, `logs/nav_reporting.json`, `logs/nav_snapshot.json`, `logs/nav_treasury.json`, `logs/nav.jsonl`, `logs/positions.jsonl`, `logs/audit_intents.jsonl`, `logs/audit_orders_*.jsonl`, `logs/veto_exec_*.json`, `leaderboard.json`, `nav_log.json`, `peak_state.json`, `synced_state.json`.
- Firestore evidence: export of `hedge/{ENV}/state/{nav,positions,leaderboard}`, audit collections (`audit_*`), and any Cloud Logging tied to the service account.
- Model provenance: latest artifacts in `models/`, ML retrain reports under `logs/ml*`, and `scripts/ml_retrain_daemon.py` output if retained.
- Deployment specs: `deploy/supervisor/*.conf`, `deploy/nginx/*`, `run_all.sh`, `scripts/go_live_now.sh`, and any crontab entries invoking repo scripts.
- Alerting trails: Telegram message history (or screenshots) for heartbeats, drawdown alerts, and daily reports to confirm trigger cadence.

## Reproduction Commands
- `ENV=prod PYTHONPATH=. python -m execution.executor_live`
- `ENV=prod PYTHONPATH=. python -m execution.sync_daemon`
- `ENV=prod PYTHONPATH=. streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0`
- `ENV=prod PYTHONPATH=. python -m execution.telegram_report --dry-run`
- `ENV=prod PYTHONPATH=. python scripts/screener_probe.py --notional 10 --lev 20 BTCUSDT ETHUSDT`
- `ENV=prod PYTHONPATH=. python telegram/heartbeat.py`
- `ENV=prod PYTHONPATH=. python scripts/ml_retrain_daemon.py`
