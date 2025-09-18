## Top-Level Layout
- `execution/`: live trading loop, Binance order adapters, Firestore publishers, sync tooling, and risk controls.
- `dashboard/`: Streamlit dashboard + helpers that read Firestore/local logs for portfolio monitoring.
- `utils/`: shared helpers (Firestore client, perf metrics, JSON utilities) used across execution and scripts.
- `scripts/`: operational CLIs for probes, ML retraining, one-off checks, and shell wrappers.
- `config/`: JSON/YAML settings for strategies, tiers, risk, dashboard, and Firebase credentials.
- `deploy/`: Supervisor + nginx configs and launch scripts for production services.
- `telegram/`: lightweight Telegram heartbeat sender alongside execution alert helpers.
- `tests/`: pytest coverage for risk gating, sizing, NAV, ML feature generation, and reporting.

## Primary Entry Points & Services
- `execution/executor_live.py`: supervisor-run loop that pulls intents (via `execution.signal_screener`), applies `RiskGate` + `check_order`, sizes via `build_order_payload`, submits to Binance, and writes Firestore/local logs.
- `execution/sync_daemon.py`: pushes local NAV/position JSON snapshots into Firestore on a cadence when Firestore access from executor is restricted.
- `execution/state_publish.py`: standalone publisher used by the executor for debounced NAV + position writes and audit trails.
- `execution/telegram_report.py`: optional daily mini-report builder (NAV PNG + AI market note) with Telegram delivery.
- `dashboard/app.py`: Streamlit UI served via Supervisor (`streamlit run ...`) rendering Firestore state with local fallbacks.
- `scripts/screener_probe.py`: CLI to inspect whether the screener would emit trades under current tier/risk caps.
- `scripts/ml_retrain_daemon.py`: background trainer triggered by Supervisor cron config (`deploy/supervisor/ml-retrain-nightly.conf`).
- `telegram/heartbeat.py`: ad-hoc heartbeat sender summarizing NAV and top positions if creds are present.

## How Services Interlock
- Executor requires Binance + Firestore creds, consumes `config/strategy_config.json`, `config/risk_limits.json`, tier/universe config, and updates `logs/`, Firestore, and Telegram.
- Sync daemon and dashboard ingest executor output (`logs/nav_log.json`, `logs/positions.jsonl`, Firestore `hedge/{ENV}/state/*`).
- Telegram utilities (`execution.telegram_utils`, `telegram/heartbeat.py`) share env-driven configuration and reuse exchange helpers.
- ML subpackage (`execution/ml`) is optional; screener degrades gracefully if models are absent.

## Core Runtime Flows
- **Signal → Order Path**: `signal_screener.generate_signals_from_config()` loads enabled strategies, filters by futures listing + tiers, probes orderbook slippage, optional ML gating, and produces intents. `executor_live._send_order()` enforces cool-downs and `RiskGate`/`check_order`, sizes via `exchange_utils.build_order_payload()`, submits with `exchange_utils.send_order()`, persists veto logs, and posts audits via `state_publish`.
- **NAV → Firestore / Dashboard**: `executor_live._compute_nav()` mixes exchange balances (`exchange_utils.get_balances/get_positions`) with treasury snapshots, writes JSON snapshots (`logs/nav_*.json`). `state_publish.StatePublisher` or the inline fallback writes NAV series and positions to Firestore `hedge/{ENV}/state`. `dashboard/data_sources` reads Firestore first, then local logs (`logs/nav.jsonl`, `logs/positions.jsonl`) for display.
- **Telegram Alerts**: `executor_live` imports `execution.telegram_utils` to send trade alerts, drawdown notices, and heartbeats based on NAV/position telemetry; `execution/telegram_report.py` composes a daily summary image + AI note; `telegram/heartbeat.py` offers a CLI heartbeat using the same exchange utils.

## Notable Utilities & Patterns
- Direct Binance REST adapter in `execution.exchange_utils` (signed requests, filter caching, protective order helpers) instead of `python-binance`.
- Layered risk controls: `RiskGate` for gross exposure caps, `RiskState` for per-symbol cooldowns, and JSON veto logging per symbol under `logs/veto_exec_*.json`.
- Firestore publishing with offline fallbacks: if Firestore is disabled, state publishers append JSONL logs under `logs/` for later sync.
- Extensive environment fallbacks: `.env` auto-loads, multiple credential sources (`FIREBASE_CREDS_PATH`, base64 JSON) handled in `utils.firestore_client`.
- Strategy config–driven sizing: `execution.sizing.determine_position_size` applied centrally in screener and executor to keep order sizing consistent.
- Optional ML gating: `execution.signal_screener` calls `execution.ml.predict` when available, carrying probability metadata through intents.
- Dashboard data pipeline mixes Firestore reads with local tail logs, providing resilience when the writer process is offline.
