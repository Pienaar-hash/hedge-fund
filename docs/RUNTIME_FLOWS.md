A) Trade signal -> order placement -> confirmations -> position state
SignalScreener (`execution/signal_screener.py`)
  -> `config/strategy_config.json`: load enabled strategies and sizing defaults
  -> UniverseResolver (`execution/universe_resolver.py`): filter to allowed/tiered symbols (`config/pairs_universe.json`, `config/symbol_tiers.json`)
  -> OrderbookFeatures (`execution/orderbook_features.py`): compute imbalance metric; veto if adverse
  -> optional `execution/ml/predict`: attach ML probabilities when models are present
  -> RiskLimits.check_order: preflight tier/gross caps for hypothetical size
  -> emit intent {symbol, side, gross_usd, leverage, metadata}
Executor Loop (`execution/executor_live.py`)
  -> receives intent; enforces local cooldown and kill switch, refreshes `PortfolioSnapshot`
  -> RiskGate.allowed_gross_notional + risk_limits.check_order: final risk/circuit verdict (publishes veto + Telegram on block)
  -> ExchangeUtils.build_order_payload: normalize qty/price, apply slippage bps, attach hedge positionSide
  -> ExchangeUtils.send_order: signed POST to Binance USD-M REST; protective SL/TP orders if configured
  -> StatePublish.publish_*: write intent/order audits to Firestore (`hedge/{ENV}/state/audit_*`) or local JSONL when offline
  -> Utils.write_nav_snapshots_pair / write_treasury_snapshot: persist fresh NAV JSON under `logs/`
  -> ExchangeUtils.get_positions: refresh open positions for next loop
  -> RiskState.note_attempt/note_fill: update cooldown counters
  -> TelegramUtils: push trade alert or drawdown alert when thresholds are met
Binance
  -> returns order status; executor logs response and updates `_last_order_ts`
  -> positions feed back into `_loop_once` and the Firestore publisher `_pub_tick`
StatePublisher (`execution/state_publish.py` or inline fallback)
  -> deduplicates and publishes positions and NAV to Firestore `hedge/{ENV}/state/{positions,nav}`

B) Balances/positions -> NAV calc -> Firestore -> dashboard render
Executor `_compute_nav`
  -> `execution/nav.compute_nav_pair`: pulls `config/strategy_config.json`, exchange balances (`get_balances`), open positions, and `config/treasury.json`
  -> `execution/utils.write_nav_snapshots_pair`: writes `logs/nav_trading.json`, `logs/nav_reporting.json`, `logs/nav_snapshot.json`
  -> `execution/utils.write_treasury_snapshot`: writes `logs/nav_treasury.json`
State publish step
  -> StatePublisher.maybe_publish_positions + publish_nav_value: debounced Firestore writes with JSONL fallback under `logs/nav.jsonl`, `logs/positions.jsonl`
  -> When Firestore is disabled, `_pub_tick` appends to local JSONL for later sync
Sync Daemon (`execution/sync_daemon.py` via Supervisor or cron)
  -> Reads `leaderboard.json`, `nav_log.json`, `peak_state.json`, `synced_state.json`
  -> Calls `execution.sync_state.sync_{leaderboard,nav,positions}` to backfill Firestore `hedge/{ENV}/state/*`
Dashboard (`dashboard/app.py` served by Supervisor/Streamlit)
  -> DashboardData loaders (`dashboard/data_sources.py`) attempt Firestore via `utils.firestore_client.get_db`
  -> On failure, fall back to local logs (`logs/nav.jsonl`, `logs/positions.jsonl`, `logs/screener_tail.log`) and `state/nav_snapshot.json`
  -> Streamlit renders KPIs, NAV chart, positions table, and veto tail using fetched datasets
  -> nginx (`deploy/nginx`) can proxy the Streamlit service; Supervisor keeps it running (`deploy/supervisor/hedge-dashboard.conf`)

C) Alerts / heartbeats / drawdown flow
Executor telemetry
  -> `execution/executor_live` tracks NAV peak/drawdown via `RiskState` and NAV snapshots
  -> On each loop, conditional calls to `execution.telegram_utils.send_heartbeat` and `send_drawdown_alert`
  -> Trade fills trigger `send_trade_alert` with fill metadata
Telegram report (`execution/telegram_report.py`)
  -> Loads NAV series from Firestore (`utils.firestore_client.get_db`)
  -> Uses `matplotlib` to render `/tmp/nav.png`, composes AI market note via `openai` when key present
  -> Sends via `execution.telegram_utils.send_telegram` unless `--dry-run`
Heartbeat CLI (`telegram/heartbeat.py`)
  -> Queries Binance via `execution.exchange_utils` for balances/positions, formats summary, hits Telegram Bot API
Supervisor / cron context
  -> `deploy/supervisor/hedge-executor.conf` keeps executor alive, ensuring heartbeats and alerts remain active
  -> `deploy/supervisor/ml-retrain-nightly.conf` optionally runs retrain daemon; outputs feed `models/` consumed by screener alerts
  -> External schedulers may invoke `scripts/ml_retrain_now.sh` or `telegram/heartbeat.py` for scheduled messaging
