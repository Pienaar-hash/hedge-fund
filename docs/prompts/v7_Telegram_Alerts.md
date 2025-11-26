# PATCH SCOPE: v7 Telegram Alerts (ATR/DD/Risk Mode/4h Close)
#
# Files:
#   - execution/telegram_utils.py        (reuse send_message helpers)
#   - (new) execution/telegram_alerts_v7.py
#   - execution/executor_live.py         (hook alerts into main loop)
#   - (new) config/telegram_v7.json      (config)
#
# GOAL:
#   Implement low-frequency, high-signal Telegram alerts for:
#     - ATR regime shifts
#     - Drawdown state changes
#     - Risk mode / guardrail changes
#     - BTC 4h close summary
#
#   Alerts should:
#     - Be driven by existing state files / snapshots (no new APIs).
#     - Use telegram_utils send helpers.
#     - Be rate-limited and stateful (no spam).
#
# --------------------------------------------------------------------
# 1) Config: config/telegram_v7.json
#
#   Create a simple JSON config with this shape:
#
#   {
#     "enabled": true,
#     "bot_token_env": "TELEGRAM_BOT_TOKEN",
#     "chat_id_env": "TELEGRAM_CHAT_ID",
#     "min_interval_seconds": 60,
#     "alerts": {
#       "atr_regime": { "enabled": true, "min_interval_seconds": 900 },
#       "dd_state":   { "enabled": true, "min_interval_seconds": 900 },
#       "risk_mode":  { "enabled": true, "min_interval_seconds": 900 },
#       "close_4h":   { "enabled": true, "min_interval_seconds": 3600 }
#     }
#   }
#
#   Notes:
#     - We will read bot token and chat id from environment variables, as
#       indicated by bot_token_env and chat_id_env (not stored in JSON).
#     - If enabled=false, all alerts are disabled.
#
# --------------------------------------------------------------------
# 2) New module: execution/telegram_alerts_v7.py
#
#   Implement a small alert engine that:
#
#   - Loads config from config/telegram_v7.json (with safe defaults).
#   - Loads & updates state at logs/state/alerts_v7_state.json.
#   - Provides a single entrypoint called from executor_live:
#
#       def run_alerts(context) -> None:
#           """
#           High-level entrypoint called from executor_live main loop.
#           Uses current risk/nav/KPI state to decide whether to send
#           any Telegram alerts.
#           """
#
#   Where `context` should include:
#     - risk_snapshot (dict)
#     - nav_snapshot (dict)
#     - kpis_snapshot (dict)
#     - now_ts (float or datetime)
#
#   The module must:
#
#   (a) Implement helpers to load/write a state file:
#
#       STATE_PATH = "logs/state/alerts_v7_state.json"
#
#       def load_state() -> dict:
#           # Read the JSON file; on failure/missing, return {}.
#
#       def save_state(state: dict) -> None:
#           # Write JSON atomically, mirroring patterns used in state_publish/sync_state.
#
#   (b) Track per-alert last values / timestamps, e.g.:
#
#       {
#         "last_sent": {
#           "atr_regime": { "value": "normal", "ts": 1764061000.0 },
#           "dd_state":   { "value": "cautious", "ts": 1764061000.0 },
#           "risk_mode":  { "value": "trading", "ts": 1764061000.0 },
#           "close_4h":   { "bar_ts": 1764061200.0 }
#         }
#       }
#
#   (c) Implement individual "maybe_send" helpers:
#
#       def maybe_send_atr_regime_alert(kpis, state, config) -> None:
#       def maybe_send_dd_state_alert(kpis, state, config) -> None:
#       def maybe_send_risk_mode_alert(risk_snapshot, state, config) -> None:
#       def maybe_send_4h_close_alert(nav_snapshot, kpis, state, config, now_ts) -> None:
#
#   Each helper should:
#     - Determine current value (e.g., atr_regime).
#     - Compare with last_sent[value] from state.
#     - Enforce min_interval_seconds from config["alerts"][...].
#     - If changed and interval passed, send a Telegram message via telegram_utils.
#     - Update state["last_sent"][...] with value + ts.
#
#   (d) Deriving values:
#
#     ATR regime:
#       atr_regime = kpis.get("atr_regime") or kpis.get("atr", {}).get("atr_regime")
#
#     DD state:
#       # Prefer top-level dd_state if present, else derive from drawdown block
#       dd_state = (
#           kpis.get("dd_state")
#           or kpis.get("drawdown", {}).get("dd_state")
#       )
#
#     Risk mode:
#       - If risk_snapshot has a summary field indicating "trading", "paused",
#         "halted", or similar, use that for risk mode.
#       - If not, compute a simple mode:
#           - If drawdown.dd_pct >= max_nav_drawdown_pct: "dd_guard"
#           - If daily_loss >= daily_loss_limit_pct: "daily_loss_guard"
#           - Else: "normal"
#
#     4h close:
#       - Use nav_snapshot["ts"] (or kpis["ts"]) and round to nearest 4h bar:
#           bar_ts = floor(ts / (4 * 3600)) * (4 * 3600)
#       - If bar_ts > last_sent["close_4h"]["bar_ts"] then:
#           - Build a summary message including:
#               * Timestamp (UTC)
#               * BTC price (if available in nav or kpis or risk; otherwise omit)
#               * nav_total
#               * dd_state
#               * atr_regime
#           - Send Telegram alert and update last_sent["close_4h"]["bar_ts"].
#
#   (e) Message formats:
#
#     Keep them concise and investor-friendly, for example:
#
#      ATR regime:
#        "ATR regime change: normal → high (dd_state=cautious, nav=$11,180)"
#
#      DD state:
#        "Drawdown state change: calm → cautious (dd=0.11%, peak=$11,200, nav=$11,188)"
#
#      Risk mode:
#        "Risk mode change: normal → dd_guard (max NAV drawdown guardrail hit)."
#
#      4h close:
#        "BTC 4h close: nav=$11,180, dd_state=cautious, atr_regime=unknown."
#
#   (f) Use telegram_utils:
#
#     - Reuse existing send_message or send_telegram helpers from execution/telegram_utils.py.
#     - If telegram_v7 config is disabled or env vars missing, skip alerts gracefully.
#     - Errors should be logged (via logging.getLogger) but must not crash the executor.
#
# --------------------------------------------------------------------
# 3) Hook alerts into executor_live.py
#
#   - In execution/executor_live.py, after state publish and sync_state have
#     written nav.json, risk.json, and kpis_v7.json, call the alert runner:
#
#       from execution import telegram_alerts_v7
#
#       ...
#       # Inside main loop after risk/nav/kpis snapshots computed and published:
#       try:
#           context = {
#               "risk_snapshot": risk_snapshot,
#               "nav_snapshot": nav_snapshot,
#               "kpis_snapshot": kpis_snapshot,
#               "now_ts": now_ts_float
#           }
#           telegram_alerts_v7.run_alerts(context)
#       except Exception:
#           logger.exception("[telegram] alerts_v7 failed; continuing loop")
#
#   - Ensure this call is non-blocking (i.e., exceptions are caught and logged).
#
# --------------------------------------------------------------------
# 4) Validation / Tests
#
#   After patch is applied:
#
#   1) Compile:
#        python -m py_compile \
#           execution/telegram_utils.py \
#           execution/telegram_alerts_v7.py \
#           execution/executor_live.py
#
#   2) Set env:
#        export TELEGRAM_BOT_TOKEN="<token>"
#        export TELEGRAM_CHAT_ID="<chat_id>"
#
#      and enable alerts in config/telegram_v7.json ("enabled": true).
#
#   3) Run executor in testnet dry-run:
#        PYTHONPATH=. BINANCE_TESTNET=1 DRY_RUN=1 python execution/executor_live.py
#
#   4) Observe:
#        - logs/state/alerts_v7_state.json being created and updated.
#        - Occasional messages in the executor log about ATR/DD/risk mode.
#        - Telegram chat receiving:
#            * a 4h close message on the next 4h boundary
#            * ATR/DD/risk-mode alerts only on actual regime changes.
#
#   5) Confirm that if "enabled" is set to false in telegram_v7.json, the
#      alert module performs no sends and exits quietly.

```
```

# == Patch Notes == #
• - Added a configurable v7 Telegram alert engine (execution/telegram_alerts_v7.py) with stateful,
    rate-limited alerts for ATR regime, drawdown state, risk mode, and 4h close summaries; persists
    state under logs/state/alerts_v7_state.json and reads env-specified bot token/chat from config/
    telegram_v7.json.
  - Hooked alerts into the executor loop (execution/executor_live.py) so alerts run after nav/risk/
    KPI publishing, without impacting the main loop on errors.
  - Added default config config/telegram_v7.json (disabled by default).

  Tests: python -m py_compile execution/telegram_alerts_v7.py execution/executor_live.py.

  Next steps: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, enable alerts in config/telegram_v7.json,
  and run the executor; monitor logs/state/alerts_v7_state.json for last_sent state.