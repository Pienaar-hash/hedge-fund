# PATCH SCOPE: Telegram Alerts v7 (state-change driven, low-noise)
#
# FILES TO MODIFY / ADD:
#   - execution/telegram_utils.py
#   - execution/executor_live.py  (non-invasive wiring)
#   - execution/state_publish.py  (read-only KPI access)
#   - Add new file: logs/state/telegram_state.json (written automatically)
#
# GOAL:
#   Implement v7-grade Telegram alerts based on *state change only*:
#     - ATR regime changes (low → medium → high; or reverse)
#     - Drawdown regime changes (none → mild → severe)
#     - Router quality changes (good → degraded → taker-only)
#     - AUM delta alerts (if total AUM moves ±N%)
#     - 4h close alerts (ONLY on 4h candle close, no lower TF)
#
#   Ensure:
#     - Alerts occur *only* when state changes from last-known state.
#     - Use logs/state/telegram_state.json as the persistence layer.
#     - Do NOT send alerts repeatedly.
#     - Do NOT spam per-tick messages.
#     - Alerts are telemetry-only: no effect on risk or strategy logic.
#
# ---------------------------------------------------------------------------
# 1) Create or extend telegram_state persistence
#    File: logs/state/telegram_state.json (auto-created if missing)
#
#    Structure example:
#    {
#      "atr_regime": "low",
#      "drawdown_state": "none",
#      "router_quality": "good",
#      "aum_total": 11173.87,
#      "last_4h_close_ts": 0
#    }
#
# Add helper in telegram_utils.py:
#
#    def load_telegram_state(path="logs/state/telegram_state.json"):
#         try: read & parse
#         except: return minimal default dict
#
#    def save_telegram_state(state):
#         write JSON atomically
#
# ---------------------------------------------------------------------------
# 2) Add state-change checker in telegram_utils.py
#
#    def detect_changes_and_build_alerts(kpis, nav_snapshot, router_snapshot):
#        - Load previous telegram_state
#        - Compare:
#             prev["atr_regime"]       vs kpis["atr"]["regime"]
#             prev["drawdown_state"]   vs kpis["drawdown"]["state"]
#             prev["router_quality"]   vs router_snapshot["quality"]
#             prev["aum_total"]        vs nav_snapshot["aum"]["total"]
#        - For each change, append a formatted alert string to alerts[]
#        - Return alerts, updated_state
#
#    AUM rule:
#        if abs(new_aum - prev_aum) / prev_aum >= 0.05 (5%) → include alert
#
# ---------------------------------------------------------------------------
# 3) Implement 4h close detection
#
# In telegram_utils.py:
#
#    def detect_4h_close(now_ts, last_kline_close_ts):
#         if last_kline_close_ts > prev["last_4h_close_ts"]:
#             return True
#
# 4h alert format:
#    "🕓 4h Close — BTCUSDT closed at {price}, ATR={atr_regime}, DD={drawdown_state}"
#
# ---------------------------------------------------------------------------
# 4) High-signal alert formatting
#
# In telegram_utils.py add:
#
#    def format_v7_alert(msg_type, payload):
#         # msg_type ∈ ["atr", "drawdown", "router", "aum", "4h"]
#         # Format with emojis, clear regime → regime transitions.
#
# Examples:
#    ATR:     "📈 ATR Regime Shift: low → high"
#    DD:      "⚠️ Drawdown State Change: none → mild"
#    Router:  "🔌 Router Quality: good → taker-only"
#    AUM:     "💰 AUM Shift: 11173 → 11800 (+5.6%)"
#    4h:      "🕓 4h Close — BTC: 102,345 | ATR=low | DD=none"
#
# ---------------------------------------------------------------------------
# 5) Executor integration (non-invasive)
#
# In execution/executor_live.py inside the main tick loop:
#
#    - After computing kpis_v7 and nav_snapshot and router_snapshot:
#
#        from execution.telegram_utils import detect_changes_and_build_alerts, send_telegram_message
#
#        alerts, new_state = detect_changes_and_build_alerts(kpis_v7, nav_snapshot, router_snapshot)
#
#        for msg in alerts:
#            send_telegram_message(msg)
#
#        save_telegram_state(new_state)
#
#    - For 4h close:
#        - Reuse existing candle feed:
#               last_kline_close_ts = kline["close_time"]
#        - If detect_4h_close(): add a single 4h alert
#
# NOTE:
#    - Must NOT alter strategy decisions.
#    - Alerts must fire AFTER execution/nav + kpis publish.
#
# ---------------------------------------------------------------------------
# 6) telegram_utils.send_telegram_message
#
# Reuse existing function if present. Otherwise:
#
#    def send_telegram_message(text):
#        if TELEGRAM_BOT_TOKEN not configured → no-op
#        send POST to API
#        log success/failure
#
# ---------------------------------------------------------------------------
# 7) Tests / Validation
#
# Manual checks:
#   - python -m py_compile execution/telegram_utils.py execution/executor_live.py execution/state_publish.py
#
# Runtime:
#   - supervisor restart executor
#   - Watch logs:
#         tail -f logs/execution/telegram.log
#         tail -f logs/state/telegram_state.json
#
# Verify behavior:
#   ✔ Only FIRST 4h close produces a message
#   ✔ ATR regime shift produces exactly ONE alert per shift
#   ✔ DD regime shift produces ONE alert per transition
#   ✔ Router quality change produces ONE alert per event
#   ✔ AUM delta only alerts if >5% move
#
# END PATCH SCOPE
