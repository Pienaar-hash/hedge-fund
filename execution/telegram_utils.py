import os
import time
import requests

TELEGRAM_ENABLED = os.environ.get("TELEGRAM_ENABLED", "0") == "1"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Cooldown state
_last_alert_time = {}
_last_dd_sent = {}

def should_send_alert(strategy, alert_type, dd_value=None, threshold=0.5, cooldown_minutes=5):
    """
    Decide whether to send an alert based on cooldown and threshold logic.
    - threshold: % difference in DD before triggering again
    - cooldown_minutes: minimum time between alerts of same type for same strategy
    """
    key = f"{strategy}:{alert_type}"
    now = time.time()
    last_time = _last_alert_time.get(key, 0)
    last_dd = _last_dd_sent.get(key, None)

    # Always allow trade alerts immediately
    if alert_type == "trade":
        _last_alert_time[key] = now
        return True

    # First time sending this alert
    if last_dd is None:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True

    # Time-based cooldown
    if now - last_time > cooldown_minutes * 60:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True

    # Significant worsening in DD
    if dd_value is not None and dd_value < last_dd - threshold:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True

    return False


def send_telegram_message(message):
    """Send a Telegram message if enabled."""
    if not TELEGRAM_ENABLED:
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not set. Skipping message.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ Telegram send failed: {r.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")


def send_trade_alert(strategy, symbol, side, qty, price, pnl_realized, pnl_unrealized, equity, dd, strategy_dd):
    """Send trade execution alert (no cooldown)."""
    if should_send_alert(strategy, "trade"):
        message = (
            f"🧰 testnet-exec\n"
            f"🚀 Trade Executed\n"
            f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"📈 Symbol: {symbol}\n"
            f"🔁 Side: {side}\n"
            f"💸 Qty: {qty:.8f}\n"
            f"💰 Price: {price} USDT\n"
            f"🧠 Strategy: {strategy}\n"
            f"📊 PnL/NAV: Realized: {pnl_realized:.2f} | Unrealized: {pnl_unrealized:.2f} | Equity: {equity:.2f} | DD: {dd:.2f}% | Strategy DD: {strategy_dd:.2f}%"
        )
        send_telegram_message(message)


def send_drawdown_alert(strategy, dd, equity):
    """Send drawdown alert with cooldown & threshold logic."""
    if should_send_alert(strategy, "drawdown", dd_value=dd, threshold=0.5, cooldown_minutes=5):
        message = (
            f"🧰 testnet-exec\n"
            f"⚠️ Drawdown Alert\n"
            f"🧠 Strategy: {strategy}\n"
            f"📉 Drawdown: {dd:.2f}%\n"
            f"💼 Equity: {equity:.2f}"
        )
        send_telegram_message(message)
