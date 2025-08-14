import os
import time
import requests
from datetime import datetime, timezone

# Environment variables for Telegram
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "0") == "1"
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# Cooldown state
_last_alert_time = {}
_last_dd_sent = {}
_last_summary_ts = None

def should_send_alert(strategy, alert_type, dd_value=None, threshold=0.5, cooldown_minutes=5):
    """Decide whether to send an alert based on cooldown and threshold logic."""
    key = f"{strategy}:{alert_type}"
    now = time.time()
    last_time = _last_alert_time.get(key, 0)
    last_dd = _last_dd_sent.get(key, None)

    if alert_type == "trade":
        _last_alert_time[key] = now
        return True
    if last_dd is None:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True
    if now - last_time > cooldown_minutes * 60:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True
    if dd_value is not None and dd_value < last_dd - threshold:
        _last_dd_sent[key] = dd_value
        _last_alert_time[key] = now
        return True
    return False

def should_send_summary(min_interval_sec: int = 3600) -> bool:
    """Rate-limit investor summaries to max one per min_interval_sec."""
    global _last_summary_ts
    now_ts = datetime.now(timezone.utc).timestamp()
    if _last_summary_ts is None or (now_ts - _last_summary_ts) >= min_interval_sec:
        _last_summary_ts = now_ts
        return True
    return False

def send_telegram(message: str, silent: bool = False) -> bool:
    """Send a Telegram message if enabled, fail gracefully."""
    if not TELEGRAM_ENABLED:
        print("❌ Telegram disabled. Skipping message.")
        return False
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram credentials not set. Skipping message.")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n{message}",
            "disable_notification": silent
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ Telegram message sent.")
            return True
        else:
            print(f"❌ Telegram send failed: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram send error: {e}")
        return False

def send_trade_alert(strategy, symbol, side, qty, price, pnl_realized, pnl_unrealized, equity, dd, strategy_dd):
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
        send_telegram(message)

def send_drawdown_alert(strategy, dd, equity):
    if should_send_alert(strategy, "drawdown", dd_value=dd, threshold=0.5, cooldown_minutes=5):
        message = (
            f"🧰 testnet-exec\n"
            f"⚠️ Drawdown Alert\n"
            f"🧠 Strategy: {strategy}\n"
            f"📉 Drawdown: {dd:.2f}%\n"
            f"💼 Equity: {equity:.2f}"
        )
        send_telegram(message)
