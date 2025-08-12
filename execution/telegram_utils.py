# execution/telegram_utils.py
import os
import requests
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TELEGRAM_ENABLED = str(os.getenv("TELEGRAM_ENABLED", "0")).lower() in ("1", "true", "yes")

def send_telegram(message: str, silent: bool = False) -> bool:
    if not TELEGRAM_ENABLED:
        print("ℹ️ Telegram disabled (TELEGRAM_ENABLED=0).")
        return False
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram BOT_TOKEN or CHAT_ID not set.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "disable_notification": silent, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ Telegram error {r.status_code}: {r.text}")
            return False
        return True
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        return False

def send_trade_alert(trade: dict, silent: bool = False) -> bool:
    ts = trade.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    symbol = trade.get("symbol", "?")
    side = trade.get("side", "?")
    price = trade.get("price", 0)
    qty = trade.get("qty", 0)
    strategy = trade.get("strategy", "?")
    realized = trade.get("realized")
    unrealized = trade.get("unrealized")
    equity = trade.get("equity")
    dd = trade.get("drawdown_pct")
    sdd = trade.get("strategy_drawdown_pct")

    msg = (
        f"🚀 <b>Trade Executed</b>\n"
        f"<b>⏰ Time:</b> {ts}\n"
        f"<b>📈 Symbol:</b> {symbol}\n"
        f"<b>🔁 Side:</b> {side}\n"
        f"<b>💸 Qty:</b> {qty}\n"
        f"<b>💰 Price:</b> {price:.2f} USDT\n"
        f"<b>🧠 Strategy:</b> {strategy}"
    )
    extras = []
    if realized is not None:  extras.append(f"Realized: {realized:.2f}")
    if unrealized is not None: extras.append(f"Unrealized: {unrealized:.2f}")
    if equity is not None:    extras.append(f"Equity: {equity:,.2f}")
    if dd is not None:        extras.append(f"DD: {dd*100:.2f}%")
    if sdd is not None: extras.append(f"Strategy DD: {sdd*100:.2f}%")
    if extras:
        msg += "\n<b>📊 PnL/NAV:</b> " + " | ".join(extras)
    return send_telegram(msg, silent=silent)

def send_drawdown_alert(strategy_name: str, drawdown_pct: float, equity: float) -> bool:
    msg = (f"⚠️ <b>Drawdown Alert</b>\n"
           f"<b>🧠 Strategy:</b> {strategy_name}\n"
           f"<b>📉 Drawdown:</b> {drawdown_pct*100:.2f}%\n"
           f"<b>💼 Equity:</b> {equity:,.2f}")
    return send_telegram(msg, silent=False)
