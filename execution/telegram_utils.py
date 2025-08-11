# execution/telegram_utils.py

import os
import requests
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(message: str, silent: bool = False) -> bool:
    """
    Sends a message to the Telegram bot if credentials are available.

    Args:
        message (str): The message content.
        silent (bool): If True, disables notification.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram BOT_TOKEN or CHAT_ID not set.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "disable_notification": silent,
        "parse_mode": "HTML",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"❌ Telegram error {response.status_code}: {response.text}")
            return False
        return True
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        return False

def send_trade_alert(trade: dict, silent: bool = False) -> bool:
    """
    Formats and sends a trade summary to Telegram.

    Args:
        trade (dict): Trade metadata including timestamp, symbol, price, qty, etc.
        silent (bool): If True, disables notification.

    Returns:
        bool: Success status
    """
    ts = trade.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    symbol = trade.get("symbol", "?")
    side = trade.get("side", "?")
    price = trade.get("price", 0)
    qty = trade.get("qty", 0)
    strategy = trade.get("strategy", "?")

    msg = (
        f"🚀 <b>Trade Executed</b>\n"
        f"<b>⏰ Time:</b> {ts}\n"
        f"<b>📈 Symbol:</b> {symbol}\n"
        f"<b>🔁 Side:</b> {side}\n"
        f"<b>💸 Qty:</b> {qty}\n"
        f"<b>💰 Price:</b> {price:.2f} USDT\n"
        f"<b>🧠 Strategy:</b> {strategy}"
    )
    return send_telegram(msg, silent=silent)
