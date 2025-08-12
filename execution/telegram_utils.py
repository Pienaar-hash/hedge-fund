# execution/telegram_utils.py
# Telegram alert helpers for Hedge — NAV summaries, drawdown breach, sync/executor errors, and trade alerts

import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def send_telegram(message: str, silent: bool = False) -> bool:
    """Low-level sender. Returns False if creds missing or API error."""
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
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Telegram error {resp.status_code}: {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        return False


# ---------- Formatters ----------

def _fmt_usd(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def _fmt_ts(ts: Optional[str]) -> str:
    if ts:
        return ts
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def fmt_nav(nav: Dict[str, Any]) -> str:
    if not isinstance(nav, dict):
        return ""
    equity = float(nav.get("equity", 0))
    peak = float(nav.get("peak_equity", 0))
    dd = float(nav.get("drawdown", 0)) * 100.0
    balance = float(nav.get("balance", 0))
    realized = float(nav.get("realized", 0))
    unrealized = float(nav.get("unrealized", 0))
    ts = _fmt_ts(nav.get("timestamp"))

    return (
        f"<b>⏰ Time:</b> {ts}\n"
        f"<b>💼 Equity:</b> {_fmt_usd(equity)} | <b>Peak:</b> {_fmt_usd(peak)} | <b>DD:</b> {dd:.2f}%\n"
        f"<b>💰 Balance:</b> {_fmt_usd(balance)} | <b>🔺 Realized:</b> {_fmt_usd(realized)} | <b>🔸 Unrealized:</b> {_fmt_usd(unrealized)}"
    )


def fmt_trade(trade: Dict[str, Any]) -> str:
    ts = _fmt_ts(trade.get("timestamp"))
    symbol = trade.get("symbol", "?")
    side = trade.get("side", "?")
    price = trade.get("price", 0)
    qty = trade.get("qty")
    strategy = trade.get("strategy", "?")
    timeframe = trade.get("timeframe", "—")

    lines = [
        f"<b>⏰ Time:</b> {ts}",
        f"<b>📈 Symbol:</b> {symbol}",
        f"<b>🔁 Side:</b> {side}",
        f"<b>💵 Price:</b> {float(price):.2f} USDT",
        f"<b>🧠 Strategy:</b> {strategy}",
        f"<b>⏱ TF:</b> {timeframe}",
    ]
    if qty is not None:
        try:
            lines.insert(3, f"<b>💸 Qty:</b> {float(qty)}")
        except Exception:
            lines.insert(3, f"<b>💸 Qty:</b> {qty}")
    return "\n".join(lines)


# ---------- High-level helpers ----------

def send_trade_alert(trade: Dict[str, Any], nav: Optional[Dict[str, Any]] = None, silent: bool = False) -> bool:
    """Formats and sends a trade summary. Optionally append NAV metrics.

    Args:
        trade: dict with timestamp, symbol, side, price, qty, strategy, timeframe, etc.
        nav: optional dict with equity/peak/drawdown/balance/realized/unrealized
        silent: if True, sends without notification
    """
    header = "🚀 <b>Trade Executed</b>"
    body = fmt_trade(trade)
    tail = ""
    if nav:
        try:
            equity = float(nav.get("equity", 0))
            peak = float(nav.get("peak_equity", 0))
            dd = float(nav.get("drawdown", 0)) * 100.0
            tail = f"\n<b>💼 Equity:</b> {_fmt_usd(equity)} | <b>Peak:</b> {_fmt_usd(peak)} | <b>DD:</b> {dd:.2f}%"
        except Exception:
            pass
    msg = f"{header}\n{body}{tail}"
    return send_telegram(msg, silent=silent)


def send_nav_summary(nav: Dict[str, Any], silent: bool = True) -> bool:
    """Compact NAV heartbeat."""
    header = "📊 <b>NAV Update</b>"
    body = fmt_nav(nav)
    return send_telegram(f"{header}\n{body}", silent=silent)


def send_dd_breach(drawdown: float, equity: float) -> bool:
    """Alert when drawdown exceeds configured threshold."""
    try:
        dd_pct = float(drawdown) * 100.0
        eq_txt = _fmt_usd(float(equity))
    except Exception:
        dd_pct = 0.0
        eq_txt = _fmt_usd(0.0)
    msg = (
        "⚠️ <b>Drawdown Breach</b>\n"
        f"<b>DD:</b> {dd_pct:.2f}%\n"
        f"<b>💼 Equity:</b> {eq_txt}"
    )
    return send_telegram(msg, silent=False)


def send_sync_error(msg_text: str) -> bool:
    """Soft-notify on Firestore sync failures."""
    ts = _fmt_ts(None)
    msg = (
        "❌ <b>Sync Failure</b>\n"
        f"<b>⏰ Time:</b> {ts}\n"
        f"<code>{msg_text}</code>"
    )
    return send_telegram(msg, silent=True)


def send_executor_error(msg_text: str) -> bool:
    """Notify on executor crashes or unexpected exceptions."""
    ts = _fmt_ts(None)
    msg = (
        "💥 <b>Executor Error</b>\n"
        f"<b>⏰ Time:</b> {ts}\n"
        f"<code>{msg_text}</code>"
    )
    return send_telegram(msg, silent=False)
