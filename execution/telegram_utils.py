# execution/telegram_utils.py

import os
import requests
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TELEGRAM_ENABLED = str(os.getenv("TELEGRAM_ENABLED", "0")).lower() in ("1", "true", "yes")
EXECUTOR_LABEL = os.getenv("EXECUTOR_LABEL", "executor")

def _fmt_float(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return None

def send_telegram(message: str, silent: bool = False) -> bool:
    """Send a raw Telegram message if enabled."""
    if not TELEGRAM_ENABLED:
        print("ℹ️ Telegram disabled (TELEGRAM_ENABLED=0).")
        return False
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
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"❌ Telegram error {r.status_code}: {r.text}")
            return False
        return True
    except Exception as e:
        print(f"❌ Telegram exception: {e}")
        return False

def send_trade_alert(trade: dict, silent: bool = False) -> bool:
    """Formats and sends a trade summary to Telegram."""
    ts = trade.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    symbol = trade.get("symbol", "?")
    side = trade.get("side", "?")
    price = _fmt_float(trade.get("price"))
    qty = _fmt_float(trade.get("qty"), nd=8)  # show more precision for qty
    strategy = trade.get("strategy", "?")

    realized = trade.get("realized")
    unrealized = trade.get("unrealized")
    equity = trade.get("equity")
    dd = trade.get("drawdown_pct")
    sdd = trade.get("strategy_drawdown_pct")

    lines = [
        f"🧰 <b>{EXECUTOR_LABEL}</b>",
        "🚀 <b>Trade Executed</b>",
        f"<b>⏰ Time:</b> {ts}",
        f"<b>📈 Symbol:</b> {symbol}",
        f"<b>🔁 Side:</b> {side}",
        f"<b>💸 Qty:</b> {qty or '—'}",
        f"<b>💰 Price:</b> {price or '—'} USDT",
        f"<b>🧠 Strategy:</b> {strategy}",
    ]

    extras = []
    rf = _fmt_float(realized);   uf = _fmt_float(unrealized);  ef = _fmt_float(equity, nd=2)
    if rf is not None: extras.append(f"Realized: {rf}")
    if uf is not None: extras.append(f"Unrealized: {uf}")
    if ef is not None: extras.append(f"Equity: {ef}")
    try:
        if dd is not None: extras.append(f"DD: {float(dd)*100:.2f}%")
    except Exception:
        pass
    try:
        if sdd is not None: extras.append(f"Strategy DD: {float(sdd)*100:.2f}%")
    except Exception:
        pass
    if extras:
        lines.append("<b>📊 PnL/NAV:</b> " + " | ".join(extras))

    return send_telegram("\n".join(lines), silent=silent)

def send_drawdown_alert(strategy_name: str, drawdown_pct: float, equity: float) -> bool:
    """Send a drawdown breach alert."""
    eq = _fmt_float(equity, nd=2) or "—"
    try:
        ddp = f"{float(drawdown_pct)*100:.2f}%"
    except Exception:
        ddp = "—"
    lines = [
        f"🧰 <b>{EXECUTOR_LABEL}</b>",
        "⚠️ <b>Drawdown Alert</b>",
        f"<b>🧠 Strategy:</b> {strategy_name}",
        f"<b>📉 Drawdown:</b> {ddp}",
        f"<b>💼 Equity:</b> {eq}",
    ]
    return send_telegram("\n".join(lines), silent=False)
