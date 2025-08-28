from __future__ import annotations
# execution/telegram_utils.py — Phase 4.1

import os
import time
from datetime import datetime
from typing import List

try:
    import requests
except Exception:
    requests = None  # handled below


# --- Env helpers ---
def _b(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "on")

def _env():
    return {
        "enabled": _b(os.getenv("TELEGRAM_ENABLED", "0")),
        "token": os.getenv("BOT_TOKEN", "").strip(),
        "chat":  os.getenv("CHAT_ID", "").strip(),
    }

def _utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# --- Core send ---
def send_telegram(message: str, silent: bool = False) -> bool:
    env = _env()
    if not env["enabled"]:
        print("❌ Telegram disabled (TELEGRAM_ENABLED!=1).", flush=True)
        return False
    if not env["token"] or not env["chat"]:
        print(f"❌ Telegram missing creds (BOT_TOKEN len={len(env['token'])}, CHAT_ID set={bool(env['chat'])}).", flush=True)
        return False
    if requests is None:
        print("❌ Telegram cannot import requests.", flush=True)
        return False
    try:
        url = f"https://api.telegram.org/bot{env['token']}/sendMessage"
        payload = {
            "chat_id": env["chat"],
            "text": f"{_utc()}\n{message}",
            "disable_notification": bool(silent),
        }
        r = requests.post(url, json=payload, timeout=15)
        if r.ok:
            print("✅ Telegram message sent.", flush=True)
            return True
        print(f"❌ Telegram send failed [{r.status_code}]: {r.text}", flush=True)
        return False
    except Exception as e:
        print(f"❌ Telegram send error: {e}", flush=True)
        return False


# --- Cadence / rate limiting ---
_last_summary_ts: float | None = None
_last_dd_ts: float | None = None

def should_send_summary(last_sent_ts: float | None, minutes: int) -> bool:
    now = time.time()
    if not last_sent_ts:
        return True
    return (now - float(last_sent_ts)) >= max(60, minutes * 60)


# --- Message helpers used by executor_live.py ---
def send_heartbeat(equity: float, peak: float, dd_pct: float, realized: float, unrealized: float, positions_top: List[str]):
    msg = (
        f"Heartbeat\n"
        f"Equity: {equity:,.2f} | Peak: {peak:,.2f} | DD: {dd_pct*100:+.2f}%\n"
        f"PnL — R: {realized:,.2f} | U: {unrealized:,.2f}\n"
        f"Top: {', '.join(positions_top) if positions_top else '—'}"
    )
    send_telegram(msg, silent=True)

def send_trade_alert(symbol: str, side: str, qty: float, fill_price: float, realized: float, unrealized: float):
    msg = (
        f"🔔 {symbol} {side} {qty:g} @ {fill_price:,.2f}\n"
        f"R: {realized:,.2f} | U: {unrealized:,.2f}"
    )
    send_telegram(msg, silent=False)

def send_drawdown_alert(drawdown_pct: float, threshold_pct: float, peak_equity: float, equity: float):
    global _last_dd_ts
    now = time.time()
    # rate‑limit to once per 15 minutes
    if _last_dd_ts and (now - _last_dd_ts) < 15 * 60:
        return
    _last_dd_ts = now
    msg = (
        f"⚠️ Drawdown Alert\n"
        f"DD: {drawdown_pct*100:.2f}% (thr={threshold_pct*100:.2f}%)\n"
        f"Equity: {equity:,.2f} | Peak: {peak_equity:,.2f}"
    )
    send_telegram(msg, silent=False)


# --- CLI smoke test ---
if __name__ == "__main__":
    ok = send_telegram("🚀 executor/telegram_utils.py smoke: hello.")
    print("send_ok:", ok)
