#!/usr/bin/env python3
"""
8h Telegram heartbeat: NAV, 24h delta, drawdown, open positions.
- Firestore-first (hedge/prod/state/nav, positions)
- Fallback to exchange (balances + positionRisk).
Read-only by design. Idempotent and safe to cron.
"""

from __future__ import annotations
import os, time, json, math, hmac, hashlib, requests, datetime as dt
from typing import Any, Dict, List, Optional

ENV = os.getenv("ENV", "prod")
RESERVE_BTC = 0.013
TELEGRAM_ENABLED = str(os.getenv("TELEGRAM_ENABLED","1")).lower() in ("1","true","yes","on")
BOT = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

BINANCE_TESTNET = str(os.getenv("BINANCE_TESTNET","1")).lower() in ("1","true","yes","on")
BASE_FUT = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"
AK = os.getenv("BINANCE_API_KEY","")
SK = os.getenv("BINANCE_API_SECRET","")

def _sig(q: str) -> str:
    return hmac.new(SK.encode(), q.encode(), hashlib.sha256).hexdigest()

def _get(path: str, signed: bool=False, **params) -> Any:
    if signed:
        params["timestamp"] = int(time.time()*1000)
        params["recvWindow"] = 5000
        q = "&".join(f"{k}={v}" for k,v in params.items())
        url = BASE_FUT + path + "?" + q + "&signature=" + _sig(q)
    else:
        url = BASE_FUT + path
    r = requests.get(url, headers={"X-MBX-APIKEY": AK} if AK else {}, timeout=10)
    r.raise_for_status()
    return r.json()

# --- Firestore (best effort) ---
def _fs_client():
    try:
        from google.cloud import firestore
        return firestore.Client()
    except Exception:
        return None

def _fs_doc(coll: str, doc: str) -> Optional[Dict[str,Any]]:
    cli = _fs_client()
    if not cli: return None
    try:
        path = f"hedge/{ENV}/state/{doc}"
        # supports nested collection path
        doc_ref = cli.document(path)
        snap = doc_ref.get()
        return snap.to_dict() if snap.exists else None
    except Exception:
        return None

def _nav_from_firestore() -> Optional[List[Dict[str,Any]]]:
    d = _fs_doc("state", "nav")  # path is already full in _fs_doc
    # tolerate both series field or list root
    if not d: return None
    if isinstance(d, dict) and "series" in d:
        return d["series"]
    if isinstance(d, dict) and "nav" in d:
        return d["nav"]
    if isinstance(d, list):
        return d
    return None

def _positions_from_firestore() -> Optional[List[Dict[str,Any]]]:
    d = _fs_doc("state", "positions")
    if isinstance(d, dict) and "rows" in d:
        return d["rows"]
    if isinstance(d, list):
        return d
    return None

# --- Fallback NAV from exchange ---
def _nav_from_exchange() -> float:
    bals = _get("/fapi/v2/balance", signed=True)
    poss = _get("/fapi/v2/positionRisk", signed=True)
    usdt = 0.0
    for b in bals:
        if b.get("asset") == "USDT":
            usdt += float(b.get("balance", 0) or 0)
            usdt += float(b.get("crossUnrealizedProfit", 0) or 0)
    # add unrealizedPnL of non-USDT assets as safety
    upnl = sum(float(p.get("unRealizedProfit", 0) or 0) for p in poss)
    return usdt + upnl

def _format_num(x: float, prec: int = 2) -> str:
    return f"{x:,.{prec}f}"

def _trend(delta_pct: float) -> str:
    if delta_pct > 0.25: return "↑"
    if delta_pct < -0.25: return "↓"
    return "→"

def build_message() -> str:
    # NAV series
    series = _nav_from_firestore()
    nav_now = None
    nav_24h = None
    if series:
        # assume items: {t:<epoch|iso>, nav: <float>} or {ts, value}
        def _ts(v):
            try:
                return float(v.get("t") or v.get("ts") or v.get("time"))
            except Exception:
                return None
        def _val(v):
            for k in ("nav","value","equity"):
                if k in v: return float(v[k])
            return None
        rows = [( _ts(x), _val(x)) for x in series if isinstance(x, dict)]
        rows = [(t,v) for t,v in rows if t and v is not None]
        rows.sort(key=lambda x: x[1] if x[0] is None else x[0])
        if rows:
            nav_now = rows[-1][1]
            # find ~24h ago
            cutoff = time.time() - 24*3600
            past = [v for (t,v) in rows if t and t <= cutoff]
            nav_24h = past[-1] if past else None

    if nav_now is None:
        nav_now = _nav_from_exchange()
    if nav_24h is None:
        nav_24h = nav_now  # best effort: flat baseline

    delta_abs = nav_now - nav_24h
    delta_pct = (delta_abs / nav_24h * 100.0) if nav_24h else 0.0

    # positions
    pos = _positions_from_firestore()
    if not pos:
        pos = _get("/fapi/v2/positionRisk", signed=True)
        pos = [
            {
                "symbol": p["symbol"],
                "side": p.get("positionSide","BOTH"),
                "qty": float(p.get("positionAmt",0) or 0),
                "uPnl": float(p.get("unRealizedProfit",0) or 0)
            }
            for p in pos if abs(float(p.get("positionAmt",0) or 0)) > 0
        ]
    pos.sort(key=lambda r: -abs(r.get("uPnl",0.0)))
    top = pos[:5]

    line_pos = "\n".join(
        f"• {r['symbol']:7} {r.get('side','?'):>5} {abs(r['qty']):g}  uPnL {_format_num(r['uPnl'], 2)}"
        for r in top
    ) or "• (none)"

    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M %Z")  # server tz

    msg = (
        f"HEDGE — 8h Update ({ENV})\n"
        f"Time: {when}\n\n"
        f"NAV: {_format_num(nav_now)} USDT  (Δ24h: {_format_num(delta_abs)}, {delta_pct:+.2f}%)\n"
        f"Trend: {_trend(delta_pct)}\n\n"
        f"Open Positions (top 5):\n{line_pos}\n\n"
        f"Notes:\n"
        f"• Reserve: {RESERVE_BTC} BTC (not in NAV)\n"
    )
    return msg

def send_telegram(msg: str) -> None:
    if not TELEGRAM_ENABLED or not BOT or not CHAT:
        return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT, "text": msg}, timeout=10)
    r.raise_for_status()

if __name__ == "__main__":
    try:
        send_telegram(build_message())
    except Exception as e:
        # never crash cron; print to stdout for /var/log/hedge-telebot.log
        print("heartbeat_error:", str(e))
