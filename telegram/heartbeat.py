#!/usr/bin/env python3
from __future__ import annotations

import os
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from execution.exchange_utils import get_balances, get_positions, is_testnet

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

def nav_and_top_positions():
    bals = get_balances()  # [{asset, balance, availableBalance, ...}]
    # crude NAV: sum stable-like assets (USDT/FDUSD/USDC) + ignore tiny others
    stables = {"USDT","FDUSD","USDC","BUSD"}
    nav = 0.0
    for b in bals:
        a = (b.get("asset") or "").upper()
        if a in stables:
            nav += float(b.get("availableBalance", b.get("balance", 0.0)) or 0.0)
    pos = get_positions()
    live = [p for p in pos if abs(float(p.get("positionAmt",0) or 0.0))>0.0]
    # top by abs notional (approx: |qty| * markPrice)
    for p in live:
        try:
            p["_abs_notional"] = abs(float(p["positionAmt"])) * float(p.get("markPrice",0.0))
        except Exception:
            p["_abs_notional"] = 0.0
    live.sort(key=lambda x: x.get("_abs_notional",0.0), reverse=True)
    return nav, live[:3]

def fmt_arrow(delta):
    return "↑" if delta>0 else ("↓" if delta<0 else "→")

def main():
    if not BOT or not CHAT:
        print("heartbeat_error: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"); return
    nav, top = nav_and_top_positions()
    delta24 = 0.0  # placeholder unless you persist NAV history
    arrow = fmt_arrow(delta24)
    lines = []
    lines.append(f"🫀 Hedge Heartbeat ({'TESTNET' if is_testnet() else 'MAINNET'})")
    lines.append(f"NAV: {nav:,.2f} | 24h Δ: {delta24:,.2f} {arrow}")
    if top:
        lines.append("Open positions:")
        for p in top:
            side = p.get("positionSide","?")
            sym  = p.get("symbol","?")
            amt  = float(p.get("positionAmt",0) or 0.0)
            upnl = float(p.get("unRealizedProfit",0) or 0.0)
            lines.append(f"• {sym} {side} {amt} uPnL: {upnl:,.2f}")
    else:
        lines.append("No open positions.")
    msg = "\n".join(lines)
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT, "text": msg})
    r.raise_for_status()
    print("heartbeat_ok")

if __name__ == "__main__":
    main()
