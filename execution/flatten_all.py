#!/usr/bin/env python3
from __future__ import annotations
import os, time, hmac, hashlib, requests, argparse, math, sys
from typing import Dict, Any, List

BINANCE_TESTNET = str(os.getenv("BINANCE_TESTNET","1")).lower() in ("1","true","yes","on")
BASE = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"
AK   = os.environ.get("BINANCE_API_KEY","")
SK   = os.environ.get("BINANCE_API_SECRET","")
H    = {"X-MBX-APIKEY": AK} if AK else {}

def _sig(q: str) -> str: return hmac.new(SK.encode(), q.encode(), hashlib.sha256).hexdigest()
def _ts() -> int: return int(time.time()*1000)

def _get(path: str, signed: bool=False, **params) -> Any:
    if signed:
        params.update(timestamp=_ts(), recvWindow=5000)
        q = "&".join(f"{k}={v}" for k,v in params.items())
        url = f"{BASE}{path}?{q}&signature={_sig(q)}"
    else:
        url = f"{BASE}{path}"
    r = requests.get(url, headers=H, timeout=10); r.raise_for_status(); return r.json()

def _post(path: str, **params) -> Any:
    params.update(timestamp=_ts(), recvWindow=5000)
    q = "&".join(f"{k}={v}" for k,v in params.items())
    url = f"{BASE}{path}?{q}&signature={_sig(q)}"
    r = requests.post(url, headers=H, timeout=10)
    print("POST", path, r.status_code, r.text)
    r.raise_for_status()
    return r.json()

def _filters_step(symbol: str) -> float:
    info = _get("/fapi/v1/exchangeInfo")
    fs = next(s for s in info["symbols"] if s["symbol"]==symbol)["filters"]
    return float(next(f for f in fs if f["filterType"]=="LOT_SIZE")["stepSize"])

def _round_down(qty: float, step: float) -> float:
    return math.floor(qty/step)*step

def _is_dual_side() -> bool:
    j = _get("/fapi/v1/positionSide/dual", signed=True)
    return bool(j.get("dualSidePosition"))

def flatten(symbols: List[str], confirm: bool, only_side: str|None):
    dual = _is_dual_side()
    pr   = _get("/fapi/v2/positionRisk", signed=True)
    rows = [p for p in pr if abs(float(p.get("positionAmt") or 0))>0]
    if symbols:
        rows = [p for p in rows if p["symbol"] in symbols]
    if only_side:
        only_side = only_side.upper()
        rows = [p for p in rows if (p.get("positionSide","BOTH").upper()==only_side)]

    if not rows:
        print("No positions to close."); return 0

    # cache steps
    steps: Dict[str,float] = {}
    for sym in {p["symbol"] for p in rows}:
        try: steps[sym] = _filters_step(sym)
        except Exception: steps[sym] = 0.0

    n_sent = 0
    for p in rows:
        sym = p["symbol"]
        side_tag = p.get("positionSide","BOTH").upper()
        amt = abs(float(p["positionAmt"]))
        step = steps.get(sym, 0.0) or 0.0
        qty  = _round_down(amt, step) if step>0 else amt

        if step>0 and qty <= 0:
            print(f"SKIP {sym} {side_tag}: amt {amt} < step {step}")
            continue

        if dual and side_tag in ("LONG","SHORT"):
            # Close LONG with SELL/LONG ; Close SHORT with BUY/SHORT
            side = "SELL" if side_tag=="LONG" else "BUY"
            params = {"symbol": sym, "side": side, "type": "MARKET",
                      "positionSide": side_tag, "quantity": qty}
        else:
            # One-way: use opposite side to flatten
            side = "SELL" if float(p["positionAmt"])>0 else "BUY"
            params = {"symbol": sym, "side": side, "type": "MARKET", "quantity": qty}

        print(("DRY-RUN " if not confirm else "") + f"CLOSE {sym} {side_tag} qty={qty} side={side}")
        if confirm:
            try:
                _post("/fapi/v1/order", **params)
                n_sent += 1
            except requests.HTTPError as e:
                t = getattr(e, "response", None)
                print("ERROR", t.text if t is not None else str(e))
                # retry without reduceOnly if any (we don't set it, but keep this pattern)
    return n_sent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flatten hedge/one-way futures positions (panic close).")
    ap.add_argument("--all", action="store_true", help="Close all symbols with open positions.")
    ap.add_argument("--symbol", action="append", default=[], help="Symbol to close (repeatable).")
    ap.add_argument("--side", choices=["LONG","SHORT"], help="Only close this hedge side.")
    ap.add_argument("--confirm", action="store_true", help="Actually send orders. Default is dry-run.")
    args = ap.parse_args()

    if not (args.all or args.symbol):
        print("Specify --all or --symbol SYMBOL (repeatable).")
        sys.exit(2)

    sent = flatten(args.symbol, args.confirm, args.side)
    print(f"Done. Orders sent: {sent}")
