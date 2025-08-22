from __future__ import annotations
import os, time, json, math, hmac, hashlib, requests, traceback
from typing import Dict, Any, Optional
from execution.signal_screener import generate_signals_from_config

# ===== ENV / BASES =====
BINANCE_TESTNET = str(os.getenv("BINANCE_TESTNET","1")).strip().lower() in ("1","true","yes","on")
API_KEY  = os.environ["BINANCE_API_KEY"]
API_SEC  = os.environ["BINANCE_API_SECRET"]
BASE_FUT = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

HEARTBEAT_SECS = float(os.getenv("HEARTBEAT_SECS","30"))
SLEEP_SECS     = float(os.getenv("EXECUTOR_LOOP_SLEEP","5"))

# ===== HTTP helpers =====
def _sig(q: str) -> str:
    return hmac.new(API_SEC.encode(), q.encode(), hashlib.sha256).hexdigest()

def _get(path: str, params: Dict[str, Any] | None = None, signed: bool = False) -> Any:
    params = dict(params or {})
    if signed:
        params["timestamp"] = int(time.time()*1000)
        q = "&".join(f"{k}={v}" for k,v in params.items())
        params["signature"] = _sig(q)
    r = requests.get(BASE_FUT + path, params=params, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    r.raise_for_status()
    return r.json()

def _post(path: str, params: Dict[str, Any]) -> Any:
    params = dict(params)
    params["timestamp"] = int(time.time()*1000)
    q = "&".join(f"{k}={v}" for k,v in params.items())
    url = BASE_FUT + path + "?" + q + "&signature=" + _sig(q)
    r = requests.post(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    print("[executor] ORDER_REQ", path, r.status_code, r.text, flush=True)
    r.raise_for_status()
    return r.json()

# ===== Exchange info / filters =====
_EXINFO: Optional[dict] = None
_EXINFO_TS = 0.0
def _exchange_info(ttl: float = 300.0) -> dict:
    global _EXINFO,_EXINFO_TS
    now = time.time()
    if _EXINFO is None or (now - _EXINFO_TS) > ttl:
        _EXINFO = _get("/fapi/v1/exchangeInfo")
        _EXINFO_TS = now
    return _EXINFO

def _filters_for(symbol: str) -> dict:
    d = next(s for s in _exchange_info()["symbols"] if s["symbol"] == symbol)
    return {f["filterType"]: f for f in d["filters"]}

def _ticker_price(symbol: str) -> float:
    return float(_get("/fapi/v1/ticker/price", {"symbol": symbol})["price"])

# ===== Mode detection =====
def _is_dual_side() -> bool:
    try:
        j = _get("/fapi/v1/positionSide/dual", signed=True)
        return bool(j.get("dualSidePosition"))
    except Exception:
        return False

# ===== Sizing & rounding =====
def _round_step(q: float, step: float) -> float:
    return math.floor(q / step) * step

def _ensure_filters(symbol: str, qty: float, px: float) -> float:
    f = _filters_for(symbol)
    step = float(f["LOT_SIZE"]["stepSize"])
    min_q = float(f["LOT_SIZE"]["minQty"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("notional", 0.0))
    qty = _round_step(qty, step)
    if qty < min_q:
        qty = min_q
    if min_notional > 0 and qty * px < min_notional:
        need = min_notional / max(px, 1e-12)
        qty = math.ceil(need / step) * step
    return _round_step(qty, step)

def _map_intent_to_order(intent: dict, dual: bool) -> dict:
    sym   = intent["symbol"]
    side  = "BUY" if intent["signal"] == "BUY" else "SELL"
    red   = bool(intent.get("reduceOnly", False))
    cap   = float(intent.get("capital_per_trade", 0.0))
    lev   = float(intent.get("leverage", 1.0))
    px    = _ticker_price(sym)
    notional = max(1.0, cap * lev)
    qty   = _ensure_filters(sym, notional / px, px)

    params = {"symbol": sym, "side": side, "type": "MARKET", "quantity": qty}
    if dual:
        # OPEN: BUY->LONG, SELL->SHORT; CLOSE: SELL closes LONG, BUY closes SHORT
        if red:
            params["positionSide"] = "LONG" if side == "SELL" else "SHORT"
            params["reduceOnly"]   = "true"
        else:
            params["positionSide"] = "LONG" if side == "BUY" else "SHORT"
    else:
        if red:
            params["reduceOnly"] = "true"

    print("[executor] SEND_ORDER", {"symbol": sym, "side": side, "qty": qty, "px": px,
                                   "dual": dual, "reduceOnly": red, "params": params}, flush=True)
    return params

# ===== Account heartbeat =====
_last_hb = 0.0
def _heartbeat():
    global _last_hb
    now = time.time()
    if now - _last_hb < HEARTBEAT_SECS:
        return
    try:
        bals = _get("/fapi/v2/balance", signed=True)
        syms = sorted([b["asset"] for b in bals if float(b.get("balance",0)) or float(b.get("crossWalletBalance",0))])
        poss = _get("/fapi/v2/positionRisk", signed=True)
        has  = [p for p in poss if abs(float(p.get("positionAmt", "0"))) > 0]
        print(f"[executor] account OK — futures=True testnet={BINANCE_TESTNET} dry_run=False balances: {syms} positions: {len(has)}", flush=True)
    except Exception as e:
        print("[executor] heartbeat error", str(e), flush=True)
    _last_hb = now

# ===== Main loop =====
def main():
    print("[executor] launch dualSide=", _is_dual_side(), "testnet=", BINANCE_TESTNET, flush=True)
    while True:
        try:
            _heartbeat()
            for intent in (generate_signals_from_config() or []):
                print("[executor] INTENT " + json.dumps(intent, sort_keys=True), flush=True)
                try:
                    dual = _is_dual_side()  # re-check each send (safe on testnet)
                    params = _map_intent_to_order(intent, dual)
                    _post("/fapi/v1/order", params)
                except Exception as e:
                    print("[executor] ORDER_ERR", str(e), "trace:", traceback.format_exc().splitlines()[-1], flush=True)
            time.sleep(SLEEP_SECS)
        except Exception as loop_err:
            print("[executor] LOOP_ERR", str(loop_err), flush=True)
            time.sleep(2)

if __name__ == "__main__":
    main()
