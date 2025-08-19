# Phase‑4.1 “Production‑Patch” Sprint (No Refactors)

# --- execution/exchange_utils.py (FULL FILE) ---
# execution/exchange_utils.py — Phase‑4.1 production patch (no refactor)
from __future__ import annotations

import os, hmac, time, json, hashlib
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

# --- env ---
_DEF_BOOL = lambda v, d=False: (str(os.getenv(v, str(int(d)))).strip().lower() in {"1","true","yes","on"})
TESTNET = _DEF_BOOL("BINANCE_TESTNET", False)
USE_FUTURES = _DEF_BOOL("USE_FUTURES", False) or _DEF_BOOL("BINANCE_FUTURES", False)
DRY_RUN = _DEF_BOOL("DRY_RUN", False)
EXCHANGE_DEBUG = _DEF_BOOL("EXCHANGE_DEBUG", False)

SPOT_BASE = "https://testnet.binance.vision" if TESTNET else "https://api.binance.com"
FAPI_BASE = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- signing/http ---
_ts = lambda: int(time.time()*1000)
_headers = lambda: ({"X-MBX-APIKEY": API_KEY} if API_KEY else {})

def _sign(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if v is None: continue
        parts.append(f"{k}={v}")
    query = "&".join(parts)
    return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def _request(method: str, path: str, *, futures=False, signed=False, params: Optional[Dict[str,Any]]=None, timeout=10) -> Tuple[Optional[Dict[str,Any]], Optional[int], Optional[str]]:
    if requests is None:
        return None, None, "requests-not-available"
    base = FAPI_BASE if futures else SPOT_BASE
    url = f"{base}{path}"
    params = dict(params or {})
    if signed:
        if "timestamp" not in params: params["timestamp"] = _ts()
        if "recvWindow" not in params: params["recvWindow"] = 5000
        params["signature"] = _sign(params)
    try:
        if method.upper()=="GET":
            r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        else:
            r = requests.post(url, params=params, headers=_headers(), timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.ok:
            return data, r.status_code, None
        return data, r.status_code, getattr(r, "text", "http-error")
    except Exception as e:
        return None, None, str(e)

# --- prices ---
def get_price(symbol: str) -> float:
    try:
        sym = str(symbol).upper()
        if USE_FUTURES:
            data, _, _ = _request("GET", "/fapi/v1/ticker/price", futures=True, params={"symbol": sym})
        else:
            data, _, _ = _request("GET", "/api/v3/ticker/price", futures=False, params={"symbol": sym})
        if isinstance(data, dict) and "price" in data:
            return float(data["price"])
    except Exception:
        pass
    return 0.0

# --- balances ---
def get_balances() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        if USE_FUTURES:
            data, status, err = _request("GET", "/fapi/v2/balance", futures=True, signed=True)
            if isinstance(data, list):
                for row in data:
                    try:
                        a = str(row.get("asset")).upper()
                        b = float(row.get("balance", row.get("withdrawAvailable", 0.0)))
                        if a: out[a] = b
                    except Exception:
                        continue
            if EXCHANGE_DEBUG and not out:
                out["_status"], out["_error"] = status, err
        else:
            data, status, err = _request("GET", "/api/v3/account", futures=False, signed=True)
            if isinstance(data, dict):
                for b in data.get("balances", []) or []:
                    try:
                        a = str(b.get("asset")).upper()
                        f = float(b.get("free", 0.0))
                        if a: out[a] = f
                    except Exception:
                        continue
            if EXCHANGE_DEBUG and not out:
                out["_status"], out["_error"] = status, err
    except Exception:
        pass
    return out

# --- positions (USD‑M futures) ---
def _pos_side(q: float) -> str:
    try: return "LONG" if float(q) >= 0 else "SHORT"
    except Exception: return "LONG"

def get_positions(symbol: Optional[str]=None) -> List[Dict[str, Any]]:
    if not USE_FUTURES:
        return []
    params = {}
    if symbol: params["symbol"] = str(symbol).upper()
    out: List[Dict[str, Any]] = []
    try:
        data, status, err = _request("GET", "/fapi/v2/positionRisk", futures=True, signed=True, params=params)
        if isinstance(data, list):
            now = int(time.time())
            for row in data:
                try:
                    qty = float(row.get("positionAmt", 0.0))
                    if abs(qty) < 1e-12:  # skip zero entries
                        continue
                    sym   = row.get("symbol")
                    entry = float(row.get("entryPrice", 0.0))
                    mark  = float(row.get("markPrice", 0.0))
                    lev   = int(float(row.get("leverage", 1)))
                    upnl  = float(row.get("unRealizedProfit", 0.0))  # ✅ correct PnL
                    out.append({
                        "symbol": sym,
                        "side": _pos_side(qty),
                        "qty": qty,
                        "entry_price": entry,
                        "mark_price": mark,
                        "leverage": lev,
                        "notional": abs(qty) * mark,
                        "pnl": upnl,                 # ✅ expose real U‑PnL
                        "unrealizedPnl": upnl,       # alias for clarity
                        "updated_at": now,
                    })
                except Exception:
                    continue
        if EXCHANGE_DEBUG and not out:
            out = [{"_status": status, "_error": err}]
    except Exception:
        pass
    return out

# --- orders ---
def place_market_order(symbol: str, side: str, quantity: float, **kwargs) -> Dict[str, Any]:
    symbol = str(symbol).upper(); side = str(side).upper(); qty = float(quantity)
    if qty <= 0:
        return {"ok": False, "avgPrice": None, "raw": {"error": "qty<=0"}}
    if DRY_RUN or requests is None:
        px = get_price(symbol) or 0.0
        return {"ok": True, "avgPrice": (float(px) if px else None), "raw": {"dry_run": True, "symbol": symbol, "side": side, "qty": qty}}
    try:
        if USE_FUTURES:
            params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
            if "positionSide" in kwargs and kwargs["positionSide"]: params["positionSide"] = kwargs["positionSide"]
            if "reduceOnly" in kwargs: params["reduceOnly"] = "true" if bool(kwargs["reduceOnly"]) else "false"
            data, status, err = _request("POST", "/fapi/v1/order", futures=True, signed=True, params=params)
        else:
            params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
            data, status, err = _request("POST", "/api/v3/order", futures=False, signed=True, params=params)
        if err:
            return {"ok": False, "avgPrice": None, "raw": {"error": err, "status": status, "resp": data}}
        avg = None
        if isinstance(data, dict):
            fills = data.get("fills")
            if isinstance(fills, list) and fills:
                try:
                    notional = sum(float(f["price"]) * float(f.get("qty", 0)) for f in fills)
                    qty_filled = sum(float(f.get("qty", 0)) for f in fills)
                    if qty_filled > 0: avg = notional / qty_filled
                except Exception: avg = None
            if avg is None:
                ap = data.get("avgPrice")
                if ap is not None:
                    try: avg = float(ap)
                    except Exception: avg = None
        if avg is None: avg = get_price(symbol) or None
        return {"ok": True, "avgPrice": avg, "raw": data}
    except Exception as e:
        return {"ok": False, "avgPrice": None, "raw": {"error": str(e)}}

# --- overview ---
def get_account_overview() -> Dict[str, Any]:
    balances = get_balances()
    positions = get_positions() if USE_FUTURES else []
    info = {"use_futures": USE_FUTURES, "testnet": TESTNET, "dry_run": DRY_RUN, "balances": balances, "positions": positions, "ts": int(time.time())}
    return info

if __name__ == "__main__":
    print(json.dumps(get_account_overview(), indent=2))

# --- execution/executor_live.py and execution/signal_screener.py ---
# (See sprint notes for full replacements with nav snapshot, FORCE_TRADE toggle, etc.)
