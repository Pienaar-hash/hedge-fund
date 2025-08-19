# execution/exchange_utils.py
from __future__ import annotations

import os, hmac, time, json, hashlib
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

TESTNET = _bool_env("BINANCE_TESTNET", False)
USE_FUTURES = _bool_env("USE_FUTURES", False) or _bool_env("BINANCE_FUTURES", False)
DRY_RUN = _bool_env("DRY_RUN", False)
EXCHANGE_DEBUG = _bool_env("EXCHANGE_DEBUG", False)

SPOT_BASE = "https://testnet.binance.vision" if TESTNET else "https://api.binance.com"
FAPI_BASE = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

def _ts() -> int:
    return int(time.time() * 1000)

def _headers() -> Dict[str, str]:
    return {"X-MBX-APIKEY": API_KEY} if API_KEY else {}

def _sign(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    query = "&".join(parts)
    return hmac.new(API_SECRET.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

def _request(
    method: str,
    path: str,
    *,
    futures: bool = False,
    signed: bool = False,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    if requests is None:
        return None, None, "requests-not-available"

    base = FAPI_BASE if futures else SPOT_BASE
    url = f"{base}{path}"
    params = dict(params or {})

    if signed:
        if "timestamp" not in params:
            params["timestamp"] = _ts()
        if "recvWindow" not in params:
            params["recvWindow"] = 5000
        params["signature"] = _sign(params)

    try:
        m = method.upper()
        if m == "GET":
            r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        elif m == "POST":
            r = requests.post(url, params=params, headers=_headers(), timeout=timeout)
        else:
            r = requests.request(m, url, params=params, headers=_headers(), timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.ok:
            return data, r.status_code, None
        return data, r.status_code, getattr(r, "text", "http-error")
    except Exception as e:
        return None, None, str(e)

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

def get_balances() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        if USE_FUTURES:
            # primary
            data, status, err = _request("GET", "/fapi/v2/balance", futures=True, signed=True)
            if isinstance(data, list) and data:
                for row in data:
                    try:
                        asset = row.get("asset")
                        bal = float(row.get("balance", row.get("withdrawAvailable", 0.0)))
                        if asset:
                            out[str(asset).upper()] = bal
                    except Exception:
                        continue
            # fallback
            if not out:
                acct, status2, err2 = _request("GET", "/fapi/v2/account", futures=True, signed=True)
                if isinstance(acct, dict):
                    for a in acct.get("assets", []) or []:
                        try:
                            asset = a.get("asset")
                            bal = float(a.get("walletBalance", 0.0))
                            if asset:
                                out[str(asset).upper()] = bal
                        except Exception:
                            continue
                if EXCHANGE_DEBUG and not out:
                    out["_status"] = status2
                    out["_error"] = err2
            elif EXCHANGE_DEBUG and not out:
                out["_status"] = status
                out["_error"] = err
        else:
            data, status, err = _request("GET", "/api/v3/account", futures=False, signed=True)
            if isinstance(data, dict):
                for b in data.get("balances", []) or []:
                    try:
                        asset = b.get("asset")
                        free = float(b.get("free", 0.0))
                        if asset:
                            out[str(asset).upper()] = free
                    except Exception:
                        continue
            if EXCHANGE_DEBUG and not out:
                out["_status"] = status
                out["_error"] = err
    except Exception:
        pass
    return out

def _pos_side(qty: float) -> str:
    try:
        return "LONG" if float(qty) >= 0 else "SHORT"
    except Exception:
        return "LONG"

def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if not USE_FUTURES:
        return []
    out: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {}
    if symbol:
        params["symbol"] = str(symbol).upper()
    try:
        data, status, err = _request("GET", "/fapi/v2/positionRisk", futures=True, signed=True, params=params)
        if isinstance(data, list):
            now = int(time.time())
            for row in data:
                try:
                    sym = row.get("symbol")
                    qty = float(row.get("positionAmt", 0.0))
                    if abs(qty) < 1e-12:
                        continue
                    entry = float(row.get("entryPrice", 0.0))
                    mark = float(row.get("markPrice", 0.0))
                    lev = int(float(row.get("leverage", 1)))
                    pnl = float(row.get("unRealizedProfit", 0.0))
                    out.append({
                        "symbol": sym,
                        "side": _pos_side(qty),
                        "qty": qty,
                        "entry_price": entry,
                        "mark_price": mark,
                        "leverage": lev,
                        "notional": abs(qty) * mark,
                        "pnl": pnl,
                        "updated_at": now,
                    })
                except Exception:
                    continue
        if EXCHANGE_DEBUG and not out:
            out = [{"_status": status, "_error": err}]
    except Exception:
        pass
    return out

def place_market_order(symbol: str, side: str, quantity: float, **kwargs) -> Dict[str, Any]:
    symbol = str(symbol).upper()
    side = str(side).upper()
    qty = float(quantity)
    if qty <= 0:
        return {"ok": False, "avgPrice": None, "raw": {"error": "qty<=0"}}

    if DRY_RUN or requests is None:
        px = get_price(symbol) or 0.0
        return {"ok": True, "avgPrice": float(px) if px else None, "raw": {"dry_run": True, "symbol": symbol, "side": side, "qty": qty}}

    try:
        if USE_FUTURES:
            params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
            if "positionSide" in kwargs and kwargs["positionSide"]:
                params["positionSide"] = kwargs["positionSide"]
            if "reduceOnly" in kwargs:
                params["reduceOnly"] = "true" if bool(kwargs["reduceOnly"]) else "false"
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
                    if qty_filled > 0:
                        avg = notional / qty_filled
                except Exception:
                    avg = None
            if avg is None:
                try:
                    ap = data.get("avgPrice")
                    if ap is not None:
                        avg = float(ap)
                except Exception:
                    avg = None
        if avg is None:
            avg = get_price(symbol) or None
        return {"ok": True, "avgPrice": avg, "raw": data}
    except Exception as e:
        return {"ok": False, "avgPrice": None, "raw": {"error": str(e)}}

def get_account_overview() -> Dict[str, Any]:
    balances = get_balances()
    positions = get_positions() if USE_FUTURES else []
    hint = None
    if USE_FUTURES and TESTNET and (not balances):
        hint = "Futures Testnet: empty balances. Check key permissions, IP whitelist, and USD‑M wallet funding."
    if USE_FUTURES and TESTNET and positions and isinstance(positions[0], dict) and positions[0].get("_error"):
        hint = f"Position fetch error: {positions[0].get('_error')}"
    info = {
        "use_futures": USE_FUTURES,
        "testnet": TESTNET,
        "dry_run": DRY_RUN,
        "balances": balances,
        "positions": positions,
        "ts": int(time.time()),
    }
    if EXCHANGE_DEBUG and hint:
        info["_hint"] = hint
    return info

if __name__ == "__main__":
    print(json.dumps(get_account_overview(), indent=2))
