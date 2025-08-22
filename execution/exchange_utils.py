from __future__ import annotations
import os, time, hmac, hashlib, math, requests
from typing import Any, Dict, List, Optional

# ===== ENV / BASES =====
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","on")

BINANCE_TESTNET = _env_bool("BINANCE_TESTNET", True)
API_KEY  = os.environ["BINANCE_API_KEY"]
API_SEC  = os.environ["BINANCE_API_SECRET"]

BASE_FUT = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

# Reuse a single Session
_SES = requests.Session()
_SES.headers.update({"X-MBX-APIKEY": API_KEY})

# ===== Signing / Request =====
def _sig(q: str) -> str:
    return hmac.new(API_SEC.encode(), q.encode(), hashlib.sha256).hexdigest()

def _request(method: str, path: str, *, futures: bool = True, signed: bool = False,
             params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, timeout: int = 10):
    base = BASE_FUT if futures else "https://api.binance.com"
    params = dict(params or {})
    if signed:
        params["timestamp"] = int(time.time()*1000)
        q = "&".join(f"{k}={v}" for k,v in params.items())
        params["signature"] = _sig(q)
    url = base + path
    try:
        r = _SES.request(method.upper(), url, params=params, json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        # Return structured error so callers can decide
        try:
            return {"error": True, "status": r.status_code, "body": r.text}
        except Exception:
            return {"error": True, "status": 0, "body": str(e)}
    except Exception as e:
        return {"error": True, "status": 0, "body": str(e)}

# ===== Public data =====
def get_price(symbol: str) -> Optional[float]:
    """Return last price for USD-M futures symbol (e.g., 'SOLUSDT')."""
    j = _request("GET", "/fapi/v1/ticker/price", params={"symbol": str(symbol).upper()})
    try:
        if isinstance(j, dict) and "price" in j:
            return float(j["price"])
    except Exception:
        pass
    return None

def get_klines(symbol: str, interval: str, limit: int = 200) -> List[tuple]:
    """
    Return list of (openTime_ms, closePrice_float) for USD-M futures.
    """
    j = _request("GET", "/fapi/v1/klines", params={"symbol": str(symbol).upper(),"interval": interval,"limit": int(limit)})
    out: List[tuple] = []
    try:
        if isinstance(j, list):
            out = [(int(row[0]), float(row[4])) for row in j]  # openTime, close
    except Exception:
        pass
    return out

# ===== Account / positions =====
def _positions_raw() -> List[Dict[str, Any]]:
    j = _request("GET", "/fapi/v2/positionRisk", signed=True)
    return j if isinstance(j, list) else []

def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns simplified open positions. Each: {symbol, qty, entryPrice, unrealized, leverage, positionSide}
    qty sign indicates direction for one-way accounts; in Hedge mode, positionSide is authoritative.
    """
    out: List[Dict[str, Any]] = []
    for p in _positions_raw():
        try:
            sym = str(p.get("symbol",""))
            if symbol and sym != symbol: 
                continue
            amt = float(p.get("positionAmt","0"))
            if abs(amt) < 1e-12:
                continue
            out.append({
                "symbol": sym,
                "qty": amt,
                "entryPrice": float(p.get("entryPrice","0") or 0),
                "unrealized": float(p.get("unRealizedProfit","0") or 0),
                "leverage": float(p.get("leverage","0") or 0),
                "positionSide": p.get("positionSide")  # LONG|SHORT for Hedge, "BOTH" for one-way
            })
        except Exception:
            continue
    return out

def get_balances() -> List[Dict[str, Any]]:
    j = _request("GET", "/fapi/v2/balance", signed=True)
    return j if isinstance(j, list) else []

def _is_dual_side() -> bool:
    j = _request("GET", "/fapi/v1/positionSide/dual", signed=True)
    try:
        return bool(j.get("dualSidePosition"))
    except Exception:
        return False

# ===== Filters helpers (optional, used by executor sizing) =====
_EXINFO_CACHE: Optional[dict] = None
_EXINFO_TS = 0.0

def _exchange_info(ttl: float = 300.0) -> dict:
    global _EXINFO_CACHE, _EXINFO_TS
    now = time.time()
    if _EXINFO_CACHE is None or (now - _EXINFO_TS) > ttl:
        j = _request("GET", "/fapi/v1/exchangeInfo")
        _EXINFO_CACHE = j if isinstance(j, dict) else {}
        _EXINFO_TS = now
    return _EXINFO_CACHE or {}

def get_symbol_filters(symbol: str) -> Dict[str, Any]:
    d = next((s for s in (_exchange_info().get("symbols") or []) if s.get("symbol")==symbol), None)
    if not d:
        return {}
    return {f["filterType"]: f for f in d.get("filters", [])}

def round_qty_to_step(qty: float, step: float) -> float:
    return math.floor(qty / step) * step

__all__ = [
    "get_price", "get_klines", "get_positions", "get_balances",
    "_is_dual_side", "get_symbol_filters", "round_qty_to_step",
]
