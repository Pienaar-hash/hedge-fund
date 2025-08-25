#!/usr/bin/env python3
from __future__ import annotations

import os, time, hmac, hashlib, json, logging, math
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlencode

# --- robust .env load (works under supervisor too) ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv() or load_dotenv("/root/hedge-fund/.env")
except Exception:
    pass
# manual fallback in case python-dotenv isn't available at runtime
_envp="/root/hedge-fund/.env"
if os.path.exists(_envp):
    try:
        with open(_envp,"r") as _f:
            for _ln in _f:
                _ln=_ln.strip()
                if not _ln or _ln.startswith("#") or "=" not in _ln: 
                    continue
                k,v=_ln.split("=",1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

import requests
from decimal import Decimal, ROUND_DOWN, ROUND_UP, localcontext

_LOG = logging.getLogger("exutil")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [exutil] %(message)s")

def is_testnet() -> bool:
    v = os.getenv("BINANCE_TESTNET", "1")
    return str(v).lower() in ("1","true","yes","y")

_BASE = "https://testnet.binancefuture.com" if is_testnet() else "https://fapi.binance.com"
_KEY  = os.getenv("BINANCE_API_KEY","")
_SEC  = os.getenv("BINANCE_API_SECRET","").encode()
_S    = requests.Session()
_S.headers["X-MBX-APIKEY"] = _KEY

def _req(method: str, path: str, *, signed: bool=False, params: Dict[str,Any]|None=None, timeout: float=8.0) -> requests.Response:
    """
    Sign EXACTLY what we send. For signed:
      - Build qs with urlencode (preserving insertion order of dict)
      - Sign qs
      - GET/DELETE: put qs+signature in URL
      - POST/PUT:   send qs+signature as x-www-form-urlencoded body
    """
    method = method.upper()
    url = _BASE + path
    params = {k: v for k, v in (params or {}).items() if v is not None}

    if signed:
        params.setdefault("timestamp", int(time.time()*1000))
        params.setdefault("recvWindow", int(os.getenv("BINANCE_RECV_WINDOW","5000")))
        kv = [(str(k), str(v)) for k, v in params.items()]
        qs = urlencode(kv, doseq=True, safe=":/")
        sig = hmac.new(_SEC, qs.encode(), hashlib.sha256).hexdigest()
        if method in ("GET","DELETE"):
            url = f"{url}?{qs}&signature={sig}"
            data = None
        else:
            data = f"{qs}&signature={sig}"
    else:
        if method in ("GET","DELETE"):
            if params:
                qs = urlencode([(str(k), str(v)) for k,v in params.items()], doseq=True, safe=":/")
                url = f"{url}?{qs}"
            data = None
        else:
            data = urlencode([(str(k), str(v)) for k,v in params.items()], doseq=True, safe=":/")

    headers = {"X-MBX-APIKEY": _KEY}
    if method in ("POST","PUT"):
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    r = _S.request(method, url, data=data, timeout=timeout, headers=headers)
    try:
        r.raise_for_status()
        return r
    except Exception as e:
        detail = ""
        try:
            detail = " :: " + r.text
        except Exception:
            pass
        # Breadcrumb on auth-ish errors
        if r.status_code in (400,401):
            try:
                code = r.json().get("code")
            except Exception:
                code = "?"
            _LOG.error("[executor] AUTH_ERR code=%s testnet=%s key=%s… sec_len=%s url=%s",
                       code, is_testnet(), (_KEY[:6] if _KEY else "NONE"), len(_SEC), url)
        raise requests.HTTPError(f"{e}{detail}", response=getattr(e, "response", None)) from None

# --- debug helpers (no need to import private vars) ---
def debug_key_head() -> Tuple[str,int,bool]:
    return (_KEY[:6] if _KEY else "NONE", len(_SEC), is_testnet())

# --- market data ---
def get_klines(symbol: str, interval: str, limit: int=150) -> List[Tuple[int, float]]:
    r = _req("GET", "/fapi/v1/klines", params={"symbol":symbol, "interval":interval, "limit":limit})
    out=[]
    for row in r.json():
        # [ openTime, open, high, low, close, volume, closeTime, ...]
        out.append( (int(row[0]), float(row[4])) )
    return out

def get_price(symbol: str) -> float:
    r = _req("GET", "/fapi/v1/ticker/price", params={"symbol":symbol})
    return float(r.json()["price"])

def get_symbol_filters(symbol: str) -> Dict[str,Any]:
    r = _req("GET", "/fapi/v1/exchangeInfo", params={"symbol":symbol})
    info = r.json()
    sym = info["symbols"][0]
    filters = { f["filterType"]: f for f in sym["filters"] }
    return filters

# --- account ---
def get_balances() -> List[Dict[str,Any]]:
    return _req("GET","/fapi/v2/balance", signed=True).json()

def get_account() -> Dict[str,Any]:
    return _req("GET","/fapi/v2/account", signed=True).json()

def _is_dual_side() -> bool:
    return bool(_req("GET","/fapi/v1/positionSide/dual", signed=True).json().get("dualSidePosition"))

def set_dual_side(flag: bool) -> Dict[str,Any]:
    try:
        return _req("POST","/fapi/v1/positionSide/dual", signed=True, params={"dualSidePosition": str(flag).lower()}).json()
    except requests.HTTPError as e:
        # -4059 No need to change position side.
        try:
            if e.response is not None and e.response.json().get("code") == -4059:
                return {"ok": True, "note":"No need to change position side."}
        except Exception:
            pass
        raise

def set_symbol_margin_mode(symbol: str, margin_type: str="CROSSED") -> Dict[str,Any]:
    try:
        return _req("POST","/fapi/v1/marginType", signed=True, params={"symbol":symbol, "marginType":margin_type}).json()
    except requests.HTTPError as e:
        try:
            if e.response is not None and e.response.json().get("code") == -4046:
                return {"ok": True, "note":"No need to change margin type."}
        except Exception:
            pass
        raise

def set_symbol_leverage(symbol: str, leverage: int) -> Dict[str,Any]:
    return _req("POST","/fapi/v1/leverage", signed=True, params={"symbol":symbol, "leverage":int(leverage)}).json()

def get_positions(symbol: Optional[str]=None) -> List[Dict[str,Any]]:
    params = {}
    if symbol: params["symbol"]=symbol
    arr = _req("GET","/fapi/v2/positionRisk", signed=True, params=params).json()
    out=[]
    for p in arr:
        qty = float(p.get("positionAmt") or 0)
        out.append({
            "symbol": p.get("symbol"),
            "positionSide": p.get("positionSide","BOTH"),
            "qty": qty,
            "entryPrice": float(p.get("entryPrice") or 0),
            "unrealized": float(p.get("unRealizedProfit") or 0),
            "leverage": float(p.get("leverage") or 0),
        })
    return out

# --- orders ---
def _floor_step(x: float, step: float) -> float:
    return math.floor(x / step) * step

def place_market_order(symbol: str, side: str, quantity: float, position_side: str, reduce_only: bool=False) -> Dict[str,Any]:
    params = {
        "symbol": symbol,
        "side": side.upper(),                # BUY/SELL
        "type": "MARKET",
        "quantity": f"{quantity:.20f}",
        "positionSide": position_side.upper() if position_side else None,
    }
    if reduce_only:
        params["reduceOnly"] = True  # include only when True

    try:
        r = _req("POST","/fapi/v1/order", signed=True, params=params)
        return r.json()
    except requests.HTTPError as e:
        # If Hedge Mode complains about reduceOnly (-1106), drop it once and retry
        try:
            j = e.response.json()
            if j.get("code") == -1106 and params.get("reduceOnly"):
                params.pop("reduceOnly", None)
                r2 = _req("POST","/fapi/v1/order", signed=True, params=params)
                return r2.json()
        except Exception:
            pass
        raise

def place_market_order_sized(symbol: str, side: str, notional: float, leverage: float, position_side: str, reduce_only: bool=False) -> Dict[str,Any]:
    price = get_price(symbol)
    filters = get_symbol_filters(symbol)
    step = float(filters["LOT_SIZE"]["stepSize"])
    minq = float(filters["LOT_SIZE"]["minQty"])
    min_notional = float(filters.get("MIN_NOTIONAL",{}).get("notional", 5.0))
    raw_qty = (float(notional) * float(leverage)) / float(price)
    qty = _floor_step(raw_qty, step)
    if qty < minq:
        qty = minq
    if qty * price < min_notional:
        qty = _floor_step((min_notional / price), step)
    if qty <= 0:
        raise ValueError(f"Computed qty <= 0 (raw={raw_qty}, step={step})")
    return place_market_order(symbol, side, qty, position_side, reduce_only=reduce_only)


# --- precise qty/step helpers ---
def _step_decimals(step) -> int:
    d = Decimal(str(step)).normalize()
    exp = -d.as_tuple().exponent
    return max(0, exp)

def _quantize_to_step(q, step, mode=ROUND_DOWN) -> Decimal:
    dstep = Decimal(str(step)).normalize()
    with localcontext() as ctx:
        ctx.rounding = mode
        # round to a multiple of step
        return (Decimal(str(q)) / dstep).to_integral_value() * dstep

def _fmt_qty(q, step) -> str:
    d = _step_decimals(step)
    # format with exactly the allowed number of decimals
    return f"{Decimal(str(q)):.{d}f}" if d>0 else str(int(Decimal(str(q))))


# --- precise order wrappers (override previous defs) ---
def place_market_order(symbol: str, side: str, quantity: float,
                       position_side: str = "BOTH", reduce_only: bool | None = None):
    f = get_symbol_filters(symbol)
    step = float(f["LOT_SIZE"]["stepSize"])
    min_qty = float(f["LOT_SIZE"]["minQty"])

    # quantize & enforce mins
    q = float(_quantize_to_step(quantity, step, ROUND_DOWN))
    if q < min_qty:
        q = float(_quantize_to_step(min_qty, step, ROUND_UP))

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": _fmt_qty(q, step),
    }
    if position_side and position_side != "BOTH":
        params["positionSide"] = position_side
    if reduce_only is True:
        params["reduceOnly"] = "true"   # only send when required

    r = _req("POST", "/fapi/v1/order", signed=True, params=params)
    return r.json()


def place_market_order_sized(symbol: str, side: str, notional: float, leverage: float,
                             position_side: str = "BOTH", reduce_only: bool = False):
    f = get_symbol_filters(symbol)
    step = float(f["LOT_SIZE"]["stepSize"])
    min_qty = float(f["LOT_SIZE"]["minQty"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("notional", 0) or 0.0)

    px = float(get_price(symbol))
    raw = (float(notional) * float(leverage)) / px

    # floor to step, then enforce mins
    qty = float(_quantize_to_step(raw, step, ROUND_DOWN))
    if qty < min_qty:
        qty = float(_quantize_to_step(min_qty, step, ROUND_UP))
    if px * qty < min_notional:
        need = min_notional / px
        qty = float(_quantize_to_step(need, step, ROUND_UP))

    return place_market_order(symbol, side, qty, position_side, reduce_only=reduce_only)
