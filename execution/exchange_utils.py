#!/usr/bin/env python3
from __future__ import annotations
import os, time, hmac, hashlib, requests, json
from typing import Any, Dict, List, Optional, Iterable, Tuple
from urllib.parse import urlencode

# Load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "1") in ("1","true","True")
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = (os.getenv("BINANCE_API_SECRET", "") or "").encode()
BASE = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

_sess = requests.Session()
if API_KEY:
    _sess.headers["X-MBX-APIKEY"] = API_KEY

def is_testnet() -> bool:
    return BINANCE_TESTNET

def _ts_ms() -> int:
    return int(time.time() * 1000)

def _sign(qs: str) -> str:
    return hmac.new(API_SECRET, qs.encode(), hashlib.sha256).hexdigest()

def _encode_pairs(pairs: Iterable[Tuple[str, Any]]) -> str:
    """URL-encode in the same order we will send to requests."""
    # Ensure all values are string/primitive types
    pairs = [(k, str(v)) for (k, v) in pairs]
    return urlencode(pairs, doseq=True, safe=":/")

def _req(method: str, path: str, *, signed: bool = False,
         params: Optional[Dict[str, Any]] = None,
         data: Optional[Dict[str, Any]] = None) -> requests.Response:
    """
    Canonical request builder:
      - Preserves param order for signature (list of tuples)
      - Computes signature over EXACT query string we send
      - On HTTPError, raises but includes server text for diagnosis
    """
    url = BASE + path
    params = dict(params or {})
    # Build ordered list of pairs (preserving caller insertion order)
    pairs: List[Tuple[str, Any]] = list(params.items())

    if signed:
        if "timestamp" not in params:
            pairs.append(("timestamp", _ts_ms()))
        if "recvWindow" not in params:
            pairs.append(("recvWindow", 10000))  # 10s window to avoid -1021
        # Encode exactly as requests will, then sign that exact string
        qs_wo_sig = _encode_pairs(pairs)
        sig = _sign(qs_wo_sig)
        pairs.append(("signature", sig))

    # Pass a list of tuples so requests preserves order
    try:
        r = _sess.request(method, url, params=pairs, data=data, timeout=15)
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        # Attach server message to help debugging (Binance puts JSON in body)
        try:
            detail = e.response.text
        except Exception:
            detail = ""
        raise requests.HTTPError(f"{e} :: {detail}", response=e.response) from None

# ---------- Market data ----------
def get_price(symbol: str) -> float:
    r = _req("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
    return float(r.json()["price"])

def get_klines(symbol: str, interval: str, limit: int = 120) -> List[tuple]:
    """Return list of (closeTime_ms, closePrice_float)."""
    r = _req("GET", "/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": int(limit)})
    out = []
    for k in r.json():
        try: out.append((int(k[6]), float(k[4])))
        except Exception: pass
    return out

def get_symbol_filters(symbol: str) -> Dict[str, Any]:
    r = _req("GET", "/fapi/v1/exchangeInfo", params={"symbol": symbol})
    info = r.json()
    fs = {}
    for f in info["symbols"][0]["filters"]:
        fs[f["filterType"]] = f
    return fs

# ---------- Account ----------
def get_balances() -> List[Dict[str, Any]]:
    return _req("GET", "/fapi/v2/balance", signed=True).json()

def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Normalized positions list. Adds 'qty' alias. If symbol provided, filter.
    """
    params = {"symbol": symbol} if symbol else None
    arr = _req("GET", "/fapi/v2/positionRisk", signed=True, params=params).json()
    out: List[Dict[str, Any]] = []
    for p in arr:
        if symbol and p.get("symbol") != symbol:  # server may ignore symbol on testnet
            continue
        try: qty = float(p.get("positionAmt", 0) or 0)
        except Exception: qty = 0.0
        q = dict(p); q["qty"] = qty
        out.append(q)
    return out

# ---------- Settings ----------
def set_dual_side(enabled: bool = True) -> Dict[str, Any]:
    return _req("POST", "/fapi/v1/positionSide/dual", signed=True,
                params={"dualSidePosition": "true" if enabled else "false"}).json()

def _is_dual_side() -> bool:
    try:
        return bool(_req("GET", "/fapi/v1/positionSide/dual", signed=True).json().get("dualSidePosition"))
    except requests.HTTPError as e:
        # Don’t crash the process if the check fails; log upstream and assume True.
        print(f"[warn] dualSide check failed: {e}")
        return True

def set_symbol_leverage(symbol: str, leverage: int) -> Dict[str, Any]:
    return _req("POST", "/fapi/v1/leverage", signed=True, params={"symbol": symbol, "leverage": int(leverage)}).json()

def set_symbol_margin_mode(symbol: str, margin_type: str = "CROSSED") -> Dict[str, Any]:
    return _req("POST", "/fapi/v1/marginType", signed=True, params={"symbol": symbol, "marginType": margin_type}).json()

# ---------- Sizing helpers ----------
def _round_step(qty: float, step: float) -> float:
    return (int(qty / step) * step) if step > 0 else qty

def _min_notional_ok(symbol: str, qty: float, price: float) -> bool:
    fs = get_symbol_filters(symbol)
    mn = fs.get("MIN_NOTIONAL", {}).get("notional")
    if mn is None: return True
    try: return (qty * price) >= float(mn)
    except Exception: return True

# ---------- Orders ----------
def place_market_order(symbol: str, side: str, quantity: float, position_side: str, reduce_only: bool = False) -> Dict[str, Any]:
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": quantity,
        "positionSide": position_side.upper(),
    }
    if reduce_only: params["reduceOnly"] = "true"
    try:
        return _req("POST", "/fapi/v1/order", signed=True, params=params).json()
    except requests.HTTPError as e:
        # On -1106 and we set reduceOnly, retry once without it
        try: j = e.response.json()
        except Exception: j = {}
        if j.get("code") == -1106 and params.get("reduceOnly") == "true":
            params.pop("reduceOnly", None)
            return _req("POST", "/fapi/v1/order", signed=True, params=params).json()
        raise

def place_market_order_sized(symbol: str, side: str, usd_capital: float, leverage: float, position_side: str, reduce_only: bool = False) -> Dict[str, Any]:
    price = float(get_price(symbol))
    fs = get_symbol_filters(symbol)
    step = float(fs.get("LOT_SIZE", {}).get("stepSize", "0.001"))
    minq = float(fs.get("LOT_SIZE", {}).get("minQty", "0.0"))
    qty = (float(usd_capital) * float(leverage)) / max(price, 1e-9)
    qty = max(qty, minq)
    qty = _round_step(qty, step)
    if qty <= 0: raise ValueError("computed qty <= 0")
    if not _min_notional_ok(symbol, qty, price): raise ValueError("MIN_NOTIONAL not satisfied")
    return place_market_order(symbol=symbol, side=side, quantity=qty, position_side=position_side, reduce_only=reduce_only)

def get_order(symbol: str, order_id: int) -> Dict[str, Any]:
    """Fetch order status (includes avgPrice/executedQty on fills)."""
    return _req("GET", "/fapi/v1/order", signed=True,
                params={"symbol": symbol, "orderId": order_id}).json()


def get_account() -> dict:
    """USDⓂ Futures account (totals include uPnL)."""
    return _req('GET','/fapi/v2/account', signed=True).json()
