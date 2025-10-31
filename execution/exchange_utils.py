#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import sys
import time
import random
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext, localcontext

import requests
from dotenv import load_dotenv  # type: ignore

from execution.utils import get_coingecko_prices, load_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [exutil] %(message)s"
)
LOGGER = logging.getLogger("exutil")
_LOG = LOGGER

if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [exutil] %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

try:
    load_dotenv(override=True)
    load_dotenv("/root/hedge-fund/.env", override=True)
    LOGGER.info("[exutil] .env loaded (override=True)")
except Exception:
    LOGGER.info("[exutil] .env not found — using existing environment")

getcontext().prec = 28

# --- Base URL + one-time environment banner ---------------------------------
def _base_url() -> str:
    """Return the USD-M futures base URL based on BINANCE_TESTNET."""
    return (
        "https://testnet.binancefuture.com"
        if os.environ.get("BINANCE_TESTNET", "0") == "1"
        else "https://fapi.binance.com"
    )

_DRY_RUN: bool = os.environ.get("DRY_RUN", "0") == "1"

_BANNER_ONCE = False
def _log_env_once() -> None:
    """Log base URL and testnet flag once per process (stderr)."""
    global _BANNER_ONCE
    if _BANNER_ONCE:
        return
    _BANNER_ONCE = True
    testnet = os.environ.get("BINANCE_TESTNET", "0") == "1"
    print(f"[exutil] base={_base_url()} testnet={testnet}", file=sys.stderr)
    LOGGER.info("[exutil] ENV context testnet=%s dry_run=%s", testnet, _DRY_RUN)

# Run banner at import time (best-effort; never fail)
try:
    _log_env_once()
except Exception:
    pass


def set_dry_run(flag: bool) -> None:
    """Toggle dry-run mode at runtime."""
    global _DRY_RUN
    new_flag = bool(flag)
    if new_flag == _DRY_RUN:
        return
    _LOG.info("[dry-run] exchange utils set to %s", "enabled" if new_flag else "disabled")
    _DRY_RUN = new_flag


def is_dry_run() -> bool:
    return _DRY_RUN


def _dry_run_stub(action: str, stub: Any) -> Any:
    _LOG.info("[dry-run] stubbed %s", action)
    return stub


def is_testnet() -> bool:
    v = os.getenv("BINANCE_TESTNET", "0")
    return str(v).lower() in ("1", "true", "yes", "y")


_BASE = _base_url()
_KEY = os.getenv("BINANCE_API_KEY", "")
_SEC = os.getenv("BINANCE_API_SECRET", "").encode()
_S = requests.Session()
_S.headers["X-MBX-APIKEY"] = _KEY

_MAX_BACKOFF_ATTEMPTS = int(os.getenv("BINANCE_MAX_RETRIES", "5") or 5)
_BACKOFF_INITIAL = float(os.getenv("BINANCE_BACKOFF_INITIAL", "0.25") or 0.25)
_BACKOFF_MAX = float(os.getenv("BINANCE_BACKOFF_MAX", "3.0") or 3.0)

TIME_OFFSET_MS: Optional[int] = None
LAST_TIME_SYNC: float = 0.0
_TIME_SYNC_INTERVAL = 600  # seconds


def _sync_server_time(force: bool = False) -> None:
    """Refresh cached Binance server time offset."""
    global TIME_OFFSET_MS, LAST_TIME_SYNC
    now = time.time()
    if not force and TIME_OFFSET_MS is not None and (now - LAST_TIME_SYNC) < _TIME_SYNC_INTERVAL:
        return

    try:
        resp = requests.get(f"{_base_url()}/fapi/v1/time", timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        server_time_ms = int(data.get("serverTime"))
        local_time_ms = int(time.time() * 1000)
        TIME_OFFSET_MS = server_time_ms - local_time_ms
        LAST_TIME_SYNC = now
        _LOG.info("[binance] server_time_offset_ms=%s", TIME_OFFSET_MS)
    except Exception as exc:  # pragma: no cover - soft failure
        _LOG.warning("[binance] time_sync_failed: %s", exc)
        if TIME_OFFSET_MS is None:
            TIME_OFFSET_MS = 0


def _req(
    method: str,
    path: str,
    *,
    signed: bool = False,
    params: Dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> requests.Response:
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
        _sync_server_time()
        params["recvWindow"] = 10000
        timestamp_ms = int(time.time() * 1000) + (TIME_OFFSET_MS or 0)
        params["timestamp"] = timestamp_ms
        kv = [(str(k), str(v)) for k, v in params.items()]
        qs = urlencode(kv, doseq=True, safe=":/")
        sig = hmac.new(_SEC, qs.encode(), hashlib.sha256).hexdigest()
        if method in ("GET", "DELETE"):
            url = f"{url}?{qs}&signature={sig}"
            data = None
        else:
            data = f"{qs}&signature={sig}"
    else:
        if method in ("GET", "DELETE"):
            if params:
                qs = urlencode(
                    [(str(k), str(v)) for k, v in params.items()], doseq=True, safe=":/"
                )
                url = f"{url}?{qs}"
            data = None
        else:
            data = urlencode(
                [(str(k), str(v)) for k, v in params.items()], doseq=True, safe=":/"
            )

    headers = {"X-MBX-APIKEY": _KEY}
    if method in ("POST", "PUT"):
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    attempt = 0
    backoff = _BACKOFF_INITIAL
    while True:
        attempt += 1
        try:
            r = _S.request(method, url, data=data, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r
        except requests.HTTPError as exc:
            resp = exc.response or r
            status = resp.status_code if resp is not None else None
            detail = ""
            try:
                detail = " :: " + (resp.text if resp is not None else "")
            except Exception:
                pass
            if status in (400, 401):
                try:
                    code = resp.json().get("code") if resp is not None else "?"
                except Exception:
                    code = "?"
                _LOG.error(
                    "[executor] AUTH_ERR code=%s testnet=%s key=%s… sec_len=%s url=%s",
                    code,
                    is_testnet(),
                    (_KEY[:6] if _KEY else "NONE"),
                    len(_SEC),
                    url,
                )
            if status is None or status >= 500 or status in (418, 429):
                if attempt < _MAX_BACKOFF_ATTEMPTS:
                    sleep_for = min(_BACKOFF_MAX, backoff)
                    jitter = random.random() * 0.1 * sleep_for
                    time.sleep(sleep_for + jitter)
                    backoff = min(_BACKOFF_MAX, backoff * 2.0)
                    continue
            raise requests.HTTPError(f"{exc}{detail}", response=exc.response) from None
        except requests.RequestException:
            if attempt >= _MAX_BACKOFF_ATTEMPTS:
                raise
            sleep_for = min(_BACKOFF_MAX, backoff)
            jitter = random.random() * 0.1 * sleep_for
            time.sleep(sleep_for + jitter)
            backoff = min(_BACKOFF_MAX, backoff * 2.0)


# --- debug helpers (no need to import private vars) ---
def debug_key_head() -> Tuple[str, int, bool]:
    return (_KEY[:6] if _KEY else "NONE", len(_SEC), is_testnet())


# --- market data ---
def get_klines(symbol: str, interval: str, limit: int = 150) -> List[List[float]]:
    """Fetch OHLCV rows for USD-M klines; clamp limit to Binance's max of 1500."""
    try:
        max_limit = 1500
        limit = 150 if limit is None else int(limit)
        if limit > max_limit:
            limit = max_limit
    except Exception:
        limit = 150

    r = _req(
        "GET",
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    out: List[List[float]] = []
    for row in r.json():
        try:
            open_time = int(row[0])
            open_p, high_p, low_p, close_p, volume = (
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
        except (IndexError, TypeError, ValueError):
            continue
        out.append([open_time, open_p, high_p, low_p, close_p, volume])
    return out


def get_price(symbol: str, venue: str = "auto", signed: bool = False) -> float:
    sym = str(symbol or "").upper()
    if not sym:
        raise ValueError("symbol_required")

    resolved_venue = venue
    if venue == "auto":
        resolved_venue = "testnet" if is_testnet() else "fapi"

    if sym == "XAUTUSDT" and resolved_venue in ("fapi", "testnet"):
        _LOG.warning(
            "[price] invalid_symbol venue=%s symbol=%s -> routing to coingecko",
            resolved_venue,
            sym,
        )
        try:
            prices = get_coingecko_prices()
            price = float(prices.get("XAUT") or 0.0)
            if price <= 0:
                raise RuntimeError("coingecko_price_missing")
            _LOG.info("[price] XAUT price source=coingecko value=%.2f", price)
            return price
        except Exception as exc:
            _LOG.error("[price] coingecko_fallback_failed: %s", exc)
            raise

    if sym in {"USDC", "USDCUSDT", "USDT", "DAI", "TUSD", "FDUSD", "USDE"}:
        return 1.0

    if resolved_venue in ("spot", "coingecko"):
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": sym},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            return float(data["price"])
        except Exception as exc:
            raise requests.RequestException(f"spot_price_failed:{sym}") from exc

    r = _req("GET", "/fapi/v1/ticker/price", params={"symbol": sym}, signed=signed)
    return float(r.json()["price"])


_EXCHANGE_FILTERS_CACHE: Dict[str, Dict[str, Any]] | None = None
_EXCHANGE_FILTERS_KEY: str | None = None


def _load_exchange_filters() -> Dict[str, Dict[str, Any]]:
    global _EXCHANGE_FILTERS_CACHE, _EXCHANGE_FILTERS_KEY
    base = _base_url()
    if _EXCHANGE_FILTERS_CACHE is not None and _EXCHANGE_FILTERS_KEY == base:
        return _EXCHANGE_FILTERS_CACHE

    resp = _req("GET", "/fapi/v1/exchangeInfo")
    payload = resp.json()
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("symbols", []) or []:
        sym_name = str(entry.get("symbol", "")).upper()
        if not sym_name:
            continue
        filters = {}
        for f in entry.get("filters", []) or []:
            ftype = str(f.get("filterType", ""))
            if not ftype:
                continue
            filters[ftype] = f
        mapping[sym_name] = filters

    if not mapping:
        raise RuntimeError("exchange_info_empty")

    _EXCHANGE_FILTERS_CACHE = mapping
    _EXCHANGE_FILTERS_KEY = base
    return mapping


def get_symbol_filters(symbol: str) -> Dict[str, Any]:
    sym = str(symbol).upper()
    filters = _load_exchange_filters()
    try:
        return filters[sym]
    except KeyError as exc:  # pragma: no cover - safety for unexpected symbols
        raise RuntimeError(f"filters_not_found:{sym}") from exc


def _dec(val: Any) -> Decimal:
    return val if isinstance(val, Decimal) else Decimal(str(val))


def _quant_floor(value: Decimal, step: Decimal) -> Decimal:
    if step == 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _quant_ceil(value: Decimal, step: Decimal) -> Decimal:
    if step == 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def _decimals_from_step(step: Decimal) -> int:
    try:
        normalized = step.normalize()
    except Exception:
        normalized = Decimal(step)
    exp = -normalized.as_tuple().exponent
    return max(0, exp)


def _format_decimal_for_step(value: Decimal, step: Decimal) -> str:
    if step == 0:
        return f"{value:f}"
    with localcontext() as ctx:
        ctx.rounding = ROUND_DOWN
        snapped = (value / step).to_integral_value() * step
    decimals = _decimals_from_step(step)
    if decimals <= 0:
        return str(int(snapped))
    return f"{snapped:.{decimals}f}"


def normalize_price_qty(
    symbol: str,
    price: float,
    desired_gross_usd: float,
) -> tuple[Decimal, Decimal, Dict[str, str]]:
    """Return a (price, qty, meta) triple snapped to Binance filters.

    Ensures price respects PRICE_FILTER.tickSize, quantity respects MARKET_LOT_SIZE/
    LOT_SIZE.stepSize, and qty >= minQty with notional >= MIN_NOTIONAL. Raises if
    qty rounds to zero after snapping.
    """
    filters = get_symbol_filters(symbol)
    tick = _dec((filters.get("PRICE_FILTER", {}) or {}).get("tickSize", "0.01"))
    lot = (filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {})
    step = _dec(lot.get("stepSize", "0.001"))
    min_qty = _dec(lot.get("minQty", "0"))
    min_notional = _dec((filters.get("MIN_NOTIONAL", {}) or {}).get("notional", "0"))

    p = _dec(price)
    if tick > 0:
        p = _quant_floor(p, tick)
    if p <= 0:
        raise ValueError(f"bad_price:{price}")

    gross = _dec(desired_gross_usd)
    qty = _dec(0)
    if p > 0:
        qty = _quant_floor(gross / p, step)
    if qty < min_qty:
        qty = _quant_ceil(min_qty, step)
    notional = qty * p
    if min_notional > 0 and notional < min_notional:
        need_qty = _quant_ceil(min_notional / p, step)
        if need_qty > qty:
            qty = need_qty
            notional = qty * p
    if qty <= 0:
        raise ValueError("qty_rounds_to_zero")
    return p, qty, {
        "tickSize": str(tick),
        "stepSize": str(step),
        "minQty": str(min_qty),
        "minNotional": str(min_notional),
        "finalNotional": str(notional),
    }


def build_order_payload(
    symbol: str,
    side: str,
    price: float,
    desired_gross_usd: float,
    reduce_only: bool,
    position_side: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    filters = get_symbol_filters(symbol)
    lot = (filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {})
    step = _dec(lot.get("stepSize", "0.001"))
    tick = _dec((filters.get("PRICE_FILTER", {}) or {}).get("tickSize", "0.01"))

    norm_price, norm_qty, meta = normalize_price_qty(symbol, price, desired_gross_usd)

    qty_str = _format_decimal_for_step(norm_qty, step)
    price_str = _format_decimal_for_step(norm_price, tick) if tick else f"{norm_price:f}"

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": qty_str,
    }

    if position_side and position_side.upper() != "BOTH":
        payload["positionSide"] = position_side.upper()
    if reduce_only:
        payload["reduceOnly"] = "true"

    meta.update(
        {
            "normalized_price": f"{norm_price:f}",
            "normalized_qty": f"{norm_qty:f}",
            "price_formatted": price_str,
            "qty_str": qty_str,
        }
    )

    return payload, meta


# --- account ---
def get_balances() -> List[Dict[str, Any]]:
    # Avoid signed USD-M calls in DRY_RUN to prevent -2015 while keys/env are being fixed
    if is_dry_run():
        return _dry_run_stub("get_balances", [])
    return _req("GET", "/fapi/v2/balance", signed=True).json()


def get_account() -> Dict[str, Any]:
    if is_dry_run():
        return _dry_run_stub(
            "get_account",
            {
                "assets": [],
                "positions": [],
                "totalMarginBalance": "0.0",
                "totalWalletBalance": "0.0",
                "totalUnrealizedProfit": "0.0",
                "dryRun": True,
            },
        )
    return _req("GET", "/fapi/v2/account", signed=True).json()


def _is_dual_side() -> bool:
    if is_dry_run():
        _dry_run_stub("_is_dual_side", {"dualSidePosition": True})
        return True
    return bool(
        _req("GET", "/fapi/v1/positionSide/dual", signed=True)
        .json()
        .get("dualSidePosition")
    )


def set_dual_side(flag: bool) -> Dict[str, Any]:
    if is_dry_run():
        return _dry_run_stub(
            "set_dual_side",
            {
                "ok": True,
                "dualSidePosition": str(flag).lower(),
                "dryRun": True,
            },
        )
    try:
        return _req(
            "POST",
            "/fapi/v1/positionSide/dual",
            signed=True,
            params={"dualSidePosition": str(flag).lower()},
        ).json()
    except requests.HTTPError as e:
        # -4059 No need to change position side.
        try:
            if e.response is not None and e.response.json().get("code") == -4059:
                return {"ok": True, "note": "No need to change position side."}
        except Exception:
            pass
        raise


def set_symbol_margin_mode(symbol: str, margin_type: str = "CROSSED") -> Dict[str, Any]:
    if is_dry_run():
        return _dry_run_stub(
            "set_symbol_margin_mode",
            {
                "ok": True,
                "symbol": symbol,
                "marginType": margin_type,
                "dryRun": True,
            },
        )
    try:
        return _req(
            "POST",
            "/fapi/v1/marginType",
            signed=True,
            params={"symbol": symbol, "marginType": margin_type},
        ).json()
    except requests.HTTPError as e:
        try:
            if e.response is not None and e.response.json().get("code") == -4046:
                return {"ok": True, "note": "No need to change margin type."}
        except Exception:
            pass
        raise


def set_symbol_leverage(symbol: str, leverage: int) -> Dict[str, Any]:
    if is_dry_run():
        return _dry_run_stub(
            "set_symbol_leverage",
            {
                "ok": True,
                "symbol": symbol,
                "leverage": int(leverage),
                "dryRun": True,
            },
        )
    return _req(
        "POST",
        "/fapi/v1/leverage",
        signed=True,
        params={"symbol": symbol, "leverage": int(leverage)},
    ).json()


def get_income_history(
    start_time_ms: int,
    end_time_ms: Optional[int] = None,
    income_type: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    if is_dry_run():
        return _dry_run_stub("get_income_history", [])
    params: Dict[str, Any] = {
        "startTime": int(start_time_ms),
        "limit": int(limit),
    }
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    if income_type:
        params["incomeType"] = str(income_type)
    return _req("GET", "/fapi/v1/income", signed=True, params=params).json()


def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    # Avoid signed USD-M calls in DRY_RUN to prevent -2015 while keys/env are being fixed
    if is_dry_run():
        return _dry_run_stub("get_positions", [])
    params: Dict[str, Any] = {}
    if symbol:
        params["symbol"] = symbol
    arr = _req("GET", "/fapi/v2/positionRisk", signed=True, params=params).json()
    out = []
    for p in arr:
        qty = float(p.get("positionAmt") or 0)
        out.append(
            {
                "symbol": p.get("symbol"),
                "positionSide": p.get("positionSide", "BOTH"),
                "qty": qty,
                "entryPrice": float(p.get("entryPrice") or 0),
                "unrealized": float(p.get("unRealizedProfit") or 0),
                "markPrice": float(p.get("markPrice") or 0),
                "leverage": float(p.get("leverage") or 0),
            }
        )
    return out


# --- orders ---
def _floor_step(x: float, step: float) -> float:
    return math.floor(x / step) * step


def send_order(
    symbol: str,
    side: str,
    type: str,
    quantity: str | float | Decimal,
    positionSide: Optional[str] = None,
    reduceOnly: Optional[str | bool] = None,
    price: Optional[str | float | Decimal] = None,
    **extra: Any,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side.upper(),
        "type": type.upper(),
    }

    if isinstance(quantity, Decimal):
        params["quantity"] = f"{quantity:f}"
    else:
        params["quantity"] = str(quantity)

    if positionSide:
        params["positionSide"] = positionSide.upper()

    if price not in (None, "", 0, 0.0):
        params["price"] = str(price)

    if isinstance(reduceOnly, str):
        reduce_flag = reduceOnly.lower() in ("1", "true", "yes", "on")
    else:
        reduce_flag = bool(reduceOnly)
    if reduce_flag:
        params["reduceOnly"] = "true"

    params.update({k: v for k, v in extra.items() if v is not None})

    if params["type"] == "MARKET":
        params.pop("price", None)

    if is_dry_run():
        qty_view = str(params.get("quantity", "0"))
        return _dry_run_stub(
            "send_order",
            {
                "dryRun": True,
                "orderId": 0,
                "symbol": params.get("symbol"),
                "side": params.get("side"),
                "type": params.get("type"),
                "status": "DRY_RUN",
                "origQty": qty_view,
                "executedQty": "0",
                "avgPrice": "0.0",
                "updateTime": int(time.time() * 1000),
            },
        )

    try:
        r = _req("POST", "/fapi/v1/order", signed=True, params=params)
        return r.json()
    except requests.HTTPError as exc:
        if params.get("reduceOnly"):
            try:
                body = exc.response.json() if exc.response is not None else {}
                if body.get("code") == -1106:
                    params.pop("reduceOnly", None)
                    r = _req("POST", "/fapi/v1/order", signed=True, params=params)
                    return r.json()
            except Exception:
                pass
        raise


def place_market_order_sized(
    symbol: str,
    side: str,
    notional: float,
    leverage: float,
    position_side: str,
    reduce_only: bool = False,
) -> Dict[str, Any]:
    price = get_price(symbol)
    filters = get_symbol_filters(symbol)
    step = float(filters["LOT_SIZE"]["stepSize"])
    minq = float(filters["LOT_SIZE"]["minQty"])
    min_notional = float(filters.get("MIN_NOTIONAL", {}).get("notional", 5.0))
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
def _quantize_to_step(q, step, mode=ROUND_DOWN) -> Decimal:
    dstep = Decimal(str(step)).normalize()
    with localcontext() as ctx:
        ctx.rounding = mode
        # round to a multiple of step
        return (Decimal(str(q)) / dstep).to_integral_value() * dstep


# --- precise order wrappers (override previous defs) ---
def place_market_order(  # noqa: F811
    symbol: str,
    side: str,
    quantity: float,
    position_side: str = "BOTH",
    reduce_only: bool | None = None,
):
    f = get_symbol_filters(symbol)
    lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE") or {}
    step = _dec(lot.get("stepSize", "1.0"))
    min_qty = _dec(lot.get("minQty", "0.0"))

    qty_dec = _quantize_to_step(quantity, step, ROUND_DOWN)
    if qty_dec < min_qty:
        qty_dec = _quantize_to_step(min_qty, step, ROUND_UP)

    qty_str = _format_decimal_for_step(qty_dec, step)
    params: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty_str,
    }
    if position_side and position_side != "BOTH":
        params["positionSide"] = position_side
    if reduce_only is True:
        params["reduceOnly"] = "true"

    return send_order(**params)


def place_market_order_sized(  # noqa: F811
    symbol: str,
    side: str,
    notional: float,
    leverage: float,
    position_side: str = "BOTH",
    reduce_only: bool = False,
):
    px = float(get_price(symbol))
    desired_gross = float(notional) * max(float(leverage), 1.0)

    floor_gross = 0.0
    try:
        cfg = load_json("config/strategy_config.json")
        sizing_cfg = (cfg.get("sizing") or {})
        floor_gross = float((sizing_cfg.get("min_gross_usd_per_order", 0.0)) or 0.0)
        per_symbol_cfg = sizing_cfg.get("per_symbol_min_gross_usd") or {}
        if isinstance(per_symbol_cfg, dict):
            sym_override = per_symbol_cfg.get(str(symbol).upper())
            if sym_override is not None:
                floor_gross = max(floor_gross, float(sym_override or 0.0))
    except Exception:
        floor_gross = max(floor_gross, 0.0)
    desired_gross = max(desired_gross, floor_gross)

    _, qty, _ = normalize_price_qty(symbol, px, desired_gross)
    return place_market_order(
        symbol,
        side,
        float(qty),
        position_side=position_side,
        reduce_only=reduce_only,
    )
