"""
Exchange utilities with safe wrappers for Spot and USD-M Futures on Binance.
- Select Futures via USE_FUTURES=1 (env). Otherwise defaults to Spot.
- Testnet via BINANCE_TESTNET=1 (env).
- Creds from API_KEY/API_SECRET or BINANCE_API_KEY/BINANCE_API_SECRET.
- Functions never hard-crash; they return neutral data and error notes.
- Outputs are normalized for executor consumption.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Tuple, List, Optional

# Spot client
try:
    from binance.client import Client as SpotClient
except Exception:  # library might not be installed
    SpotClient = None  # type: ignore

# USD-M futures client
try:
    from binance.um_futures import UMFutures
except Exception:
    UMFutures = None  # type: ignore


# ------------------ helpers ------------------

def _get_env(name: str, alt: str | None = None, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        if alt is not None:
            v = os.getenv(alt, default)
        else:
            v = default
    return v


def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0") in ("1", "true", "TRUE", "yes", "YES")


def _clients() -> Tuple[Optional[SpotClient], Optional[UMFutures], Dict[str, Any]]:
    key = _get_env("API_KEY", "BINANCE_API_KEY")
    sec = _get_env("API_SECRET", "BINANCE_API_SECRET")
    testnet = _bool_env("BINANCE_TESTNET", False)
    use_futures = _bool_env("USE_FUTURES", False)

    info = {"testnet": testnet, "use_futures": use_futures}

    spot, fut = None, None
    try:
        if not use_futures and SpotClient is not None and key and sec:
            spot = SpotClient(api_key=key, api_secret=sec, testnet=testnet)
    except Exception as e:
        info["spot_error"] = str(e)
        spot = None
    try:
        if use_futures and UMFutures is not None and key and sec:
            fut = UMFutures(key=key, secret=sec, testnet=testnet)
    except Exception as e:
        info["futures_error"] = str(e)
        fut = None

    return spot, fut, info


# ------------------ public API ------------------

def get_price(symbol: str) -> Dict[str, Any]:
    """Return {symbol, price} or {symbol, price:0, _error}. Works for both modes.
    """
    spot, fut, meta = _clients()
    try:
        if meta.get("use_futures") and fut is not None:
            r = fut.ticker_price(symbol=symbol)
            return {"symbol": symbol, "price": float(r["price"]) }
        if spot is not None:
            r = spot.get_symbol_ticker(symbol=symbol)
            return {"symbol": symbol, "price": float(r["price"]) }
        return {"symbol": symbol, "price": 0.0, "_error": "client_unavailable"}
    except Exception as e:
        return {"symbol": symbol, "price": 0.0, "_error": str(e)}


def get_balances() -> Dict[str, Any]:
    """Spot: returns {asset: free}
       Futures: returns {asset: balance} (wallet balance map)
       On error, returns minimal neutral map with _error
    """
    spot, fut, meta = _clients()
    try:
        if meta.get("use_futures") and fut is not None:
            # futures account balances
            bals = fut.balance()  # list of {asset, balance, ...}
            return {b["asset"]: float(b.get("balance", 0.0)) for b in bals}
        if spot is not None:
            acct = spot.get_account()
            return {b["asset"]: float(b.get("free", 0.0)) for b in acct.get("balances", [])}
        return {"USDT": 0.0, "_error": "client_unavailable"}
    except Exception as e:
        return {"USDT": 0.0, "_error": str(e)}


def get_positions() -> List[Dict[str, Any]]:
    """Futures: list of open positions with qty != 0.
       Spot: returns empty list (spot doesn't have persistent positions in the same sense).
    """
    spot, fut, meta = _clients()
    try:
        if meta.get("use_futures") and fut is not None:
            out: List[Dict[str, Any]] = []
            for p in fut.get_position_risk():
                amt = float(p.get("positionAmt", 0.0))
                if amt == 0.0:
                    continue
                entry = float(p.get("entryPrice", 0.0))
                sym = p.get("symbol")
                out.append({
                    "symbol": sym,
                    "qty": amt,
                    "entry_price": entry,
                    "side": "LONG" if amt > 0 else "SHORT",
                    "leverage": int(float(p.get("leverage", 1))),
                })
            return out
        # Spot equivalent: return empty; executor handles inventory separately if needed.
        return []
    except Exception:
        return []


def cancel_all_orders(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Cancel open orders. If symbol None, cancel all supported symbols.
    Works for both modes, best-effort.
    """
    spot, fut, meta = _clients()
    try:
        if meta.get("use_futures") and fut is not None:
            if symbol:
                try:
                    fut.cancel_open_orders(symbol=symbol)
                except Exception:
                    pass
            else:
                for o in fut.get_open_orders():
                    try:
                        fut.cancel_order(symbol=o["symbol"], orderId=o["orderId"])  # type: ignore
                    except Exception:
                        pass
            return {"ok": True}
        if spot is not None:
            if symbol:
                try:
                    spot.cancel_open_orders(symbol=symbol)
                except Exception:
                    pass
            else:
                # Best-effort: fetch tickers to derive symbols and cancel
                for t in spot.get_all_tickers():
                    sym = t.get("symbol")
                    try:
                        spot.cancel_open_orders(symbol=sym)
                    except Exception:
                        pass
            return {"ok": True}
        return {"ok": False, "_error": "client_unavailable"}
    except Exception as e:
        return {"ok": False, "_error": str(e)}


def place_market_order(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    """Place a market order across modes. Returns {ok, order_id?, _error?}.
    side: 'BUY' or 'SELL'
    """
    spot, fut, meta = _clients()
    try:
        side = side.upper()
        if meta.get("use_futures") and fut is not None:
            r = fut.new_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
            return {"ok": True, "order_id": r.get("orderId")}
        if spot is not None:
            if side == "BUY":
                r = spot.order_market_buy(symbol=symbol, quantity=quantity)
            else:
                r = spot.order_market_sell(symbol=symbol, quantity=quantity)
            return {"ok": True, "order_id": r.get("orderId")}
        return {"ok": False, "_error": "client_unavailable"}
    except Exception as e:
        return {"ok": False, "_error": str(e)}


__all__ = [
    "get_price",
    "get_balances",
    "get_positions",
    "cancel_all_orders",
    "place_market_order",
]
