"""
Precision Normalization Engine (v7.2-alpha2)

Applies Binance/KuCoin exchange filters (tickSize, stepSize, minQty, minNotional)
to ensure orders comply with exchange precision requirements.

This module does NOT modify:
- risk_limits or veto logic
- sizing contract (qty = gross_usd / mark_price)
- strategy adaptation logic
- router policy or maker autoselection
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("exchange_precision")

PRECISION_CACHE_PATH = Path("config/exchange_precision_cache.json")


def load_precision_table() -> Dict[str, Dict[str, Any]]:
    """Load the precision cache from disk."""
    if not PRECISION_CACHE_PATH.exists():
        LOGGER.warning("[precision] cache not found at %s", PRECISION_CACHE_PATH)
        return {}
    try:
        with open(PRECISION_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        LOGGER.info("[precision] loaded %d symbols from cache", len(data))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        LOGGER.error("[precision] failed to load cache: %s", e)
        return {}


_PRECISION_TABLE: Dict[str, Dict[str, Any]] = load_precision_table()


def reload_precision_table() -> None:
    """Reload the precision table from disk (useful after cache update)."""
    global _PRECISION_TABLE
    _PRECISION_TABLE = load_precision_table()


def get_filters(symbol: str) -> Dict[str, Any]:
    """Get exchange filters for a symbol."""
    return _PRECISION_TABLE.get(symbol, {})


# -----------------------
# Normalization helpers
# -----------------------


def normalize_price(symbol: str, price: float) -> float:
    """
    Normalize price to comply with tickSize filter.
    
    Floors to the nearest tick increment.
    """
    if price <= 0:
        return price
    f = get_filters(symbol)
    tick_str = f.get("tickSize")
    if not tick_str:
        return price
    try:
        tick = float(tick_str)
        if tick > 0:
            # Floor to tick size
            normalized = math.floor(price / tick) * tick
            # Round to avoid floating point artifacts
            decimals = _count_decimals(tick_str)
            return round(normalized, decimals)
    except (ValueError, TypeError):
        pass
    return price


def normalize_qty(symbol: str, qty: float) -> float:
    """
    Normalize quantity to comply with stepSize (LOT_SIZE) filter.
    
    Floors to the nearest step increment.
    """
    if qty <= 0:
        return qty
    f = get_filters(symbol)
    step_str = f.get("stepSize")
    if not step_str:
        return qty
    try:
        step = float(step_str)
        if step > 0:
            # Floor to step size
            normalized = math.floor(qty / step) * step
            # Round to avoid floating point artifacts
            decimals = _count_decimals(step_str)
            return round(normalized, decimals)
    except (ValueError, TypeError):
        pass
    return qty


def get_min_qty(symbol: str) -> float:
    """Get minimum quantity for a symbol."""
    f = get_filters(symbol)
    try:
        return float(f.get("minQty", 0))
    except (ValueError, TypeError):
        return 0.0


def get_min_notional(symbol: str) -> float:
    """Get minimum notional value for a symbol."""
    f = get_filters(symbol)
    try:
        return float(f.get("minNotional", 0))
    except (ValueError, TypeError):
        return 0.0


def meets_min_notional(symbol: str, price: float, qty: float) -> bool:
    """Check if order meets minimum notional requirement."""
    min_notional = get_min_notional(symbol)
    if min_notional <= 0:
        return True
    notional = price * qty
    return notional >= min_notional


def clamp_to_min_notional(symbol: str, price: float, qty: float) -> float:
    """
    Adjust quantity upward to meet minimum notional if needed.
    
    Returns the minimum qty required to meet minNotional at given price.
    """
    if price <= 0:
        return qty
    min_notional = get_min_notional(symbol)
    if min_notional <= 0:
        return qty
    notional = price * qty
    if notional >= min_notional:
        return qty
    # Calculate minimum qty needed
    min_qty_for_notional = min_notional / price
    return min_qty_for_notional


def normalize_order(symbol: str, price: float, qty: float) -> tuple:
    """
    Fully normalize an order's price and quantity.
    
    Returns (normalized_price, normalized_qty).
    
    Steps:
    1. Normalize price to tickSize
    2. Normalize qty to stepSize
    3. If below minNotional, adjust qty upward
    4. Re-normalize qty after minNotional adjustment
    """
    # Step 1: normalize price
    norm_price = normalize_price(symbol, price)
    
    # Step 2: normalize qty
    norm_qty = normalize_qty(symbol, qty)
    
    # Step 3: ensure minNotional
    if not meets_min_notional(symbol, norm_price, norm_qty):
        norm_qty = clamp_to_min_notional(symbol, norm_price, norm_qty)
        # Step 4: re-normalize qty after minNotional adjustment
        norm_qty = normalize_qty(symbol, norm_qty)
    
    return norm_price, norm_qty


def _count_decimals(value_str: str) -> int:
    """Count decimal places in a string representation of a number."""
    if not value_str or "." not in value_str:
        return 0
    try:
        # Handle scientific notation
        if "e" in value_str.lower():
            return abs(int(value_str.lower().split("e")[1]))
        return len(value_str.split(".")[1].rstrip("0")) or 0
    except (IndexError, ValueError):
        return 8  # Default to 8 decimals


def get_qty_precision(symbol: str) -> int:
    """Get the number of decimal places for quantity."""
    f = get_filters(symbol)
    step_str = f.get("stepSize", "")
    if step_str:
        return _count_decimals(step_str)
    return 8  # Default


def get_price_precision(symbol: str) -> int:
    """Get the number of decimal places for price."""
    f = get_filters(symbol)
    tick_str = f.get("tickSize", "")
    if tick_str:
        return _count_decimals(tick_str)
    return 8  # Default


def format_qty(symbol: str, qty: float) -> str:
    """Format quantity as string with correct precision."""
    precision = get_qty_precision(symbol)
    normalized = normalize_qty(symbol, qty)
    if precision == 0:
        return str(int(normalized))
    return f"{normalized:.{precision}f}"


def format_price(symbol: str, price: float) -> str:
    """Format price as string with correct precision."""
    precision = get_price_precision(symbol)
    normalized = normalize_price(symbol, price)
    if precision == 0:
        return str(int(normalized))
    return f"{normalized:.{precision}f}"


def refresh_precision_cache() -> bool:
    """
    Refresh the precision cache by fetching latest exchange info.
    
    Fetches all USDT perpetual pairs from Binance Futures and writes
    to PRECISION_CACHE_PATH.
    
    Returns True if successful, False otherwise.
    """
    import os
    import requests
    
    try:
        testnet = os.getenv("BINANCE_TESTNET", "0").lower() in ("1", "true", "yes")
        base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        endpoint = f"{base_url}/fapi/v1/exchangeInfo"
        
        resp = requests.get(endpoint, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        symbols = data.get("symbols", [])
        cache: Dict[str, Dict[str, Any]] = {}
        
        for sym_info in symbols:
            symbol = sym_info.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            if sym_info.get("contractType") != "PERPETUAL":
                continue
            if sym_info.get("status") != "TRADING":
                continue
            
            filters = sym_info.get("filters", [])
            entry: Dict[str, Any] = {}
            
            for f in filters:
                ftype = f.get("filterType")
                if ftype == "PRICE_FILTER":
                    entry["tickSize"] = f.get("tickSize")
                    entry["minPrice"] = f.get("minPrice")
                    entry["maxPrice"] = f.get("maxPrice")
                elif ftype == "LOT_SIZE":
                    entry["stepSize"] = f.get("stepSize")
                    entry["minQty"] = f.get("minQty")
                    entry["maxQty"] = f.get("maxQty")
                elif ftype == "MIN_NOTIONAL":
                    entry["minNotional"] = f.get("notional")
            
            if entry:
                cache[symbol] = entry
        
        # Write to file
        PRECISION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PRECISION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        
        LOGGER.info("[precision] refreshed cache with %d symbols", len(cache))
        
        # Reload into memory
        reload_precision_table()
        
        return True
        
    except Exception as e:
        LOGGER.error("[precision] failed to refresh cache: %s", e)
        return False
