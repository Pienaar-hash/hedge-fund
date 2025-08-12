# execution/exchange_utils.py
import os
import math
import traceback
from binance.client import Client
from binance.exceptions import BinanceAPIException
from execution.utils import load_env_var

# --- Binance client setup ---
BINANCE_API_KEY = load_env_var("BINANCE_API_KEY")
BINANCE_API_SECRET = load_env_var("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("⚠️ BINANCE_API_KEY / BINANCE_API_SECRET not set — private endpoints will fail.")

TESTNET = str(os.getenv("BINANCE_TESTNET", "1")).lower() in ("1", "true", "yes")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=TESTNET)

# --- Public helpers ---
def get_balances():
    try:
        account_info = client.get_account()
        balances = {
            a["asset"]: float(a["free"]) + float(a["locked"])
            for a in account_info["balances"]
            if float(a["free"]) + float(a["locked"]) > 0
        }
        return balances
    except BinanceAPIException as e:
        print(f"❌ Binance API error: {e.message}")
        return {}
    except Exception:
        print("❌ Error fetching balances:")
        print(traceback.format_exc())
        return {}

def get_price(symbol: str) -> float:
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception:
        print(f"❌ Error fetching price for {symbol}")
        return 0.0

# --- Filters & formatting ---
def _symbol_filters(symbol: str):
    """Return (stepSize, minQty, minNotional, tickSize) as floats for symbol."""
    try:
        info = client.get_symbol_info(symbol)
        fs = {f["filterType"]: f for f in info.get("filters", [])}
        lot = fs.get("LOT_SIZE", {})
        min_notional = fs.get("MIN_NOTIONAL", {}).get("minNotional") or fs.get("NOTIONAL", {}).get("minNotional")
        price_filter = fs.get("PRICE_FILTER", {})
        step = float(lot.get("stepSize", 0.0) or 0.0)
        min_qty = float(lot.get("minQty", 0.0) or 0.0)
        min_notional = float(min_notional or 0.0)
        tick = float(price_filter.get("tickSize", 0.0) or 0.0)
        return step, min_qty, min_notional, tick
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def _step_decimals(step: float) -> int:
    """Number of decimal places implied by stepSize (e.g., 0.001 -> 3)."""
    if step <= 0:
        return 8  # safe fallback
    s = f"{step:.20f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0

def _floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def _format_qty(qty: float, step: float) -> str:
    """Floor to step and format with exact decimals to avoid 'too much precision'."""
    dec = _step_decimals(step)
    floored = _floor_step(qty, step)
    return f"{floored:.{dec}f}"

# --- Trade entry point ---
def execute_trade(
    symbol: str,
    side: str,
    capital: float,
    balances: dict,
    desired_qty: float | None = None,
    min_notional_usdt: float = 0.0,
):
    """
    BUY (spot): spend 'capital' USDT via quoteOrderQty (must be >= min_notional_usdt).
    SELL: sell 'desired_qty' (or all available) formatted to LOT_SIZE, enforcing exchange MIN_NOTIONAL and config min_notional_usdt.
    """
    price = get_price(symbol)
    if price == 0.0:
        return {"error": "Price unavailable"}

    base_asset = symbol.replace("USDT", "")
    step, min_qty, ex_min_notional, _tick = _symbol_filters(symbol)
    cfg_min_notional = float(min_notional_usdt or 0.0)

    try:
        if side == "BUY":
            if capital <= 0.0:
                return {"error": "Invalid capital for BUY"}
            if cfg_min_notional and capital < cfg_min_notional:
                return {"error": f"Below config MIN_NOTIONAL ({cfg_min_notional} USDT)"}
            # Exchange enforces its own min notional; quoteOrderQty avoids LOT_SIZE qty precision issues
            order = client.order_market_buy(symbol=symbol, quoteOrderQty=round(float(capital), 2))

        elif side == "SELL":
            available = float(balances.get(base_asset, 0.0))
            qty_target = float(desired_qty) if desired_qty is not None else available
            qty_floored = _floor_step(min(qty_target, available), step)
            if qty_floored <= 0.0:
                return {"error": f"No {base_asset} available to sell"}
            if qty_floored < min_qty:
                return {"error": f"Below minQty ({min_qty})"}
            notional = qty_floored * price
            min_required = max(ex_min_notional or 0.0, cfg_min_notional or 0.0)
            if min_required and notional < min_required:
                return {"error": f"Below MIN_NOTIONAL ({min_required} USDT)"}
            qty_str = _format_qty(qty_floored, step)  # strict decimals to avoid precision error
            order = client.order_market_sell(symbol=symbol, quantity=qty_str)

        else:
            return {"error": f"Invalid side: {side}"}

        ts = order.get("transactTime")
        qty_exec = float(order.get("executedQty") or order.get("origQty") or 0.0)
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty_exec if qty_exec > 0 else (float(_format_qty(capital / price, step)) if side == "BUY" else qty_floored),
            "price": price,
            "order_id": order.get("orderId"),
            "timestamp": ts,
        }

    except BinanceAPIException as e:
        return {"error": f"Binance API Exception: {e.message}"}
    except Exception:
        return {"error": traceback.format_exc()}
