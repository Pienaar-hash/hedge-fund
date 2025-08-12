# execution/exchange_utils.py
import os
import math
import traceback
from binance.client import Client
from binance.exceptions import BinanceAPIException
from execution.utils import load_env_var

BINANCE_API_KEY = load_env_var("BINANCE_API_KEY")
BINANCE_API_SECRET = load_env_var("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("⚠️ BINANCE_API_KEY / BINANCE_API_SECRET not set — private endpoints will fail.")

TESTNET = str(os.getenv("BINANCE_TESTNET", "1")).lower() in ("1", "true", "yes")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=TESTNET)

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

# --- Helpers for symbol filters ---
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

def _floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def execute_trade(symbol: str, side: str, capital: float, balances: dict):
    """
    For BUY (spot): prefer quoteOrderQty=capital in USDT to avoid LOT_SIZE.
    For SELL: ignore `capital` and sell available (rounded to step), ensuring minQty & minNotional.
    """
    price = get_price(symbol)
    if price == 0.0:
        return {"error": "Price unavailable"}

    base_asset = symbol.replace("USDT", "")
    step, min_qty, min_notional, _tick = _symbol_filters(symbol)

    try:
        if side == "BUY":
            # Spend 'capital' USDT; Binance computes the qty.
            # Round capital to 2 dp for safety; Binance handles internal precision.
            qoq = max(0.0, float(capital))
            if qoq <= 0.0:
                return {"error": "Invalid capital for BUY"}
            order = client.order_market_buy(symbol=symbol, quoteOrderQty=round(qoq, 2))

        elif side == "SELL":
            available = float(balances.get(base_asset, 0.0))
            # Sell only what we actually hold (ignore capital here)
            sell_qty = _floor_step(available, step)
            if sell_qty <= 0.0:
                return {"error": f"No {base_asset} available to sell"}
            # Enforce minQty and minNotional
            if sell_qty < min_qty:
                return {"error": f"Below minQty ({min_qty})"}
            notional = sell_qty * price
            if min_notional and notional < min_notional:
                return {"error": f"Below MIN_NOTIONAL ({min_notional} USDT)"}
            order = client.order_market_sell(symbol=symbol, quantity=sell_qty)

        else:
            return {"error": f"Invalid side: {side}"}

        ts = order.get("transactTime")
        # For BUY with quoteOrderQty, Binance returns executedQty; compute qty & price from fills if needed
        qty = float(order.get("executedQty") or order.get("origQty") or 0.0)
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty if qty > 0 else (_floor_step(capital / price, step) if side == "BUY" else 0.0),
            "price": price,
            "order_id": order.get("orderId"),
            "timestamp": ts
        }

    except BinanceAPIException as e:
        return {"error": f"Binance API Exception: {e.message}"}
    except Exception:
        return {"error": traceback.format_exc()}
