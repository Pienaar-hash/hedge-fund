# execution/exchange_utils.py
import os
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

def execute_trade(symbol: str, side: str, capital: float, balances: dict):
    price = get_price(symbol)
    if price == 0.0:
        return {"error": "Price unavailable"}

    base_asset = symbol.replace("USDT", "")
    qty = round(capital / price, 6)

    try:
        if side == "BUY":
            order = client.order_market_buy(symbol=symbol, quantity=qty)
        elif side == "SELL":
            available = balances.get(base_asset, 0)
            if available < qty:
                return {"error": f"Insufficient {base_asset} to sell"}
            order = client.order_market_sell(symbol=symbol, quantity=min(qty, available))
        else:
            return {"error": f"Invalid side: {side}"}

        ts = order.get("transactTime")
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "order_id": order.get("orderId"),
            "timestamp": ts
        }

    except BinanceAPIException as e:
        return {"error": f"Binance API Exception: {e.message}"}
    except Exception:
        return {"error": traceback.format_exc()}
