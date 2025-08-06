import time
import json
import os
from binance.client import Client
from pathlib import Path

# Testnet credentials
API_KEY = "Qb2r9fZI7CMgE05cE1L0Bq6NfzIofzVzU0rYJMCJ3yAtbkPpI6iElikiBfUt1k7V"
API_SECRET = "2DB812uJt1f0nBnjSLSydPf8DCNiB9DtEXg8gQDDu4RvdwtWT5KMZymo4OheQiOu"
BASE_URL = "https://testnet.binance.vision"

client = Client(API_KEY, API_SECRET)
client.API_URL = BASE_URL

signal_dir = "data/signals"
executed_log = "data/state/executed.json"
trade_log = "data/state/trade_log.json"
pnl_log = "data/state/pnl_log.json"
unrealized_file = "data/state/unrealized.json"
balance_file = "data/state/balances.json"

# Strategy settings
position_size_pct = 0.05  # 5% of balance per trade
leverage = 2

# Load executed + trade logs
executed = {}
if os.path.exists(executed_log):
    with open(executed_log, "r") as f:
        executed = json.load(f)

trade_history = []
if os.path.exists(trade_log):
    with open(trade_log, "r") as f:
        trade_history = json.load(f)

realized_pnl = []
if os.path.exists(pnl_log):
    with open(pnl_log, "r") as f:
        realized_pnl = json.load(f)

def place_order(symbol, side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side.upper(),
            type="MARKET",
            quantity=quantity
        )
        print(f"✅ Order placed: {symbol} {side.upper()} qty={quantity}")
        return order
    except Exception as e:
        print(f"❌ Failed to place order: {symbol} → {e}")
        return None

def get_min_qty(symbol):
    info = client.get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            return float(f["minQty"])
    return 0.001

def get_price(symbol):
    return float(client.get_symbol_ticker(symbol=symbol)['price'])

def get_usdt_balance():
    balance = client.get_asset_balance(asset='USDT')
    return float(balance['free']) if balance else 0.0

def print_order_history(symbol):
    try:
        orders = client.get_all_orders(symbol=symbol)
        print(f"📄 Last 5 orders for {symbol}:")
        for o in orders[-5:]:
            print(f"  ↪️ {o['side']} | {o['status']} | Qty: {o['origQty']} | Price: {o['price']}")
    except Exception as e:
        print(f"⚠️ Could not fetch order history for {symbol}: {e}")

def print_usdt_balance():
    try:
        balance = client.get_asset_balance(asset='USDT')
        print(f"💰 USDT Balance: {balance['free']} free, {balance['locked']} locked")
    except Exception as e:
        print(f"⚠️ Could not fetch balance: {e}")

def log_trade(symbol, side, qty, price, ts):
    entry = {
        "symbol": symbol,
        "side": side.upper(),
        "quantity": qty,
        "price": price,
        "timestamp": ts
    }
    trade_history.append(entry)
    with open(trade_log, "w") as f:
        json.dump(trade_history, f, indent=2)

def log_realized_pnl(symbol, pnl, ts):
    entry = {
        "symbol": symbol,
        "realized_pnl": pnl,
        "timestamp": ts
    }
    realized_pnl.append(entry)
    with open(pnl_log, "w") as f:
        json.dump(realized_pnl, f, indent=2)

def get_avg_entry(symbol):
    entries = [t for t in trade_history if t["symbol"] == symbol]
    if not entries:
        return 0
    total_qty = sum(float(t["quantity"]) for t in entries)
    if total_qty == 0:
        return 0
    total_val = sum(float(t["quantity"]) * float(t["price"]) for t in entries)
    return round(total_val / total_qty, 2)

def calc_position_size(symbol):
    price = get_price(symbol)
    balance = get_usdt_balance()
    exposure = balance * position_size_pct * leverage
    qty = round(exposure / price, 6)
    return max(qty, get_min_qty(symbol))

def calc_unrealized_pnl(symbol):
    avg_entry = get_avg_entry(symbol)
    current_price = get_price(symbol)
    entries = [t for t in trade_history if t["symbol"] == symbol]
    total_qty = sum(float(t["quantity"]) for t in entries)
    side = entries[-1]["side"] if entries else ""
    if avg_entry == 0 or total_qty == 0:
        return 0
    pnl = (current_price - avg_entry) * total_qty if side == "BUY" else (avg_entry - current_price) * total_qty
    return round(pnl, 2)

def get_net_realized_pnl():
    return round(sum(entry['realized_pnl'] for entry in realized_pnl), 2)

def get_net_unrealized_pnl():
    symbols = set(t["symbol"] for t in trade_history)
    return round(sum(calc_unrealized_pnl(sym) for sym in symbols), 2)

def log_unrealized():
    symbols = set(t["symbol"] for t in trade_history)
    output = {sym: calc_unrealized_pnl(sym) for sym in symbols}
    with open(unrealized_file, "w") as f:
        json.dump(output, f, indent=2)

def log_balance():
    try:
        balances = client.get_account()["balances"]
        usdt = next(b for b in balances if b["asset"] == "USDT")
        with open(balance_file, "w") as f:
            json.dump({"USDT": usdt}, f, indent=2)
    except Exception as e:
        print(f"⚠️ Balance logging failed: {e}")

def close_position(symbol):
    entries = [t for t in trade_history if t["symbol"] == symbol]
    if not entries:
        return
    qty = sum(float(t["quantity"]) for t in entries)
    avg_entry = get_avg_entry(symbol)
    side = "SELL" if entries[-1]["side"] == "BUY" else "BUY"
    result = place_order(symbol, side, qty)
    if result:
        exit_price = float(result['fills'][0]['price'])
        pnl = (exit_price - avg_entry) * qty if side == "SELL" else (avg_entry - exit_price) * qty
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_realized_pnl(symbol, round(pnl, 2), ts)
        print(f"💼 Realized PnL for {symbol}: {round(pnl, 2)} USDT")

while True:
    for file in os.listdir(signal_dir):
        if not file.endswith(".json"):
            continue

        path = os.path.join(signal_dir, file)
        with open(path, "r") as f:
            signal = json.load(f)

        symbol = signal["symbol"]
        signal_id = f"{symbol}-{signal['timestamp']}"

        if signal_id in executed:
            continue  # Already executed

        side = signal["signal"]
        qty = calc_position_size(symbol)

        result = place_order(symbol, side, qty)

        if result:
            executed[signal_id] = time.time()
            with open(executed_log, "w") as f:
                json.dump(executed, f)

            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            executed_price = result['fills'][0]['price']
            log_trade(symbol, side, qty, executed_price, ts)

            print_order_history(symbol)
            print_usdt_balance()
            avg = get_avg_entry(symbol)
            unrealized = calc_unrealized_pnl(symbol)
            realized = get_net_realized_pnl()
            total_unrealized = get_net_unrealized_pnl()
            print(f"📊 {symbol} Avg Entry: {avg} | Unrealized PnL: {unrealized} USDT")
            print(f"📈 Total Unrealized: {total_unrealized} | 💼 Total Realized: {realized} USDT")

            log_unrealized()
            log_balance()

    time.sleep(30)
