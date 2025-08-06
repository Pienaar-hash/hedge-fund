# executor_live.py — Portfolio Strategy Runner
import time
import numpy as np
import pandas as pd
import sys
import json
import logging
import os
from datetime import datetime
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from execution.telegram_utils import send_telegram, send_portfolio_summary_with_pnl
from execution.testnet_keys import API_KEY, API_SECRET

# === Logging ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]
sys.stdout.reconfigure(encoding='utf-8')

# === Settings ===
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
interval = Client.KLINE_INTERVAL_1MINUTE
lookback = 20
quantity_fraction = 0.99
min_order_size = 10
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# === Capital Allocation (20% per symbol default) ===
capital_allocation = {
    "BTCUSDT": 0.2,
    "ETHUSDT": 0.2,
    "SOLUSDT": 0.2,
}

# === Time Sync ===
def sync_time():
    try:
        server_time = client.get_server_time()["serverTime"]
        local_time = int(time.time() * 1000)
        client.timestamp_offset = server_time - local_time
        print(f"TIME OFFSET APPLIED: {client.timestamp_offset} ms")
    except Exception as e:
        print(f"ERROR: Failed to sync time: {e}")

sync_time()

# === Helpers ===
def get_step_size(symbol):
    info = client.get_symbol_info(symbol)
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return float(f['stepSize'])
    return 0.000001

def get_balance(asset):
    balance = client.get_asset_balance(asset=asset)
    return float(balance['free']) if balance else 0

def round_step_size(quantity, step_size):
    precision = int(round(-np.log10(step_size)))
    return round(quantity, precision)

def fetch_ohlcv(symbol):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=100)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_vol', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['log_return'].rolling(lookback).sum()
    df['vol'] = df['log_return'].rolling(lookback).std()
    df['zscore'] = df['momentum'] / df['vol']
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    df.dropna(inplace=True)
    return df

def write_nav_log(symbol, equity):
    # Deprecated — no longer used
    pass

def write_nav_snapshot():
    snapshot = {}
    for symbol in symbols:
        asset = symbol.replace("USDT", "")
        price = fetch_ohlcv(symbol).iloc[-1]['close']
        qty = get_balance(asset)
        usdt = get_balance("USDT") if asset == "USDT" else 0
        snapshot[symbol] = round(qty * price + usdt, 2)

    log_path = "nav_log.json"
    try:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),
            "equity": snapshot
        })

        with open(log_path, "w") as f:
            json.dump(logs[-1000:], f, indent=2)

        print(f"✅ NAV snapshot saved: {snapshot}")
    except Exception as e:
        print(f"NAV LOG ERROR: {e}")

def write_trade_log(entry):
    try:
        with open("trade_log.json", "r") as f:
            data = json.load(f)
    except:
        data = []
    data.append(entry)
    with open("trade_log.json", "w") as f:
        json.dump(data[-1000:], f, indent=2)

def place_order(symbol, side, quantity):
    try:
        order = client.order_market(symbol=symbol, side=side, quantity=quantity)
        print(f"ORDER: {side} {quantity} of {symbol}")
        send_telegram(f"✅ {side} {symbol} @ qty {quantity}")
        return order
    except BinanceAPIException as e:
        print(f"ORDER ERROR: {e}")
        send_telegram(f"❌ Order Error for {symbol}: {e}")
        return None

def trade_logic(symbol, state):
    asset = symbol.replace("USDT", "")
    df = fetch_ohlcv(symbol)
    latest = df.iloc[-1]
    price = latest['close']
    z = latest['zscore']
    trend = "UP" if latest['ema_fast'] > latest['ema_slow'] else "DOWN"
    vol = latest['vol']
    signal = z > 0.5 and trend == "UP" and vol > 0.0003

    print(f"{symbol} | Close: {price:.2f} | Z: {z:+.2f} | Trend: {trend} | Vol: {vol:.4f} | Signal: {'BUY' if signal else '-'}")

    step = get_step_size(symbol)
    usdt = get_balance("USDT")
    coin = get_balance(asset)
    equity = usdt + coin * price

    alloc = capital_allocation.get(symbol, 0.2)

    if not state['in_position'] and signal and usdt > min_order_size:
        qty = round_step_size((usdt * alloc) / price, step)
        order = place_order(symbol, SIDE_BUY, qty)
        if order:
            state.update({"in_position": True, "entry": price, "qty": qty})
            write_trade_log({"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "BUY", "price": price, "qty": qty})

    elif state['in_position'] and (price < state['entry'] * 0.995 or z < -0.5):
        order = place_order(symbol, SIDE_SELL, state['qty'])
        if order:
            pnl = (price - state['entry']) * state['qty']
            print(f"EXIT {symbol} @ {price:.2f} | PnL: {pnl:.2f}")
            send_telegram(f"💰 EXIT {symbol} @ {price:.2f} | PnL: {pnl:.2f}")
            write_trade_log({"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "SELL", "price": price, "qty": state['qty'], "pnl": round(pnl, 2)})
            state.update({"in_position": False, "entry": 0, "qty": 0})

# === Main Loop ===
def run_portfolio_loop():
    print("STARTING PORTFOLIO EXECUTOR...")
    send_telegram("🚀 Portfolio Executor started on Binance Testnet.")
    state_map = {}
    last_alert_hour = -1

    for s in symbols:
        asset = s.replace("USDT", "")
        coin = get_balance(asset)
        price = fetch_ohlcv(s).iloc[-1]['close']
        in_position = coin > 0.0001
        state_map[s] = {
            "in_position": in_position,
            "entry": price if in_position else 0,
            "qty": coin if in_position else 0,
            "latest_price": price,
            "pnl": 0.0
        }

    while True:
        for sym in symbols:
            try:
                trade_logic(sym, state_map[sym])
            except Exception as e:
                print(f"ERROR in {sym}: {e}")

        try:
            write_nav_snapshot()
        except Exception as e:
            print(f"NAV SNAPSHOT ERROR: {e}")

        current_hour = datetime.utcnow().hour
        if current_hour % 4 == 0 and current_hour != last_alert_hour:
            send_portfolio_summary_with_pnl(state_map)
            last_alert_hour = current_hour

        time.sleep(60)

if __name__ == "__main__":
    run_portfolio_loop()
