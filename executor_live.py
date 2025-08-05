# executor_live.py
import time
import numpy as np
import pandas as pd
import sys
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from config.testnet_keys import API_KEY, API_SECRET

sys.stdout.reconfigure(encoding='utf-8')

symbol = "BTCUSDT"
asset = "BTC"
quote = "USDT"
interval = Client.KLINE_INTERVAL_1MINUTE
lookback = 20
quantity_fraction = 0.99
min_order_size = 10

client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# === Time Sync Helper ===
def sync_time():
    try:
        server_time = client.get_server_time()["serverTime"]
        local_time = int(time.time() * 1000)
        offset = server_time - local_time
        client.timestamp_offset = offset
        print(f"TIME OFFSET APPLIED: {offset} ms")
    except Exception as e:
        print(f"ERROR: Failed to sync time: {e}")

sync_time()

def fetch_ohlcv(symbol, interval, limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
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

def get_balance(asset):
    balance = client.get_asset_balance(asset=asset)
    return float(balance['free']) if balance else 0

def print_balances():
    usdt = get_balance("USDT")
    btc = get_balance("BTC")
    print(f"BALANCES — USDT: {usdt:.2f} | BTC: {btc:.6f}")

def place_order(side, quantity):
    try:
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity
        )
        print(f"ORDER EXECUTED: {side} {quantity} {asset}")
        return order
    except BinanceAPIException as e:
        print(f"ORDER ERROR: {e}")
        return None

def run_loop():
    in_position = False
    entry_price = 0
    bought_qty = 0

    while True:
        try:
            df = fetch_ohlcv(symbol, interval)
            latest = df.iloc[-1]
            price = latest['close']
            momentum = latest['momentum']
            vol = latest['vol']
            z = latest['zscore']
            trend = "UP" if latest['ema_fast'] > latest['ema_slow'] else "DOWN"
            signal_flag = "BUY" if z > 0.5 and trend == "UP" and vol > 0.0003 else "-"

            print(f"\n{latest.name} | Close: {price:.2f} | Momentum: {momentum:+.4f} | Vol: {vol:.4f} | Z: {z:+.2f} | Trend: {trend} | Signal: {signal_flag}")
            print_balances()
            print(f"EQUITY: {price:.2f} {quote}")

            if not in_position and signal_flag == "BUY":
                usdt_balance = get_balance(quote)
                notional = usdt_balance * quantity_fraction
                quantity = round(notional / price, 6)

                if notional >= min_order_size:
                    order = place_order(SIDE_BUY, quantity)
                    if order:
                        in_position = True
                        entry_price = price
                        bought_qty = quantity
                        print(f"LONG ENTRY @ {entry_price:.2f} | Qty: {bought_qty}")
                else:
                    print(f"WARNING: Not enough USDT to buy (Available: {usdt_balance:.2f})")

            elif in_position and (price < entry_price * 0.995 or z < -0.5):
                order = place_order(SIDE_SELL, bought_qty)
                if order:
                    pnl = (price - entry_price) * bought_qty
                    print(f"EXIT @ {price:.2f} | PnL: {pnl:.2f} USDT")
                    in_position = False
                    entry_price = 0
                    bought_qty = 0

            else:
                print("INFO: No action. Monitoring...")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(60)

if __name__ == "__main__":
    print("STARTING LIVE EXECUTOR ON BINANCE TESTNET...")
    run_loop()
