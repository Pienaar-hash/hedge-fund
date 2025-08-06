import json
from datetime import datetime
import os
import requests
from binance.client import Client

import pandas as pd
import numpy as np

# === Fetch OHLCV Data ===
def fetch_ohlcv(client, symbol, interval="1m", limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_vol', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['log_return'].rolling(20).sum()
    df['vol'] = df['log_return'].rolling(20).std()
    df['zscore'] = df['momentum'] / df['vol']
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    df.dropna(inplace=True)
    return df

# === Binance Client ===
client = Client()  # Set your keys if needed
SYNCED_STATE_PATH = "synced_state.json"
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Telegram Sender ===
def send_telegram(message):
    print(f"📨 BOT_TOKEN: {BOT_TOKEN}")
    print(f"📨 CHAT_ID: {CHAT_ID}")
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram BOT_TOKEN or CHAT_ID not set.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        print(f"📨 Request URL: {url}")
        print(f"📨 Payload: {payload}")
        print(f"📨 Status Code: {response.status_code}")
        print(f"📨 Response: {response.text}")
        if response.status_code != 200:
            print(f"Telegram send failed: {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

# === Portfolio Summary Sender ===
def send_portfolio_summary_with_pnl(state_map):
    nav_path = "nav_log.json"
    try:
        if not os.path.exists(nav_path):
            send_telegram("⚠️ No NAV log available.")
            return

        with open(nav_path, "r") as f:
            logs = json.load(f)
        latest = logs[-1]
        timestamp = latest["timestamp"]
        equity = latest["equity"]

        total_equity = sum(equity.values())
        lines = [f"📊 *Portfolio Summary* — {timestamp}"]
        for symbol, eq in equity.items():
            lines.append(f"{symbol}: ${eq:,.2f}")
        lines.append(f"🧾 *Total Equity*: ${total_equity:,.2f}\n")

        account_info = client.get_account()
        balances = {item['asset']: float(item['free']) for item in account_info['balances'] if float(item['free']) > 0}

        open_orders = client.get_open_orders()
        open_positions = {}

        for order in open_orders:
            symbol = order['symbol']
            side = order['side']
            qty = float(order['origQty'])
            if symbol not in open_positions:
                open_positions[symbol] = {"BUY": 0.0, "SELL": 0.0}
            open_positions[symbol][side] += qty

        synced_state = {}

        for symbol in state_map:
            asset = symbol.replace("USDT", "")
            if asset in balances:
                qty = balances[asset]
                price = fetch_ohlcv(client, symbol).iloc[-1]['close']
                entry = state_map[symbol].get("entry", price)
                pnl = (price - entry) * qty

                if qty > 0.0001:
                    state_map[symbol].update({
                        "in_position": True,
                        "qty": qty,
                        "latest_price": price,
                        "pnl": pnl
                    })
                    synced_state[symbol] = state_map[symbol]
                    lines.append(f"📈 {symbol} POS: Qty: {qty:.4f} | Entry: {entry:.2f} | Now: {price:.2f} | PnL: ${pnl:.2f}")

        if synced_state:
            with open(SYNCED_STATE_PATH, "w") as f:
                json.dump(synced_state, f, indent=2)

        msg = "\n".join(lines)
        send_telegram(msg)
    except Exception as e:
        send_telegram(f"❌ Portfolio summary error: {e}")
