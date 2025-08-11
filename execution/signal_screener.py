import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

BINANCE_BASE = "https://api.binance.com"

def fetch_candles(symbol="BTCUSDT", interval="4h", limit=200):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["z_score"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    df["momentum"] = df["close"] - df["close"].shift(4)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signal(df):
    latest = df.iloc[-1]
    signal = None

    if latest["z_score"] > 1.5 and latest["rsi"] > 60 and latest["momentum"] > 0:
        signal = "BUY"
    elif latest["z_score"] < -1.5 and latest["rsi"] < 40 and latest["momentum"] < 0:
        signal = "SELL"

    return {
        "signal": signal,
        "z_score": round(latest["z_score"], 2),
        "rsi": round(latest["rsi"], 2),
        "momentum": round(latest["momentum"], 2),
        "price": round(latest["close"], 2)
    }

def get_trade_signal():
    return {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "price": 120000,
        "z_score": 2.5,
        "rsi": 65,
        "momentum": 150
    }

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def log_trade(entry, path="logs/trade_log.json"):
    log = load_json(path)
    timestamp = datetime.utcnow().isoformat()
    log[timestamp] = entry
    save_json(path, log)
