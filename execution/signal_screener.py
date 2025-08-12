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

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["z_score"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    df["momentum"] = df["close"] - df["close"].shift(4)
    return df

def screen_symbol(symbol: str, tf: str, thresholds: dict):
    df = fetch_candles(symbol, interval=tf, limit=200)
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    sig = None
    if (latest["z_score"] > thresholds.get("zscore_threshold", 1.5)
        and latest["rsi"] > thresholds.get("rsi_buy", 60)
        and latest["momentum"] > 0):
        sig = "BUY"
    elif (latest["z_score"] < -thresholds.get("zscore_threshold", 1.5)
          and latest["rsi"] < thresholds.get("rsi_sell", 40)
          and latest["momentum"] < 0):
        sig = "SELL"
    return sig, round(latest["close"],2), round(latest["z_score"],2), round(latest["rsi"],2), round(latest["momentum"],2)

def generate_signals_from_config(config: dict):
    for strat in config.get("strategies", []):
        name = strat.get("name")
        params = strat.get("params", {})
        if name == "momentum":
            tf = params.get("timeframe", "4h")
            for sym in params.get("symbols", ["BTCUSDT"]):
                sig, price, z, rsi, mom = screen_symbol(sym, tf, params)
                if sig:
                    yield {
                        "strategy": f"{name}_{sym.lower()}",
                        "strategy_name": name,
                        "symbol": sym,
                        "signal": sig,
                        "price": price,
                        "z_score": z,
                        "rsi": rsi,
                        "momentum": mom,
                        "timeframe": tf
                    }
