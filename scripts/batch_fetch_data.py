import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define fetch configuration
fetch_config = {
    "eth_btc_arb": {
        "symbols": ["ETH/USDT", "BTC/USDT", "ETH/BTC"],
        "timeframes": ["1h", "15m"],
        "months": 18
    },
    "momentum": {
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "timeframes": ["1h", "4h"],
        "months": 18
    },
    "volatility_targeting": {
        "symbols": ["BTC/USDT"],
        "timeframes": ["1d", "1h"],
        "months": 36
    },
    "factor_based": {
        "symbols": [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "SOL/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "TRX/USDT"
        ],
        "timeframes": ["1d"],
        "months": 36
    }
}

exchange = ccxt.binance()

def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return []

# Start fetching data
for strategy, config in fetch_config.items():
    for symbol in config["symbols"]:
        for timeframe in config["timeframes"]:
            duration_days = config["months"] * 30
            since = int((datetime.utcnow() - timedelta(days=duration_days)).timestamp() * 1000)
            print(f"Fetching {symbol} | {timeframe} | {config['months']} months...")
            data = fetch_ohlcv(symbol, timeframe, since)
            if data:
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                filename = f"{strategy}_{symbol.replace('/', '-')}_{timeframe}.csv"
                filepath = RAW_DATA_DIR / filename
                df.to_csv(filepath, index=False)
                print(f"Saved to {filepath}")
