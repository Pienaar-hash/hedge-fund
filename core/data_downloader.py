# === core/data_downloader.py ===
import os
import time
import requests
import pandas as pd
from datetime import datetime

BINANCE_API = "https://api.binance.com/api/v3/klines"
DATA_DIR = "data/processed"
START_DATE = "2022-01-01"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_klines(symbol, interval="1h", start_date=START_DATE, end_date=None, limit=1000):
    df_all = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)

    while start_ts < end_ts:
        url = f"{BINANCE_API}?symbol={symbol}&interval={interval}&startTime={start_ts}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if not data or "code" in data:
                break

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            df_all.append(df)

            start_ts = int(data[-1][0]) + 1
            time.sleep(0.4)  # Binance rate limit safety
        except Exception as e:
            print(f"❌ Failed to fetch {symbol} {interval}: {e}")
            break

    if df_all:
        full_df = pd.concat(df_all).drop_duplicates(subset="timestamp").sort_values("timestamp")
        return full_df
    return pd.DataFrame()

def save_symbol(symbol, interval):
    print(f"📥 Downloading {symbol} {interval}")
    df = fetch_klines(symbol, interval)
    if not df.empty:
        out_path = os.path.join(DATA_DIR, f"{symbol.lower()}_{interval}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved to {out_path}")
    else:
        print(f"⚠️ No data fetched for {symbol} {interval}")

if __name__ == "__main__":
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT",
        "NEARUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT", "MKRUSDT", "INJUSDT", "GMXUSDT",
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "MATICUSDT", "ARBUSDT", "OPUSDT",
        "SUIUSDT", "LDOUSDT", "RUNEUSDT", "SNXUSDT"
    ]
    intervals = ["1h", "4h", "1d"]

    for symbol in symbols:
        for interval in intervals:
            save_symbol(symbol, interval)
