# fetch_binance_data.py
import ccxt
import pandas as pd
from datetime import datetime, timedelta

binance = ccxt.binance()

SYMBOLS = ['ETH/USDT', 'BTC/USDT']
SAVE_PATHS = {
    'ETH/USDT': 'data/eth_usdt.csv',
    'BTC/USDT': 'data/btc_usdt.csv'
}

def fetch_ohlcv(symbol, since, timeframe='1h'):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'close']].rename(columns={'close': 'price'})
    return df

def save_to_csv():
    since = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)
    for symbol in SYMBOLS:
        df = fetch_ohlcv(symbol, since)
        df.to_csv(SAVE_PATHS[symbol], index=False)
        print(f"✅ Saved {symbol} to {SAVE_PATHS[symbol]}")

if __name__ == '__main__':
    save_to_csv()
    