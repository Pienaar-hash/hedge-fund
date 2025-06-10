import pandas as pd
import numpy as np
import os

# Config
ASSETS = [
    ('BTCUSDT', '1H'),
    ('ETHUSDT', '1H'),
    ('SOLUSDT', '1H'),
    ('BTCUSDT', '1D'),
    ('ETHUSDT', '1D'),
    ('SOLUSDT', '1D'),
    ('BNBUSDT', '1D'),
    ('ADAUSDT', '1D'),
    ('AVAXUSDT', '1D'),
    ('DOGEUSDT', '1D')
]

MOMENTUM_WINDOW = 14

def compute_momentum(df):
    df['momentum'] = df['close'] - df['close'].shift(MOMENTUM_WINDOW)
    df.dropna(inplace=True)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'momentum']]

def process_asset(symbol, tf):
    file_name = f"{symbol}_{tf.lower()}.csv"
    input_path = os.path.join("data/processed", file_name)
    if not os.path.exists(input_path):
        print(f"❌ Source file not found: {input_path}. Skipping...")
        return

    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    df = compute_momentum(df)

    out_path = f"data/processed/momentum_{symbol.lower()}_{tf.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    for symbol, tf in ASSETS:
        process_asset(symbol, tf)
