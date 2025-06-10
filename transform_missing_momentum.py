import pandas as pd
import os

# Assets we need to transform
TARGETS = [
    ("BTCUSDT", "1D"),
    ("ETHUSDT", "1D"),
    ("SOLUSDT", "1D")
]

for symbol, tf in TARGETS:
    input_file = f"data/processed/{symbol}_{tf.lower()}.csv"
    output_file = f"data/processed/momentum_{symbol.lower()}_{tf.lower()}.csv"

    if not os.path.exists(input_file):
        print(f"❌ Missing source: {input_file}")
        continue

    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.sort_values('timestamp', inplace=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Transformed: {output_file}")
