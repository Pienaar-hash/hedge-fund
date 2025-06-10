import pandas as pd
from pathlib import Path

# Define input/output directories
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Resample frequency mapping (customize per use case)
DEFAULT_FREQ = {
    "15m": "15T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D"
}

# Process all CSVs in /raw
for file in RAW_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Infer timeframe from filename (e.g., momentum_BTC-USDT_1h.csv)
        parts = file.stem.split("_")
        timeframe = parts[-1]
        freq = DEFAULT_FREQ.get(timeframe, "1H")

        df_resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        out_file = PROCESSED_DIR / file.name
        df_resampled.to_csv(out_file)
        print(f"Processed: {file.name} → {out_file.name}")

    except Exception as e:
        print(f"Failed to process {file.name}: {e}")
