import pandas as pd
import numpy as np
import os
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Config
base = "ETHUSDT"
pairs = ["BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
timeframe = "1d"
data_folder = "data/processed"
log_path = "logs/relative_value_monte_carlo.csv"
z_entry_vals = [1.0, 1.5, 2.0]
z_exit_vals = [0.1, 0.2, 0.3]
lookback = 60

results = []

for quote in pairs:
    file1 = f"{data_folder}/{base}_{timeframe}.csv"
    file2 = f"{data_folder}/{quote}_{timeframe}.csv"

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"❌ Missing data for {base}/{quote}. Skipping.")
        continue

    x_df = pd.read_csv(file1, parse_dates=['timestamp'])
    y_df = pd.read_csv(file2, parse_dates=['timestamp'])
    merged = pd.merge(x_df[['timestamp', 'close']], y_df[['timestamp', 'close']], on='timestamp', suffixes=("_x", "_y"))
    merged = merged.dropna()

    # Estimate hedge ratio
    X = add_constant(merged['close_y'])
    model = OLS(merged['close_x'], X).fit()
    hedge_ratio = model.params.iloc[1]

    # Create spread
    merged['spread'] = merged['close_x'] - hedge_ratio * merged['close_y']
    merged['zscore'] = (merged['spread'] - merged['spread'].rolling(lookback).mean()) / merged['spread'].rolling(lookback).std()

    for ze in z_entry_vals:
        for zx in z_exit_vals:
            in_position = False
            direction = 0
            entry_spread = 0
            pnl = []

            for i in range(lookback, len(merged)):
                z = merged['zscore'].iloc[i]
                spread = merged['spread'].iloc[i]

                if not in_position and abs(z) > ze:
                    entry_spread = spread
                    direction = -1 if z > 0 else 1
                    in_position = True

                elif in_position and abs(z) < zx:
                    profit = (entry_spread - spread) * direction
                    pnl.append(profit)
                    in_position = False

            if pnl:
                pnl = np.array(pnl)
                sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() != 0 else 0
                results.append({
                    "pair": f"{base}/{quote}",
                    "z_entry": ze,
                    "z_exit": zx,
                    "num_trades": len(pnl),
                    "cumulative_return": pnl.sum(),
                    "sharpe_ratio": sharpe
                })

# Save
results_df = pd.DataFrame(results)
results_df.to_csv(log_path, index=False)
print(f"✅ Saved sweep results to {log_path}")
