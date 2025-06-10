# volatility_optimize.py

import pandas as pd
import numpy as np
from itertools import product

# Load data
df = pd.read_csv("data/btcusdt_cleaned_backtest.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Define parameter grid
lookback_vol_range = [12, 24, 48, 72]
vol_target_range = [0.005, 0.01, 0.02]

results = []

for lookback_vol, vol_target in product(lookback_vol_range, vol_target_range):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=lookback_vol).std()
    df['position'] = np.where(df['volatility'] < vol_target, 1, 0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    total_return = df['strategy_returns'].add(1).prod() - 1
    avg_return = df['strategy_returns'].mean()
    win_rate = (df['strategy_returns'] > 0).sum() / df['strategy_returns'].count()

    results.append({
        'lookback_vol': lookback_vol,
        'vol_target': vol_target,
        'total_return_pct': total_return * 100,
        'avg_return_per_trade': avg_return * 100,
        'win_rate_pct': win_rate * 100
    })

results_df = pd.DataFrame(results)
results_df.to_csv("logs/volatility_optimization_results.csv", index=False)
print("Volatility optimization complete. Results saved to logs/volatility_optimization_results.csv")
