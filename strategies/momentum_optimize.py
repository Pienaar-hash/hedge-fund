import pandas as pd
import numpy as np
from itertools import product

# Load backtest data
df = pd.read_csv("data/btcusdt_cleaned_backtest.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Define parameter grid
short_windows = range(3, 15, 2)  # 3 to 13
long_windows = range(10, 30, 5)  # 10 to 25

# Store results
results = []

for short, long in product(short_windows, long_windows):
    if short >= long:
        continue

    df['short_ma'] = df['close'].rolling(window=short).mean()
    df['long_ma'] = df['close'].rolling(window=long).mean()

    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
    df['position'] = df['signal'].shift(1)

    df['return'] = df['close'].pct_change()
    df['strategy'] = df['position'] * df['return']

    trades = df[df['position'].diff() != 0].copy()
    trades['cumulative'] = (1 + df['strategy']).cumprod()

    total_return = df['strategy'].sum() * 100
    win_rate = (df['strategy'] > 0).mean() * 100
    avg_return = df['strategy'].mean() * 100

    results.append({
        'short_ma': short,
        'long_ma': long,
        'total_return_pct': round(total_return, 2),
        'win_rate_pct': round(win_rate, 2),
        'avg_return_pct': round(avg_return, 4),
    })

# Save results
results_df = pd.DataFrame(results)
results_df.sort_values(by='total_return_pct', ascending=False, inplace=True)
results_df.to_csv("logs/momentum_optimization_results.csv", index=False)

print("Optimization complete. Results saved to logs/momentum_optimization_results.csv")
