# volatility_targeting.py

import pandas as pd
import numpy as np

# Load cleaned BTC/USDT data
df = pd.read_csv("data/btcusdt_cleaned_backtest.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Compute returns and rolling volatility
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=24).std()

# Define volatility targeting threshold
vol_thresh = 0.005

# Generate signals based on volatility
# Buy (1) when vol < threshold, Sell (-1) when vol > threshold
df['signal'] = np.where(df['volatility'] < vol_thresh, 1, -1)

# Shift signal for execution and simulate trades
df['position'] = df['signal'].shift(1)
df['strategy_returns'] = df['position'] * df['returns']

# Detect entries and exits
entries = df[(df['position'] == 1) & (df['position'].shift(1) != 1)]
exits = df[(df['position'] != 1) & (df['position'].shift(1) == 1)]

entry_times = entries.index.to_list()
exit_times = exits.index.to_list()

# Match lengths and calculate trade returns
net_returns = []
for entry, exit in zip(entry_times, exit_times):
    entry_price = df.at[entry, 'close']
    exit_price = df.at[exit, 'close']
    net_returns.append((exit_price - entry_price) / entry_price * 100)

# ✅ PATCH: Ensure all arrays are equal length
min_len = min(len(entry_times), len(exit_times), len(net_returns))
trades = pd.DataFrame({
    "entry_time": entry_times[:min_len],
    "exit_time": exit_times[:min_len],
    "net_return_pct": net_returns[:min_len],
})

# Summary stats
total_trades = len(trades)
avg_return = trades['net_return_pct'].mean()
win_rate = (trades['net_return_pct'] > 0).mean() * 100
total_return = trades['net_return_pct'].sum()

print(f"Total Trades: {total_trades}")
print(f"Average Return per Trade: {avg_return:.2f}%")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total Return: {total_return:.2f}%")

# Save trades
trades.to_csv("logs/volatility_backtest_trades.csv", index=False)
