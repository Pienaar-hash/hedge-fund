import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_dir = "logs"
trade_logs = []

for file in sorted(os.listdir(log_dir)):
    if file.startswith("vol_target_backtest_trades_") and file.endswith(".csv"):
        path = os.path.join(log_dir, file)
        symbol = file.replace("vol_target_backtest_trades_", "").replace(".csv", "").upper()
        df = pd.read_csv(path, parse_dates=['entry_time'])
        df['Symbol'] = symbol
        trade_logs.append(df)

if not trade_logs:
    print("⚠️ No trade logs found in logs/")
    exit()

combined = pd.concat(trade_logs)

# Plot: Return distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=combined, x='net_return_pct', hue='Symbol', kde=True, bins=100, element='step')
plt.title("Distribution of Net Trade Returns by Asset")
plt.xlabel("Net Return %")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/vol_target_return_distribution.png")
print("✅ Saved return distribution to logs/vol_target_return_distribution.png")

# Plot: Cumulative trade count
plt.figure(figsize=(12, 6))
combined['count'] = 1
combined = combined.sort_values('entry_time')
combined['cum_trades'] = combined.groupby('Symbol')['count'].cumsum()
sns.lineplot(data=combined, x='entry_time', y='cum_trades', hue='Symbol')
plt.title("Cumulative Trades Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Trades")
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/vol_target_cumulative_trades.png")
print("✅ Saved cumulative trade count to logs/vol_target_cumulative_trades.png")
