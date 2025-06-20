import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

log_folder = "logs"
summary = []

# Parse filenames
for file in os.listdir(log_folder):
    if file.startswith("backtest_relative_value_") and file.endswith(".csv"):
        filepath = os.path.join(log_folder, file)
        print(f"\U0001F4C4 Checking: {file}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
            continue

        if not df.empty and 'pnl' in df.columns:
            parts = file.replace("backtest_relative_value_", "").replace(".csv", "").split("_")
            if len(parts) == 3:
                base, quote, tf = parts
                pair = f"{base}/{quote}"
            else:
                print(f"⚠️ Unexpected filename format: {file}")
                continue

            pnl = df['pnl']
            sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0
            cumulative = pnl.sum()
            summary.append({
                'pair': pair,
                'timeframe': tf,
                'sharpe_ratio': sharpe_ratio,
                'cumulative_return': cumulative,
                'num_trades': len(pnl)
            })
        else:
            print(f"⚠️ Missing 'pnl' in {file}")

# Create summary DataFrame
summary_df = pd.DataFrame(summary)
if summary_df.empty:
    print("⚠️ No valid backtest files found or all files missing 'pnl'.")
    exit()

summary_df.to_csv("logs/relative_value_pair_summary.csv", index=False)

# Plot top pairs per timeframe
top_n = 10
for tf in summary_df['timeframe'].unique():
    subset = summary_df[summary_df['timeframe'] == tf]
    top = subset.sort_values(by='sharpe_ratio', ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top, x='pair', y='sharpe_ratio', palette="RdYlGn")
    plt.title(f"Top {top_n} Pairs by Sharpe Ratio – {tf}")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"logs/top_relative_value_pairs_{tf}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved plot for {tf} to {filename}")

print("✅ Pair summary and all bar plots saved.")
