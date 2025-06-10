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
        print(f"📄 Checking: {file}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
            continue

        print(f"   Rows: {len(df)}, Columns: {df.columns.tolist()}")

        if not df.empty and 'pnl' in df.columns:
            parts = file.replace("backtest_relative_value_", "").replace(".csv", "").split("_")
            pair = f"{parts[0]}/{parts[1]}"

            pnl = df['pnl']
            sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0
            cumulative = pnl.sum()
            summary.append({
                'pair': pair,
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

# Bar plot of Sharpe ratios
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df.sort_values(by='sharpe_ratio', ascending=False), x='pair', y='sharpe_ratio', palette="RdYlGn")
plt.xticks(rotation=45)
plt.title("Relative Value Strategy - Sharpe Ratios by Pair")
plt.tight_layout()
plt.savefig("logs/relative_value_pair_sharpe_barplot.png")
plt.close()

print("✅ Pair summary and bar plot saved.")
