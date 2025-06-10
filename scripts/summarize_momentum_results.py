import os
import pandas as pd

LOG_DIR = "logs/"
SUMMARY_PATH = os.path.join(LOG_DIR, "momentum_trade_summary.csv")

trade_files = [f for f in os.listdir(LOG_DIR) if f.startswith("momentum_trades_") and f.endswith(".csv")]

summary_data = []

for file in trade_files:
    df = pd.read_csv(os.path.join(LOG_DIR, file))
    if df.empty:
        continue

    asset = file.replace("momentum_trades_", "").replace(".csv", "")
    trades = len(df)
    win_rate = (df['return_pct'] > 0).mean()
    avg_return = df['return_pct'].mean()
    std_dev = df['return_pct'].std()
    max_win = df['return_pct'].max()
    max_loss = df['return_pct'].min()
    cumulative_return = (1 + df['return_pct']).prod() - 1

    summary_data.append({
        "Asset": asset,
        "Trades": trades,
        "Win Rate": win_rate,
        "Avg Return": avg_return,
        "Std Dev": std_dev,
        "Max Win": max_win,
        "Max Loss": max_loss,
        "Cumulative Return": cumulative_return
    })

summary_df = pd.DataFrame(summary_data)
summary_df.sort_values(by="Cumulative Return", ascending=False, inplace=True)
summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"\u2705 Trade summary saved to {SUMMARY_PATH}")
print(summary_df)
