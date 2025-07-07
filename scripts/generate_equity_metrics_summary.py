import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
OUTPUT_PATH = os.path.join(LOG_DIR, "trade_metrics_summary.csv")

records = []

for fname in os.listdir(LOG_DIR):
    if "trades" in fname and fname.endswith(".csv"):
        path = os.path.join(LOG_DIR, fname)
        df = pd.read_csv(path)
        if df.empty:
            continue

        strategy = fname.replace(".csv", "").replace("_trades", "").replace("momentum_", "").replace("relative_value_", "").replace("vol_target_", "")

        pnl_col = None
        for col in ["pnl_log_return", "net_ret", "pnl_pct"]:
            if col in df.columns:
                pnl_col = col
                break
        if not pnl_col:
            continue

        wins = df[df[pnl_col] > 0]
        losses = df[df[pnl_col] < 0]
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        avg_win = wins[pnl_col].mean() if not wins.empty else 0
        avg_loss = losses[pnl_col].mean() if not losses.empty else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss if len(df) > 0 else 0
        profit_factor = wins[pnl_col].sum() / abs(losses[pnl_col].sum()) if not losses.empty else float('inf')
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

        records.append({
            "strategy": strategy,
            "Total Trades": len(df),
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Expectancy": expectancy,
            "Profit Factor": profit_factor,
            "Payoff Ratio": payoff_ratio
        })

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved trade metrics to {OUTPUT_PATH}")
