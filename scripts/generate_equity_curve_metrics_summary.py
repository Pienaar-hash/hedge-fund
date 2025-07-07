import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
OUTPUT_PATH = os.path.join(LOG_DIR, "equity_metrics_summary.csv")

records = []

for fname in os.listdir(LOG_DIR):
    if fname.startswith("equity_curve_") and fname.endswith(".csv"):
        path = os.path.join(LOG_DIR, fname)
        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns or "equity" not in df.columns:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df["equity"] = df["equity"].astype(float)

            df["returns"] = df["equity"].pct_change().fillna(0)
            days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
            years = days / 365.25 if days > 0 else 1

            cagr = (df["equity"].iloc[-1] / df["equity"].iloc[0])**(1 / years) - 1 if years > 0 else 0
            sharpe = df["returns"].mean() / df["returns"].std() * np.sqrt(252) if df["returns"].std() > 0 else 0
            mdd = (df["equity"] / df["equity"].cummax() - 1).min()
            vol = df["returns"].std() * np.sqrt(252)
            total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1

            label = fname.replace("equity_curve_", "").replace(".csv", "")
            records.append({
                "strategy": label,
                "CAGR": cagr,
                "Sharpe": sharpe,
                "MaxDrawdown": mdd,
                "Volatility": vol,
                "Total Return": total_return
            })
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved equity metrics to {OUTPUT_PATH}")
