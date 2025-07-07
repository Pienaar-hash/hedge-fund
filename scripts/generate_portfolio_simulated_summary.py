import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
OUTPUT_PATH = os.path.join(LOG_DIR, "portfolio_simulated_summary.csv")
INITIAL_CAPITAL = 10000

def compute_metrics(df, label):
    df = df.copy()
    df = df.sort_values("timestamp")
    df["returns"] = df["equity"].pct_change().fillna(0)
    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25

    equity = df["equity"]
    cagr = (equity.iloc[-1] / equity.iloc[0])**(1 / years) - 1 if years > 0 else 0
    sharpe = df["returns"].mean() / df["returns"].std() * np.sqrt(252) if df["returns"].std() > 0 else 0
    mdd = (equity / equity.cummax() - 1).min()
    vol = df["returns"].std() * np.sqrt(252)
    return {
        "strategy": label,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "Volatility": vol
    }

records = []

# Step 1: Load all equity curves
curves = {}
for fname in os.listdir(LOG_DIR):
    if fname.startswith("portfolio_simulated_equity_") and fname.endswith(".csv"):
        label = fname.replace("portfolio_simulated_equity_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(LOG_DIR, fname), parse_dates=["timestamp"])
        curves[label] = df
        records.append(compute_metrics(df, label))

# Step 2: Blended portfolio
if len(curves) > 0:
    # Align all timestamps
    aligned = pd.concat([df.set_index("timestamp")["equity"].rename(k) for k, df in curves.items()], axis=1)
    aligned = aligned.ffill().dropna()
    blended = aligned.mean(axis=1)
    blended_df = pd.DataFrame({"timestamp": blended.index, "equity": blended.values})
    blended_df = blended_df.reset_index()
    records.append(compute_metrics(blended_df, "portfolio_simulated_equity"))

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved summary to {OUTPUT_PATH}")
