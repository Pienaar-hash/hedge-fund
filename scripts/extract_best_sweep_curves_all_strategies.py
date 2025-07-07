import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
OUT_PREFIX = "portfolio_simulated_equity_"
records = []

def extract_label_from_filename(fname):
    if "vol" in fname:
        return fname.split("_")[3].lower()
    elif "relative_value" in fname:
        return fname.split("_")[3].lower()
    elif "momentum" in fname:
        parts = fname.split("_")
        if len(parts) > 3:
            return "_".join(parts[2:4]).lower()
        return "momentum"
    else:
        return "misc"

# 1. Load and evaluate each equity curve
for fname in os.listdir(LOG_DIR):
    if fname.startswith("equity_curve_") and fname.endswith(".csv"):
        path = os.path.join(LOG_DIR, fname)
        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns or "equity" not in df.columns or df.empty:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            ret = df["equity"].pct_change().dropna()
            if ret.std() == 0:
                continue

            sharpe = ret.mean() / ret.std() * np.sqrt(252)
            cagr = (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (365 / len(df)) - 1
            label = extract_label_from_filename(fname)

            # === Outlier filter ===
            if sharpe > 10 or cagr > 10:
                print(f"⚠️ Skipping outlier: {fname} (Sharpe={sharpe:.2f}, CAGR={cagr:.2%})")
                continue

            records.append({
                "file": fname,
                "label": label,
                "sharpe": sharpe,
                "cagr": cagr
            })
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

# 2. Pick best file per label
df_perf = pd.DataFrame(records)
best_per_label = df_perf.sort_values("sharpe", ascending=False).groupby("label").first().reset_index()

# 3. Save standard-formatted files
for _, row in best_per_label.iterrows():
    src = os.path.join(LOG_DIR, row["file"])
    dst = os.path.join(LOG_DIR, f"{OUT_PREFIX}{row['label']}.csv")
    pd.read_csv(src).to_csv(dst, index=False)
    print(f"✅ Saved best curve for {row['label']} → {dst}")

print("🏑 Done. Now run: python -m core.portfolio_simulator")
