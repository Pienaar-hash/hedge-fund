import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
OUT_PREFIX = "portfolio_simulated_equity_"
records = []

def score_curve(path):
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns or "equity" not in df.columns or df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        returns = df["equity"].pct_change().dropna()
        if returns.std() == 0:
            return None
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        cagr = (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (365 / len(df)) - 1
        if sharpe > 15 or cagr > 10:
            return None
        return sharpe, cagr, df
    except:
        return None

def extract_group_key(fname, strategy):
    parts = fname.replace(".csv", "").split("_")
    if strategy == "momentum":
        if len(parts) >= 4:
            return parts[2].lower()  # asset like btcusdt
        return "unknown"
    elif strategy == "relative_value":
        if len(parts) >= 5:
            base = parts[2].lower()
            quote = parts[3].lower()
            return f"{base}_{quote}"
        return "unknown"
    return "misc"

# === Scan and evaluate all relval and momentum equity curves ===
for fname in os.listdir(LOG_DIR):
    if not fname.startswith("equity_curve_") or not fname.endswith(".csv"):
        continue
    strategy = None
    if "momentum" in fname:
        strategy = "momentum"
    elif "relative_value" in fname:
        strategy = "relative_value"
    else:
        continue

    path = os.path.join(LOG_DIR, fname)
    result = score_curve(path)
    if result is None:
        continue
    sharpe, cagr, df = result

    group = extract_group_key(fname, strategy)
    key = f"{strategy}_{group}"

    records.append({
        "file": fname,
        "key": key,
        "strategy": strategy,
        "group": group,
        "sharpe": sharpe,
        "cagr": cagr
    })

# === Pick best per strategy + group ===
df_perf = pd.DataFrame(records)
best_per_group = df_perf.sort_values("sharpe", ascending=False).groupby("key").first().reset_index()

# === Promote to portfolio_simulated_equity_<strategy>_<group>.csv ===
for _, row in best_per_group.iterrows():
    src = os.path.join(LOG_DIR, row["file"])
    dst = os.path.join(LOG_DIR, f"{OUT_PREFIX}{row['strategy']}_{row['group']}.csv")
    df = pd.read_csv(src)[["timestamp", "equity"]].dropna()
    df.to_csv(dst, index=False)
    print(f"✅ Promoted {row['strategy']} {row['group']} → {dst}")

print("🏁 Done. You can now run: python -m core.portfolio_simulator")
