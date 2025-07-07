# === clean_logs_after_sweep.py ===
import os
import pandas as pd

LOG_DIR = "logs"
TOP_FILE = os.path.join(LOG_DIR, "momentum_top_configs.csv")

# Load top config labels
if not os.path.exists(TOP_FILE):
    print("❌ Top config file not found. Please run the sweep first.")
    exit()

print("📂 Cleaning logs...")
df = pd.read_csv(TOP_FILE)
df["Symbol"] = df["Label"].str.extract(r"momentum_(.*?)_")
df = (
    df.groupby("Symbol", group_keys=False)
    .apply(lambda g: g.sort_values("Sharpe", ascending=False).head(5))
    .reset_index(drop=True)
)
top_labels = set(df["Label"])

# Allowed files to keep
allowed_keywords = [
    "momentum_top_configs.csv"
] + [
    f"momentum_trades_{label}.csv" for label in top_labels
] + [
    f"equity_curve_momentum_{label}.csv" for label in top_labels
]

# Delete all other momentum-related logs
for fname in os.listdir(LOG_DIR):
    if any(keyword in fname for keyword in allowed_keywords):
        continue
    if fname.startswith("momentum") or "equity_curve_momentum" in fname:
        os.remove(os.path.join(LOG_DIR, fname))
        print(f"🗑️ Removed {fname}")

print("✅ Cleanup complete. Only top configs retained.")
