# === momentum_post_sweep_ranker.py ===
import os
import pandas as pd
import numpy as np

LOG_DIR = "logs"
TRADE_PREFIX = "momentum_trades_"
SUMMARY_OUT = os.path.join(LOG_DIR, "momentum_top3_per_symbol.csv")

print("🔍 Scanning trade logs...")
rows = []

for fname in os.listdir(LOG_DIR):
    if fname.startswith(TRADE_PREFIX) and fname.endswith(".csv"):
        label = fname[len(TRADE_PREFIX):-4]  # strip prefix and .csv
        symbol = label.split("_")[0]  # extract e.g. btcusdt_1h

        try:
            df = pd.read_csv(os.path.join(LOG_DIR, fname))
            if df.empty or "pnl_log_return" not in df.columns:
                continue

            pnl = df["pnl_log_return"].dropna()
            if len(pnl) < 10:
                continue

            sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
            win = pnl[pnl > 0]
            loss = pnl[pnl < 0]

            rows.append({
                "Label": label,
                "Symbol": symbol,
                "Trades": len(pnl),
                "Sharpe": sharpe,
                "WinRate": len(win) / len(pnl),
                "ProfitFactor": win.sum() / abs(loss.sum()) if not loss.empty else np.inf
            })
        except Exception as e:
            print(f"⚠️ Failed on {fname}: {e}")

if not rows:
    print("❌ No valid trade logs found.")
    exit()

# Build summary dataframe
df = pd.DataFrame(rows)
top3 = df.groupby("Symbol", group_keys=False).apply(lambda g: g.sort_values("Sharpe", ascending=False).head(3))
top3.to_csv(SUMMARY_OUT, index=False)
top_labels = set(top3["Label"])
print(f"✅ Saved top 3 per symbol to {SUMMARY_OUT}")

# Clean up unused logs
for fname in os.listdir(LOG_DIR):
    if fname.startswith("momentum") or "equity_curve_momentum" in fname:
        if not any(label in fname for label in top_labels):
            os.remove(os.path.join(LOG_DIR, fname))
            print(f"🗑️ Removed {fname}")

# Clean rolling metrics, summaries, and other clutter
for fname in os.listdir(LOG_DIR):
    if (
        fname.startswith("rolling_metrics_momentum_")
        or fname.startswith("summary_momentum_")
        or fname.startswith("pnl_by_strength_")
        or fname.startswith("equity_curve_momentum_")
    ):
        if not any(label in fname for label in top_labels):
            os.remove(os.path.join(LOG_DIR, fname))
            print(f"🧽 Removed {fname}")
