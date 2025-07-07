import os
import pandas as pd
import numpy as np
import json
import altair as alt

LOG_DIR = "logs"
OUTPUT_FILE = os.path.join(LOG_DIR, "portfolio_simulated_equity.csv")
SUMMARY_FILE = os.path.join(LOG_DIR, "portfolio_summary.json")
CLEAN_LIST = os.path.join(LOG_DIR, "clean_equity_curves.csv")

print("\n[INFO] Building portfolio equity from cleaned strategies...")

# Load clean equity list
try:
    survivors = pd.read_csv(CLEAN_LIST)["file"].tolist()
except Exception as e:
    print(f"[ERROR] Failed to load clean_equity_curves.csv: {e}")
    survivors = []

curves = []

for fname in survivors:
    if not fname.startswith("equity_curve_") or not fname.endswith(".csv"):
        continue

    path = os.path.join(LOG_DIR, fname)
    try:
        df = pd.read_csv(path)
        if "equity" not in df.columns or "timestamp" not in df.columns:
            print(f"[WARN] Skipping {fname}: missing 'equity' or 'timestamp'")
            continue

        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        norm = df["equity"] / df["equity"].iloc[0]
        norm.name = fname.replace("equity_curve_", "").replace(".csv", "")
        curves.append(norm)
    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

if len(curves) == 0:
    print("[ERROR] No valid strategy equity curves found. Portfolio not built.")
else:
    combined = pd.concat(curves, axis=1, join="outer").sort_index()
    combined = combined.ffill().dropna()
    combined["portfolio"] = combined.mean(axis=1)

    portfolio_df = combined[["portfolio"]].reset_index().rename(columns={"portfolio": "equity", "index": "timestamp"})
    portfolio_df["rolling_max"] = portfolio_df["equity"].cummax()
    portfolio_df["drawdown"] = portfolio_df["equity"] / portfolio_df["rolling_max"] - 1
    portfolio_df["returns"] = portfolio_df["equity"].pct_change()
    portfolio_df["volatility"] = portfolio_df["returns"].rolling(window=30).std()

    # Save portfolio equity curve
    out = portfolio_df[["timestamp", "equity"]]
    out.to_csv(OUTPUT_FILE, index=False)

    # Compute summary stats
    total_return = portfolio_df["equity"].iloc[-1] / portfolio_df["equity"].iloc[0] - 1
    days = (portfolio_df["timestamp"].iloc[-1] - portfolio_df["timestamp"].iloc[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan
    sharpe = np.sqrt(252) * portfolio_df["returns"].mean() / portfolio_df["returns"].std() if portfolio_df["returns"].std() else np.nan
    max_dd = portfolio_df["drawdown"].min()

    summary = {
        "CAGR": round(cagr, 6),
        "Sharpe": round(sharpe, 3),
        "Return": round(total_return, 6),
        "MaxDrawdown": round(max_dd, 6),
        "Start": str(portfolio_df["timestamp"].iloc[0].date()),
        "End": str(portfolio_df["timestamp"].iloc[-1].date()),
        "Length": len(portfolio_df)
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Portfolio simulated equity saved to: {OUTPUT_FILE}")
    print(f"[INFO] Portfolio summary saved to: {SUMMARY_FILE}")
    print(summary)

    # Altair visual QC
    equity_chart = alt.Chart(portfolio_df).mark_line().encode(
        x="timestamp:T",
        y="equity:Q"
    ).properties(title="Portfolio Equity", height=300)

    drawdown_chart = alt.Chart(portfolio_df).mark_area(color="darkred").encode(
        x="timestamp:T",
        y="drawdown:Q"
    ).properties(title="Rolling Drawdown", height=100)

    volatility_chart = alt.Chart(portfolio_df).mark_line(color="orange").encode(
        x="timestamp:T",
        y="volatility:Q"
    ).properties(title="Rolling Volatility (30d)", height=100)

    (equity_chart & drawdown_chart & volatility_chart).show()
