import os
import pandas as pd
import numpy as np
from datetime import datetime

LOG_DIR = "logs"
OUTPUT = []
EXCLUDE = []

print("\n🔍 Scanning equity curve files in 'logs/'...")

for fname in os.listdir(LOG_DIR):
    if not fname.startswith("equity_curve_") or not fname.endswith(".csv"):
        continue

    path = os.path.join(LOG_DIR, fname)
    try:
        df = pd.read_csv(path)
        row_count = len(df)
        has_equity = "equity" in df.columns
        has_timestamp = "timestamp" in df.columns
        is_constant = False
        has_nan = df.isna().sum().sum() > 0
        days = np.nan
        total_return = np.nan
        start_date = end_date = "-"
        cagr = sharpe = np.nan
        exclude_reason = ""

        if has_equity and has_timestamp and row_count >= 2:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            start_date = df["timestamp"].iloc[0].date()
            end_date = df["timestamp"].iloc[-1].date()
            days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
            is_constant = df["equity"].nunique() <= 1
            total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
            years = days / 365.25 if days > 0 else np.nan
            cagr = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan
            df["returns"] = df["equity"].pct_change()
            sharpe = np.sqrt(252) * df["returns"].mean() / df["returns"].std() if df["returns"].std() else np.nan

            # Exclusion logic
            if df["equity"].iloc[0] <= 0:
                exclude_reason = "initial_equity <= 0"
            elif total_return < -1:
                exclude_reason = "return < -100%"
            elif cagr < -0.5:
                exclude_reason = "CAGR < -50%"
            elif row_count < 100:
                exclude_reason = "<100 rows"
            elif is_constant:
                exclude_reason = "constant equity"

        else:
            exclude_reason = "missing columns or too few rows"

        record = {
            "file": fname,
            "rows": row_count,
            "start": start_date,
            "end": end_date,
            "days": days,
            "return": total_return,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "exclude_reason": exclude_reason
        }

        if exclude_reason:
            EXCLUDE.append(record)
        else:
            OUTPUT.append(record)

    except Exception as e:
        print(f"❌ Failed to read {fname}: {e}")

# Convert to DataFrames
survivors_df = pd.DataFrame(OUTPUT)
exclude_df = pd.DataFrame(EXCLUDE)

# Save reports
survivors_df.to_csv(os.path.join(LOG_DIR, "clean_equity_curves.csv"), index=False)
combined = pd.concat([survivors_df.assign(status="included"), exclude_df.assign(status="excluded")], ignore_index=True)
combined.to_csv(os.path.join(LOG_DIR, "equity_log_validation_report.csv"), index=False)

print("\n✅ Validation complete.")
print(f"✔️ Survivors saved to logs/clean_equity_curves.csv")
print(f"📋 Full report saved to logs/equity_log_validation_report.csv")
print(survivors_df.sort_values("Sharpe", ascending=False).head(10))
