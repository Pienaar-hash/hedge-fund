# fix_relative_value_equity_curves.py
import os
import pandas as pd
import numpy as np

FOLDER = "logs"
PREFIX = "equity_curve_relative_value_"
SUFFIX = "_1d.csv"

for fname in os.listdir(FOLDER):
    if fname.startswith(PREFIX) and fname.endswith(SUFFIX):
        path = os.path.join(FOLDER, fname)
        try:
            df = pd.read_csv(path)
            original_len = len(df)
            
            # Clean equity: remove inf, NaN, extreme outliers
            df = df[np.isfinite(df['equity'])]
            df = df[df['equity'].abs() < 1e5]

            # Save cleaned
            df.to_csv(path, index=False)
            print(f"✅ Cleaned {fname}: {original_len} → {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Failed on {fname}: {e}")
