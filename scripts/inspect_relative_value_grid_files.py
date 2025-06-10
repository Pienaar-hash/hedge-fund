import os
import pandas as pd

log_folder = "logs"

for file in sorted(os.listdir(log_folder)):
    if file.startswith("backtest_ETH_BNB_1d_") and file.endswith(".csv"):
        path = os.path.join(log_folder, file)
        df = pd.read_csv(path)
        print(f"\n📄 {file}")
        print(df.head(2))
        print(f"Columns: {df.columns.tolist()}")
