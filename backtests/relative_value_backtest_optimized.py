import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from datetime import datetime

# Ensure project root is in path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import compute_sharpe, compute_volatility, compute_max_drawdown, compute_cagr

# --- CONFIG ---
base = "ETHUSDT"
pairs = ["BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
timeframes = ["1d"]
data_folder = "data/processed"
log_folder = "logs"

results = []

for quote in pairs:
    for tf in timeframes:
        file1 = f"{data_folder}/{base}_{tf}.csv"
        file2 = f"{data_folder}/{quote}_{tf}.csv"

        print(f"📂 Checking: {file1} vs {file2}")

        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"❌ Missing one or both files: {file1} or {file2}")
            continue

        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            df1['timestamp'] = pd.to_datetime(df1['timestamp'])
            df2['timestamp'] = pd.to_datetime(df2['timestamp'])

            merged = pd.merge(df1[['timestamp', 'close']], df2[['timestamp', 'close']], on='timestamp', suffixes=(f'_{base}', f'_{quote}'))
            merged = merged.dropna()

            # Hedge ratio via OLS
            X = add_constant(merged[f'close_{quote}'])
            model = OLS(merged[f'close_{base}'], X).fit()
            hedge_ratio = model.params.iloc[1]

            # Spread and z-score
            merged['spread'] = merged[f'close_{base}'] - hedge_ratio * merged[f'close_{quote}']
            merged['zscore'] = (merged['spread'] - merged['spread'].rolling(30).mean()) / merged['spread'].rolling(30).std()

            # Entry/exit logic
            z_entry = 1.5
            z_exit = 0.1
            in_position = False
            trades = []
            cumulative_pnl = 0
            equity = [1]

            for i in range(1, len(merged)):
                z = merged.loc[i, 'zscore']
                ts = merged.loc[i, 'timestamp']
                spread = merged.loc[i, 'spread']

                if not in_position and abs(z) > z_entry:
                    entry = spread
                    direction = -1 if z > 0 else 1
                    in_position = True

                elif in_position and abs(z) < z_exit:
                    exit = spread
                    pnl = (entry - exit) * direction
                    cumulative_pnl += pnl
                    trades.append({'timestamp': ts, 'pnl': pnl, 'cumulative_pnl': cumulative_pnl})
                    equity.append(equity[-1] * (1 + pnl))
                    in_position = False

            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df.to_csv(f"{log_folder}/backtest_relative_value_{base}_{quote}_{tf}.csv", index=False)

                equity_df = pd.DataFrame({
                    'timestamp': trades_df['timestamp'],
                    'equity': equity[1:]
                })
                equity_df.to_csv(f"{log_folder}/equity_curve_relative_value_{base.lower()}_{quote.lower()}_{tf}.csv", index=False)

                sharpe = compute_sharpe(trades_df['pnl'])
                result = {
                    'pair': f"{base}/{quote}",
                    'timeframe': tf,
                    'total_pnl': cumulative_pnl,
                    'num_trades': len(trades_df),
                    'sharpe_ratio': sharpe
                }
                results.append(result)

        except Exception as e:
            print(f"⚠️ Error processing {base}/{quote} at {tf}: {e}")

# --- Summary ---
if results:
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f"{log_folder}/relative_value_summary.csv", index=False)
    print("=== Comparative Summary ===")
    print(summary_df.sort_values(by='sharpe_ratio', ascending=False).to_string(index=False))
else:
    print("=== Comparative Summary ===")
    print("No valid backtests were run. Check for missing or misnamed files.")
