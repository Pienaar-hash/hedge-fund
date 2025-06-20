# === strategies/relative_value.py ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

DATA_FOLDER = "data/processed"
LOG_FOLDER = "logs"

class StrategyImpl(Strategy):
    def configure(self, params):
        self.base = params.get("base", "ETHUSDT")
        self.pairs = params.get("pairs", [])
        self.lookback = params.get("lookback", 30)
        self.z_entry = params.get("z_entry", 1.5)
        self.z_exit = params.get("z_exit", 0.1)

    def run(self):
        self.results = []
        for quote in self.pairs:
            file1 = f"{DATA_FOLDER}/{self.base}_1d.csv"
            file2 = f"{DATA_FOLDER}/{quote}_1d.csv"

            if not os.path.exists(file1) or not os.path.exists(file2):
                print(f"❌ Missing data for {self.base}/{quote}. Skipping.")
                continue

            df1 = pd.read_csv(file1, parse_dates=['timestamp'])
            df2 = pd.read_csv(file2, parse_dates=['timestamp'])
            merged = pd.merge(df1[['timestamp', 'close']], df2[['timestamp', 'close']], on='timestamp', suffixes=(f'_{self.base}', f'_{quote}')).dropna()

            X = add_constant(merged[f'close_{quote}'])
            model = OLS(merged[f'close_{self.base}'], X).fit()
            hedge_ratio = model.params.iloc[1]

            merged['spread'] = merged[f'close_{self.base}'] - hedge_ratio * merged[f'close_{quote}']
            merged['zscore'] = (merged['spread'] - merged['spread'].rolling(self.lookback).mean()) / merged['spread'].rolling(self.lookback).std()

            in_position, cumulative_pnl, equity = False, 0, [1]
            trades = []

            for i in range(1, len(merged)):
                z = merged.loc[i, 'zscore']
                ts = merged.loc[i, 'timestamp']
                spread = merged.loc[i, 'spread']

                if not in_position and abs(z) > self.z_entry:
                    entry = spread
                    entry_ts = ts
                    direction = -1 if z > 0 else 1
                    in_position = True

                elif in_position and abs(z) < self.z_exit:
                    exit = spread
                    exit_ts = ts
                    pnl = (entry - exit) * direction
                    duration = (exit_ts - entry_ts).days
                    cumulative_pnl += pnl
                    trades.append({
                        'entry_time': entry_ts,
                        'exit_time': exit_ts,
                        'pnl': pnl,
                        'duration_days': duration,
                        'cumulative_pnl': cumulative_pnl
                    })
                    equity.append(equity[-1] + pnl)
                    in_position = False

            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_csv(f"{LOG_FOLDER}/backtest_relative_value_{self.base}_{quote}_1d.csv", index=False)

                equity_df = pd.DataFrame({
                    'timestamp': trades_df['exit_time'],
                    'equity': equity[1:]
                })
                equity_df.to_csv(f"{LOG_FOLDER}/equity_curve_relative_value_{self.base.lower()}_{quote.lower()}_1d.csv", index=False)

                sharpe = (trades_df['pnl'].mean() / trades_df['pnl'].std()) * np.sqrt(252) if trades_df['pnl'].std() > 0 else 0
                self.results.append({
                    'pair': f"{self.base}/{quote}",
                    'total_pnl': cumulative_pnl,
                    'num_trades': len(trades_df),
                    'sharpe_ratio': sharpe
                })

        all_curves = []
        for quote in self.pairs:
            path = f"logs/equity_curve_relative_value_{self.base.lower()}_{quote.lower()}_1d.csv"
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.rename(columns={"equity": quote})
                all_curves.append(df)
        if all_curves:
            combined = pd.concat(all_curves, axis=1).mean(axis=1)
            out = pd.DataFrame({"timestamp": combined.index, "equity": combined.values})
            out.to_csv("logs/equity_curve_relative_value.csv", index=False)
        self.log_results()

    def log_results(self):
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(f"{LOG_FOLDER}/relative_value_summary.csv", index=False)
            print(df.sort_values(by='sharpe_ratio', ascending=False).to_string(index=False))
        else:
            print("⚠️ No valid backtests were run. Check for missing files or data errors.")
