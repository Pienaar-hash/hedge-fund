# === strategies/relative_value.py ===
import os
import numpy as np
import pandas as pd
from core.strategy_base import Strategy
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant

DATA_FOLDER = "data/processed"
LOG_FOLDER = "logs"

class RelativeValue(Strategy):
    def configure(self, params):
        self.base = params.get("base", "ETHUSDT")
        self.pairs = params.get("pairs", [])
        self.lookback = params.get("lookback", 20)
        self.z_entry = params.get("z_entry", 1.5)
        self.z_exit = params.get("z_exit", 0.5)
        self.hold_period = params.get("hold_period", 3)
        self.capital = params.get("capital", 100000)
        self.cap_weight = params.get("capital_weight", 0.05)
        self.timeframe = params.get("timeframe", "1d")
        self.fee = params.get("fee", 0.001)
        self.dd_mod = params.get("asymmetric_drawdown", True)

    def run(self):
        all_trades = []

        for quote in self.pairs:
            f1 = f"{DATA_FOLDER}/{self.base.lower()}_{self.timeframe}.csv"
            f2 = f"{DATA_FOLDER}/{quote.lower()}_{self.timeframe}.csv"
            if not os.path.exists(f1) or not os.path.exists(f2):
                continue

            df1 = pd.read_csv(f1, parse_dates=['timestamp'])
            df2 = pd.read_csv(f2, parse_dates=['timestamp'])
            merged = pd.merge(df1[['timestamp', 'close']], df2[['timestamp', 'close']], on='timestamp', suffixes=(f'_{self.base}', f'_{quote}'))
            merged.set_index("timestamp", inplace=True)

            X = add_constant(merged[f'close_{quote}'])
            rolling_model = RollingOLS(merged[f'close_{self.base}'], X, window=self.lookback).fit()
            merged['rolling_beta'] = rolling_model.params[f'close_{quote}']
            merged = merged.dropna().copy()

            merged['spread'] = merged[f'close_{self.base}'] - merged['rolling_beta'] * merged[f'close_{quote}']
            merged['zscore'] = (merged['spread'] - merged['spread'].rolling(self.lookback).mean()) / merged['spread'].rolling(self.lookback).std()
            merged['trend'] = merged['spread'].rolling(5).mean() - merged['spread'].rolling(20).mean()

            pair_trades = []

            for i in range(self.lookback, len(merged) - self.hold_period):
                row = merged.iloc[i]
                z = row['zscore']
                trend = row['trend']
                timestamp = row.name

                if abs(z) <= self.z_entry:
                    continue
                if (z > 0 and trend >= 0) or (z < 0 and trend <= 0):
                    continue

                direction = -1 if z > 0 else 1
                entry_price_base = row[f'close_{self.base}']
                entry_price_pair = row[f'close_{quote}']
                entry_ts = timestamp
                exit_idx = i + self.hold_period
                stopped_out = False

                max_dd = 0.1 if self.dd_mod and abs(z) < 2.0 else 0.25 if self.dd_mod else 0.2

                for j in range(i + 1, exit_idx + 1):
                    cur_base = merged.iloc[j][f'close_{self.base}']
                    cur_pair = merged.iloc[j][f'close_{quote}']
                    pnl = direction * ((cur_base / entry_price_base - 1) - (entry_price_pair / cur_pair - 1))
                    if pnl < -max_dd:
                        exit_idx = j
                        stopped_out = True
                        break

                exit_row = merged.iloc[exit_idx]
                exit_price_base = exit_row[f'close_{self.base}']
                exit_price_pair = exit_row[f'close_{quote}']
                gross_ret = direction * ((exit_price_base / entry_price_base - 1) - (entry_price_pair / exit_price_pair - 1))
                net_ret = gross_ret - 2 * self.fee
                capital_weight_dynamic = min(0.02 + abs(z) * 0.01, 0.1)
                capital_used = self.capital * capital_weight_dynamic
                pnl = capital_used * net_ret

                pair_trades.append({
                    "pair": f"{self.base}/{quote}",
                    "entry_time": entry_ts,
                    "exit_time": exit_row.name,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "capital_used": capital_used,
                    "pnl": pnl,
                    "z_entry": z,
                    "stopped_out": stopped_out
                })

            df = pd.DataFrame(pair_trades)
            if not df.empty:
                self.trades_df = df
                logfile = f"{LOG_FOLDER}/relative_value_trades_{self.base}_{quote}_z{self.z_entry}_h{self.hold_period}_ddmod{int(self.dd_mod)}.csv"
                df.to_csv(logfile, index=False)
                print(f"✅ Trades logged to {logfile}")

                df['timestamp'] = pd.to_datetime(df['exit_time'])
                df = df.sort_values('timestamp')
                df = df.drop_duplicates(subset='timestamp', keep='last')
                equity = df.set_index('timestamp')['pnl'].cumsum()
                equity = equity - equity.min() + self.capital
                equity_df = equity.reset_index().rename(columns={'pnl': 'equity'})
                equity_df.columns = ['timestamp', 'equity']
                equity_logfile = f"{LOG_FOLDER}/equity_curve_relative_value_{self.base}_{quote}_z{self.z_entry}_h{self.hold_period}_ddmod{int(self.dd_mod)}.csv"
                equity_df.to_csv(equity_logfile, index=False)
                print(f"✅ Equity curve saved to {equity_logfile}")
            else:
                print(f"⚠️ No trades for {self.base}/{quote}")
                gross_ret = direction * ((exit_price_base / entry_price_base - 1) - (entry_price_pair / exit_price_pair - 1))
                net_ret = gross_ret - 2 * self.fee
                capital_weight_dynamic = min(0.02 + abs(z) * 0.01, 0.1)
                capital_used = self.capital * capital_weight_dynamic
                pnl = capital_used * net_ret

                all_trades.append({
                    "pair": f"{self.base}/{quote}",
                    "entry_time": entry_ts,
                    "exit_time": exit_row.name,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "capital_used": capital_used,
                    "pnl": pnl,
                    "z_entry": z,
                    "stopped_out": stopped_out
                })

        df = pd.DataFrame(all_trades)
        self.trades_df = df  # ✅ Retain trades in memory for sweep use
        if not df.empty:
            logfile = f"{LOG_FOLDER}/relative_value_trades_{self.base}_{quote}_z{self.z_entry}_h{self.hold_period}_ddmod{int(self.dd_mod)}.csv"
            df.to_csv(logfile, index=False)
            print("✅ Trades logged to relative_value_trades.csv")

            # ✅ Patch equity curve for simulator compatibility
            df['timestamp'] = pd.to_datetime(df['exit_time'])
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset='timestamp', keep='last')
            equity = df.set_index('timestamp')['pnl'].cumsum()
            equity = equity - equity.min() + self.capital
            equity_df = equity.reset_index().rename(columns={'pnl': 'equity'})
            equity_df.columns = ['timestamp', 'equity']
            equity_logfile = f"{LOG_FOLDER}/equity_curve_relative_value_{self.base}_{quote}_z{self.z_entry}_h{self.hold_period}_ddmod{int(self.dd_mod)}.csv"
            equity_df.to_csv(equity_logfile, index=False)
            print("✅ Equity curve saved to equity_curve_relative_value.csv")
        else:
            print("⚠️ No trades executed.")


StrategyImpl = RelativeValue
