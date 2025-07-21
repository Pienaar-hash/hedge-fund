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
    def run(self):
        all_trades = []

        for quote in self.pairs:
            equity = self.starting_equity * self.capital_weight

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

            for i in range(self.lookback, len(merged) - self.hold_period):
                row = merged.iloc[i]
                z = row['zscore']
                trend = row['trend']
                spread = row['spread']
                timestamp = row.name

                if abs(z) < self.z_entry or abs(spread) < self.min_spread:
                    continue
                if (z > 0 and trend > 0) or (z < 0 and trend < 0):
                    continue

                direction = -1 if z > 0 else 1
                entry_price_base = row[f'close_{self.base}']
                entry_price_pair = row[f'close_{quote}']
                entry_spread = row['spread']
                entry_ts = timestamp
                exit_idx = i + self.hold_period
                stopped_out = False
                z_revert_exit = False
                max_dd = 0.1 if self.dd_mod and abs(z) < 2.0 else 0.25 if self.dd_mod else 0.2
                max_spread_move = 0

                for j in range(i + 1, min(exit_idx + 1, len(merged))):
                    cur_row = merged.iloc[j]
                    cur_base = cur_row[f'close_{self.base}']
                    cur_pair = cur_row[f'close_{quote}']
                    cur_spread = cur_row['spread']
                    cur_z = cur_row['zscore']

                    pnl = direction * ((cur_base / entry_price_base - 1) - (entry_price_pair / cur_pair - 1))
                    spread_move = abs(cur_spread - entry_spread)
                    max_spread_move = max(max_spread_move, spread_move)

                    if pnl < -max_dd:
                        exit_idx = j
                        stopped_out = True
                        break
                    if abs(cur_z) < self.z_exit:
                        exit_idx = j
                        z_revert_exit = True
                        break

                exit_row = merged.iloc[exit_idx]
                exit_price_base = exit_row[f'close_{self.base}']
                exit_price_pair = exit_row[f'close_{quote}']
                gross_ret = direction * ((exit_price_base / entry_price_base - 1) - (entry_price_pair / exit_price_pair - 1))
                net_ret = gross_ret - 2 * self.fee
                pnl = equity * net_ret
                equity += pnl

                all_trades.append({
                    "pair": f"{self.base}/{quote}",
                    "entry_time": entry_ts,
                    "exit_time": exit_row.name,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "capital_used": self.starting_equity * self.capital_weight,
                    "pnl_log_return": net_ret,
                    "z_entry": z,
                    "z_exit": merged.iloc[exit_idx]['zscore'],
                    "spread_move": max_spread_move,
                    "stopped_out": stopped_out,
                    "z_revert_exit": z_revert_exit
                })

        self.trades_df = pd.DataFrame(all_trades)
        self.log_results()

StrategyImpl = RelativeValue
