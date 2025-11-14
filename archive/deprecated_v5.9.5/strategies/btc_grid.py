# === strategies/btc_grid.py ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy, log_strategy_outputs

class StrategyImpl(Strategy):
    def configure(self, params):
        self.symbol = params.get("symbol", "BTCUSDT")
        self.timeframe = params.get("timeframe", "30m")
        self.grid_tp = params.get("take_profits", [0.01, 0.015, 0.02])
        self.grid_sl = params.get("stop_losses", [0.005, 0.01])
        self.min_rr = params.get("min_reward_risk", 1.5)
        self.data_path = f"data/processed/{self.symbol.lower()}_{self.timeframe}.csv"
        self.starting_equity = params.get("starting_equity", 100000)
        self.label = params.get("label", "btc_grid_search")
        self.cooldown_period = params.get("cooldown_period", 10)
        self.vol_threshold = params.get("vol_threshold", 0.002)

    def run(self):
        if not os.path.exists(self.data_path):
            print(f"❌ Missing data: {self.data_path}")
            return

        df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.dropna(inplace=True)

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df.dropna(inplace=True)

        all_trades = []
        last_exit_idx = -100

        for tp in self.grid_tp:
            for sl in self.grid_sl:
                for direction in [1, -1]:  # 1 = long, -1 = short
                    for i in range(20, len(df) - 1):
                        if i - last_exit_idx < self.cooldown_period:
                            continue
                        if df.iloc[i]['volatility'] < self.vol_threshold:
                            continue

                        entry_price = df.iloc[i]["close"]
                        entry_time = df.index[i]

                        if direction == 1:
                            stop_price = entry_price * (1 - sl)
                            target_price = entry_price * (1 + tp)
                        else:
                            stop_price = entry_price * (1 + sl)
                            target_price = entry_price * (1 - tp)

                        rr = tp / sl
                        if rr < self.min_rr:
                            continue

                        exit_time = None
                        exit_price = None
                        for j in range(i + 1, min(i + 20, len(df))):
                            low = df.iloc[j]["low"]
                            high = df.iloc[j]["high"]

                            if direction == 1:
                                if low <= stop_price:
                                    exit_time = df.index[j]
                                    exit_price = stop_price
                                    last_exit_idx = j
                                    break
                                elif high >= target_price:
                                    exit_time = df.index[j]
                                    exit_price = target_price
                                    last_exit_idx = j
                                    break
                            else:
                                if high >= stop_price:
                                    exit_time = df.index[j]
                                    exit_price = stop_price
                                    last_exit_idx = j
                                    break
                                elif low <= target_price:
                                    exit_time = df.index[j]
                                    exit_price = target_price
                                    last_exit_idx = j
                                    break

                        if exit_time and exit_price:
                            log_ret = direction * np.log(exit_price / entry_price)
                            all_trades.append({
                                "entry_time": entry_time,
                                "exit_time": exit_time,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "pnl_log_return": log_ret,
                                "take_profit": tp,
                                "stop_loss": sl,
                                "reward_risk": rr,
                                "position": "long" if direction == 1 else "short"
                            })

        if all_trades:
            self.trades_df = pd.DataFrame(all_trades)
            log_strategy_outputs(self.trades_df, self.label)

            # Equity reconstruction
            self.trades_df = self.trades_df.sort_values("exit_time")
            self.trades_df['capital_used'] = self.starting_equity * 0.1
            self.trades_df['pnl'] = self.trades_df['capital_used'] * self.trades_df['pnl_log_return']
            self.trades_df['equity'] = self.starting_equity + self.trades_df['pnl'].cumsum()

            equity_df = self.trades_df[['exit_time', 'equity']].rename(columns={'exit_time': 'timestamp'})
            equity_df.to_csv(f"logs/portfolio_simulated_equity_{self.label}.csv", index=False)
            self.results = equity_df
