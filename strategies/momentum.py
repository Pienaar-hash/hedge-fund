# === Patched momentum.py with capital scaling ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy

class StrategyImpl(Strategy):
    def run(self):
        price_data = {}
        for symbol in self.symbols:
            path = f"data/processed/{symbol.lower()}_{self.timeframe}.csv"
            if not os.path.exists(path):
                print(f"❌ File not found: {path}")
                continue
            df = pd.read_csv(path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['momentum'] = df['log_return'].rolling(self.lookback).sum()
            df['score'] = df['momentum'].ewm(span=10).mean()
            df['score_std'] = df['momentum'].rolling(50).std()
            df['zscore'] = df['score'] / df['score_std']
            df['ema_fast'] = df['close'].ewm(span=10).mean()
            df['ema_slow'] = df['close'].ewm(span=50).mean()
            df['vol'] = df['log_return'].rolling(20).std()
            df['atr'] = df['close'].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
            df.dropna(inplace=True)
            price_data[symbol] = df

        all_trades = []
        rebalance_indices = price_data[self.symbols[0]].iloc[self.lookback::self.rebalance_period].index

        for rebalance_time in rebalance_indices:
            scored = []
            for symbol, df in price_data.items():
                if rebalance_time not in df.index:
                    continue
                row = df.loc[rebalance_time]
                if row['vol'] < self.vol_threshold:
                    continue
                if self.use_trend_filter and (row['ema_fast'] < row['ema_slow'] or row['zscore'] < 0):
                    continue
                if abs(row['zscore']) < self.zscore_threshold:
                    continue
                rr_estimate = abs(row['score']) / (self.atr_multiplier * row['atr'])
                if rr_estimate < self.min_reward_risk:
                    continue
                scored.append((symbol, row['zscore']))

            top_symbols = sorted(scored, key=lambda x: -abs(x[1]))[:self.top_n]

            if not top_symbols:
                continue

            # Capital allocation scaling based on zscore
            total_weight = sum(abs(z) for _, z in top_symbols)
            capital_alloc = {
                symbol: (abs(z) / total_weight) * self.starting_equity for symbol, z in top_symbols
            }

            for symbol, _ in top_symbols:
                df = price_data[symbol]
                i = df.index.get_loc(rebalance_time)
                if i + self.hold_period >= len(df):
                    continue
                row = df.iloc[i]
                is_long = row['zscore'] > 0
                entry_time = df.index[i]
                entry_price = row['close']
                atr = row['atr']
                stop_loss = entry_price - self.atr_multiplier * atr if is_long else entry_price + self.atr_multiplier * atr
                trail_pct = 0.08
                max_price = entry_price

                exit_time, exit_price = None, None
                for j in range(i + 1, i + self.hold_period):
                    price = df.iloc[j]['close']
                    max_price = max(max_price, price) if is_long else min(max_price, price)
                    trail_trigger = max_price * (1 - trail_pct) if is_long else max_price * (1 + trail_pct)

                    if (is_long and price < stop_loss) or (not is_long and price > stop_loss):
                        exit_time = df.index[j]
                        exit_price = price
                        break
                    elif (is_long and price < trail_trigger) or (not is_long and price > trail_trigger):
                        exit_time = df.index[j]
                        exit_price = price
                        break

                if not exit_time:
                    exit_time = df.index[i + self.hold_period]
                    exit_price = df.iloc[i + self.hold_period]['close']

                capital = capital_alloc[symbol]
                log_ret = np.log(exit_price / entry_price) if is_long else np.log(entry_price / exit_price)
                pnl = capital * log_ret
                all_trades.append({
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_log_return": log_ret,
                    "zscore": row['zscore'],
                    "vol": row['vol'],
                    "atr": atr,
                    "position": "long" if is_long else "short",
                    "rebalance_index": i
                })

        self.trades_df = pd.DataFrame(all_trades)
        self.log_results()

StrategyImpl = StrategyImpl
