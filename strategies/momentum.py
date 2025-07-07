# === strategies/momentum.py ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.strategy_base import Strategy

class StrategyImpl(Strategy):
    def configure(self, params):
        self.params = params or {}
        raw_symbols = self.params.get("symbols", ["BTCUSDT"])
        self.symbols = [s.replace("_1h", "").upper() for s in (raw_symbols if isinstance(raw_symbols, list) else [raw_symbols])]
        self.timeframe = self.params.get("timeframe", "1h")
        self.lookback = self.params.get("lookback", 20)
        self.rebalance_period = self.params.get("rebalance_period", 12)
        self.hold_period = self.params.get("hold_period", 48)
        self.zscore_threshold = self.params.get("zscore_threshold", 2.0)
        self.use_trend_filter = self.params.get("use_trend_filter", True)
        self.vol_threshold = self.params.get("vol_threshold", 0.015)
        self.top_n = self.params.get("top_n", 1)
        self.allow_shorts = self.params.get("allow_shorts", True)
        self.atr_multiplier = self.params.get("atr_multiplier", 1.5)
        self.min_reward_risk = self.params.get("min_reward_risk", 1.5)
        self.starting_equity = self.params.get("starting_equity", 100000)
        self.label = self.params.get("label", "ranked")

    def run(self):
        price_data = {}
        for symbol in self.symbols:
            data_path = f"data/processed/{symbol.lower()}_{self.timeframe.lower()}.csv"
            if not os.path.exists(data_path):
                print(f"❌ File not found: {data_path}")
                continue
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['close']].copy()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['momentum_score'] = df['log_return'].rolling(self.lookback).sum()
            df['score_ewma'] = df['momentum_score'].ewm(span=10).mean()
            df['score_std'] = df['momentum_score'].rolling(50).std()
            df['momentum_z'] = (df['score_ewma'] / df['score_std']).replace([np.inf, -np.inf], np.nan)
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['volatility'] = df['log_return'].rolling(20).std()
            df['atr'] = df['close'].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
            price_data[symbol] = df

        if not price_data:
            print("⚠️ No valid symbols loaded. Aborting momentum strategy.")
            return

        index = list(price_data.values())[0].index
        all_trades = []
        symbol_equities = {s: [] for s in self.symbols}
        symbol_equity_values = {s: self.starting_equity for s in self.symbols}

        for i in range(self.lookback, len(index), self.rebalance_period):
            zscores = {}
            for symbol, df in price_data.items():
                if index[i] not in df.index:
                    continue
                if df.loc[index[i], 'volatility'] < self.vol_threshold:
                    continue
                if self.use_trend_filter and df.loc[index[i], 'ema_10'] < df.loc[index[i], 'ema_50']:
                    continue
                if pd.isna(df.loc[index[i], 'atr']) or df.loc[index[i], 'atr'] == 0:
                    continue
                if pd.notna(df.loc[index[i], 'momentum_z']):
                    zscores[symbol] = df.loc[index[i], 'momentum_z']

            if not zscores:
                continue

            sorted_symbols = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
            long_candidates = [(s, z) for s, z in sorted_symbols if z > self.zscore_threshold]
            short_candidates = [(s, z) for s, z in sorted_symbols if z < -self.zscore_threshold]

            long_symbols = long_candidates[:self.top_n]
            short_symbols = short_candidates[:self.top_n] if self.allow_shorts else []

            for symbol, z in long_symbols + short_symbols:
                df = price_data[symbol]
                if i + self.hold_period >= len(df):
                    continue
                entry_time = index[i]
                entry_price = df.loc[entry_time, 'close']
                atr = df.loc[entry_time, 'atr']

                is_long = (z > 0)
                if is_long:
                    stop_loss = entry_price - self.atr_multiplier * atr
                else:
                    stop_loss = entry_price + self.atr_multiplier * atr

                future_price = df.iloc[min(i + self.hold_period, len(df) - 1)]['close']
                expected_reward = (future_price - entry_price) if is_long else (entry_price - future_price)
                risk = abs(entry_price - stop_loss)

                if risk <= 0 or (expected_reward / risk) < self.min_reward_risk:
                    continue

                exit_time, exit_price = None, None
                for j in range(i + 1, i + self.hold_period):
                    price = df.iloc[j]['close']
                    if (is_long and price < stop_loss) or (not is_long and price > stop_loss):
                        exit_time = df.index[j]
                        exit_price = price
                        break
                else:
                    exit_time = index[i + self.hold_period]
                    exit_price = df.loc[exit_time, 'close']

                ret = np.log(exit_price / entry_price) if is_long else np.log(entry_price / exit_price)
                all_trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_log_return': ret,
                    'zscore': z,
                    'atr': atr,
                    'volatility': df.loc[entry_time, 'volatility'],
                    'ema_10': df.loc[entry_time, 'ema_10'],
                    'ema_50': df.loc[entry_time, 'ema_50'],
                    'reward_risk': expected_reward / risk,
                    'position': 'long' if is_long else 'short',
                    'rebalance_index': i
                })
                symbol_equity_values[symbol] *= np.exp(ret)
                symbol_equities[symbol].append({"timestamp": exit_time, "equity": symbol_equity_values[symbol]})

        self.trades_df = pd.DataFrame(all_trades)
        if not self.trades_df.empty:
            self.save_trades(self.trades_df, f"momentum_trades_{self.label}.csv")
            self.log_results(label=f"momentum_{self.label}")

            self.trades_df = self.trades_df.sort_values("exit_time")
            self.trades_df["timestamp"] = pd.to_datetime(self.trades_df["exit_time"])
            self.trades_df = self.trades_df.drop_duplicates(subset="timestamp", keep="last")
            returns_df = self.trades_df.sort_values("exit_time")[["exit_time", "pnl_log_return"]]
            raw_returns = np.exp(returns_df["pnl_log_return"]) - 1
            equity = self.starting_equity * (1 + raw_returns).cumprod()
            equity_df = pd.DataFrame({
                "timestamp": returns_df["exit_time"].values,
                "equity": equity.values
            })
            equity_df.to_csv(f"logs/equity_curve_momentum_{self.label}.csv", index=False)
            print(f"✅ Equity curve saved to logs/equity_curve_momentum_{self.label}.csv")

            for symbol, entries in symbol_equities.items():
                if not entries:
                    continue
                df = pd.DataFrame(entries)
                df.to_csv(f"logs/equity_curve_momentum_{symbol.lower()}_{self.label}.csv", index=False)
                print(f"✅ Saved individual equity curve for {symbol} to logs/equity_curve_momentum_{symbol.lower()}_{self.label}.csv")

        else:
            print("⚠️ No trades generated.")

    def save_trades(self, df, fname):
        df.to_csv(f"logs/{fname}", index=False)
        print(f"✅ Saved {len(df)} trades to logs/{fname}")
