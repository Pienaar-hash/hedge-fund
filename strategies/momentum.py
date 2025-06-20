# === strategies/momentum.py ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.strategy_base import Strategy

class StrategyImpl(Strategy):
    def configure(self, params):
        self.params = params or {}
        self.symbols = self.params.get("symbols", ["BTCUSDT"])
        self.timeframe = self.params.get("timeframe", "1h")
        self.lookback = self.params.get("lookback", 20)
        self.rebalance_period = self.params.get("rebalance_period", 24)
        self.hold_period = self.params.get("hold_period", 24)
        self.top_n = self.params.get("top_n", 3)

    def run(self):
        price_data = {}

        for symbol in self.symbols:
            data_path = f"data/processed/{symbol.lower()}.csv"
            if not os.path.exists(data_path):
                print(f"❌ File not found: {data_path}")
                continue

            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['close']].copy()
            df['return'] = df['close'].pct_change(self.lookback)
            df['vol'] = df['close'].pct_change().rolling(self.lookback).std()
            price_data[symbol] = df

        index = price_data[self.symbols[0]].index
        all_trades = []

        for i in range(self.lookback, len(index), self.rebalance_period):
            scores = []
            for symbol, df in price_data.items():
                if index[i] not in df.index:
                    continue
                score = df.loc[index[i], 'return']
                vol = df.loc[index[i], 'vol']
                if pd.notna(score) and pd.notna(vol) and vol > 0:
                    scores.append((symbol, score / vol))

            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            longs = [s for s, _ in ranked[:self.top_n]]
            shorts = [s for s, _ in ranked[-self.top_n:]]

            for position_type, selected in [('long', longs), ('short', shorts)]:
                for symbol in selected:
                    df = price_data[symbol]
                    if i + self.hold_period >= len(df):
                        continue
                    entry_time = index[i]
                    exit_time = index[i + self.hold_period]
                    entry_price = df.loc[entry_time, 'close']
                    exit_price = df.loc[exit_time, 'close']
                    ret = (exit_price - entry_price) / entry_price
                    pnl_pct = ret if position_type == 'long' else -ret
                    all_trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'position': position_type
                    })

        self.trades_df = pd.DataFrame(all_trades)
        if not self.trades_df.empty:
            self.save_trades(self.trades_df, "momentum_trades_ranked.csv")
            self.save_equity_curve(self.trades_df, "momentum_ranked")
            self.plot_equity_curve("momentum_ranked")
        else:
            print("⚠️ No trades generated.")

    def save_trades(self, df, fname):
        df.to_csv(f"logs/{fname}", index=False)
        print(f"✅ Saved {len(df)} trades to logs/{fname}")

    def save_equity_curve(self, df, name):
        equity = (1 + df['pnl_pct']).cumprod()
        out = pd.DataFrame({
            "timestamp": df['exit_time'],
            "equity": equity
        })
        out.to_csv(f"logs/equity_curve_{name}.csv", index=False)

    def plot_equity_curve(self, name):
        df = pd.read_csv(f"logs/equity_curve_{name}.csv", parse_dates=['timestamp'])
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['equity'])
        plt.title("Equity Curve - Ranked Momentum")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"logs/equity_curve_{name}.png")
        plt.close()
        print(f"📈 Equity plot saved to logs/equity_curve_{name}.png")

    def log_results(self):
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            print(f"✅ Logged {len(self.trades_df)} ranked momentum trades")
        else:
            print("⚠️ No ranked momentum trades to log")
