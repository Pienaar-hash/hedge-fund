# === strategies/volatility_target.py ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy

class StrategyImpl(Strategy):
    def configure(self, params):
        self.params = params or {}
        self.assets = [(a["symbol"], a["tf"]) for a in self.params.get("assets", [])]
        self.lookback = self.params.get("lookback", 20)
        self.target_vol = self.params.get("target_vol", 0.015)
        self.max_leverage = self.params.get("max_leverage", 3)
        self.fee = self.params.get("fee", 0.001)
        self.rebalance_period = self.params.get("rebalance_period", 1)
        self.starting_equity = self.params.get("starting_equity", 100000)
        self.use_trend_filter = self.params.get("use_trend_filter", False)

    def run(self):
        all_equity_curves = []
        all_logs = []

        for symbol, tf in self.assets:
            df = self.load_data(symbol, tf)
            if df is None or len(df) < self.lookback:
                continue

            result = self.backtest(df, symbol)
            if result is not None:
                eq, trades = result
                equity_filename = f"logs/equity_curve_vol_target_{symbol.lower()}.csv"
                eq.to_csv(equity_filename, index=False)
                print(f"📉 Saved equity curve for {symbol} to {equity_filename}")
                all_equity_curves.append(eq.set_index("timestamp").rename(columns={"equity": f"{symbol.lower()}_{tf.lower()}"}))
                all_logs.append(trades)

        if all_equity_curves:
            blended = pd.concat(all_equity_curves, axis=1).mean(axis=1)
            blended_df = pd.DataFrame({"timestamp": blended.index, "equity": blended.values})
            blended_df.to_csv("logs/equity_curve_vol_target_btcusdt.csv", index=False)
            print("✅ Saved blended portfolio to logs/equity_curve_vol_target_btcusdt.csv")

        if all_logs:
            all_trades = pd.concat(all_logs)
            all_trades.to_csv("logs/vol_target_trades_ranked.csv", index=False)
            self.trades_df = all_trades
            self.log_results(label="vol_target_btcusdt")

    def load_data(self, symbol, tf):
        path = f"data/processed/{symbol.lower()}_{tf.lower()}.csv"
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            return None
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol"] = df["log_return"].rolling(self.lookback).std()
        if self.use_trend_filter:
            df["ema"] = df["close"].ewm(span=20).mean()
        df.dropna(inplace=True)
        return df

    def backtest(self, df, symbol):
        trades = []
        equity = self.starting_equity
        leverage = 1.0
        equity_series = []

        for i in range(self.lookback, len(df), self.rebalance_period):
            row = df.iloc[i]

            if self.use_trend_filter and row["close"] < row.get("ema", row["close"]):
                continue

            vol = df.iloc[i - self.lookback:i]["log_return"].std()
            if vol == 0 or pd.isna(vol):
                continue

            leverage = min(self.target_vol / vol, self.max_leverage)
            daily_ret = row["log_return"]
            fee_cost = self.fee * abs(leverage)

            net_ret = (daily_ret * leverage) - fee_cost
            equity = max(equity * (1 + net_ret), 1e-6)

            trades.append({
                "timestamp": df.index[i],
                "symbol": symbol,
                "entry_time": df.index[i],
                "exit_time": df.index[i],
                "log_return": daily_ret,
                "leverage": leverage,
                "net_return": net_ret,
                "pnl_log_return": net_ret,
                "equity": equity,
                "vol": vol
            })

            equity_series.append({"timestamp": df.index[i], "equity": equity})

        if not trades:
            print(f"⚠️ No trades executed for {symbol}")
            return None

        trades_df = pd.DataFrame(trades).replace([np.inf, -np.inf], np.nan).dropna()
        trades_df = trades_df[trades_df["pnl_log_return"] != 0]
        equity_df = pd.DataFrame(equity_series)
        return equity_df, trades_df
