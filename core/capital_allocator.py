# === core/capital_allocator.py ===
import pandas as pd
import numpy as np
import os
import sys
import argparse
from glob import glob

class CapitalAllocator:
    def __init__(self, log_dir, capital_base=100_000, capital_per_trade_pct=0.05, variable_sizing=False):
        self.log_dir = log_dir
        self.capital_base = capital_base
        self.capital_per_trade_pct = capital_per_trade_pct
        self.variable_sizing = variable_sizing
        self.capital_per_trade = capital_base * capital_per_trade_pct

    def load_trades(self):
        files = glob(os.path.join(self.log_dir, "relative_value_trades_*.csv"))
        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
            df["filename"] = os.path.basename(f)
            dfs.append(df)
        return pd.concat(dfs).sort_values("exit_time").reset_index(drop=True)

    def compute_trade_size(self, trade):
        if self.variable_sizing:
            weight = trade.get("capital_weight", 1.0)
            base_weight = 3.0
            capital = self.capital_per_trade * (weight / base_weight)
        else:
            capital = self.capital_per_trade
        return min(capital, self.capital_base)

    def simulate_equity(self, trades_df):
        all_days = pd.date_range(start=trades_df["entry_time"].min(), end=trades_df["exit_time"].max(), freq="D")

        available_equity = self.capital_base
        open_positions = []
        equity_log = []

        for day in all_days:
            closed_today = []
            for pos in open_positions:
                if pos["exit_time"].date() == day.date():
                    ret = np.exp(pos["pnl_log_return"]) - 1
                    pnl = pos["capital"] * ret
                    available_equity += pos["capital"] + pnl
                    closed_today.append(pos)
            open_positions = [p for p in open_positions if p not in closed_today]

            trades_today = trades_df[trades_df["entry_time"].dt.date == day.date()]
            for _, trade in trades_today.iterrows():
                capital = self.compute_trade_size(trade)
                if available_equity >= capital:
                    open_positions.append({
                        "entry_time": trade["entry_time"],
                        "exit_time": trade["exit_time"],
                        "pnl_log_return": trade["pnl_log_return"],
                        "capital": capital
                    })
                    available_equity -= capital

            locked = sum(p["capital"] for p in open_positions)
            equity_today = available_equity + locked
            equity_log.append((day, equity_today))

        equity_df = pd.DataFrame(equity_log, columns=["timestamp", "equity"])
        equity_df["return"] = equity_df["equity"].pct_change().fillna(0)
        return equity_df

    def performance_metrics(self, equity_df):
        cagr = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) ** (252 / len(equity_df)) - 1
        sharpe = equity_df["return"].mean() / equity_df["return"].std() * np.sqrt(252)
        sortino = equity_df["return"].mean() / equity_df["return"][equity_df["return"] < 0].std() * np.sqrt(252)
        volatility = equity_df["return"].std() * np.sqrt(252)
        max_dd = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min()

        return {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Volatility": volatility,
            "Max Drawdown": max_dd
        }

def unified_backtest(strategy_params, log_dir="logs", allocator_params=None):
    from core.strategy_base import StrategyImpl  # lazy import for CLI compatibility
    print("🚀 Running Strategy: Relative Value")
    strategy = StrategyImpl()
    strategy.configure(strategy_params)
    strategy.run()

    if allocator_params is None:
        allocator_params = {}

    allocator = CapitalAllocator(log_dir, **allocator_params)
    trades = allocator.load_trades()
    equity = allocator.simulate_equity(trades)
    metrics = allocator.performance_metrics(equity)

    equity.to_csv(os.path.join(log_dir, "capital_weighted_equity_curve.csv"), index=False)
    print("📊 Capital Allocator Metrics:", metrics)
    return equity, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory of trade logs")
    parser.add_argument("--capital", type=float, default=100000, help="Starting capital")
    parser.add_argument("--pct", type=float, default=0.05, help="Capital per trade percent")
    parser.add_argument("--variable", action="store_true", help="Enable variable sizing")
    args = parser.parse_args()

    allocator = CapitalAllocator(
        log_dir=args.log_dir,
        capital_base=args.capital,
        capital_per_trade_pct=args.pct,
        variable_sizing=args.variable
    )
    trades = allocator.load_trades()
    equity = allocator.simulate_equity(trades)
    metrics = allocator.performance_metrics(equity)
    equity.to_csv(os.path.join(args.log_dir, "capital_weighted_equity_curve.csv"), index=False)

    print("\n✅ Capital-Weighted Equity Curve saved to:", os.path.join(args.log_dir, "capital_weighted_equity_curve.csv"))
    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
