# === core/strategy_base.py ===
import pandas as pd
import os

class Strategy:
    def configure(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def run(self):
        raise NotImplementedError("Strategy must implement run method.")

    def log_results(self, label=None):
        label = label or getattr(self, "label", "default")
        if not getattr(self, "log_output", True):
            print(f"⚠️ Logging disabled for {label}.")
            return

        if hasattr(self, "trades_df") and not self.trades_df.empty:
            self.compute_and_save_metrics(self.trades_df, label)
            trades_path = f"logs/trades_{label}.csv"
            self.trades_df.to_csv(trades_path, index=False)
            print(f"📄 Trade log saved to {trades_path}")
        else:
            print(f"⚠️ No trades found for logging under {label}.")

    def compute_and_save_metrics(self, trades_df, label):
        pnl_col = next((col for col in ["pnl_log_return", "net_return", "pnl_pct"] if col in trades_df.columns), None)
        if not pnl_col:
            print(f"❌ No recognized PnL column for {label}.")
            return

        if "exit_time" not in trades_df.columns or "entry_time" not in trades_df.columns:
            print(f"❌ Entry or Exit timestamps missing for {label}.")
            return

        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df = trades_df.sort_values("exit_time")
        trades_df.set_index("exit_time", inplace=True)

        # Daily-aligned equity curve
        returns = trades_df[pnl_col]
        equity_curve = (1 + returns).cumprod()
        equity_df = equity_curve.resample("1D").last().ffill().reset_index()
        equity_df.columns = ["timestamp", "equity"]
        equity_df["equity"] = equity_df["equity"] / equity_df["equity"].iloc[0] * getattr(self, "starting_equity", 100000)

        os.makedirs("logs", exist_ok=True)
        equity_path = f"logs/portfolio_simulated_equity_{label}.csv"
        equity_df.to_csv(equity_path, index=False)
        print(f"📈 Saved daily-aligned equity to {equity_path}")

        # Compute performance metrics
        win_trades = trades_df[trades_df[pnl_col] > 0]
        loss_trades = trades_df[trades_df[pnl_col] <= 0]
        win_rate = len(win_trades) / len(trades_df)
        avg_win = win_trades[pnl_col].mean()
        avg_loss = loss_trades[pnl_col].mean()
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        profit_factor = abs(win_trades[pnl_col].sum() / loss_trades[pnl_col].sum()) if loss_trades[pnl_col].sum() != 0 else float('inf')
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        summary = pd.DataFrame([{
            "Label": label,
            "Trades": len(trades_df),
            "WinRate": win_rate,
            "AvgWin": avg_win,
            "AvgLoss": avg_loss,
            "Expectancy": expectancy,
            "ProfitFactor": profit_factor,
            "PayoffRatio": payoff_ratio
        }])
        summary_path = f"logs/summary_{label}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"📊 Metrics summary saved to {summary_path}")

        # Rolling Metrics
        rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std()
        rolling_dd = equity_curve / equity_curve.cummax() - 1
        rolling_metrics_df = pd.DataFrame({
            "timestamp": rolling_sharpe.index,
            "rolling_sharpe": rolling_sharpe,
            "rolling_drawdown": rolling_dd
        }).dropna()

        rolling_metrics_path = f"logs/rolling_metrics_{label}.csv"
        rolling_metrics_df.to_csv(rolling_metrics_path, index=False)
        print(f"📉 Rolling metrics saved to {rolling_metrics_path}")
