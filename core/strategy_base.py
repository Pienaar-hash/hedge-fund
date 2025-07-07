# === core/strategy_base.py ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Strategy:
    def configure(self, params):
        # Default configure method to avoid frozen config errors
        for k, v in params.items():
            setattr(self, k, v)

    def run(self):
        raise NotImplementedError("Strategy must implement run method.")

    def log_results(self, label="default"):
        if not getattr(self, "log_output", True):
            return
        if hasattr(self, "trades_df") and not self.trades_df.empty:
            self.compute_and_save_metrics(self.trades_df, label=label)

    def compute_and_save_metrics(self, df, label="default"):
        if "pnl_log_return" in df.columns:
            pnl_col = "pnl_log_return"
        elif "pnl_pct" in df.columns:
            pnl_col = "pnl_pct"
        elif "net_return" in df.columns:
            pnl_col = "net_return"
        else:
            print(f"❌ No recognized PnL column in dataframe for {label}.")
            return

        if "exit_time" not in df.columns:
            print(f"❌ Missing 'exit_time' in trades dataframe for {label}.")
            return

        df = df.copy()
        df = df.sort_values("exit_time")
        df.set_index("exit_time", inplace=True)

        equity = (1 + df[pnl_col]).cumprod()
        returns = df[pnl_col]

        equity_df = pd.DataFrame({
            "timestamp": equity.index,
            "equity": equity / equity.iloc[0]
        })
        equity_df.to_csv(f"logs/equity_curve_{label}.csv", index=False)

        win_trades = df[df[pnl_col] > 0]
        loss_trades = df[df[pnl_col] < 0]
        win_rate = len(win_trades) / len(df)
        avg_win = win_trades[pnl_col].mean()
        avg_loss = loss_trades[pnl_col].mean()
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        profit_factor = win_trades[pnl_col].sum() / abs(loss_trades[pnl_col].sum()) if not loss_trades.empty else float('inf')
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss else float('inf')

        summary_df = pd.DataFrame([{
            "Label": label,
            "Trades": len(df),
            "WinRate": win_rate,
            "AvgWin": avg_win,
            "AvgLoss": avg_loss,
            "Expectancy": expectancy,
            "ProfitFactor": profit_factor,
            "PayoffRatio": payoff_ratio
        }])
        summary_df.to_csv(f"logs/summary_{label}.csv", index=False)

        # Optional diagnostic plot if entry_strength is available
        if "entry_strength" in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df.reset_index(), x="entry_strength", y=pnl_col)
            plt.title(f"PnL by Entry Strength - {label}")
            plt.ylabel("PnL")
            plt.xlabel("Entry Strength")
            plt.grid(True)
            plt.tight_layout()
            plot_path = f"logs/pnl_by_strength_{label}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"📊 Saved diagnostic plot to {plot_path}")

        # Rolling Sharpe and Drawdown
        rolling_window = 30
        rolling_sharpe = returns.rolling(rolling_window).mean() / returns.rolling(rolling_window).std()
        rolling_dd = equity / equity.cummax() - 1

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        rolling_sharpe.plot(title=f"Rolling Sharpe ({rolling_window} trades) - {label}", grid=True)
        plt.ylabel("Sharpe Ratio")

        plt.subplot(2, 1, 2)
        rolling_dd.plot(title=f"Rolling Drawdown - {label}", grid=True)
        plt.ylabel("Drawdown")
        plt.xlabel("Date")

        plt.tight_layout()
        plot_path = f"logs/rolling_metrics_{label}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Saved rolling Sharpe and drawdown plot to {plot_path}")
