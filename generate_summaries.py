from pathlib import Path
import pandas as pd
import numpy as np

def generate_summary_csvs(log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    summary_equity = []
    summary_trade = []

    trade_files = list(Path(log_dir).glob("trades_*.csv"))
    for path in trade_files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        label = path.stem.replace("trades_", "")
        col = next((c for c in ["pnl_log_return", "net_ret", "pnl_pct"] if c in df.columns), None)
        if not col:
            continue

        df = df.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time").set_index("exit_time")
        returns = df[col]

        # Equity metrics
        equity = (1 + returns).cumprod()
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        duration_days = (equity.index[-1] - equity.index[0]).days
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365 / duration_days) - 1 if duration_days > 0 else 0
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() else 0
        max_dd = (equity / equity.cummax() - 1).min()
        vol = returns.std() * np.sqrt(365) if len(returns) >= 20 else np.nan

        summary_equity.append({
            "DurationDays": duration_days,
            "Label": label,
            "Total Return": total_return,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDrawdown": max_dd,
            "Volatility": vol
        })

        # Trade metrics
        wins = df[df[col] > 0]
        losses = df[df[col] <= 0]
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        avg_win = wins[col].mean() if not wins.empty else 0
        avg_loss = losses[col].mean() if not losses.empty else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        profit_factor = wins[col].sum() / abs(losses[col].sum()) if not losses.empty else np.inf
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf

        summary_trade.append({
            "DurationDays": duration_days,
            "Label": label,
            "Trades": len(df),
            "WinRate": win_rate,
            "AvgWin": avg_win,
            "AvgLoss": avg_loss,
            "Expectancy": expectancy,
            "ProfitFactor": profit_factor,
            "PayoffRatio": payoff_ratio
        })

    pd.DataFrame(summary_equity).to_csv(Path(log_dir) / "equity_metrics_summary.csv", index=False)
    pd.DataFrame(summary_trade).to_csv(Path(log_dir) / "trade_metrics_summary.csv", index=False)

if __name__ == "__main__":
    generate_summary_csvs()
