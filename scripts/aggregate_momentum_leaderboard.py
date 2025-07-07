import os
import pandas as pd
import numpy as np
from glob import glob

LOG_DIR = "logs"
TRADE_GLOB = os.path.join(LOG_DIR, "momentum_trades_*.csv")

def compute_sharpe(returns, periods_per_year=252):
    mean_ret = returns.mean()
    std_ret = returns.std()
    return (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0

def compute_drawdown(equity_series):
    drawdown = equity_series / equity_series.cummax() - 1
    return drawdown.min()

def compute_cagr(equity_series, timestamps):
    start = pd.to_datetime(timestamps.iloc[0])
    end = pd.to_datetime(timestamps.iloc[-1])
    n_years = (end - start).days / 365.25
    if n_years <= 0:
        return 0
    return (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / n_years) - 1

leaderboard = []

for trade_file in glob(TRADE_GLOB):
    try:
        label = os.path.basename(trade_file).replace("momentum_trades_", "").replace(".csv", "")
        equity_file = os.path.join(LOG_DIR, f"equity_curve_momentum_{label}.csv")
        summary_file = os.path.join(LOG_DIR, f"summary_momentum_{label}.csv")

        # Load required files
        df_trades = pd.read_csv(trade_file)
        df_equity = pd.read_csv(equity_file)
        df_summary = pd.read_csv(summary_file)

        if df_trades.empty or df_equity.empty or df_summary.empty:
            continue

        # PnL-based metrics
        total_trades = len(df_trades)
        win_rate = (df_trades["exit_price"] > df_trades["entry_price"]).mean()

        # Equity-based metrics
        equity = df_equity["equity"]
        timestamps = df_equity["timestamp"]
        returns = np.log(equity).diff().dropna()

        sharpe = compute_sharpe(returns)
        max_dd = compute_drawdown(equity)
        cagr = compute_cagr(equity, timestamps)

        # From summary file
        pf = df_summary["ProfitFactor"].iloc[0]
        pr = df_summary["PayoffRatio"].iloc[0]
        expectancy = df_summary["Expectancy"].iloc[0]

        leaderboard.append({
            "Label": label,
            "Trades": total_trades,
            "WinRate": round(win_rate, 4),
            "Sharpe": round(sharpe, 4),
            "MaxDrawdown": round(max_dd, 4),
            "CAGR": round(cagr, 4),
            "ProfitFactor": round(pf, 2),
            "PayoffRatio": round(pr, 2),
            "Expectancy": round(expectancy, 4)
        })

    except Exception as e:
        print(f"⚠️ Error processing {trade_file}: {e}")

# Build leaderboard
leader_df = pd.DataFrame(leaderboard)

# Add tags
leader_df["Tag"] = ""
leader_df.loc[leader_df["Sharpe"].nlargest(5).index, "Tag"] += "🚀 Top Sharpe "
leader_df.loc[leader_df["CAGR"].nlargest(5).index, "Tag"] += "🌱 High CAGR "
leader_df.loc[leader_df["MaxDrawdown"] < -0.3, "Tag"] += "🛑 High Drawdown"

# Sort and save
leader_df = leader_df.sort_values(by=["Sharpe", "CAGR"], ascending=False)
leader_df.to_csv(os.path.join(LOG_DIR, "momentum_leaderboard.csv"), index=False)
leader_df.to_markdown(os.path.join(LOG_DIR, "momentum_leaderboard.md"), index=False)

print("✅ Momentum leaderboard saved as CSV and Markdown.")
