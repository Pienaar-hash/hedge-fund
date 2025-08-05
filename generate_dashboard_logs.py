# === generate_dashboard_logs.py ===
import os
from portfolio.backtest_runner import run_strategy
from portfolio.portfolio_simulator import PortfolioSimulator
import shutil
import pandas as pd
from datetime import datetime

# === Step 1: Clean logs folder and archive old logs ===
LOGS_DIR = "logs"
ARCHIVE_DIR = f"logs_archive/{datetime.now().strftime('%Y%m%d_%H%M')}"

os.makedirs(ARCHIVE_DIR, exist_ok=True)
for f in os.listdir(LOGS_DIR):
    shutil.move(os.path.join(LOGS_DIR, f), os.path.join(ARCHIVE_DIR, f))

print(f"📦 Archived old logs to {ARCHIVE_DIR}")

# === Step 2: Define strategies to run ===
strategies = [
    {"name": "momentum", "params": {"label": "ranked"}},
    {"name": "volatility_target", "params": {}},
    {"name": "relative_value", "params": {}},
    {"name": "factor_allocation", "params": {}}
]

# === Step 3: Run each strategy ===
for strat in strategies:
    print(f"🚀 Running {strat['name']}...")
    run_strategy(strat["name"], strat.get("params", {}))

# === Step 4: Simulate Portfolio ===
sim = PortfolioSimulator()
sim.simulate_independent()

# Optional: simulate capital-weighted (equal weight)
equal_weights = {name: 1 for name in sim.equity_curves}
total = sum(equal_weights.values())
weights = {k: v / total for k, v in equal_weights.items()}
weighted_df = sim.simulate_capital_weighted(weights)
weighted_df.to_csv("logs/portfolio_simulated_equity_weighted.csv", index=False)
print("📈 Saved capital-weighted portfolio to logs/portfolio_simulated_equity_weighted.csv")

# === Step 5: Generate rolling metrics and pnl heatmaps ===
def generate_rolling_metrics(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df["returns"] = df["equity"].pct_change().fillna(0)
    df["rolling_sharpe"] = df["returns"].rolling(30).mean() / df["returns"].rolling(30).std()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    df[["timestamp", "rolling_sharpe", "drawdown"]].to_csv("logs/rolling_metrics_portfolio.csv", index=False)
    print("✅ Saved rolling Sharpe and drawdown to logs/rolling_metrics_portfolio.csv")

def generate_pnl_heatmap(trade_file):
    df = pd.read_csv(trade_file, parse_dates=["exit_time"])
    df["hour"] = df["exit_time"].dt.hour
    df["dayofweek"] = df["exit_time"].dt.dayofweek
    heatmap = df.groupby(["dayofweek", "hour"])["pnl_log_return"].mean().unstack().fillna(0)
    heatmap.to_csv("logs/pnl_clusters.csv")
    print("✅ Saved PnL cluster heatmap to logs/pnl_clusters.csv")

generate_rolling_metrics("logs/portfolio_simulated_equity.csv")
generate_pnl_heatmap("logs/momentum_trades_ranked.csv")

# === Step 6: Export portfolio summary table ===
def generate_summary(path, label):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["returns"] = df["equity"].pct_change().fillna(0)
    r = df["returns"]
    equity = df["equity"]
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    sharpe = r.mean() / r.std() * (252 ** 0.5) if r.std() > 0 else 0
    mdd = (equity / equity.cummax() - 1).min()
    summary = pd.DataFrame([{
        "Label": label,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "FinalEquity": equity.iloc[-1]
    }])
    return summary

summary_files = [
    ("logs/portfolio_simulated_equity.csv", "EqualWeight Portfolio"),
    ("logs/portfolio_simulated_equity_weighted.csv", "CapitalWeighted Portfolio")
]

summary_all = pd.concat([generate_summary(p, l) for p, l in summary_files])
summary_all.to_csv("logs/portfolio_summary.csv", index=False)
print("📊 Saved portfolio summary to logs/portfolio_summary.csv")

print("🎯 Dashboard logs regenerated with summary.")