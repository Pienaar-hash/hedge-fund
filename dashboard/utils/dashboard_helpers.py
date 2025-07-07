import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

LOG_DIR = "logs"

STRATEGY_COLORS = {
    "momentum": "#1f77b4",
    "volatility_target": "#2ca02c",
    "relative_value": "#ff7f0e",
    "portfolio": "#7FDBFF",
    "btc": "orange",
    "spy": "green"
}

def get_color(strategy_name):
    return STRATEGY_COLORS.get(strategy_name.lower(), "#888")

def load_equity_curve(name, normalize=True):
    path = os.path.join(LOG_DIR, f"equity_curve_{name}.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    if normalize and "equity" in df.columns:
        df["norm_equity"] = df["equity"] / df["equity"].iloc[0]
        df["drawdown"] = df["norm_equity"] / df["norm_equity"].cummax() - 1
    return df

def load_metric_summary(name="equity_metrics_summary.csv"):
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def compute_rolling_metrics(returns, window=30):
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
    rolling_dd = returns.cumsum() / returns.cumsum().cummax() - 1
    return rolling_sharpe.fillna(0), rolling_dd.fillna(0)

# === Log File Utilities ===
def list_trade_logs():
    return [f for f in os.listdir(LOG_DIR) if "trades" in f and f.endswith(".csv")]

def load_trade_log(file_name):
    path = os.path.join(LOG_DIR, file_name)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time", ascending=False)
    return df

# === Tear Sheet Utilities ===
def compute_trade_summary(df):
    summary = {}
    col = next((c for c in ["pnl_log_return", "net_ret", "pnl_pct"] if c in df.columns), None)
    if not col or df.empty:
        return {}

    wins = df[df[col] > 0]
    losses = df[df[col] < 0]
    summary["Win Rate"] = len(wins) / len(df) if len(df) > 0 else 0
    summary["Avg Win"] = wins[col].mean() if not wins.empty else 0
    summary["Avg Loss"] = losses[col].mean() if not losses.empty else 0
    summary["Expectancy"] = summary["Win Rate"] * summary["Avg Win"] + (1 - summary["Win Rate"]) * summary["Avg Loss"]
    summary["Profit Factor"] = wins[col].sum() / abs(losses[col].sum()) if not losses.empty else float('inf')
    summary["Payoff Ratio"] = summary["Avg Win"] / abs(summary["Avg Loss"]) if summary["Avg Loss"] != 0 else float('inf')
    return summary

# === Heatmap + Clustering Utilities ===
def compute_pnl_correlation_heatmap(trade_logs):
    pnl_data = {}
    for file in trade_logs:
        name = file.replace("trades_", "").replace(".csv", "")
        df = load_trade_log(file)
        col = next((c for c in ["pnl_log_return", "net_ret", "pnl_pct"] if c in df.columns), None)
        if col and not df.empty:
            series = df[["exit_time", col]].dropna()
            series = series.groupby("exit_time")[col].sum()
            pnl_data[name] = series

    pnl_df = pd.DataFrame(pnl_data).fillna(0)
    corr = pnl_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("PnL Correlation Heatmap")
    return fig

def compute_pnl_distribution_clusters(df, col="pnl_log_return", bins=50):
    if col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[col], bins=bins, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"PnL Distribution: {col}")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    return fig
