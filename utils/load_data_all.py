import os
import pandas as pd

def load_equity_curve(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if "timestamp" not in df.columns or "equity" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def load_all_summary_metrics():
    summary_files = [f for f in os.listdir("logs") if f.endswith("summary.csv")]
    dfs = {}

    for file in summary_files:
        name = file.replace("_summary.csv", "")
        path = os.path.join("logs", file)

        # Group volatility_targeting into one DataFrame
        if name.startswith("volatility_targeting"):
            base = "volatility_target"
            if base not in dfs:
                dfs[base] = []
            dfs[base].append(pd.read_csv(path))
        else:
            dfs[name] = pd.read_csv(path)

    # Combine grouped volatility_target summaries
    for k in dfs:
        if isinstance(dfs[k], list):
            dfs[k] = pd.concat(dfs[k], ignore_index=True)

    return dfs

def load_all_equity_curves():
    equity_files = [f for f in os.listdir("logs") if f.startswith("equity_curve_") and f.endswith(".csv")]
    data = {}
    for file in equity_files:
        name = file.replace("equity_curve_", "").replace(".csv", "")
        df = load_equity_curve(os.path.join("logs", file))
        if df is not None:
            data[name] = df
    return data

def load_trade_log(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def load_trade_logs():
    logs = {}
    for file in os.listdir("logs"):
        if file.endswith(".csv") and "_trades_" in file:
            df = pd.read_csv(os.path.join("logs", file))
            logs[file] = df
    return logs

def load_portfolio_equity():
    path = "logs/portfolio_simulated_equity.csv"
    return load_equity_curve(path)

def load_drawdown_series():
    df = load_portfolio_equity()
    if df is None or df.empty:
        return None
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df[["drawdown"]]

def load_monte_carlo_summary():
    path = "logs/factor_monte_carlo.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)
