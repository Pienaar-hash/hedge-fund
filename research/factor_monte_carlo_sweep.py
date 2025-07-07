# === research/factor_monte_carlo_sweep.py ===
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import os
import json

# Load config to get starting_equity per strategy
with open("config/strategy_config.json") as f:
    config = json.load(f)
    equity_start = {
        strat["name"]: strat["params"].get("starting_equity", 1.0)
        for strat in config["strategies"]
    }

# Load actual equity curves and compute log returns
LOG_PATHS = {
    "momentum": "logs/equity_curve_momentum_ranked.csv",
    "volatility": "logs/equity_curve_vol_target_btcusdt.csv",
    "value": "logs/equity_curve_relative_value.csv"
}

returns_data = {}
for factor, path in LOG_PATHS.items():
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')
        df = df[~df.index.duplicated(keep='first')]
        col = [c for c in df.columns if c != 'timestamp'][0]
        equity = df[col]
        equity = equity[equity > 0]  # drop invalid values
        start_value = equity_start.get(factor, equity.iloc[0])
        equity = equity / start_value
        returns = np.log(equity / equity.shift(1)).dropna()
        returns.name = factor
        returns_data[factor] = returns
    else:
        print(f"⚠️ Missing equity curve: {path}")

# Join all returns on their common timestamps
if len(returns_data) < 3:
    print("❌ Not enough factors loaded to run simulation.")
    exit()

returns_df = pd.concat(returns_data.values(), axis=1, join="inner").fillna(0)
if returns_df.shape[0] < 30:
    print("⚠️ Warning: Too few overlapping timestamps in equity curves.")

# Simulate weighted returns
def simulate_return(weights):
    if returns_df.empty:
        return 0, 0, 0, 0

    weighted = returns_df.copy()
    for i, factor in enumerate(["momentum", "volatility", "value"]):
        weighted[factor] = weighted[factor] * weights[i]

    portfolio_returns = weighted.sum(axis=1)
    equity = (1 + portfolio_returns).cumprod()
    if equity.empty:
        return 0, 0, 0, 0

    cumulative_return = equity.iloc[-1] - 1
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    cagr = (equity.iloc[-1] / equity.iloc[0])**(252 / len(portfolio_returns)) - 1 if len(portfolio_returns) > 0 else 0

    return cumulative_return, sharpe, max_drawdown, cagr

# Monte Carlo sweep with finer resolution and constraints
grid = []
steps = np.round(np.arange(0.0, 1.01, 0.05), 3).tolist()  # finer granularity (5%)

now = datetime.utcnow()

for w in product(steps, repeat=3):
    total = sum(w)
    if abs(total - 1.0) < 0.01 and w[0] >= 0.2 and w[1] >= 0.1:
        ret, sharpe, mdd, cagr = simulate_return(w)
        grid.append({
            "strategy": "factor_blend",
            "momentum": w[0],
            "volatility": w[1],
            "value": w[2],
            "cumulative_return": ret,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "cagr": cagr,
            "timestamp": now
        })

df = pd.DataFrame(grid)
df.to_csv("logs/factor_monte_carlo.csv", index=False)
print("✅ Monte Carlo sweep saved to logs/factor_monte_carlo.csv")
