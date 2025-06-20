# === research/factor_monte_carlo_sweep.py ===
import pandas as pd
import numpy as np
from itertools import product

# Simulate returns from weighted factors
def simulate_return(weights):
    np.random.seed(int(sum(weights) * 1000))  # fixed seed for repeatability
    base = 1.0
    noise = np.random.normal(0, 0.01, 252)
    daily_returns = np.dot(weights, [0.02, 0.015, 0.01]) / 252 + noise
    cumulative = (1 + daily_returns).cumprod()

    cumulative_return = cumulative[-1] - 1
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    cagr = cumulative[-1]**(252 / len(daily_returns)) - 1

    return cumulative_return, sharpe, max_drawdown, cagr

# Monte Carlo sweep of weight combinations
grid = []
steps = np.round(np.arange(0.0, 1.01, 0.1), 2).tolist()

for w in product(steps, repeat=3):
    if abs(sum(w) - 1.0) < 0.01:
        ret, sharpe, mdd, cagr = simulate_return(w)
        grid.append({
            "momentum": w[0],
            "volatility": w[1],
            "value": w[2],
            "cumulative_return": ret,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "cagr": cagr
        })

df = pd.DataFrame(grid)
df.to_csv("logs/factor_monte_carlo.csv", index=False)
print("✅ Monte Carlo sweep saved to logs/factor_monte_carlo.csv")
