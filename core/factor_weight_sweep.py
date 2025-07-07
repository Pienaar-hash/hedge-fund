# research/factor_weight_sweep.py
import numpy as np
import pandas as pd
import itertools
import os
from core.portfolio_simulator import PortfolioSimulator

factors = ["momentum", "volatility_target", "relative_value", "factor_allocation"]
weight_range = np.linspace(0, 1, 11)

results = []

for weights in itertools.product(weight_range, repeat=len(factors)):
    if abs(sum(weights) - 1.0) > 1e-6:
        continue

    weight_dict = dict(zip(factors, weights))

    sim = PortfolioSimulator()
    equity_df = sim.simulate_with_factor_returns(weight_dict)
    
    if equity_df.empty or equity_df['equity'].std() == 0:
        continue

    returns = equity_df['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    results.append({"sharpe": sharpe, **weight_dict})

results_df = pd.DataFrame(results)
results_df.to_csv("logs/factor_monte_carlo.csv", index=False)
print("✅ Saved factor Monte Carlo grid to logs/factor_monte_carlo.csv")
