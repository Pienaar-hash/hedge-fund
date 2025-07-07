# factor_weight_sweep.py
import pandas as pd
import numpy as np
import os
from itertools import product
from core.portfolio_simulator import PortfolioSimulator

# Load available equity curves
simulator = PortfolioSimulator()
equity_curves = simulator.load_equity_curves()

# Define sweep ranges
momentum_weights = np.arange(0.0, 1.1, 0.2)
volatility_weights = np.arange(0.0, 1.1, 0.2)
value_weights = np.arange(0.0, 1.1, 0.2)

results = []

for m, v, val in product(momentum_weights, volatility_weights, value_weights):
    w_total = m + v + val
    if w_total == 0:
        continue

    weights = {
        "momentum_ranked": m / w_total,
        "vol_target_btcusdt": v / w_total,
        "relative_value_rebuild": val / w_total
    }

    try:
        equity = simulator.simulate_capital_weighted(weights, normalize=True)
        returns = equity["equity"].pct_change().fillna(0)

        cagr = (equity["equity"].iloc[-1] / equity["equity"].iloc[0]) ** (252 / len(equity)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        dd = (equity["equity"] / equity["equity"].cummax() - 1).min()
        vol = returns.std() * np.sqrt(252)
        cumulative_return = equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1

        results.append({
            "momentum": weights["momentum_ranked"],
            "volatility": weights["vol_target_btcusdt"],
            "value": weights["relative_value_rebuild"],
            "sharpe": sharpe,
            "cagr": cagr,
            "drawdown": dd,
            "volatility": vol,
            "cumulative_return": cumulative_return
        })

    except Exception as e:
        print(f"⚠️ Skipped combo m={m}, v={v}, val={val} due to error: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("logs/factor_monte_carlo.csv", index=False)
print("✅ Saved factor Monte Carlo grid to logs/factor_monte_carlo.csv")
