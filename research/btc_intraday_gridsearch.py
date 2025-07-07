# === research/gridsearch_relative_value.py ===
import os
import json
import pandas as pd
import numpy as np
from importlib import import_module

# Strategy setup
StrategyImpl = import_module("strategies.relative_value").StrategyImpl

# Sweep parameters
lookbacks = [5, 10, 21]
z_entries = [1.0, 1.5, 2.0]
capital_weights = [1.0, 2.0, 3.0]

# Base config template
base_config = {
    "base": "ETHUSDT",
    "pairs": ["BTCUSDT", "BNBUSDT", "AVAXUSDT", "SOLUSDT", "LTCUSDT"],
    "z_exit": 0.1,
    "z_entry_strong": 2.0,
    "strong_weight": 6.0,
    "moderate_weight": 3.0,
    "max_spread_std": 0.15,
    "trend_filter": False,
    "trailing_exit_threshold": 0.5,
    "timeframe": "1d",
    "fee": 0.001,
    "starting_equity": 100000,
    "monthly_rotation": True,
    "ranking_metric": "sharpe",
    "top_n_pairs": 5
}

results = []

for lookback in lookbacks:
    for z_entry in z_entries:
        for capital_weight in capital_weights:
            print(f"🔍 Testing lookback={lookback}, z_entry={z_entry}, capital_weight={capital_weight}")

            params = base_config.copy()
            params.update({
                "lookback": lookback,
                "z_entry": z_entry,
                "strong_weight": capital_weight,
                "moderate_weight": capital_weight
            })

            strat = StrategyImpl()
            strat.configure(params)
            strat.run()

            if hasattr(strat, "results"):
                for r in strat.results:
                    results.append({
                        **r,
                        "lookback": lookback,
                        "z_entry": z_entry,
                        "capital_weight": capital_weight
                    })

df = pd.DataFrame(results)
output_path = "logs/relative_value_gridsearch.csv"
df.to_csv(output_path, index=False)
print(f"✅ Grid search results saved to {output_path}")
