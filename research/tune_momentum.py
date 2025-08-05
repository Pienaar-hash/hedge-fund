# === research/tune_momentum.py ===
import itertools
import json
import os
from datetime import datetime
import pandas as pd
from portfolio.backtest_runner import run_strategy

# === Grid Search Space ===
search_space = {
    "zscore_threshold": [0.8, 1.0, 1.2],
    "rebalance_period": [6, 12],
    "top_n": [1, 2],
    "atr_multiplier": [1.0, 1.2, 1.5]
}

# === Static Parameters ===
base_params = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "timeframe": "1h",
    "lookback": 20,
    "hold_period": 24,
    "vol_threshold": 0.007,
    "use_trend_filter": True,
    "allow_shorts": True,
    "min_reward_risk": 1.0,
    "fee": 0.001,
    "starting_equity": 100000
}

# === Log Setup ===
os.makedirs("logs/tune", exist_ok=True)
log_path = "logs/tune/momentum_tuning_results.csv"
results = []

def label_for(combo):
    return f"tune_z{combo['zscore_threshold']}_r{combo['rebalance_period']}_n{combo['top_n']}_atr{combo['atr_multiplier']}"

# === Run Grid Search ===
param_keys = list(search_space.keys())
combinations = list(itertools.product(*[search_space[k] for k in param_keys]))
print(f"🔍 Tuning {len(combinations)} combinations...")

for values in combinations:
    combo = dict(zip(param_keys, values))
    run_params = base_params.copy()
    run_params.update(combo)
    run_params["label"] = label_for(combo)

    try:
        run_strategy("momentum", run_params)

        # Load summary output
        summary_file = f"logs/summary_{run_params['label']}.csv"
        if os.path.exists(summary_file):
            summary = pd.read_csv(summary_file)
            if not summary.empty:
                row = summary.iloc[0].to_dict()
                row.update(combo)
                results.append(row)
    except Exception as e:
        print(f"❌ Error for combo {combo}: {e}")

# === Save Results ===
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by="Expectancy", ascending=False)
    df.to_csv(log_path, index=False)
    print(f"✅ Saved tuning results to {log_path}")
else:
    print("⚠️ No valid tuning results saved.")
