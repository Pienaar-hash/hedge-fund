import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.factor_based import fetch_data as fetch_factor_data, generate_scores, simulate_trades
from strategies.relative_value import fetch_data as fetch_rv_data, generate_signals
from strategies.eth_btc_arb import run_loop as run_eth_btc

import json
import pandas as pd

os.makedirs("logs", exist_ok=True)

# ---- Relative Value
with open("logs/relative_value_best_params.json") as f:
    rv_params = json.load(f)
rv_data = fetch_rv_data()
rv_trades = generate_signals(rv_data, z_entry=rv_params['z_entry'], z_exit=rv_params['z_exit'])
pd.DataFrame(rv_trades).to_csv("logs/relative_value_trades_live.csv", index=False)

# ---- Factor-Based
with open("logs/factor_best_params.json") as f:
    f_params = json.load(f)
factor_data = fetch_factor_data()
f_scored = generate_scores(factor_data, f_params)
f_trades = simulate_trades(f_scored)
pd.DataFrame(f_trades).to_csv("logs/factor_based_trades_live.csv", index=False)

# ---- ETH/BTC Spread Arb
run_eth_btc()

print("✅ All strategies executed and logs saved.")
