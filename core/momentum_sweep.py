# === core/momentum_sweep.py ===
import os
import pandas as pd
import numpy as np
import itertools
import json
from strategies.momentum import StrategyImpl

# Load JSON config
def load_config():
    with open("config/strategy_config.json", "r") as f:
        config = json.load(f)
    for strat in config.get("strategies", []):
        if strat.get("name") == "momentum":
            return strat["params"]
    raise KeyError("Momentum strategy config not found.")

def compute_summary(df, label, pnl_col="pnl_log_return"):
    win = df[df[pnl_col] > 0]
    loss = df[df[pnl_col] < 0]
    win_rate = len(win) / len(df) if len(df) > 0 else 0
    avg_win = win[pnl_col].mean()
    avg_loss = loss[pnl_col].mean()
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    profit_factor = win[pnl_col].sum() / abs(loss[pnl_col].sum()) if not loss.empty else float('inf')
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss else float('inf')
    sharpe = df[pnl_col].mean() / df[pnl_col].std() * (252 ** 0.5) if df[pnl_col].std() > 0 else 0

    return {
        "Label": label,
        "Trades": len(df),
        "WinRate": win_rate,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "Expectancy": expectancy,
        "ProfitFactor": profit_factor,
        "PayoffRatio": payoff_ratio,
        "Sharpe": sharpe
    }

configs = [
    {
        "lookback": 10,
        "reward_risk": r,
        "tp": tp,
        "sl": sl,
        "use_vol_filter": bool(vf),
        "use_ema_filter": bool(ema),
        "use_volatility_zscore": bool(vol),
        "zscore_threshold": z,
    }
    for r, tp, sl, vf, ema, vol, z in itertools.product(
        [0.3, 0.5, 0.7],
        [10, 15],
        [5, 10],
        [0, 1],
        [0, 1],
        [0, 1],
        [1.0, 1.5, 2.0],
    )
]

base = load_config()
summaries = []
label_config_map = {}

# Phase 1: run and collect summaries only (no logging to disk)
for cfg in configs:
    label = f"{base['symbols'][0]}_lb{base['lookback']}_r{cfg['reward_risk']}_tp{cfg['tp']}_sl{cfg['sl']}_vf{int(cfg['use_vol_filter'])}_ema{int(cfg['use_ema_filter'])}_vol{int(cfg['use_volatility_zscore'])}_z{cfg['zscore_threshold']}"
    params = base.copy()
    params.update(cfg)
    params["label"] = label
    params["log"] = False  # suppress logging during Phase 1
    strategy = StrategyImpl()
    strategy.configure(params)
    strategy.run()

    if hasattr(strategy, "trades_df") and strategy.trades_df is not None and not strategy.trades_df.empty:
        summary = compute_summary(strategy.trades_df, label)
        summaries.append(summary)
        label_config_map[label] = cfg

# Rank and filter: trades >= 100 and sharpe > 1.0, keep top 5
if summaries:
    df = pd.DataFrame(summaries)
    df = df[(df["Trades"] >= 100) & (df["Sharpe"] > 1.0)].sort_values("Sharpe", ascending=False).head(5)
    df.to_csv("logs/momentum_top_configs.csv", index=False)
    print("\n✅ Top configs saved to logs/momentum_top_configs.csv")

    # Phase 2: rerun only selected configs and save equity + trade logs
    for row in df.itertuples():
        label = row.Label
        cfg = label_config_map[label]
        params = base.copy()
        params.update(cfg)
        params["label"] = label
        params["log"] = True  # enable logging for Phase 2

        print(f"📈 Logging: {label}")
        strategy = StrategyImpl()
        strategy.configure(params)
        strategy.run()

        if hasattr(strategy, "trades_df") and strategy.trades_df is not None and not strategy.trades_df.empty:
            trades_path = f"logs/momentum_trades_{label}.csv"
            strategy.trades_df.to_csv(trades_path, index=False)
            print(f"✅ Saved trades to {trades_path}")

        if hasattr(strategy, "log_results"):
            strategy.log_results(label=f"momentum_{label}")
else:
    print("⚠️ No valid trade summaries generated.")
