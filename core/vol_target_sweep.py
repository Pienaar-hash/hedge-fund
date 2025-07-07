# === core/vol_target_sweep.py ===
import importlib
from strategies import volatility_target
import os
import pandas as pd
import numpy as np

# Parameter ranges to sweep
target_vols = [0.005, 0.01, 0.015, 0.02]
rebalance_periods = [1, 3, 5, 10]

# Asset universe to test
symbols = ["BTCUSDT", "DOGEUSDT", "INJUSDT", "ARBUSDT", "SOLUSDT", "XRPUSDT", "MATICUSDT"]

# Fixed parameters
common_params = {
    "lookback": 20,
    "max_leverage": 3,
    "fee": 0.001,
    "starting_equity": 100000,
    "use_trend_filter": True  # ✅ Enable EMA-based trend filter
}

all_results = []

for symbol in symbols:
    asset_results = []
    print(f"\n🔎 Sweeping for {symbol}...")

    for target_vol in target_vols:
        for rebalance_period in rebalance_periods:
            trend_tag = "trend" if common_params["use_trend_filter"] else "notrend"
            label = f"{symbol.lower()}_vol_{target_vol}_rebalance_{rebalance_period}_{trend_tag}"
            print(f"  ▶ Combo: {label}")

            # Configure and run strategy
            StrategyImpl = volatility_target.StrategyImpl
            strat = StrategyImpl()
            params = common_params.copy()
            params.update({
                "assets": [{"symbol": symbol, "tf": "4H"}],
                "target_vol": target_vol,
                "rebalance_period": rebalance_period
            })
            strat.configure(params)
            strat.run()

            if hasattr(strat, "trades_df") and not strat.trades_df.empty:
                df = strat.trades_df.copy()
                df = df.sort_values("exit_time")
                df.set_index("exit_time", inplace=True)
                raw_returns = np.exp(df["pnl_log_return"]) - 1
                equity = (1 + raw_returns).cumprod()

                # Save equity curve
                equity_df = pd.DataFrame({
                    "timestamp": df.index,
                    "equity": equity.values
                })
                equity_df.to_csv(f"logs/equity_curve_vol_sweep_{label}.csv", index=False)

                sharpe = raw_returns.mean() / raw_returns.std() * np.sqrt(252) if raw_returns.std() > 0 else np.nan
                mdd = (equity / equity.cummax() - 1).min()
                cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1

                asset_results.append({
                    "Symbol": symbol,
                    "Target Vol": target_vol,
                    "Rebalance Period": rebalance_period,
                    "Sharpe": sharpe,
                    "Max Drawdown": mdd,
                    "CAGR": cagr,
                    "Trend Filter": common_params["use_trend_filter"]
                })

    # Save per-asset sweep results
    df_asset = pd.DataFrame(asset_results)
    df_asset.to_csv(f"logs/vol_target_sweep_{symbol.lower()}.csv", index=False)
    all_results.extend(asset_results)

# Save global sweep leaderboard
df_all = pd.DataFrame(all_results)
df_all.to_csv("logs/vol_target_sweep_all_assets.csv", index=False)
print("\n✅ Full multi-asset sweep completed. Results saved to logs/vol_target_sweep_all_assets.csv")
