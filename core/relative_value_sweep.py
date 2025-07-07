# === core/relative_value_sweep.py ===
import os
import pandas as pd
import numpy as np
from strategies.relative_value import StrategyImpl

# Sweep parameters
z_entries = [1.0, 1.25, 1.5]
hold_periods = [2, 3, 5]
dd_mods = [True, False]

# Expanded ETH-based pairs
pairs = ["BTCUSDT", "AVAXUSDT", "OPUSDT", "MATICUSDT", "LDOUSDT"]

# Static config
base = "ETHUSDT"
timeframe = "1d"
capital = 100000
fee = 0.001
lookback = 20

LOG_FOLDER = "logs"

results = []

for quote in pairs:
    for z_entry in z_entries:
        for hold_period in hold_periods:
            for dd_mod in dd_mods:
                label = f"{base}_{quote}_z{z_entry}_h{hold_period}_ddmod{int(dd_mod)}"
                print(f"🔍 Running: {label}")

                params = {
                    "base": base,
                    "pairs": [quote],
                    "lookback": lookback,
                    "z_entry": z_entry,
                    "z_exit": 0.5,
                    "hold_period": hold_period,
                    "capital": capital,
                    "capital_weight": 0.05,
                    "fee": fee,
                    "timeframe": timeframe,
                    "asymmetric_drawdown": dd_mod
                }

                strat = StrategyImpl()
                strat.configure(params)
                strat.run()

                # Load from saved log file instead of strat.trades_df
                log_path = f"{LOG_FOLDER}/relative_value_trades_{base}_{quote}_z{z_entry}_h{hold_period}_ddmod{int(dd_mod)}.csv"
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path, parse_dates=["exit_time"])
                    if not df.empty:
                        df = df.sort_values("exit_time")
                        df.set_index("exit_time", inplace=True)
                        pnl = df["net_ret"]
                        equity = (1 + pnl).cumprod()

                        # Corrected time-based Sharpe & CAGR
                        days = (df.index[-1] - df.index[0]).days
                        years = days / 365.25 if days > 0 else 1
                        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(len(pnl) / years) if pnl.std() > 0 else 0
                        cagr = (equity.iloc[-1] / equity.iloc[0])**(1 / years) - 1 if years > 0 else 0
                        mdd = (equity / equity.cummax() - 1).min()

                        results.append({
                            "Pair": f"{base}/{quote}",
                            "Z_Entry": z_entry,
                            "Hold_Period": hold_period,
                            "AsymDD": dd_mod,
                            "Sharpe": sharpe,
                            "MaxDD": mdd,
                            "CAGR": cagr,
                            "Trades": len(df)
                        })

                        print(f"✅ Saved result: Sharpe={sharpe:.2f}, MaxDD={mdd:.2%}, CAGR={cagr:.2%}, Trades={len(df)}")
                    else:
                        print(f"⚠️ No trades found in {log_path}. Appending zeroed row.")
                        results.append({
                            "Pair": f"{base}/{quote}",
                            "Z_Entry": z_entry,
                            "Hold_Period": hold_period,
                            "AsymDD": dd_mod,
                            "Sharpe": 0,
                            "MaxDD": 0,
                            "CAGR": 0,
                            "Trades": 0
                        })
                else:
                    print(f"⚠️ Trade log not found: {log_path}. Appending zeroed row.")
                    results.append({
                        "Pair": f"{base}/{quote}",
                        "Z_Entry": z_entry,
                        "Hold_Period": hold_period,
                        "AsymDD": dd_mod,
                        "Sharpe": 0,
                        "MaxDD": 0,
                        "CAGR": 0,
                        "Trades": 0
                    })

# Always save sweep results
df_results = pd.DataFrame(results)
df_results.to_csv("logs/relative_value_sweep_results.csv", index=False)
print("✅ Sweep completed. Results saved to logs/relative_value_sweep_results.csv")
