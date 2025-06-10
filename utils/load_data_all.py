# utils/load_data_all.py
import pandas as pd
import os

def load_all_summary_metrics():
    files = [
        ("logs/momentum_btcusdt_1h_results.csv", "momentum"),
        ("logs/momentum_ethusdt_1h_results.csv", "momentum"),
        ("logs/momentum_solusdt_1h_results.csv", "momentum"),
        ("logs/momentum_btcusdt_1d_results.csv", "momentum"),
        ("logs/momentum_ethusdt_1d_results.csv", "momentum"),
        ("logs/momentum_solusdt_1d_results.csv", "momentum"),
        ("logs/momentum_bnbusdt_1d_results.csv", "momentum"),
        ("logs/momentum_adausdt_1d_results.csv", "momentum"),
        ("logs/momentum_avaxusdt_1d_results.csv", "momentum"),
        ("logs/momentum_dogeusdt_1d_results.csv", "momentum"),
        ("logs/volatility_trade_summary.csv", "volatility"),
        ("logs/relative_value_trade_summary.csv", "relative_value"),
    ]

    all_dfs = []
    for path, strategy in files:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['strategy'] = strategy
            if 'symbol' in df.columns and 'timeframe' in df.columns:
                df['asset'] = df['symbol'].str.upper() + "_" + df['timeframe'].str.upper()
            elif 'asset' not in df.columns:
                df['asset'] = os.path.basename(path).replace("_results.csv", "").replace("momentum_", "").upper()
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()