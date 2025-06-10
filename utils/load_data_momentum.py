# --- load_data_momentum.py ---

import os
import pandas as pd
import altair as alt

def load_momentum_results():
    df_list = []
    logs_path = "logs"
    for file in os.listdir(logs_path):
        if file.startswith("momentum_") and file.endswith("_results.csv"):
            parts = file.replace("momentum_", "").replace("_results.csv", "").split("_")
            if len(parts) < 2:
                continue
            symbol = parts[0]
            tf = parts[1]
            df = pd.read_csv(os.path.join(logs_path, file))
            df['asset'] = symbol.upper()
            df['timeframe'] = tf.upper()
            df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def load_equity_curve_momentum(asset, timeframe):
    fname = f"logs/equity_curve_momentum_{asset.lower()}_{timeframe.lower()}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname, parse_dates=['timestamp'])
        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='equity:Q'
        ).properties(height=300)
        return chart
    else:
        return alt.Chart(pd.DataFrame({'timestamp': [], 'equity': []})).mark_line()

def load_momentum_trades(asset, timeframe):
    path = f"logs/momentum_trades_{asset.lower()}_{timeframe.lower()}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None