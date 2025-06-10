import os
import pandas as pd
import altair as alt

def load_relative_value_results():
    path = "logs/relative_value_summary_all.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return pd.DataFrame()

def load_equity_curve_relative(asset):
    fname = f"logs/equity_curve_relative_value_{asset.lower()}_1d.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname, parse_dates=['timestamp'])
        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='equity:Q'
        ).properties(height=300)
        return chart
    else:
        return alt.Chart(pd.DataFrame({'timestamp': [], 'equity': []})).mark_line()
