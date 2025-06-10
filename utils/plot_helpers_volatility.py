import os
import pandas as pd
import altair as alt

def plot_equity_curve_volatility(asset):
    fname = f"logs/equity_curve_volatility_{asset.lower()}_1d.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname, parse_dates=['timestamp'])
        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='equity:Q'
        ).properties(height=300)
        return chart
    else:
        return alt.Chart(pd.DataFrame({'timestamp': [], 'equity': []})).mark_line()

def plot_trade_distribution_volatility(df):
    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({'return': []})).mark_bar()
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('return:Q', bin=alt.Bin(maxbins=50)),
        y='count()',
    ).properties(title="Trade Return Distribution", height=300)
    return chart
