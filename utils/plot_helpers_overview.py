# utils/plot_helpers_overview.py
import pandas as pd
import altair as alt
import os

def plot_equity_curve(strategy: str, asset: str) -> alt.Chart:
    filename = f"logs/equity_curve_{strategy.lower()}_{asset.lower()}.csv"
    if not os.path.exists(filename):
        return alt.Chart(pd.DataFrame({"timestamp": [], "equity": []})).mark_line()

    df = pd.read_csv(filename, parse_dates=['timestamp'])
    return alt.Chart(df).mark_line().encode(
        x='timestamp:T',
        y='equity:Q'
    ).properties(
        title=f"Equity Curve for {strategy.upper()} - {asset.upper()}"
    )
