# plot_helpers_momentum.py
import altair as alt
import pandas as pd
import os

def plot_equity_curve_momentum(asset, strategy, timeframe):
    filename = f"logs/equity_curve_{strategy}_{asset}_{timeframe}.csv"
    if not os.path.exists(filename):
        return alt.Chart(pd.DataFrame({"timestamp": [], "equity": []})).mark_line()

    df = pd.read_csv(filename, parse_dates=['timestamp'])
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('equity:Q', title='Equity'),
        tooltip=['timestamp:T', 'equity']
    ).properties(title=f"Equity Curve — {asset} {timeframe}", height=300)
    return chart

def plot_trade_distribution_momentum(trades_df):
    if trades_df.empty:
        return alt.Chart(pd.DataFrame({"return": []})).mark_bar()

    # Use 'net_return_pct' if available, else fall back to 'return_pct'
    return_col = 'net_return_pct' if 'net_return_pct' in trades_df.columns else 'return_pct'

    if return_col not in trades_df.columns:
        return alt.Chart(pd.DataFrame({return_col: []})).mark_bar()

    chart = alt.Chart(trades_df).mark_bar(opacity=0.7).encode(
        x=alt.X(return_col, bin=alt.Bin(maxbins=30), title="Trade Return %"),
        y=alt.Y('count()', title='Count of Records'),
        tooltip=[return_col, 'count()']
    ).properties(title="Trade Return Distribution")
    return chart

def plot_momentum_heatmap(df, metric='Sharpe Ratio'):
    if df.empty or metric not in df.columns:
        return alt.Chart(pd.DataFrame({"TP": [], "SL": [], metric: []})).mark_rect()

    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('TP:O', title='Take Profit'),
        y=alt.Y('SL:O', title='Stop Loss'),
        color=alt.Color(metric, scale=alt.Scale(scheme='redyellowgreen')),
        tooltip=['TP', 'SL', metric]
    ).properties(title=f"{metric} Heatmap", height=300)
    return chart
