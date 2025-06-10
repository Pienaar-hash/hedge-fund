# 3_Volatility_Targeting.py
import streamlit as st
import pandas as pd
import altair as alt
import os

st.title("📉 Volatility Targeting")

st.markdown("""
Volatility targeting strategies adjust position sizes based on rolling volatility estimates to achieve more stable risk-adjusted returns.
This dashboard highlights performance metrics, equity growth, and trade distributions.
""")

def load_volatility_results():
    dfs = []
    folder = "logs"
    for fname in os.listdir(folder):
        if fname.startswith("volatility_targeting_summary_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            asset_name = fname.replace("volatility_targeting_summary_", "").replace(".csv", "")
            df['asset'] = asset_name.upper()
            df['timeframe'] = '1d'  # assuming 1D for all
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            dfs.append(df)
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    return pd.DataFrame(columns=['sharpe_ratio', 'cagr', 'max_drawdown', 'asset', 'timeframe'])

def load_equity_curve_volatility(asset, timeframe="1d"):
    fname = f"logs/equity_curve_vol_target_{asset.lower()}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname, parse_dates=['timestamp'])
        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='equity:Q'
        ).properties(height=300)
        return chart
    else:
        return alt.Chart(pd.DataFrame({'timestamp': [], 'equity': []})).mark_line()

def load_volatility_trades(asset, timeframe="1d"):
    fname = f"logs/vol_target_backtest_trades_{asset.lower()}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.rename(columns={'net_return_pct': 'return'}, inplace=True)
        df['asset'] = asset.upper()
        return df
    return pd.DataFrame()

def plot_equity_curve_volatility(asset, timeframe="1d"):
    fname = f"logs/equity_curve_vol_target_{asset.lower()}.csv"
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

vol_df = load_volatility_results()
if vol_df.empty:
    st.warning("No volatility targeting summary data found.")
else:
    assets = vol_df['asset'].unique()
    timeframes = vol_df['timeframe'].unique()

    col1, col2 = st.columns(2)
    selected_asset = col1.selectbox("Select Asset", assets)
    selected_tf = col2.selectbox("Select Timeframe", timeframes)

    filtered = vol_df[(vol_df['asset'] == selected_asset) & (vol_df['timeframe'] == selected_tf)]

    if not filtered.empty:
        filtered.columns = filtered.columns.str.strip().str.lower().str.replace(' ', '_')
        if 'sharpe_ratio' in filtered.columns:
            best = filtered.sort_values("sharpe_ratio", ascending=False).iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Sharpe Ratio", f"{best['sharpe_ratio']:.2f}")
            k2.metric("CAGR", f"{best['cagr']:.2%}")
            k3.metric("Max Drawdown", f"{best['max_drawdown']:.2%}")

        st.markdown("---")

        # Equity Curve
        st.subheader("📈 Equity Curve")
        equity_chart = plot_equity_curve_volatility(asset=selected_asset, timeframe=selected_tf)
        st.altair_chart(equity_chart, use_container_width=True)

        # Trade Return Distribution
        st.subheader("📂 Trade Return Distribution")
        trades_df = load_volatility_trades(asset=selected_asset, timeframe=selected_tf)
        if trades_df is not None and not trades_df.empty:
            dist_chart = plot_trade_distribution_volatility(trades_df)
            st.altair_chart(dist_chart, use_container_width=True)
        else:
            st.info("No trade data available for this configuration.")
    else:
        st.warning("No results found for the selected asset and timeframe.")
