import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from pathlib import Path

# Parameters
TF = "1D"
WINDOW = 10
TOP_N = 3
HOLD_PERIOD = 5  # number of days to hold position
TRANSACTION_COST = 0.001  # 0.1% cost per side
DATA_PATH = Path("data/processed")
LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT"
]

def load_price_series(symbol: str):
    file = DATA_PATH / f"momentum_{symbol.lower()}_{TF.lower()}.csv"
    if not file.exists():
        print(f"❌ Missing file: {file}")
        return None
    df = pd.read_csv(file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df["close"]

def compute_cross_sectional_signals():
    prices = {}
    for symbol in ASSETS:
        series = load_price_series(symbol)
        if series is not None:
            prices[symbol] = series

    price_df = pd.DataFrame(prices).dropna()
    returns = price_df.pct_change()
    rolling = returns.rolling(WINDOW).mean()

    signals = []
    for i in range(WINDOW, len(rolling)):
        row = rolling.iloc[i]
        long_assets = row.nlargest(TOP_N).index.tolist()
        short_assets = row.nsmallest(TOP_N).index.tolist()
        signals.append({
            "timestamp": rolling.index[i],
            "long": long_assets,
            "short": short_assets
        })

    signals_df = pd.DataFrame(signals)
    signals_df.to_csv(LOG_PATH / "cross_sectional_signals.csv", index=False)
    return signals_df, price_df

def simulate_equity_curve(signals_df, price_df):
    def compute_profit_factor(trades):
        gains = trades[trades["return"] > 0]["return"].sum()
        losses = -trades[trades["return"] < 0]["return"].sum()
        return gains / losses if losses > 0 else np.nan

    equity = [1]
    timestamps = []
    trade_log = []
    daily_returns = []

    for i in range(len(signals_df) - HOLD_PERIOD):
        row = signals_df.iloc[i]
        date = pd.to_datetime(row["timestamp"])
        next_date = pd.to_datetime(signals_df.iloc[i + HOLD_PERIOD]["timestamp"])

        long_assets = row["long"]
        short_assets = row["short"]

        if date not in price_df.index or next_date not in price_df.index:
            continue

        pnl = 0
        position_returns = []

        for asset in long_assets:
            ret = price_df[asset].loc[next_date] / price_df[asset].loc[date] - 1 - TRANSACTION_COST * 2
            pnl += ret
            position_returns.append({"entry_time": date, "exit_time": next_date, "asset": asset, "side": "long", "return": ret, "holding_days": HOLD_PERIOD})

        for asset in short_assets:
            ret = price_df[asset].loc[date] / price_df[asset].loc[next_date] - 1 - TRANSACTION_COST * 2
            pnl += ret
            position_returns.append({"entry_time": date, "exit_time": next_date, "asset": asset, "side": "short", "return": ret, "holding_days": HOLD_PERIOD})

        pnl /= (len(long_assets) + len(short_assets))
        new_equity = equity[-1] * (1 + pnl)
        equity.append(new_equity)
        daily_returns.append(pnl)
        timestamps.append(next_date)
        trade_log.extend(position_returns)

    equity_df = pd.DataFrame({"timestamp": timestamps, "equity": equity[1:]})
    trades_df = pd.DataFrame(trade_log)

    # Metrics
    ret_series = pd.Series(daily_returns)
    sharpe = ret_series.mean() / ret_series.std() * np.sqrt(252 / HOLD_PERIOD) if ret_series.std() > 0 else np.nan
    max_dd = (np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)
    max_drawdown = max_dd.max() if len(max_dd) else 0
    win_rate = (ret_series > 0).mean()
    start_value = equity[0]
    end_value = equity[-1]
    num_years = len(timestamps) * HOLD_PERIOD / 252
    cagr = (end_value / start_value) ** (1 / num_years) - 1 if num_years > 0 else np.nan
    num_trades = len(trade_log)

    profit_factor = compute_profit_factor(trades_df)
    return equity_df, trades_df, ret_series, sharpe, max_drawdown, win_rate, cagr, num_trades, profit_factor

def add_benchmark_curves(equity_df, price_df):
    aligned_index = pd.to_datetime(equity_df["timestamp"])
    btc_series = price_df["BTCUSDT"]
    btc_series.index = pd.to_datetime(btc_series.index)

    btc_aligned = btc_series.reindex(btc_series.index.union(aligned_index)).sort_index().ffill()
    btc_curve = btc_aligned.loc[aligned_index] / btc_aligned.loc[aligned_index[0]]

    btc_df = pd.DataFrame({"timestamp": aligned_index, "equity": btc_curve.values, "asset": "BTC Buy&Hold"})
    strategy_df = equity_df.copy()
    strategy_df["asset"] = "Cross-Sectional"

    all_df = pd.concat([strategy_df, btc_df])
    return all_df

def grid_search_sensitivity(price_df, top_n_default, hold_default, window_default):
    st.sidebar.subheader("🎛️ Parameter Tuning")
    top_n_input = st.sidebar.slider("Top N Assets", min_value=1, max_value=5, value=top_n_default)
    hold_period_input = st.sidebar.slider("Hold Period (days)", min_value=1, max_value=10, value=hold_default)
    window_input = st.sidebar.slider("Lookback Window (days)", min_value=5, max_value=30, value=window_default)

    st.subheader("🔍 Parameter Sensitivity: Sharpe Ratio")
    results = []
    for top_n in [2, 3, 4, 5]:
        for hold in [3, 5, 7, 10]:
            global TOP_N, HOLD_PERIOD, WINDOW
            TOP_N, HOLD_PERIOD, WINDOW = top_n, hold, window_input
            signals_df, _ = compute_cross_sectional_signals()
            _, _, ret_series, sharpe, *_ = simulate_equity_curve(signals_df, price_df)
            results.append({"TOP_N": top_n, "HOLD": hold, "Sharpe": sharpe})

    grid_df = pd.DataFrame(results)
    pivot = grid_df.pivot(index="HOLD", columns="TOP_N", values="Sharpe")

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Sharpe Ratio Heatmap")
    st.pyplot(fig)

    return top_n_input, hold_period_input, window_input


def main():
    st.title("Cross-Sectional Momentum Strategy")

    st.markdown("""
    ### 🧠 Strategy Summary
    This strategy ranks assets by their average returns over the past **{WINDOW}** days and selects the top **{TOP_N}** as long positions and bottom **{TOP_N}** as short positions. Positions are held for **{HOLD_PERIOD}** days and then rotated. PnL is adjusted for **{TRANSACTION_COST:.2%}** transaction cost per side.

    - **Universe:** {', '.join(ASSETS)}
    - **Timeframe:** {TF}
    - **Signal:** Rolling mean of daily returns (cross-sectional ranking)
    - **Execution:** Equal-weighted long/short basket
    """)
    st.sidebar.subheader("🎛️ Parameter Tuning")
    global TOP_N, HOLD_PERIOD, WINDOW
    TOP_N = st.sidebar.slider("Top N Assets", min_value=1, max_value=5, value=TOP_N)
    HOLD_PERIOD = st.sidebar.slider("Hold Period (days)", min_value=1, max_value=10, value=HOLD_PERIOD)
    WINDOW = st.sidebar.slider("Lookback Window (days)", min_value=5, max_value=30, value=WINDOW)

    signals_df, price_df = compute_cross_sectional_signals()
    equity_df, trades_df, ret_series, sharpe, max_drawdown, win_rate, cagr, num_trades, profit_factor = simulate_equity_curve(signals_df, price_df)

    st.subheader("📊 Performance Metrics")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
    st.write(f"**Max Drawdown:** {max_drawdown:.2%}")
    st.write(f"**Win Rate:** {win_rate:.2%}")
    st.write(f"**CAGR:** {cagr:.2%}")
    st.write(f"**Number of Trades:** {num_trades}")
    st.write(f"**Profit Factor:** {profit_factor:.2f}")

    st.subheader("📈 Equity Curve vs BTC")
    all_df = add_benchmark_curves(equity_df, price_df)
    chart = alt.Chart(all_df).mark_line().encode(
        x="timestamp:T",
        y="equity:Q",
        color="asset:N"
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.caption("This chart compares the cumulative equity of the cross-sectional momentum strategy to a simple buy-and-hold of BTC over the same period. Both are normalized to start at 1.")

    st.subheader("📋 Sample Trade Log")
    st.download_button("📥 Download Full Trade Log (CSV)", trades_df.to_csv(index=False), file_name="cross_sectional_trades.csv")
    trades_df['outcome'] = trades_df['return'].apply(lambda x: '✅ Win' if x > 0 else '❌ Loss')
    st.dataframe(
        trades_df[['entry_time', 'exit_time', 'asset', 'side', 'return', 'outcome']]
        .sort_values(by='entry_time', ascending=False)
        .reset_index(drop=True)
    )

    rolling_sharpe_df = pd.DataFrame({
        "timestamp": equity_df["timestamp"].iloc[-len(ret_series):],
        "rolling_sharpe": ret_series.rolling(20).mean() / ret_series.rolling(20).std() * np.sqrt(252 / HOLD_PERIOD)
    }).dropna()

    rolling_chart = alt.Chart(rolling_sharpe_df).mark_line(color='orange').encode(
        x="timestamp:T",
        y="rolling_sharpe:Q"
    ).properties(height=300)
    st.altair_chart(rolling_chart, use_container_width=True)

if __name__ == "__main__":
    main()
