import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dashboard.utils.dashboard_helpers import load_equity_curve, compute_rolling_metrics

st.set_page_config(page_title="Portfolio Equity", layout="wide")
st.title("📈 Portfolio Equity Curve vs Benchmarks")

try:
    eq = pd.read_csv("logs/portfolio_simulated_equity_blended.csv", parse_dates=["timestamp"])
    eq = eq.sort_values("timestamp")
    eq["norm_equity"] = eq["equity"] / eq["equity"].iloc[0]
    eq["drawdown"] = eq["norm_equity"] / eq["norm_equity"].cummax() - 1
    if eq.empty:
        st.warning("No equity curve data found. Please run the simulator or check filenames.")
        st.stop()

    
    # Determine common start timestamp
    min_timestamp = eq["timestamp"].min()

    # Load optional strategy overlays
    strategy_files = [
        f for f in os.listdir("logs")
        if f.startswith("portfolio_simulated_equity_") and f.endswith(".csv")
    ]
    selected_strategies = st.multiselect("Overlay strategies on equity curve:", strategy_files)

    eq["equity"] = eq["equity"] / eq["equity"].iloc[0]  # Normalize for chart comparison
    overlay_df = eq.copy()
    overlay_df["source"] = "Portfolio"
    drawdown_df = eq[["timestamp", "drawdown"]].copy()
    drawdown_df["source"] = "Portfolio"

    for file in selected_strategies:
        path = os.path.join("logs", file)
        df = pd.read_csv(path)
        if "timestamp" in df.columns and "equity" in df.columns:
            df = df[["timestamp", "equity"]].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= min_timestamp]
            df["equity"] = df["equity"] / df["equity"].iloc[0]
            label = file.replace("portfolio_simulated_equity_", "").replace(".csv", "")
            df["source"] = label
            overlay_df = pd.concat([overlay_df, df], axis=0)
            df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
            dd = df[["timestamp", "drawdown"]].copy()
            dd["source"] = label
            drawdown_df = pd.concat([drawdown_df, dd], axis=0)

    # Load and append benchmark equity curves
    def load_benchmark_series(filename, label):
        path = f"data/processed/{filename}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= min_timestamp]
            df = df.sort_values("timestamp")
            df["equity"] = df["close"] / df["close"].iloc[0]
            df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
            equity_df = df[["timestamp", "equity"]].copy()
            equity_df["source"] = label
            dd_df = df[["timestamp", "drawdown"]].copy()
            dd_df["source"] = label
            return equity_df, dd_df
        return None, None

    for label, file in {"BTC": "btcusdt_1d.csv", "SPY": "sp500_1d.csv", "ETH": "ethusdt_1d.csv"}.items():
        df_bench, df_dd = load_benchmark_series(file, label)
        if df_bench is not None:
            overlay_df = pd.concat([overlay_df, df_bench], axis=0)
        if df_dd is not None:
            drawdown_df = pd.concat([drawdown_df, df_dd], axis=0)

    st.altair_chart(
        alt.Chart(overlay_df).mark_line().encode(
            x="timestamp:T",
            y=alt.Y("equity:Q", title="Normalized Equity"),
            color="source:N"
        ).properties(height=300),
        use_container_width=True
    )

    st.altair_chart(
        alt.Chart(drawdown_df).mark_area(opacity=0.5).encode(
            x="timestamp:T",
            y=alt.Y("drawdown:Q", title="Drawdown"),
            color="source:N"
        ).properties(height=150),
        use_container_width=True
    )

    st.subheader("📊 Performance Metrics")

    
    eq["returns"] = eq["norm_equity"].pct_change().fillna(0)
    total_return = eq["norm_equity"].iloc[-1] - 1
    cagr = (eq["norm_equity"].iloc[-1]) ** (365 / len(eq)) - 1
    volatility = eq["returns"].std() * np.sqrt(365)
    sharpe = cagr / volatility if volatility != 0 else 0
    max_dd = eq["drawdown"].min()

    # Load benchmarks
    def load_benchmark(filename):
        path = f"data/processed/{filename}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= min_timestamp]
            df = df.sort_values("timestamp")
            df["equity"] = df["close"] / df["close"].iloc[0]
            df["returns"] = df["equity"].pct_change().fillna(0)
            drawdown = df["equity"] / df["equity"].cummax() - 1
            metrics = {
                "Total Return": df["equity"].iloc[-1] - 1,
                "CAGR": (df["equity"].iloc[-1]) ** (365 / len(df)) - 1,
                "Volatility": df["returns"].std() * np.sqrt(365),
                "Sharpe": 0,
                "Max DD": drawdown.min()
            }
            if metrics["Volatility"] != 0:
                metrics["Sharpe"] = metrics["CAGR"] / metrics["Volatility"]
            return metrics
        else:
            return None

    benchmarks = {
        "BTC": load_benchmark("btcusdt_1d.csv"),
        "SPY": load_benchmark("sp500_1d.csv"),
        "ETH": load_benchmark("ethusdt_1d.csv"),
        "GOLD": load_benchmark("goldusd_1d.csv")
    }

    st.markdown("### 📈 Portfolio vs Benchmarks")
    rows = [
        {"Asset": "Portfolio", "Total Return": total_return, "CAGR": cagr, "Volatility": volatility, "Sharpe": sharpe, "Max DD": max_dd}
    ]
    for label, metric in benchmarks.items():
        if metric:
            rows.append({"Asset": label, **metric})

    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.set_index("Asset").style.format("{:.2%}"), use_container_width=True)

except Exception as e:
    st.warning(f"Error loading portfolio or benchmark data: {e}")
