import streamlit as st
import os
import pandas as pd
import altair as alt
from utils.dashboard_helpers import get_color

st.set_page_config(page_title="Strategy Leaderboard", layout="wide")
st.title("🏆 Strategy Leaderboard")

try:
    equity_metrics_path = "logs/equity_metrics_summary.csv"
    trade_metrics_path = "logs/trade_metrics_summary.csv"
    equity_metrics = pd.read_csv(equity_metrics_path) if os.path.exists(equity_metrics_path) else pd.DataFrame()
    trade_metrics = pd.read_csv(trade_metrics_path) if os.path.exists(trade_metrics_path) else pd.DataFrame()

    col1, col2 = st.columns(2)

    col1.subheader("📊 Equity Performance")
    if not equity_metrics.empty:
        equity_metrics_filtered = equity_metrics[(equity_metrics['CAGR'] < 10) & (equity_metrics['Sharpe'] < 10) & (equity_metrics['Total Return'] < 10)]
        equity_metrics_sorted = equity_metrics_filtered.sort_values("Sharpe", ascending=False).reset_index(drop=True)
        top3 = equity_metrics_sorted.head(3).copy()
        top3.index = ["🥇", "🥈", "🥉"]
        equity_metrics_sorted = pd.concat([top3, equity_metrics_sorted.iloc[3:]]).reset_index(drop=True)

        col1.download_button("📥 Download Equity Metrics", equity_metrics_sorted.to_csv(index=False).encode(), file_name="equity_metrics_summary.csv")
        col1.dataframe(equity_metrics_sorted.style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "Total Return": "{:.2%}",
            "MaxDrawdown": "{:.2%}", "Volatility": "{:.2%}"
        }), use_container_width=True)

    col2.subheader("📋 Trade Performance")
    if not trade_metrics.empty:
        trade_metrics_sorted = trade_metrics.sort_values("Expectancy", ascending=False).reset_index(drop=True)
        col2.download_button("📥 Download Trade Metrics", trade_metrics_sorted.to_csv(index=False).encode(), file_name="trade_metrics_summary.csv")
        col2.dataframe(trade_metrics_sorted.style.format({
            "Win Rate": "{:.2%}", "Avg Win": "{:.4f}", "Avg Loss": "{:.4f}",
            "Expectancy": "{:.4f}", "Profit Factor": "{:.2f}", "Payoff Ratio": "{:.2f}"
        }), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Strategy Equity Curve + Drawdown")
    eq_files = [f for f in os.listdir("logs") if f.startswith("equity_curve_") and f.endswith(".csv")]
    filtered_eq_files = []
    for file in eq_files:
        try:
            df_tmp = pd.read_csv(os.path.join("logs", file))
            if "equity" in df_tmp.columns and df_tmp.shape[0] > 1:
                multiple = df_tmp["equity"].iloc[-1] / df_tmp["equity"].iloc[0]
                if multiple <= 5:
                    filtered_eq_files.append(file)
        except Exception:
            continue

    if filtered_eq_files:
        selected_eq = st.selectbox("Select a strategy:", filtered_eq_files)
        df_eq = pd.read_csv(os.path.join("logs", selected_eq))
        df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
        df_eq = df_eq.sort_values("timestamp")
        df_eq["norm_equity"] = df_eq["equity"] / df_eq["equity"].iloc[0]
        df_eq["drawdown"] = df_eq["norm_equity"] / df_eq["norm_equity"].cummax() - 1

        base = alt.Chart(df_eq).encode(x="timestamp:T")
        line = base.mark_line(color=get_color("momentum")).encode(y=alt.Y("norm_equity:Q", title="Equity"))
        dd = base.mark_area(color="darkred", opacity=0.5).encode(y=alt.Y("drawdown:Q", title="Drawdown"))

        st.altair_chart(line.properties(height=300), use_container_width=True)
        st.altair_chart(dd.properties(height=120), use_container_width=True)

except Exception as e:
    st.warning(f"Error loading leaderboard data: {e}")
