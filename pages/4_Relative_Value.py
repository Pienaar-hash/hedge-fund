# 4_Relative_Value.py (Updated with Seaborn heatmap and interactivity)
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("\U0001F501 Relative Value Strategy")

st.markdown("""
Relative value strategies take long and short positions in asset pairs to exploit temporary mispricings. This dashboard presents top-performing pairs with interactive filters and visual analytics.
""")

@st.cache_data
def load_relative_value_results():
    path = "logs/relative_value_summary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    return pd.DataFrame()

rv_df = load_relative_value_results()

if rv_df.empty:
    st.warning("No relative value summary data found.")
else:
    with st.sidebar:
        st.subheader("Filters")
        available_timeframes = sorted(rv_df["timeframe"].dropna().unique())
        timeframe = st.selectbox("Timeframe", available_timeframes)

        metrics = ["sharpe_ratio", "total_pnl", "num_trades"]
        available_metrics = [m for m in metrics if m in rv_df.columns]
        selected_metric = st.selectbox("Heatmap Metric", available_metrics)

    filtered = rv_df[rv_df["timeframe"] == timeframe]

    # Heatmap Section
    st.subheader(f"\U0001F4CA Heatmap of {selected_metric.replace('_', ' ').title()} by Pair")
    heatmap_data = filtered.pivot(index="pair", columns="timeframe", values=selected_metric)

    fig, ax = plt.subplots(figsize=(10, len(heatmap_data) * 0.5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Full Table View
    st.markdown("---")
    st.subheader("\U0001F4DD Full Summary Table")
    st.dataframe(filtered.sort_values(by=selected_metric, ascending=False), use_container_width=True)
