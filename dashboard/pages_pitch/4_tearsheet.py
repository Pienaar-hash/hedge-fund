import streamlit as st
from dashboard.utils.dashboard_helpers import list_trade_logs, load_trade_log, compute_trade_summary, compute_pnl_correlation_heatmap, compute_pnl_distribution_clusters, compute_rolling_metrics
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📄 Strategy Tear Sheet", layout="wide")
st.title("📄 Strategy Tear Sheet")

st.markdown("""
This tear sheet summarizes the performance of selected strategy logs, including trade stats, distributional analysis, rolling metrics, and correlation heatmaps.
""")

trade_logs = list_trade_logs()
summary_table = []
if trade_logs:
    selected_logs = st.multiselect("Select trade logs for analysis:", trade_logs, default=trade_logs[:2])
    if selected_logs:
        for file in selected_logs:
            df = load_trade_log(file)
            label = file.replace('_', ' ').replace('.csv', '')
            st.markdown(f"### 📊 {label}")

            summary = compute_trade_summary(df)
            summary["Label"] = label
            summary_table.append(summary)

            col1, col2, col3 = st.columns(3)
            col1.metric("Win Rate", f"{summary.get('Win Rate', 0):.2%}")
            col1.metric("Avg Win", f"{summary.get('Avg Win', 0):.4f}")
            col1.metric("Avg Loss", f"{summary.get('Avg Loss', 0):.4f}")
            col2.metric("Expectancy", f"{summary.get('Expectancy', 0):.4f}")
            col2.metric("Profit Factor", f"{summary.get('Profit Factor', 0):.2f}")
            col2.metric("Payoff Ratio", f"{summary.get('Payoff Ratio', 0):.2f}")

            fig = compute_pnl_distribution_clusters(df)
            if fig:
                st.pyplot(fig)

            

                if "capital_used" in df.columns:
                    total_cap = df["capital_used"].sum()
                    avg_cap = df["capital_used"].mean()
                    col3.metric("Total Capital Used", f"{total_cap:,.0f}")
                    col3.metric("Avg Capital/Trade", f"{avg_cap:,.0f}")

            # Holding Time Analysis
            if "entry_time" in df.columns and "exit_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"])
                df["exit_time"] = pd.to_datetime(df["exit_time"])
                df["holding_time"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600  # in hours

                holding_fig, ax_ht = plt.subplots(figsize=(6, 2.5))
                sns.histplot(df["holding_time"], bins=30, kde=True, ax=ax_ht, color="skyblue")
                ax_ht.set_title("Holding Time Distribution (hrs)")
                ax_ht.set_xlabel("Hours")
                ax_ht.set_ylabel("Frequency")
                st.pyplot(holding_fig)

        

        # Summary table export
        st.subheader("📋 Exportable Summary Table")
        df_summary = pd.DataFrame(summary_table)
        st.dataframe(df_summary, use_container_width=True)

        csv = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Summary", csv, file_name="tearsheet_summary.csv")

        # Placeholder for PDF export
        
else:
    st.info("No trade logs found in /logs directory.")
