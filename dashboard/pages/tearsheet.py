import streamlit as st
from utils.dashboard_helpers import list_trade_logs, load_trade_log, compute_trade_summary, compute_pnl_correlation_heatmap, compute_pnl_distribution_clusters, compute_rolling_metrics
import pandas as pd
import io
import matplotlib.pyplot as plt

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

            if "pnl_log_return" in df.columns:
                df = df.sort_values("exit_time")
                df.set_index("exit_time", inplace=True)
                returns = df["pnl_log_return"].fillna(0)
                rolling_sharpe, rolling_dd = compute_rolling_metrics(returns)

                fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                rolling_sharpe.plot(ax=axes[0], title=f"Rolling Sharpe - {label}")
                rolling_dd.plot(ax=axes[1], title="Rolling Drawdown", color="red")
                plt.tight_layout()
                st.pyplot(fig)

        st.subheader("🧠 PnL Correlation Heatmap")
        fig_corr = compute_pnl_correlation_heatmap(selected_logs)
        st.pyplot(fig_corr)

        # Summary table export
        st.subheader("📋 Exportable Summary Table")
        df_summary = pd.DataFrame(summary_table)
        st.dataframe(df_summary, use_container_width=True)

        csv = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Summary", csv, file_name="tearsheet_summary.csv")

        # Placeholder for PDF export
        st.subheader("📄 Export Tear Sheet PDF (Coming Soon)")
        st.button("📤 Export PDF", disabled=True)
else:
    st.info("No trade logs found in /logs directory.")
