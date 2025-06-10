import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategies.factor_based import fetch_data, generate_scores, simulate_trades

st.set_page_config(page_title="Tune Factor Strategy", layout="wide")
st.title("📊 Factor-Based Strategy Tuner")

# Load data
df = fetch_data()

# Sliders for weights
st.markdown("### 🎛️ Adjust Factor Weights")
momentum = st.slider("Momentum Weight", 0.0, 1.0, 0.33, 0.01)
volatility = st.slider("Volatility Weight", 0.0, 1.0, 0.33, 0.01)
value = st.slider("Value Weight", 0.0, 1.0, 0.34, 0.01)

# Normalize weights
total = momentum + volatility + value
weights = {
    'momentum': round(momentum / total, 4),
    'volatility': round(volatility / total, 4),
    'value': round(value / total, 4)
}
st.markdown(f"🔁 Normalized Weights: `{weights}`")

if st.button("🚀 Run Simulation"):
    df_scored = generate_scores(df.copy(), weights)
    trades = simulate_trades(df_scored)

    if trades:
        signals_df = pd.DataFrame(trades)
        st.success(f"✅ {len(trades)} trades generated.")
        st.dataframe(signals_df.tail(10))

        # --- Summary Stats ---
        returns = signals_df['price'].pct_change().dropna()
        sharpe = 0
        if not returns.empty and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        cumulative = (1 + returns).cumprod()
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()

        st.markdown(f"""
        ### 📈 Performance Summary  
        - **Sharpe Ratio**: `{sharpe:.2f}`  
        - **Win Rate**: `{win_rate:.1%}`  
        - **Max Drawdown**: `{max_dd:.1%}`  
        """)

        # Score distribution
        st.markdown("### 📊 Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        signals_df['score'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Score Distribution (Top Ranked Assets)')
        st.pyplot(fig)

        # Top symbols
        st.markdown("### 🔝 Top Selected Symbols")
        top_assets = signals_df.groupby("symbol").size().sort_values(ascending=False).head(10)
        st.bar_chart(top_assets)

        # --- Capital Growth ---
        st.markdown("### 💹 Cumulative Capital Growth")
        capital = (1 + returns).cumprod()
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        capital.plot(ax=ax2, title="Capital Growth Over Time", color='green')
        ax2.set_ylabel("Portfolio Value")
        ax2.set_xlabel("Trade Index")
        st.pyplot(fig2)

        # CSV export
        csv = signals_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Trade Log as CSV",
            data=csv,
            file_name='factor_trades.csv',
            mime='text/csv'
        )

        # Save config
        if st.button("💾 Save as Best Config"):
            weights['source'] = 'Streamlit Manual Tuner'
            with open('logs/factor_best_params.json', 'w') as f:
                json.dump(weights, f, indent=2)
            st.success("✅ Best config saved to logs/factor_best_params.json")

    else:
        st.warning("No trades generated. Try different weights.")
