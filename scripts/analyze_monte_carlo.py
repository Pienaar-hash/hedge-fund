# analyze_monte_carlo.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Monte Carlo results
df = pd.read_csv("logs/relative_value_monte_carlo.csv")

# Display top 10 configurations sorted by cumulative return
print("\n\U0001F3C6 Top 10 Configurations:")
top_10 = df.sort_values(by='cumulative_return', ascending=False).head(10)
print(top_10.to_string(index=False))

# Plot cumulative return vs number of trades
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="num_trades",
    y="cumulative_return",
    hue="window",
    palette="viridis",
    s=100,
    edgecolor="black"
)
plt.title("Monte Carlo Sweep: Return vs Trade Count")
plt.xlabel("Number of Trades")
plt.ylabel("Cumulative Return")
plt.axhline(0, color='red', linestyle='--', alpha=0.6)
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/monte_carlo_sweep_plot.png")
print("\U0001F4CA Scatterplot saved to logs/monte_carlo_sweep_plot.png")
