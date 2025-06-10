# analyze_factor_monte_carlo.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sweep results
df = pd.read_csv('logs/factor_monte_carlo.csv')
top = df.sort_values(by='cumulative_return', ascending=False).head(10)

# Print Top 10
print("\n🏆 Top 10 Weight Combinations:")
print(top.to_string(index=False))

# Plot: 3D scatter of weight combinations
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='momentum',
    y='volatility',
    size='cumulative_return',
    hue='value',
    palette='coolwarm',
    sizes=(20, 200)
)
plt.title('Monte Carlo Sweep – Factor Weight Optimization')
plt.xlabel('Momentum Weight')
plt.ylabel('Volatility Weight')
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/factor_weight_sweep_plot.png')

print("📊 Scatterplot saved to logs/factor_weight_sweep_plot.png")
