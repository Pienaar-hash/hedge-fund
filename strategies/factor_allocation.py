# === strategies/factor_allocation.py ===
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.strategy_base import Strategy
from core.portfolio_simulator import PortfolioSimulator

LOG_PATH = 'logs/factor_monte_carlo.csv'
PLOT_PATH = 'logs/factor_weight_sweep_plot.png'

class StrategyImpl(Strategy):
    def configure(self, params):
        global LOG_PATH, PLOT_PATH
        LOG_PATH = params.get("log_path", LOG_PATH)
        PLOT_PATH = params.get("plot_path", PLOT_PATH)
        print("[DEBUG] factor_allocation configure() received:", params)

    def run(self):
        print("[DEBUG] factor_allocation run() starting...")
        if not os.path.exists(LOG_PATH):
            print(f"❌ File not found: {LOG_PATH}")
            self.df = pd.DataFrame()
            return

        self.df = pd.read_csv(LOG_PATH)
        if self.df.empty or 'cagr' not in self.df.columns:
            print(f"⚠️ File exists but has missing or invalid data: {LOG_PATH}")
            return

        self.top_cagr = self.df.sort_values(by='cagr', ascending=False).head(10)
        self.top_sharpe = self.df.sort_values(by='sharpe', ascending=False).head(10)

        print("\n🏆 Top 10 by CAGR:")
        print(self.top_cagr.to_string(index=False))

        print("\n📈 Top 10 by Sharpe:")
        print(self.top_sharpe.to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df,
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
        plt.savefig(PLOT_PATH)
        plt.close()

        best_weights = self.top_sharpe.iloc[0][['momentum', 'volatility', 'value']].to_dict()
        print("\n📦 Plugged top Sharpe weights into simulated portfolio allocation:")
        print(best_weights)

        factor_to_strategy = {
            "momentum": "momentum_ranked",
            "volatility": "vol_target_btcusdt",
            "value": "relative_value"
        }

        strategy_weights = {factor_to_strategy[k]: v for k, v in best_weights.items() if k in factor_to_strategy}
        simulator = PortfolioSimulator(list(strategy_weights.keys()))

        # Deduplicate timestamps before simulating
        for name in simulator.strategies:
            if name in simulator.equity_curves:
                df = simulator.equity_curves[name]
                simulator.equity_curves[name] = df[~df.index.duplicated(keep='first')]

        try:
            simulator.simulate_capital_weighted(strategy_weights)
        except Exception as e:
            print(f"❌ Portfolio simulation failed: {e}")

    def log_results(self):
        if not self.df.empty:
            print(f"📊 Scatterplot saved to {PLOT_PATH}")
        else:
            print("⚠️ No results to log for factor allocation strategy.")
