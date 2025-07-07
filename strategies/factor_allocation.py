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
        self.mode = params.get("mode", "static")  # "static" or "rolling"
        self.rolling_window_days = params.get("rolling_window_days", 60)
        self.rebalance_days = params.get("rebalance_days", 30)
        self.factor_map = params.get("factor_map", {
            "momentum": "momentum_ranked",
            "volatility": "vol_target_btcusdt",
            "value": "relative_value_eth_blend"
        })
        print("[DEBUG] factor_allocation configure() received:", params)

    def run(self):
        print("[DEBUG] factor_allocation run() starting...")
        if not os.path.exists(LOG_PATH):
            print(f"❌ File not found: {LOG_PATH}")
            self.df = pd.DataFrame()
            return

        self.df = pd.read_csv(LOG_PATH, parse_dates=['timestamp'])

        if self.df.empty or 'cagr' not in self.df.columns:
            print(f"⚠️ File exists but has missing or invalid data: {LOG_PATH}")
            return

        self.top_cagr = self.df.sort_values(by='cagr', ascending=False).head(10)
        self.top_sharpe = self.df.sort_values(by='sharpe', ascending=False).head(10)

        print("\n🏆 Top 10 by CAGR:")
        print(self.top_cagr.to_string(index=False))

        print("\n📈 Top 10 by Sharpe:")
        print(self.top_sharpe.to_string(index=False))

        self.plot_weights(self.df)

        simulator = PortfolioSimulator()
        simulator.load_equity_curves()

        if self.mode == "rolling":
            print("\n🔁 Running ROLLING rebalancing mode")
            self.run_rolling(simulator)
            self.evaluate_portfolio("logs/portfolio_rolling_equity.csv")
        else:
            print("\n📦 Running STATIC allocation mode")
            best_weights = self.top_sharpe.iloc[0][['momentum', 'volatility', 'value']].to_dict()
            print("\n📦 Plugged top Sharpe weights into simulated portfolio allocation:")
            print(best_weights)

            strategy_weights = {
                self.factor_map[factor]: weight
                for factor, weight in best_weights.items()
                if factor in self.factor_map
            }

            try:
                simulator.simulate_capital_weighted(strategy_weights)
                self.evaluate_portfolio("logs/portfolio_simulated_equity.csv")
            except Exception as e:
                print(f"❌ Portfolio simulation failed: {e}")

    # === Enhanced Portfolio Rebalancer ===
    def run_rolling(self, simulator):
        df = self.df.copy()
        if 'timestamp' not in df.columns:
            print("❌ Missing 'timestamp' column in factor log for rolling weights")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        def weight_lookup(date):
            window_start = date - pd.Timedelta(days=self.rolling_window_days)
            window_df = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= date)]
            if window_df.empty:
                return {}

            # Optional filters for volatility targeting or drawdown limits
            top = window_df.sort_values(by='sharpe', ascending=False).head(1)
            weights = top[['momentum', 'volatility', 'value']].iloc[0].to_dict()

            # Normalize weights (optional)
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items() if v > 0}
            return {
                self.factor_map[k]: v for k, v in weights.items() if k in self.factor_map
            }

        try:
            simulator.simulate_rolling_weights(weight_lookup, self.rebalance_days)
        except Exception as e:
            print(f"❌ Rolling portfolio simulation failed: {e}")

    def evaluate_portfolio(self, path):
        if not os.path.exists(path):
            print(f"⚠️ Portfolio equity file missing: {path}")
            return

        df = pd.read_csv(path, parse_dates=['timestamp'])
        if 'equity' not in df.columns and 'portfolio' in df.columns:
            df['equity'] = df['portfolio']
        df.set_index('timestamp', inplace=True)
        df['return'] = df['equity'].pct_change()
        df.dropna(inplace=True)

        cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0])**(252 / len(df)) - 1
        sharpe = (df['return'].mean() / df['return'].std()) * (252 ** 0.5)
        sortino = (df['return'].mean() / df[df['return'] < 0]['return'].std()) * (252 ** 0.5)
        volatility = df['return'].std() * (252 ** 0.5)
        dd = (df['equity'] / df['equity'].cummax() - 1).min()

        print(f"\n📊 Performance Summary for {os.path.basename(path)}:")
        print(f"  CAGR: {cagr:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Sortino Ratio: {sortino:.2f}")
        print(f"  Volatility: {volatility:.2%}")
        print(f"  Max Drawdown: {dd:.2%}")

        summary = pd.DataFrame([{
            'file': os.path.basename(path),
            'cagr': cagr,
            'sharpe': sharpe,
            'sortino': sortino,
            'volatility': volatility,
            'max_drawdown': dd
        }])
        summary.to_csv("logs/factor_portfolio_summary.csv", mode='a', index=False, header=not os.path.exists("logs/factor_portfolio_summary.csv"))

    def plot_weights(self, df):
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
        plt.savefig(PLOT_PATH)
        plt.close()
        print(f"📊 Scatterplot saved to {PLOT_PATH}")

    def log_results(self):
        if not self.df.empty:
            print("\n✅ Factor allocation strategy completed.")
        else:
            print("⚠️ No results to log for factor allocation strategy.")
