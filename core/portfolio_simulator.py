# === core/portfolio_simulator.py ===
import pandas as pd
import os
import numpy as np

LOG_DIR = "logs"

class PortfolioSimulator:
    def __init__(self, strategies, capital=100000):
        self.strategies = strategies
        self.capital = capital
        self.equity_curves = {}

    def load_equity(self, name):
        path = os.path.join(LOG_DIR, f"equity_curve_{name}.csv")
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'])
            if df.empty or 'equity' not in df.columns:
                raise ValueError("Missing or invalid equity data")
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df = df.interpolate(method='linear').ffill().bfill()
            df['equity'] = df['equity'].clip(lower=1e-8)
            df['equity'] = df['equity'] / df['equity'].iloc[0] * self.capital
            df['strategy'] = name
            self.equity_curves[name] = df[['equity']]
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")

    def simulate_capital_weighted(self, weights):
        for strat in self.strategies:
            self.load_equity(strat)

        available = list(self.equity_curves.keys())
        if not available:
            print("❌ No valid equity curves loaded. Cannot simulate portfolio.")
            return pd.DataFrame()

        combined = pd.concat([self.equity_curves[s].rename(columns={'equity': s}) for s in available], axis=1)

        # Print overlap diagnostics
        print("\n🔍 Timestamp overlaps:")
        for strat in available:
            print(f"{strat}: {self.equity_curves[strat].index.min()} → {self.equity_curves[strat].index.max()}")

        # Fill forward and backward to handle partial overlap
        combined = combined.sort_index().ffill().bfill()

        # Normalize each strategy equity curve
        combined = combined.apply(lambda x: x / x.iloc[0] if x.iloc[0] != 0 else x)

        weight_vec = pd.Series({k: weights[k] for k in available})
        weighted = combined.multiply(weight_vec, axis=1)
        combined['portfolio'] = weighted.sum(axis=1)

        print("\n📉 Portfolio Preview:")
        print(combined.describe())
        print(combined.tail(10))

        combined.dropna(subset=['portfolio'], inplace=True)

        equity_path = os.path.join(LOG_DIR, "equity_curve_factor_blended.csv")
        combined.reset_index()[['timestamp', 'portfolio']].rename(columns={'portfolio': 'equity'}).to_csv(equity_path, index=False)

        metrics = self.compute_metrics(combined[['portfolio']].rename(columns={"portfolio": "equity"}))
        metrics_path = os.path.join(LOG_DIR, "factor_blended_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        print("📊 Portfolio Metrics:", metrics)
        return combined

    def compute_metrics(self, portfolio_df):
        portfolio_df = portfolio_df.dropna()
        portfolio_returns = portfolio_df['equity'].pct_change().dropna()

        print("🔎 Sample returns:", portfolio_returns.head())
        print("🔎 Return stats — Mean:", portfolio_returns.mean(), "Std:", portfolio_returns.std(), "N:", len(portfolio_returns))

        cumulative_return = portfolio_df['equity'].iloc[-1] / portfolio_df['equity'].iloc[0] - 1
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0

        valid_equity = portfolio_df['equity'].replace(0, np.nan).dropna()
        peak = valid_equity.cummax()
        drawdown = (valid_equity - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        n_years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365 if len(portfolio_df) > 1 else 1
        cagr = (portfolio_df['equity'].iloc[-1] / portfolio_df['equity'].iloc[0])**(1 / n_years) - 1 if n_years > 0 else 0

        return {
            'Cumulative Return': cumulative_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'CAGR': cagr
        }
