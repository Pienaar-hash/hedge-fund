# === core/portfolio_simulator.py ===
import os
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 10000   # consistent normalization baseline

class PortfolioSimulator:
    def __init__(self):
        self.equity_curves = {}
        self.strategies = {}
        self.load_equity_curves()

    def load_equity_curves(self):
        self.equity_curves = {}
        logs_dir = "logs"

        for fname in os.listdir(logs_dir):
            if fname.startswith("equity_curve_") and fname.endswith(".csv"):
                strategy_name = fname.replace("equity_curve_", "").replace(".csv", "")
                path = os.path.join(logs_dir, fname)
                df = pd.read_csv(path, parse_dates=True)

                if 'equity' in df.columns:
                    df = df.rename(columns={'equity': strategy_name})
                else:
                    equity_col = df.columns[1]
                    df = df.rename(columns={equity_col: strategy_name})

                df = df.drop_duplicates(subset=df.columns[0])
                df.set_index(df.columns[0], inplace=True)
                df = df.sort_index()
                self.equity_curves[strategy_name] = df[[strategy_name]]

        return self.equity_curves

    def simulate_independent(self, periods_per_year=365):
        all_summaries = []
        aligned = pd.concat(self.equity_curves.values(), axis=1).ffill().dropna()

        for strat, df in self.equity_curves.items():
            df = df.copy()
            df = df.reindex(aligned.index).ffill()
            df = df / df.iloc[0] * INITIAL_CAPITAL
            returns = df.pct_change().fillna(0)

            equity = INITIAL_CAPITAL * (1 + returns).cumprod()
            out = equity.copy()
            out.columns = ["equity"]
            out.to_csv(f"logs/portfolio_simulated_equity_{strat}.csv")

            r = returns[strat]
            e = equity[strat]
            cagr = (e.iloc[-1] / e.iloc[0])**(periods_per_year / len(e)) - 1
            sharpe = r.mean() / r.std() * (periods_per_year ** 0.5) if r.std() > 0 else 0
            dd = (e / e.cummax() - 1).min()
            vol = r.std() * (periods_per_year ** 0.5)

            all_summaries.append({
                "strategy": strat,
                "CAGR": cagr,
                "Sharpe": sharpe,
                "MaxDrawdown": dd,
                "Volatility": vol
            })

        # Save blended portfolio curve
        blended = aligned.mean(axis=1)
        blended_df = pd.DataFrame({"timestamp": blended.index, "equity": blended.values})
        blended_df.to_csv("logs/portfolio_simulated_equity.csv", index=False)

        # Compute blended portfolio metrics
        blended_ret = blended.pct_change().fillna(0)
        cagr = (blended.iloc[-1] / blended.iloc[0])**(periods_per_year / len(blended)) - 1
        sharpe = blended_ret.mean() / blended_ret.std() * (periods_per_year ** 0.5) if blended_ret.std() > 0 else 0
        dd = (blended / blended.cummax() - 1).min()
        vol = blended_ret.std() * (periods_per_year ** 0.5)

        all_summaries.append({
            "strategy": "portfolio_simulated_equity",
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDrawdown": dd,
            "Volatility": vol
        })

        summary = pd.DataFrame(all_summaries)
        summary.to_csv("logs/portfolio_simulated_summary.csv", index=False)
        print("📊 Saved individual strategy summaries to logs/portfolio_simulated_summary.csv")

    def simulate_capital_weighted(self, weights, normalize=True, save_path=None):
        dfs = []
        for name, weight in weights.items():
            if name not in self.equity_curves:
                raise ValueError(f"Missing equity curve: {name}")
            df = self.equity_curves[name].copy()
            if normalize:
                df = df / df.iloc[0] * INITIAL_CAPITAL
            df *= weight
            dfs.append(df)

        combined = pd.concat(dfs, axis=1).fillna(method="ffill").dropna()
        combined["equity"] = combined.sum(axis=1)
        combined = combined[["equity"]]

        if save_path:
            combined.reset_index().to_csv(save_path, index=False)

        return combined.reset_index()

if __name__ == "__main__":
    print("🔁 Running portfolio simulation...")
    sim = PortfolioSimulator()
    sim.simulate_independent()
    print("✅ Finished portfolio simulation.")
