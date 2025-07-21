# === core/portfolio_simulator.py ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INITIAL_CAPITAL = 100000  # consistent normalization baseline aligned with strategies

class PortfolioSimulator:
    def __init__(self):
        self.equity_curves = {}
        self.load_equity_curves()

    def load_equity_curves(self):
        self.equity_curves = {}
        logs_dir = "logs"

        for fname in os.listdir(logs_dir):
            if fname.startswith("portfolio_simulated_equity_") and fname.endswith(".csv") and "summary" not in fname:
                strategy_name = fname.replace("portfolio_simulated_equity_", "").replace(".csv", "")
                path = os.path.join(logs_dir, fname)

                try:
                    df = pd.read_csv(path, parse_dates=["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    df = df.resample("1D").last().ffill()
                    df = df / df.iloc[0] * INITIAL_CAPITAL
                    df.rename(columns={"equity": strategy_name}, inplace=True)
                    self.equity_curves[strategy_name] = df[[strategy_name]]
                    print(f"✅ Loaded and aligned equity for {strategy_name}")
                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")

        return self.equity_curves

    def simulate_blended_portfolio(self, periods_per_year=365, log_scale=False):
        # Load BTC Hodl Benchmark
        try:
            btc_df = pd.read_csv("data/processed/btcusdt_1d.csv", parse_dates=["timestamp"])
            btc_df = btc_df.set_index("timestamp").resample("1D").last().ffill()
            btc_df = btc_df[btc_df["close"].notna()].copy()
            btc_df["equity"] = btc_df["close"] / btc_df["close"].iloc[0] * INITIAL_CAPITAL
            btc_df.reset_index()[["timestamp", "equity"]].to_csv("logs/benchmark_btc_hodl.csv", index=False)
            print("📈 BTC benchmark saved to logs/benchmark_btc_hodl.csv")
        except Exception as e:
            print(f"⚠️ Could not generate BTC benchmark: {e}")


        all_equity = pd.concat(self.equity_curves.values(), axis=1).ffill().dropna()
        returns = all_equity.pct_change().fillna(0)
        blended_ret = returns.mean(axis=1)
        blended_equity = (1 + blended_ret).cumprod() * INITIAL_CAPITAL

        blended_df = pd.DataFrame({"timestamp": blended_equity.index, "equity": blended_equity.values})
        blended_df.to_csv("logs/portfolio_simulated_equity_blended.csv", index=False)

        summary = []
        for strat, equity_curve in self.equity_curves.items():
            r = equity_curve.pct_change().dropna().squeeze()
            e = equity_curve.iloc[:, 0]
            summary.append({
                "strategy": strat,
                "CAGR": (e.iloc[-1] / e.iloc[0])**(periods_per_year / len(e)) - 1,
                "Sharpe": (r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() != 0 else 0,
                "MaxDrawdown": (e / e.cummax() - 1).min(),
                "Volatility": r.std() * np.sqrt(periods_per_year)
            })

        blended_summary = {
            "strategy": "blended",
            "CAGR": (blended_equity.iloc[-1] / blended_equity.iloc[0])**(periods_per_year / len(blended_equity)) - 1,
            "Sharpe": (blended_ret.mean() / blended_ret.std() * np.sqrt(periods_per_year)) if blended_ret.std() != 0 else 0,
            "MaxDrawdown": (blended_equity / blended_equity.cummax() - 1).min(),
            "Volatility": blended_ret.std() * np.sqrt(periods_per_year)
        }
        summary.append(blended_summary)

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv("logs/summary_portfolio_combined.csv", index=False)

        plt.figure(figsize=(10, 6))
        for col in all_equity.columns:
            plt.plot(all_equity.index, np.log(all_equity[col]) if log_scale else all_equity[col], label=col)
        plt.plot(blended_equity.index, np.log(blended_equity) if log_scale else blended_equity, label="Blended", linewidth=2, linestyle="--")
        plt.legend()
        plt.grid(True)
        plt.title("Portfolio Equity Curves" + (" (Log Scale)" if log_scale else ""))
        plt.xlabel("Date")
        plt.ylabel("Log Equity" if log_scale else "Equity")
        plt.tight_layout()
        plt.savefig("logs/portfolio_equity_plot.png")
        plt.close()
        print("📊 Blended equity curves saved to logs/portfolio_equity_plot.png")

if __name__ == "__main__":
    print("🔁 Running portfolio simulation...")
    sim = PortfolioSimulator()
    sim.simulate_blended_portfolio(log_scale=True)
    print("✅ Finished portfolio simulation.")
