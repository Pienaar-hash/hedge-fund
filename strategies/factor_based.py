# factor_optimize.py
import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime
import argparse


def fetch_data():
    df = pd.read_csv("data/factor_assets.csv", parse_dates=['timestamp'])
    return df

def generate_scores(df, weights):
    momentum_w = weights['momentum']
    volatility_w = weights['volatility']
    value_w = weights['value']

    df['momentum'] = df['price'].pct_change(10)
    df['volatility'] = df['price'].rolling(10).std()
    df['value'] = 1 / df['price']

    df['score'] = (
        momentum_w * df['momentum'].rank(pct=True) +
        volatility_w * (1 - df['volatility'].rank(pct=True)) +
        value_w * df['value'].rank(pct=True)
    )

    return df

def simulate_trades(df, top_n=5):
    df = df.dropna()
    trades = []
    grouped = df.groupby('timestamp')
    for ts, group in grouped:
        top_assets = group.sort_values(by='score', ascending=False).head(top_n)
        for _, row in top_assets.iterrows():
            trades.append({
                'timestamp': ts,
                'strategy': 'factor_based',
                'symbol': row['symbol'],
                'price': row['price'],
                'score': row['score']
            })
    return trades

def log_trades(trades, path='logs/factor_based_trades.csv'):
    df = pd.DataFrame(trades)
    df.to_csv(path, index=False)
    print(f"✅ Logged {len(df)} trades to {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run Monte Carlo sweep for factor weights')
    args = parser.parse_args()

    df = fetch_data()

    if args.sweep:
        results = []
        for i in range(20):
            weights = {
                'momentum': round(random.uniform(0.0, 1.0), 2),
                'volatility': round(random.uniform(0.0, 1.0), 2),
                'value': round(random.uniform(0.0, 1.0), 2),
            }
            total = sum(weights.values())
            for k in weights:
                weights[k] /= total

            scored = generate_scores(df.copy(), weights)
            trades = simulate_trades(scored)
            cumulative_return = sum(t['price'] for t in trades)  # placeholder

            results.append({
                'momentum': weights['momentum'],
                'volatility': weights['volatility'],
                'value': weights['value'],
                'num_trades': len(trades),
                'cumulative_return': cumulative_return
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv('logs/factor_monte_carlo.csv', index=False)

        best = results_df.sort_values(by='cumulative_return', ascending=False).iloc[0]
        best_weights = {
            'momentum': best['momentum'],
            'volatility': best['volatility'],
            'value': best['value'],
            'cumulative_return': best['cumulative_return'],
            'num_trades': int(best['num_trades'])
        }
        with open('logs/factor_best_params.json', 'w') as f:
            json.dump(best_weights, f, indent=2)

        print("\n🏆 Top 10 Weight Combinations:")
        print(results_df.sort_values(by='cumulative_return', ascending=False).head(10).to_string(index=False))
        print("✅ Optimization complete. Results saved to logs/factor_monte_carlo.csv")
        print("🏆 Best parameters saved to logs/factor_best_params.json")

    else:
        try:
            with open('logs/factor_best_params.json', 'r') as f:
                weights = json.load(f)
                print(f"🚀 Running strategy with best weights: {weights}")
        except FileNotFoundError:
            weights = {'momentum': 0.33, 'volatility': 0.33, 'value': 0.34}
            print("⚠️ Best config not found. Using default weights.")

        scored_df = generate_scores(df.copy(), weights)
        trades = simulate_trades(scored_df)
        log_trades(trades)
