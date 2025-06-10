import pandas as pd
import numpy as np
from pathlib import Path
import os

# Config
ASSETS = [
    ('BTCUSDT', '1D'),
    ('ETHUSDT', '1D'),
    ('BNBUSDT', '1D'),
    ('ADAUSDT', '1D'),
    ('AVAXUSDT', '1D'),
    ('DOGEUSDT', '1D')
]

VOL_LOOKBACKS = [10, 20, 30]
TARGET_VOLS = [0.005, 0.01, 0.02]
FEE = 0.001
STARTING_EQUITY = 100000

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

summary_records = []

def load_price_data(symbol, tf):
    path = f"data/processed/{symbol}_{tf.lower()}.csv"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}. Skipping...")
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[['close']].copy()
    df.rename(columns={'close': 'price'}, inplace=True)
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df.dropna(inplace=True)
    return df

def run_backtest(df, lookback, target_vol):
    df['vol'] = df['log_return'].rolling(lookback).std()
    df['leverage'] = target_vol / df['vol']
    df['leverage'] = df['leverage'].clip(upper=3)
    df['strategy_return'] = df['log_return'] * df['leverage']
    df['strategy_return_net'] = df['strategy_return'] - FEE * abs(df['leverage'].diff().fillna(0))
    df['cum_return'] = (1 + df['strategy_return_net']).cumprod()
    return df

def compute_metrics(df):
    returns = df['strategy_return_net']
    cum_returns = (1 + returns).cumprod()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else np.nan
    mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
    cagr = cum_returns.iloc[-1]**(252 / len(returns)) - 1 if len(returns) > 1 else np.nan
    return sharpe, mdd, cagr

def run():
    for symbol, tf in ASSETS:
        df = load_price_data(symbol, tf)
        if df is None:
            continue
        for lookback in VOL_LOOKBACKS:
            for target_vol in TARGET_VOLS:
                result_df = run_backtest(df.copy(), lookback, target_vol)
                sharpe, mdd, cagr = compute_metrics(result_df)
                summary_records.append({
                    'Symbol': symbol,
                    'Lookback': lookback,
                    'TargetVol': target_vol,
                    'Sharpe': sharpe,
                    'MaxDrawdown': mdd,
                    'CAGR': cagr
                })
                print(f"✅ {symbol} LB={lookback} TV={target_vol:.3f} | Sharpe={sharpe:.2f} | CAGR={cagr:.2%} | MDD={mdd:.2%}")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv("logs/volatility_optimization_results.csv", index=False)
    print("\n📈 Sweep complete. Results saved to logs/volatility_optimization_results.csv")

if __name__ == "__main__":
    run()
