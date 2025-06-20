# === strategies/volatility_target.py ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy

ASSETS = []
VOL_LOOKBACK = 20
TARGET_VOL = 0.01
FEE = 0.001
STARTING_EQUITY = 100000

class StrategyImpl(Strategy):
    def configure(self, params):
        global VOL_LOOKBACK, TARGET_VOL
        VOL_LOOKBACK = params.get("lookback", VOL_LOOKBACK)
        TARGET_VOL = params.get("target_vol", TARGET_VOL)
        global ASSETS
        if "assets" in params:
            ASSETS = [(a["symbol"], a["tf"]) for a in params["assets"]]

    def run(self):
        self.results = []
        for symbol, tf in ASSETS:
            df = self.load_price_data(symbol, tf)
            if df is None:
                continue
            df = self.backtest_vol_target(df)
            self.save_trades(df, symbol)
            self.save_equity_curve(symbol)
            self.save_summary_metrics(symbol)

    def load_price_data(self, symbol, tf):
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

    def backtest_vol_target(self, df):
        df['vol'] = df['log_return'].rolling(VOL_LOOKBACK).std()
        df['leverage'] = TARGET_VOL / df['vol']
        df['leverage'] = df['leverage'].clip(upper=3)
        df['strategy_return'] = df['log_return'] * df['leverage']

        fee = FEE * abs(df['leverage'].diff())
        fee.fillna(0, inplace=True)
        df['strategy_return_net'] = df['strategy_return'] - fee

        df.dropna(subset=['strategy_return_net'], inplace=True)
        df['cum_return'] = (1 + df['strategy_return_net']).cumprod()
        return df

    def save_trades(self, df, symbol):
        trades = df[['strategy_return_net']].copy()
        trades['timestamp'] = df.index
        trades['net_return_pct'] = trades['strategy_return_net'] * 100
        trades['entry_time'] = trades['timestamp']
        trades['exit_time'] = trades['timestamp']
        trades = trades[['entry_time', 'exit_time', 'net_return_pct']]
        trades.dropna(inplace=True)
        out_path = f"logs/vol_target_backtest_trades_{symbol.lower()}.csv"
        trades.to_csv(out_path, index=False)
        print(f"✅ Saved {len(trades)} trades to {out_path}")

    def save_equity_curve(self, symbol):
        path = f"logs/vol_target_backtest_trades_{symbol.lower()}.csv"
        df = pd.read_csv(path, parse_dates=['entry_time'])
        df = df.sort_values('entry_time')
        df['net_return'] = df['net_return_pct'] / 100
        df['equity'] = (1 + df['net_return']).cumprod() * STARTING_EQUITY
        df['timestamp'] = df['entry_time']
        df = df[['timestamp', 'equity']]
        df.dropna(inplace=True)
        out_path = f"logs/equity_curve_vol_target_{symbol.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved equity curve to {out_path}")

    def save_summary_metrics(self, symbol):
        path = f"logs/vol_target_backtest_trades_{symbol.lower()}.csv"
        df = pd.read_csv(path, parse_dates=['entry_time'])
        df.set_index('entry_time', inplace=True)
        returns = df['net_return_pct'] / 100
        cum_returns = (1 + returns).cumprod()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else np.nan
        mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
        cagr = cum_returns.iloc[-1]**(252 / len(returns)) - 1 if len(returns) > 1 else np.nan
        summary = pd.DataFrame([{
            'Symbol': symbol,
            'Lookback': VOL_LOOKBACK,
            'TargetVol': TARGET_VOL,
            'Sharpe': sharpe,
            'MaxDrawdown': mdd,
            'CAGR': cagr
        }])
        out_path = f"logs/volatility_targeting_summary_{symbol.lower()}.csv"
        summary.to_csv(out_path, index=False)
        print(f"✅ Saved summary metrics to {out_path}")

    def log_results(self):
        print("🔁 Volatility Target Strategy Run Complete")
