# === utils/vectorbt_helpers.py ===
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.io as pio

# Force safe Plotly mode
pio.templates.default = None
vbt.settings['plotting']['enabled'] = False

def generate_vbt_signals(df, rsi_threshold=30, ema_fast=20, ema_slow=50):
    rsi = vbt.RSI.run(df['close'], window=14).rsi
    ema_fast_series = vbt.MA.run(df['close'], window=ema_fast).ma
    ema_slow_series = vbt.MA.run(df['close'], window=ema_slow).ma

    entries = (ema_fast_series > ema_slow_series) & (rsi < rsi_threshold)
    exits = entries.vbt.fshift(1)  # exit on next candle

    return entries, exits

def run_vbt_portfolio(df, entries, exits, tp=0.03, sl=0.02):
    pf = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        size=1.0,
        sl_stop=sl,
        tp_stop=tp,
        fees=0.001,
        direction='longonly',
        init_cash=1.0,
        freq='1h'
    )
    return pf

def run_vbt_volatility_target(df, target_vol=0.01, lookback=20, fee=0.001):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vol'] = df['log_return'].rolling(lookback).std()
    df['leverage'] = (target_vol / df['vol']).clip(upper=3)
    df['strategy_return'] = df['log_return'] * df['leverage']
    df['strategy_return_net'] = df['strategy_return'] - fee * abs(df['leverage'].diff().fillna(0))
    df['cum_return'] = (1 + df['strategy_return_net']).cumprod()
    df.dropna(inplace=True)
    return df

def run_vbt_relative_value(spread_series, lookback=30, z_entry=1.5, z_exit=0.1):
    mean = spread_series.rolling(lookback).mean()
    std = spread_series.rolling(lookback).std()
    zscore = (spread_series - mean) / std

    entries = (zscore > z_entry) | (zscore < -z_entry)
    exits = (zscore < z_exit) & (zscore > -z_exit)
    direction = np.where(zscore > z_entry, -1, np.where(zscore < -z_entry, 1, 0))

    return entries, exits, direction, zscore

def save_vbt_logs(pf, strategy_name, symbol):
    trades = pf.trades.records_readable.copy()
    trades['pnl_pct'] = trades['PnL'] / trades['Entry Price']
    trades.rename(columns={
        'Entry Timestamp': 'entry_time',
        'Exit Timestamp': 'exit_time',
        'Entry Price': 'entry_price',
        'Exit Price': 'exit_price'
    }, inplace=True)
    trades.to_csv(f"logs/{strategy_name}_trades_{symbol}_vectorbt.csv", index=False)

    equity = pf.value()
    equity_df = pd.DataFrame({
        'timestamp': equity.index,
        'equity': equity.values
    })
    equity_df.to_csv(f"logs/equity_curve_{strategy_name}_{symbol}_vectorbt.csv", index=False)

def save_vbt_equity(df, strategy_name, symbol):
    equity_df = pd.DataFrame({
        'timestamp': df.index,
        'equity': df['cum_return']
    })
    equity_df.to_csv(f"logs/equity_curve_{strategy_name}_{symbol}_vectorbt.csv", index=False)

def compute_metrics_from_equity(equity):
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    drawdown = (equity / equity.cummax() - 1).min()
    duration_days = (equity.index[-1] - equity.index[0]).days if len(equity.index) > 1 else 1
    cagr = (equity.iloc[-1] / equity.iloc[0])**(365 / duration_days) - 1 if duration_days > 0 else np.nan

    return {
        'Sharpe': sharpe,
        'Max Drawdown': drawdown,
        'CAGR': cagr
    }

def aggregate_portfolio_metrics(equity_dict):
    summary = {}
    for name, equity in equity_dict.items():
        if isinstance(equity, pd.Series):
            equity = equity.to_frame("equity")
        summary[name] = compute_metrics_from_equity(equity['equity'])
    return pd.DataFrame(summary).T.reset_index().rename(columns={'index': 'Strategy'})
