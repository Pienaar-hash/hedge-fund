import pandas as pd
from datetime import datetime
import os

def fetch_data():
    eth = pd.read_csv("data/eth_usdt.csv", parse_dates=['timestamp'])
    btc = pd.read_csv("data/btc_usdt.csv", parse_dates=['timestamp'])
    eth['timestamp'] = eth['timestamp'].dt.round('h')
    btc['timestamp'] = btc['timestamp'].dt.round('h')
    df = pd.merge(eth, btc, on='timestamp', suffixes=('_eth', '_btc'))
    df['eth_btc'] = df['price_eth'] / df['price_btc']
    return df

def generate_signals(df, z_entry=1.0, z_exit=0.2):
    df['zscore'] = (df['eth_btc'] - df['eth_btc'].rolling(30).mean()) / df['eth_btc'].rolling(30).std()
    df = df.dropna(subset=['zscore'])
    position = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        z = row['zscore']

        if position is None:
            if z > z_entry:
                position = ('short_eth', row['timestamp'], row['eth_btc'])
            elif z < -z_entry:
                position = ('long_eth', row['timestamp'], row['eth_btc'])
        else:
            if abs(z) < z_exit:
                _, entry_time, entry_price = position
                exit_price = row['eth_btc']
                gross_return = (entry_price - exit_price) if position[0] == 'short_eth' else (exit_price - entry_price)
                net_return = gross_return - 0.002  # estimated slippage and fees
                trades.append({
                    'timestamp': row['timestamp'],
                    'strategy': 'relative_value',
                    'symbol': 'ETH/BTC',
                    'side': position[0],
                    'price': 1 + net_return,
                    'z_score': z
                })
                position = None
    return trades
