# utils/data_loader.py

import pandas as pd
import os

def load_price(symbol: str, freq: str = '1d', path: str = 'data/processed') -> pd.DataFrame:
    """
    Load OHLCV data from the processed directory.

    Parameters:
    - symbol (str): e.g., 'BTCUSDT'
    - freq (str): e.g., '1d', '1h', '15m'
    - path (str): base folder, default is 'data/processed'

    Returns:
    - pd.DataFrame: indexed by timestamp
    """
    file = os.path.join(path, f'{symbol}_{freq}.csv')
    df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
    return df
