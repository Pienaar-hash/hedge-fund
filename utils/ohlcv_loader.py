"""
OHLCV Data Loader for Backtesting (v1.0)

Utility functions to load collected OHLCV data for analysis and backtesting.

Usage:
    from utils.ohlcv_loader import OHLCVLoader
    
    loader = OHLCVLoader()
    df = loader.load("BTCUSDT", "15m", days=7)
    
    # Multi-symbol DataFrame
    dfs = loader.load_multi(["BTCUSDT", "ETHUSDT"], "15m", days=7)
    
    # Get OHLCV matrix (aligned timestamps)
    matrix = loader.get_price_matrix(["BTCUSDT", "ETHUSDT"], "1h", days=30)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("OHLCV_DATA_DIR") or PROJECT_ROOT / "data" / "ohlcv")


class OHLCVLoader:
    """Load and manipulate collected OHLCV data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
    
    def load(
        self,
        symbol: str,
        interval: str,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "15m", "1h")
            start: Start datetime or string
            end: End datetime or string
            days: Number of days back from now (alternative to start/end)
        
        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume,
                                   close_time, quote_volume, trades, datetime
        """
        # Parse dates
        if days is not None:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
        else:
            if isinstance(start, str):
                start = pd.to_datetime(start).to_pydatetime()
            if isinstance(end, str):
                end = pd.to_datetime(end).to_pydatetime()
        
        # Find and load parquet files
        data_path = self.data_dir / symbol / interval
        if not data_path.exists():
            return pd.DataFrame()
        
        files = sorted(data_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        
        # Filter by date range
        filtered_files = []
        for f in files:
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
                if start and file_date < start.date():
                    continue
                if end and file_date > end.date():
                    continue
                filtered_files.append(f)
            except ValueError:
                continue
        
        if not filtered_files:
            return pd.DataFrame()
        
        # Load and concatenate
        dfs = [pd.read_parquet(f) for f in filtered_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["open_time"], keep="last")
        df = df.sort_values("open_time").reset_index(drop=True)
        
        # Add datetime column
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        
        # Filter by exact time range
        if start:
            start_ms = int(start.timestamp() * 1000)
            df = df[df["open_time"] >= start_ms]
        if end:
            end_ms = int(end.timestamp() * 1000)
            df = df[df["open_time"] <= end_ms]
        
        return df.reset_index(drop=True)
    
    def load_multi(
        self,
        symbols: List[str],
        interval: str,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        return {
            symbol: self.load(symbol, interval, **kwargs)
            for symbol in symbols
        }
    
    def get_price_matrix(
        self,
        symbols: List[str],
        interval: str,
        price_col: str = "close",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get aligned price matrix for multiple symbols.
        
        Returns:
            DataFrame with datetime index and symbol columns
        """
        dfs = self.load_multi(symbols, interval, **kwargs)
        
        price_dfs = []
        for symbol, df in dfs.items():
            if df.empty:
                continue
            price_df = df.set_index("datetime")[[price_col]].rename(columns={price_col: symbol})
            price_dfs.append(price_df)
        
        if not price_dfs:
            return pd.DataFrame()
        
        # Outer join all price series
        result = price_dfs[0]
        for df in price_dfs[1:]:
            result = result.join(df, how="outer")
        
        return result.sort_index()
    
    def get_returns_matrix(
        self,
        symbols: List[str],
        interval: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Get returns matrix (percentage change)."""
        prices = self.get_price_matrix(symbols, interval, **kwargs)
        return prices.pct_change().dropna()
    
    def get_volatility(
        self,
        symbol: str,
        interval: str,
        window: int = 20,
        annualize: bool = True,
        **kwargs,
    ) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            symbol: Trading symbol
            interval: Candlestick interval
            window: Rolling window size
            annualize: Whether to annualize (assumes 24/7 trading)
        """
        df = self.load(symbol, interval, **kwargs)
        if df.empty:
            return pd.Series()
        
        returns = df.set_index("datetime")["close"].pct_change()
        vol = returns.rolling(window).std()
        
        if annualize:
            # Annualization factor based on interval
            intervals_per_year = {
                "1m": 365 * 24 * 60,
                "5m": 365 * 24 * 12,
                "15m": 365 * 24 * 4,
                "30m": 365 * 24 * 2,
                "1h": 365 * 24,
                "4h": 365 * 6,
                "1d": 365,
            }
            factor = np.sqrt(intervals_per_year.get(interval, 365 * 24 * 4))
            vol = vol * factor
        
        return vol
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available data."""
        if not self.data_dir.exists():
            return []
        return [d.name for d in self.data_dir.iterdir() 
                if d.is_dir() and not d.name.startswith(".")]
    
    def get_available_intervals(self, symbol: str) -> List[str]:
        """Get list of intervals with available data for a symbol."""
        data_path = self.data_dir / symbol
        if not data_path.exists():
            return []
        return [d.name for d in data_path.iterdir() if d.is_dir()]
    
    def get_date_range(self, symbol: str, interval: str) -> tuple:
        """Get (min_date, max_date) for a symbol/interval."""
        data_path = self.data_dir / symbol / interval
        if not data_path.exists():
            return (None, None)
        
        files = sorted(data_path.glob("*.parquet"))
        if not files:
            return (None, None)
        
        min_date = datetime.strptime(files[0].stem, "%Y-%m-%d").date()
        max_date = datetime.strptime(files[-1].stem, "%Y-%m-%d").date()
        return (min_date, max_date)
    
    def resample(
        self,
        df: pd.DataFrame,
        target_interval: str,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a higher timeframe.
        
        Args:
            df: Source DataFrame with OHLCV data
            target_interval: Target interval (e.g., "1h" from "15m")
        
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Map intervals to pandas resample strings
        resample_map = {
            "1m": "1min", "3m": "3min", "5m": "5min",
            "15m": "15min", "30m": "30min",
            "1h": "1h", "2h": "2h", "4h": "4h",
            "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
        }
        
        rule = resample_map.get(target_interval)
        if not rule:
            raise ValueError(f"Unknown interval: {target_interval}")
        
        df = df.copy()
        df = df.set_index("datetime")
        
        resampled = df.resample(rule).agg({
            "open_time": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
            "quote_volume": "sum",
            "trades": "sum",
        }).dropna()
        
        resampled = resampled.reset_index()
        return resampled


# --- Technical Indicators ---

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to OHLCV DataFrame.
    
    Adds:
        - SMA_20, SMA_50, SMA_200
        - EMA_12, EMA_26
        - RSI_14
        - MACD, MACD_signal, MACD_hist
        - BB_upper, BB_lower, BB_middle (Bollinger Bands)
        - ATR_14
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # SMAs
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()
    
    # EMAs
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger Bands
    df["BB_middle"] = df["SMA_20"]
    std = close.rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + (2 * std)
    df["BB_lower"] = df["BB_middle"] - (2 * std)
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    
    return df


# Convenience function
def load_ohlcv(symbol: str, interval: str = "15m", days: int = 30) -> pd.DataFrame:
    """Quick loader function."""
    return OHLCVLoader().load(symbol, interval, days=days)
