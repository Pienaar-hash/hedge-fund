from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "compute_rsi",
    "compute_zscore",
    "compute_volatility",
    "prepare_factor_frame",
]


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder smoothing."""
    diff = prices.diff().fillna(0.0)
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=5).mean()
    rolling_std = series.rolling(window=window, min_periods=5).std(ddof=0)
    z = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return z.fillna(0.0)


def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    realized = returns.rolling(window=window, min_periods=5).std(ddof=0)
    return realized.ffill().fillna(returns.std(ddof=0))


def prepare_factor_frame(
    prices: pd.Series,
    *,
    ml_signal: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Convenience helper assembling a blended feature frame."""
    prices = prices.astype(float)
    returns = prices.pct_change().fillna(0.0)

    frame = pd.DataFrame(
        {
            "ta_rsi": compute_rsi(prices, window=max(5, window // 2)),
            "ta_momentum": compute_zscore(prices.pct_change(), window=window),
            "risk_volatility": compute_volatility(returns, window=window),
        },
        index=prices.index,
    )
    if ml_signal is not None:
        frame["ml_signal"] = ml_signal.reindex(prices.index).ffill().bfill().fillna(0.0)
    if volume is not None:
        frame["vol_zscore"] = compute_zscore(volume.astype(float), window=max(5, window // 2))
    return frame.fillna(0.0)
