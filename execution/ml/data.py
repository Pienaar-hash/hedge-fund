from __future__ import annotations

import pandas as pd

import os
import numpy as np
from execution.exchange_utils import get_klines
# NOTE: Binance USD-M klines limit is 1500; paginate when requesting deeper history.


def load_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if os.environ.get("ML_SIMULATOR", "0") == "1":
        freq = "H" if interval.endswith("h") or interval.endswith("H") else "H"
        idx = pd.date_range("2024-01-01", periods=limit, freq=freq, tz="UTC")
        drift = np.linspace(0, 0.003 * limit, limit)
        noise = np.random.normal(0, 0.2, size=limit).cumsum()
        base = 100 + drift + noise
        df = pd.DataFrame(
            {
                "open": base,
                "high": base * 1.001,
                "low": base * 0.999,
                "close": base,
                "volume": 1.0,
            },
            index=idx,
        )
        df.index.name = "open_time"
        return df

    rows = get_klines(symbol, interval, limit)
    if not rows:
        raise ValueError(f"no klines returned for {symbol} interval={interval}")
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    df = pd.DataFrame(rows)
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for key in ["open", "high", "low", "close", "volume"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna().set_index("open_time").sort_index()
    return df
