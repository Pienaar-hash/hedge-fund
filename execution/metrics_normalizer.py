from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "NormalizedMetrics",
    "compute_normalized_metrics",
    "rolling_sharpe",
    "confidence_weighted_cumsum",
    "extract_returns",
]

EPSILON = 1e-9
DEFAULT_ANNUALIZATION = 252  # daily samples by default


def _to_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def extract_returns(records: Iterable[Mapping[str, object]], *, key: str = "pnl_pct") -> List[float]:
    """Extract return-like values from a stream of dictionaries.

    The parser looks for `key` first (defaults to ``pnl_pct``) and falls back to a
    handful of common hedging metrics if absent. Values are coerced to floats and
    filtered for finiteness. The return list preserves the input order.
    """
    returns: List[float] = []
    fallback_keys = ("pnl_pct", "return_pct", "pnl", "pnl_usd", "pnl_at_close_usd")
    for record in records:
        if not isinstance(record, Mapping):
            continue
        value = record.get(key)
        if value is None:
            for fb_key in fallback_keys:
                if fb_key == key:
                    continue
                if fb_key in record:
                    value = record.get(fb_key)
                    break
        numeric = _to_float(value)
        if numeric is None:
            continue
        returns.append(float(numeric))
    return returns


def confidence_weighted_cumsum(values: Sequence[float], confidences: Sequence[Optional[float]], *, default_conf: float = 0.5) -> np.ndarray:
    """Return the cumulative sum of values weighted by doctor/model confidence.

    Unknown confidence values fall back to ``default_conf``.
    """
    if len(values) != len(confidences):
        raise ValueError("values and confidences must be the same length")
    adjusted = []
    for value, confidence in zip(values, confidences):
        conf = _to_float(confidence)
        if conf is None:
            conf = default_conf
        conf = max(0.0, min(1.0, conf))
        adjusted.append(float(value) * conf)
    return np.cumsum(np.asarray(adjusted, dtype=float))


@dataclass(slots=True)
class NormalizedMetrics:
    raw_sharpe: float
    normalized_sharpe: float
    volatility_scale: float
    realized_vol: float
    mean_return: float
    sample_size: int
    window: int


def compute_normalized_metrics(
    returns: Sequence[float],
    *,
    target_vol: float = 0.02,
    annualization: int = DEFAULT_ANNUALIZATION,
    min_observations: int = 5,
    window: Optional[int] = None,
) -> NormalizedMetrics:
    """Compute raw/normalized Sharpe ratios and the recommended volatility scale.

    ``returns`` should be expressed as fractional returns (e.g. 0.01 == 1%). When
    raw PnL values are supplied, the caller should normalise them prior to calling
    (for instance dividing by notional or capital at risk).
    """
    if isinstance(returns, pd.Series):
        series = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        clean = series.to_numpy(dtype=float)
    else:
        clean = np.asarray([_to_float(val) for val in returns], dtype=float)
        clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return NormalizedMetrics(0.0, 0.0, 1.0, 0.0, 0.0, 0, window or 0)

    series = clean[-window:] if window and window > 0 else clean
    sample_size = int(series.size)
    if sample_size < min_observations:
        vol = float(np.std(series, ddof=1)) if sample_size > 1 else 0.0
        mean = float(np.mean(series)) if sample_size else 0.0
        volatility_scale = 1.0 if vol <= EPSILON else max(0.1, min(3.0, target_vol / vol)) if target_vol > EPSILON else 1.0
        return NormalizedMetrics(0.0, 0.0, volatility_scale, vol, mean, sample_size, window or sample_size)

    mean = float(np.mean(series))
    vol = float(np.std(series, ddof=1))
    if vol <= EPSILON:
        volatility_scale = 1.0
        return NormalizedMetrics(0.0, 0.0, volatility_scale, 0.0, mean, sample_size, window or sample_size)

    raw_sharpe = (mean / vol) * math.sqrt(float(annualization))

    if target_vol <= EPSILON:
        volatility_scale = 1.0
        normalized_sharpe = raw_sharpe
    else:
        volatility_scale = max(0.1, min(3.0, target_vol / vol))
        normalized_sharpe = raw_sharpe * (vol / target_vol)

    return NormalizedMetrics(
        raw_sharpe=raw_sharpe,
        normalized_sharpe=normalized_sharpe,
        volatility_scale=volatility_scale,
        realized_vol=vol,
        mean_return=mean,
        sample_size=sample_size,
        window=window or sample_size,
    )


def rolling_sharpe(
    pnl: Sequence[float],
    *,
    window: int = 20,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> pd.Series:
    """Return a rolling Sharpe series computed over ``window`` samples."""
    if window <= 1:
        raise ValueError("window must be at least 2 to compute Sharpe")

    if isinstance(pnl, pd.Series):
        series = pnl.copy()
    else:
        series = pd.Series(list(pnl), dtype=float)
    series = series.replace([np.inf, -np.inf], np.nan)

    def _sharpe(sub: pd.Series) -> float:
        clean = sub.dropna()
        if clean.size < 2:
            return 0.0
        mean = float(clean.mean())
        vol = float(clean.std(ddof=1))
        if vol <= EPSILON:
            return 0.0
        return (mean / vol) * math.sqrt(float(annualization))

    return series.rolling(window=window, min_periods=2).apply(_sharpe, raw=False)
