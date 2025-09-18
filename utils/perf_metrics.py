from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PerformanceStats:
    sharpe: float
    calmar: float
    cagr: float
    max_drawdown: float
    psr: float


def _annualization_factor(periods_per_year: float) -> float:
    return math.sqrt(periods_per_year)


def compute_sharpe(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return 0.0
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(mean / std * _annualization_factor(periods_per_year))


def compute_cagr(equity: Sequence[float], periods_per_year: float = 252.0) -> float:
    arr = np.asarray(equity, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2 or arr[0] <= 0:
        return 0.0
    total_return = arr[-1] / arr[0]
    if total_return <= 0:
        return 0.0
    years = arr.size / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def compute_max_drawdown(equity: Sequence[float]) -> float:
    arr = np.asarray(equity, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdowns = arr / peaks - 1.0
    return float(drawdowns.min())


def compute_calmar(equity: Sequence[float], periods_per_year: float = 252.0) -> float:
    cagr = compute_cagr(equity, periods_per_year)
    mdd = compute_max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr / abs(mdd))


def compute_psr(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    sr_ref: float = 0.0,
) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n <= 1:
        return 0.0
    sr_hat = compute_sharpe(arr, periods_per_year)
    if sr_hat <= -10 or sr_hat >= 10:
        sr_hat = max(min(sr_hat, 10.0), -10.0)
    z_num = (sr_hat - sr_ref) * math.sqrt(n - 1)
    denom = math.sqrt(1.0 - min(sr_hat * sr_hat, 0.99))
    if denom <= 0:
        denom = 1.0
    z = z_num / denom
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def compute_performance_stats(
    returns: Sequence[float],
    equity: Sequence[float],
    periods_per_year: float = 252.0,
    sr_ref: float = 0.0,
) -> PerformanceStats:
    sharpe = compute_sharpe(returns, periods_per_year)
    calmar = compute_calmar(equity, periods_per_year)
    cagr = compute_cagr(equity, periods_per_year)
    mdd = compute_max_drawdown(equity)
    psr = compute_psr(returns, periods_per_year, sr_ref)
    return PerformanceStats(sharpe=sharpe, calmar=calmar, cagr=cagr, max_drawdown=mdd, psr=psr)


__all__ = [
    "PerformanceStats",
    "compute_sharpe",
    "compute_cagr",
    "compute_max_drawdown",
    "compute_calmar",
    "compute_psr",
    "compute_performance_stats",
]
