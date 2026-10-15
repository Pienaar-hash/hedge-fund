"""Pre-registered inference for H-LIQ-001: UTC-week cluster bootstrap, matched-pair
permutation test, and the primary/secondary statistics table.

Cluster unit is the UTC ISO calendar week, deliberately retaining simultaneous
BTC/ETH/SOL macro shocks together rather than treating symbol-events as
independent draws.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Hashable, Optional, Sequence

import numpy as np


def utc_week_key(t_ms: int) -> tuple[int, int]:
    d = dt.datetime.fromtimestamp(t_ms / 1000, tz=dt.timezone.utc)
    iso = d.isocalendar()
    return (iso[0], iso[1])


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    seed: int
    n_observations: int
    n_clusters: int


def cluster_bootstrap(
    values: Sequence[float],
    cluster_keys: Sequence[Hashable],
    statistic_fn: Callable[[np.ndarray], float] = lambda x: float(np.median(x)),
    n_resamples: int = 10_000,
    seed: int = 20260903,
) -> Optional[BootstrapResult]:
    if len(values) == 0:
        return None
    values_arr = np.asarray(values, dtype=np.float64)
    keys = list(cluster_keys)
    unique_clusters = sorted(set(keys))
    n_clusters = len(unique_clusters)
    cluster_to_indices: dict[Hashable, list[int]] = {c: [] for c in unique_clusters}
    for i, k in enumerate(keys):
        cluster_to_indices[k].append(i)

    point_estimate = statistic_fn(values_arr)
    if n_clusters == 0:
        return BootstrapResult(point_estimate, point_estimate, point_estimate, n_resamples, seed, len(values_arr), 0)

    rng = np.random.default_rng(seed)
    cluster_list = unique_clusters
    stats = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idxs: list[int] = []
        for ci in chosen:
            idxs.extend(cluster_to_indices[cluster_list[ci]])
        stats[r] = statistic_fn(values_arr[idxs]) if idxs else float("nan")

    valid = stats[~np.isnan(stats)]
    if valid.shape[0] == 0:
        return BootstrapResult(point_estimate, float("nan"), float("nan"), n_resamples, seed, len(values_arr), n_clusters)
    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    return BootstrapResult(point_estimate, ci_low, ci_high, n_resamples, seed, len(values_arr), n_clusters)


def matched_permutation_test(
    diffs: Sequence[float], n_resamples: int = 10_000, seed: int = 20260903
) -> Optional[float]:
    """Two-sided paired sign-flip permutation test on matched differences.
    Returns the p-value of observing |mean(diffs)| this large or larger under
    the null that each sign is equally likely, via deterministic Monte Carlo."""
    if len(diffs) == 0:
        return None
    d = np.asarray(diffs, dtype=np.float64)
    observed = float(np.mean(d))
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    count_ge = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=n)
        resampled_mean = float(np.mean(d * signs))
        if abs(resampled_mean) >= abs(observed):
            count_ge += 1
    return count_ge / n_resamples


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def hit_rate(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr > 0))


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    if not values:
        return {str(q): None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.percentile(arr, q * 100)) for q in qs}
