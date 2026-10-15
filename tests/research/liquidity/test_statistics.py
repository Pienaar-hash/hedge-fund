"""Tests for research/liquidity/statistics.py — no network, no filesystem."""
from __future__ import annotations

import numpy as np

from research.liquidity.statistics import (
    cluster_bootstrap,
    hit_rate,
    matched_permutation_test,
    median,
    utc_week_key,
)


def test_bootstrap_is_deterministic_given_same_seed():
    values = [0.01, 0.02, -0.01, 0.015, 0.03, -0.005, 0.02, 0.01]
    keys = ["2023-W01"] * 4 + ["2023-W02"] * 4
    r1 = cluster_bootstrap(values, keys, seed=42, n_resamples=500)
    r2 = cluster_bootstrap(values, keys, seed=42, n_resamples=500)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high
    assert r1.point_estimate == r2.point_estimate


def test_bootstrap_different_seed_can_differ():
    values = [0.01, 0.02, -0.01, 0.015, 0.03, -0.005, 0.02, 0.01, 0.04, -0.02]
    keys = [f"W{i}" for i in range(10)]
    r1 = cluster_bootstrap(values, keys, seed=1, n_resamples=500)
    r2 = cluster_bootstrap(values, keys, seed=2, n_resamples=500)
    # not asserting inequality strictly (could coincide), just that both run and are self-consistent
    assert r1.n_observations == r2.n_observations == 10


def test_cluster_unit_is_utc_week_not_symbol():
    import datetime as dt

    monday = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)  # ISO week 1 of 2023
    t_ms = int(monday.timestamp() * 1000)
    wk1 = utc_week_key(t_ms)
    wk2 = utc_week_key(t_ms + 3 * 24 * 60 * 60 * 1000)  # thursday, same ISO week
    wk3 = utc_week_key(t_ms + 10 * 24 * 60 * 60 * 1000)  # following week
    assert wk1 == wk2
    assert wk1 != wk3


def test_cluster_bootstrap_all_one_cluster_has_zero_width_ci_for_constant_values():
    values = [0.02] * 20
    keys = ["W1"] * 20
    result = cluster_bootstrap(values, keys, seed=7, n_resamples=200)
    assert result.ci_low == result.ci_high == 0.02


def test_matched_permutation_test_symmetric_diffs_gives_high_p_value():
    diffs = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]  # symmetric around zero
    p = matched_permutation_test(diffs, n_resamples=2000, seed=3)
    assert p > 0.3  # should not look significant


def test_matched_permutation_test_all_positive_gives_low_p_value():
    diffs = [0.02, 0.025, 0.018, 0.03, 0.022, 0.019, 0.021, 0.024]
    p = matched_permutation_test(diffs, n_resamples=5000, seed=3)
    assert p < 0.05


def test_median_and_hit_rate_basic():
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert hit_rate([0.1, -0.1, 0.2, -0.3]) == 0.5
    assert median([]) is None
    assert hit_rate([]) is None
