from __future__ import annotations

import numpy as np
import pytest

from execution.factor_diagnostics import compute_factor_covariance, orthogonalize_factors
from execution.intel.symbol_score_v6 import build_factor_vector

pytestmark = pytest.mark.unit


def test_compute_factor_covariance_orthogonal_inputs():
    factors = ["trend", "carry"]
    vecs = [
        build_factor_vector("BTCUSDT", {"trend": 1.0, "carry": 0.0}, hybrid_score=0.5),
        build_factor_vector("ETHUSDT", {"trend": -1.0, "carry": 0.0}, hybrid_score=0.5),
        build_factor_vector("SOLUSDT", {"trend": 0.0, "carry": 1.0}, hybrid_score=0.5),
        build_factor_vector("LTCUSDT", {"trend": 0.0, "carry": -1.0}, hybrid_score=0.5),
    ]
    snapshot = compute_factor_covariance(vecs, factors)
    cov = snapshot.covariance
    assert cov.shape == (2, 2)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-8)
    assert cov[1, 0] == pytest.approx(0.0, abs=1e-8)
    assert snapshot.correlation[0, 0] == pytest.approx(1.0, rel=1e-6)


def test_orthogonalize_factors_makes_columns_uncorrelated():
    factors = ["trend", "carry"]
    vecs = [
        build_factor_vector("BTCUSDT", {"trend": 1.0, "carry": 1.0}, hybrid_score=0.5),
        build_factor_vector("ETHUSDT", {"trend": 2.0, "carry": 2.0}, hybrid_score=0.5),
        build_factor_vector("SOLUSDT", {"trend": -1.0, "carry": -1.0}, hybrid_score=0.5),
    ]

    ortho = orthogonalize_factors(vecs, factors)
    X = np.array([[vals[f] for f in factors] for vals in ortho.per_symbol.values()])
    col0 = X[:, 0]
    col1 = X[:, 1]
    dot = float(np.dot(col0, col1))
    assert abs(dot) < 1e-6
