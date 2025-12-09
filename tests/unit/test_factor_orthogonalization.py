from __future__ import annotations

import math

from execution.factor_diagnostics import OrthogonalizedFactorVectors, orthogonalize_factors
from execution.intel.symbol_score_v6 import FactorVector


def _vec(symbol: str, trend: float, carry: float, expectancy: float) -> FactorVector:
    return FactorVector(
        symbol=symbol,
        factors={"trend": trend, "carry": carry, "expectancy": expectancy},
        hybrid_score=0.0,
        direction="LONG",
        regime="normal",
    )


def test_gram_schmidt_produces_orthonormal_basis_with_degenerate_handling() -> None:
    # trend and carry independent, expectancy collinear with carry
    vectors = [
        _vec("BTCUSDT", 1.0, 0.0, 0.0),
        _vec("ETHUSDT", 0.0, 1.0, 1.0),
    ]

    ortho: OrthogonalizedFactorVectors = orthogonalize_factors(vectors, ["trend", "carry", "expectancy"])

    # Non-degenerate columns should be orthonormal
    assert math.isclose(ortho.dot_products["trend"]["trend"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ortho.dot_products["carry"]["carry"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert abs(ortho.dot_products["trend"]["carry"]) < 1e-6

    # Collinear expectancy is marked degenerate and has near-zero norm
    assert "expectancy" in ortho.degenerate
    assert ortho.norms["expectancy"] < 1e-6
    assert all(abs(v) < 1e-6 for v in ortho.dot_products["expectancy"].values())
