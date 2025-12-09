from __future__ import annotations

import pytest

from execution.factor_diagnostics import FactorDiagnosticsConfig, build_factor_diagnostics_snapshot
from execution.intel.symbol_score_v6 import FactorVector

pytestmark = [pytest.mark.integration]


def _vec(symbol: str, trend: float, carry: float, expectancy: float) -> FactorVector:
    return FactorVector(
        symbol=symbol,
        factors={"trend": trend, "carry": carry, "expectancy": expectancy},
        hybrid_score=0.0,
        direction="LONG",
        regime="normal",
    )


def test_factor_diagnostics_surface_shape() -> None:
    cfg = FactorDiagnosticsConfig(
        enabled=True,
        factors=["trend", "carry", "expectancy"],
        normalization_mode="zscore",
        covariance_lookback_days=7,
        pnl_attribution_lookback_days=5,
    )
    snapshot = build_factor_diagnostics_snapshot(
        factor_vectors=[
            _vec("BTCUSDT", 1.0, 0.2, 0.1),
            _vec("ETHUSDT", 0.5, 0.9, -0.2),
        ],
        cfg=cfg,
    ).to_dict()

    for key in [
        "updated_ts",
        "raw_factors",
        "normalization_coeffs",
        "covariance",
        "orthogonalized",
        "factor_ir",
        "weights",
        "pnl_attribution",
    ]:
        assert key in snapshot

    assert isinstance(snapshot["normalization_coeffs"], dict)
    assert isinstance(snapshot.get("covariance", {}).get("covariance_matrix"), list)
