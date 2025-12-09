from __future__ import annotations

from execution.factor_diagnostics import (
    FactorDiagnosticsConfig,
    FactorWeights,
    build_factor_diagnostics_snapshot,
)
from execution.intel.symbol_score_v6 import FactorVector


def _vec(symbol: str, trend: float, carry: float) -> FactorVector:
    return FactorVector(
        symbol=symbol,
        factors={"trend": trend, "carry": carry},
        hybrid_score=0.0,
        direction="LONG",
        regime="normal",
    )


def test_covariance_ir_and_weights_respect_clamps_and_smoothing() -> None:
    cfg = FactorDiagnosticsConfig(
        enabled=True,
        factors=["trend", "carry"],
        normalization_mode="zscore",
        covariance_lookback_days=10,
        pnl_attribution_lookback_days=5,
        max_abs_zscore=3.0,
    )
    strategy_config = {
        "factor_diagnostics": {
            "auto_weighting": {
                "enabled": True,
                "mode": "vol_inverse_ir",
                "min_weight": 0.05,
                "max_weight": 0.6,
                "normalize_to_one": True,
                "smoothing_alpha": 0.5,
            },
            "orthogonalization": {"enabled": False},
        }
    }

    vectors = [_vec("BTCUSDT", 1.0, 0.1), _vec("ETHUSDT", 0.5, 1.0)]
    prev = FactorWeights(weights={"trend": 0.5, "carry": 0.5})
    snapshot = build_factor_diagnostics_snapshot(
        factor_vectors=vectors,
        cfg=cfg,
        factor_pnl={"trend": 0.02, "carry": 0.01},
        prev_weights=prev,
        strategy_config=strategy_config,
    )

    assert snapshot.covariance is not None
    assert snapshot.covariance.lookback_days == 10
    weights = snapshot.weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(0.05 <= w <= 0.6 for w in weights.values())
    assert snapshot.factor_ir
    assert snapshot.pnl_attribution.get("window_days") == 5
