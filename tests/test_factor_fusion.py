from __future__ import annotations

import numpy as np
import pandas as pd

from research.factor_fusion import FactorFusion, FactorFusionConfig, prepare_factor_frame


def test_factor_fusion_positive_signal() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=220, freq="D")
    price_shocks = rng.normal(0.05, 1.0, len(dates))
    prices = pd.Series(100 + np.cumsum(price_shocks), index=dates)
    ml_signal = pd.Series(np.sin(np.linspace(0, 6, len(dates))) + rng.normal(0, 0.05, len(dates)), index=dates)
    volume = pd.Series(1_000_000 + rng.normal(0, 5_000, len(dates)), index=dates)

    factors = prepare_factor_frame(prices, ml_signal=ml_signal, volume=volume, window=20)
    factors = factors.ffill().fillna(0.0)

    future_returns = prices.pct_change().shift(-1).fillna(0.0)
    target = (
        future_returns * 0.6
        + ml_signal.shift(-1).fillna(0.0) * 0.4
        + 0.0005
    ).loc[factors.index]

    fusion = FactorFusion(FactorFusionConfig(regularization=5e-3, positive_weights=False, clip_output=4.0))
    result = fusion.fit(factors.iloc[30:], target.iloc[30:])

    assert result.ic > 0.3
    assert abs(result.signal_sharpe) < 1e-6
    assert abs(float(result.weights.sum()) - 1.0) < 1e-9

    transformed = fusion.transform(factors.iloc[40:])
    assert not transformed.empty
    assert transformed.name == "fused_alpha"
