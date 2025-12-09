from __future__ import annotations

import math
import pytest

from execution.utils import vol

pytestmark = pytest.mark.unit


def test_compute_log_returns_basic():
    prices = [100, 110, 121]
    returns = vol.compute_log_returns(prices)
    assert len(returns) == 2
    assert pytest.approx(returns[0], rel=1e-6) == math.log(1.1)


def test_compute_ewma_vol_constant_series_zero():
    returns = [0.0] * 10
    result = vol.compute_ewma_vol(returns, halflife_bars=3)
    assert result == 0.0


def test_compute_ewma_vol_weights_recent_more():
    # Small set with clear variance; ensure positive vol
    returns = [0.0, 0.01, 0.02, -0.01]
    vol_short = vol.compute_ewma_vol(returns, halflife_bars=2)
    vol_long = vol.compute_ewma_vol(returns, halflife_bars=10)
    assert vol_short > 0
    assert vol_long > 0
    # shorter halflife -> more weight on latest => typically differs from longer window
    assert vol_short != vol_long
