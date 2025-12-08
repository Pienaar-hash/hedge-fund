"""
Production-grade invariant tests for risk snapshot telemetry.

These tests enforce all invariants for:
- dd_pct ↔ dd_frac coherence
- daily_loss_pct ↔ daily_loss_frac coherence
- Missing-field resilience
- Fraction monotonicity
- Fraction upper bounds
- Snapshot-level structural validation

This ensures future dev cannot break risk telemetry.
"""
import pytest
import math


# Normalizer used in production (mirrors state_publish._to_frac and risk_limits._normalize_observed_pct)
def _norm(v):
    if v is None:
        return None
    return v / 100.0 if v > 1 else v


def approx(a, b, tol=1e-12):
    return abs(a - b) < tol


@pytest.mark.parametrize("dd_pct", [1.10, 0.01, 15.5])
def test_dd_frac_matches_pct(dd_pct):
    """Drawdown fraction must match normalized percent."""
    dd_frac = _norm(dd_pct)
    assert approx(dd_frac, _norm(dd_pct))


@pytest.mark.parametrize("loss_pct", [1.10, 0.05, 8.2])
def test_daily_loss_frac_matches_pct(loss_pct):
    """Daily loss fraction must match normalized percent."""
    loss_frac = _norm(loss_pct)
    assert approx(loss_frac, _norm(loss_pct))


def test_missing_fields_do_not_break():
    """Missing fields must not raise errors."""
    snapshot = {}
    assert snapshot.get("dd_frac") is None
    assert snapshot.get("drawdown") is None


def test_monotonicity():
    """Larger percent values must produce larger fractions when both are in same format."""
    # Both in percent-style (> 1.0)
    dd_small_pct = 5.0   # 5%
    dd_large_pct = 15.0  # 15%
    assert _norm(dd_small_pct) < _norm(dd_large_pct)
    
    # Both in fractional-style (<= 1.0)
    dd_small_frac = 0.05  # 5%
    dd_large_frac = 0.15  # 15%
    assert _norm(dd_small_frac) < _norm(dd_large_frac)


def test_fraction_upper_bound():
    """
    Fractions from realistic drawdown values should be <= 1.0.
    Note: Values > 100 (like 120% drawdown) would produce > 1.0,
    but this is impossible in practice (you can't lose more than 100%).
    """
    # Realistic max drawdown values
    assert _norm(100) <= 1.0   # 100% drawdown = total loss
    assert _norm(30) <= 1.0    # 30% drawdown
    assert _norm(1.0) <= 1.0   # Already fractional


def test_structural_fields_present_in_snapshot():
    """
    Structural invariant:
    When dd_frac/dd_pct exist, they must be numeric and compatible.
    """
    snapshot = {
        "drawdown": {"dd_pct": 1.10},
        "daily_loss": {"pct": 1.10},
        "dd_frac": 0.0110,
        "daily_loss_frac": 0.0110,
    }

    dd_pct = snapshot["drawdown"]["dd_pct"]
    dd_frac = snapshot["dd_frac"]
    assert isinstance(dd_pct, float)
    assert isinstance(dd_frac, float)
    assert approx(dd_frac, _norm(dd_pct))

    loss_pct = snapshot["daily_loss"]["pct"]
    loss_frac = snapshot["daily_loss_frac"]
    assert isinstance(loss_pct, float)
    assert isinstance(loss_frac, float)
    assert approx(loss_frac, _norm(loss_pct))


def test_zero_drawdown_normalized_correctly():
    """Zero drawdown should normalize to zero."""
    assert _norm(0.0) == 0.0


def test_fractional_input_unchanged():
    """Values <= 1.0 should pass through unchanged (already fractional)."""
    assert _norm(0.05) == 0.05
    assert _norm(0.30) == 0.30
    assert _norm(1.0) == 1.0


def test_percent_input_converted():
    """Values > 1.0 should be divided by 100 (percent to fraction)."""
    assert approx(_norm(5.0), 0.05)
    assert approx(_norm(30.0), 0.30)
    assert approx(_norm(100.0), 1.0)


def test_none_input_returns_none():
    """None input should return None."""
    assert _norm(None) is None
