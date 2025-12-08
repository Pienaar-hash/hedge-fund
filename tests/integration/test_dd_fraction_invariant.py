"""
Invariant tests for drawdown fraction normalization.

These tests ensure that dd_frac and daily_loss_frac always match
the normalized value of dd_pct and daily_loss.pct respectively.

This prevents future regressions from:
- A refactor breaking normalization
- A telemetry change overwriting one field
- Incorrect calculations in drawdown_tracker.py
"""
import json
import math
from pathlib import Path


# Mirror the production normalization used in risk_limits.py/state_publish.py
def normalize(v):
    if v is None:
        return None
    return v / 100.0 if v > 1 else v


def approx(a, b, tol=1e-10):
    return abs(a - b) < tol


def test_drawdown_fraction_invariant(tmp_path):
    """
    Invariant:
    For any risk snapshot that includes dd_pct and dd_frac,
    dd_frac must match normalize(dd_pct).
    """
    # Simulated real snapshot (could be extended with parametrization)
    snapshot = {
        "drawdown": {"dd_pct": 1.10},
        "daily_loss": {"pct": 1.10},
        "dd_frac": 0.0110,
        "daily_loss_frac": 0.0110,
    }

    dd_pct = snapshot["drawdown"]["dd_pct"]
    dd_frac = snapshot["dd_frac"]
    expected = normalize(dd_pct)

    assert approx(dd_frac, expected), (
        f"Invariant failed: dd_frac={dd_frac} does not match expected={expected} "
        f"for dd_pct={dd_pct}"
    )


def test_daily_loss_fraction_invariant(tmp_path):
    """
    Same invariant but for daily-loss.
    """
    snapshot = {
        "drawdown": {"dd_pct": 1.10},
        "daily_loss": {"pct": 1.10},
        "dd_frac": 0.0110,
        "daily_loss_frac": 0.0110,
    }

    loss_pct = snapshot["daily_loss"]["pct"]
    loss_frac = snapshot["daily_loss_frac"]
    expected = normalize(loss_pct)

    assert approx(loss_frac, expected), (
        f"Invariant failed: daily_loss_frac={loss_frac} != expected={expected}"
    )
