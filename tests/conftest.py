"""
Pytest configuration and shared fixtures for test suite.
"""
from __future__ import annotations

import os
import pytest
from typing import Any, Dict
from unittest import mock

# Force safe defaults even if .env sets production values.
os.environ["ENV"] = "test"
os.environ.setdefault("ALLOW_PROD_WRITE", "1")


@pytest.fixture
def mock_clean_drawdown_state(monkeypatch):
    """
    Fixture that mocks _drawdown_snapshot to return a clean state.
    
    Use this in tests that should not be affected by stale peak_state.json
    on disk. This prevents the daily_loss_limit veto from triggering
    unexpectedly due to leftover state from previous runs.
    """
    from execution import risk_limits as risk_limits_module
    
    clean_snapshot = {
        "drawdown": {
            "pct": 0.0,
            "peak_nav": 10000.0,
            "nav": 10000.0,
            "abs": 0.0,
        },
        "daily_loss": {
            "pct": 0.0,
        },
        "dd_pct": 0.0,
        "peak": 10000.0,
        "nav": 10000.0,
        "usable": True,
        "stale_flags": {},
        "nav_health": {"fresh": True, "sources_ok": True},
        "peak_state": {},
        "assets": {},
    }
    
    monkeypatch.setattr(
        risk_limits_module,
        "_drawdown_snapshot",
        lambda g_cfg=None: clean_snapshot,
    )
    
    return clean_snapshot


@pytest.fixture
def mock_empty_nav_history(monkeypatch):
    """
    Fixture that mocks _nav_history_from_log to return empty list.
    
    Use this in tests that should not trigger portfolio DD circuit breaker.
    """
    from execution import risk_limits as risk_limits_module
    
    monkeypatch.setattr(
        risk_limits_module,
        "_nav_history_from_log",
        lambda limit=200: [],
    )


@pytest.fixture(autouse=False)
def reset_telegram_state():
    """
    Fixture to reset telegram rate limit state before/after tests.
    
    Use this in telegram tests to ensure clean state.
    """
    try:
        from execution import telegram_utils as tu
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()
    except ImportError:
        pass
    yield
    try:
        from execution import telegram_utils as tu
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()
    except ImportError:
        pass


@pytest.fixture(autouse=True, scope="function")
def _reset_global_state():
    """
    Auto-reset certain global state that bleeds between tests.
    Runs BEFORE and AFTER each test.
    """
    # Enforce test environment even if .env overrides to prod.
    os.environ["ENV"] = "test"
    # Clean up telegram rate limiting state BEFORE test
    try:
        from execution import telegram_utils as tu
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()
    except (ImportError, AttributeError):
        pass
    try:
        from execution import diagnostics_metrics
        diagnostics_metrics.reset_diagnostics()
    except Exception:
        pass
    yield
    # Clean up telegram rate limiting state AFTER test
    try:
        from execution import telegram_utils as tu
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()
    except (ImportError, AttributeError):
        pass
    try:
        from execution import diagnostics_metrics
        diagnostics_metrics.reset_diagnostics()
    except Exception:
        pass
