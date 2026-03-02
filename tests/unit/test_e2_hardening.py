"""
Tests for E2.1-E2.3 hardening patches:
  - E2.1: churn_guard.bootstrap_from_positions()
  - E2.3: exchange_utils.cancel_all_open_orders()
  - E2.4: process_fill_for_quality callsite (signature fix)
"""

import json
import os
import tempfile
import time

import pytest
from unittest.mock import patch, MagicMock

from execution.churn_guard import (
    bootstrap_from_positions,
    check_exit_allowed,
    ChurnConfig,
    reset_state,
    _key,
)


@pytest.fixture(autouse=True)
def _clean_state():
    reset_state()
    yield
    reset_state()


# ── E2.1: Bootstrap from positions ────────────────────────────────────────

class TestBootstrapFromPositions:
    """churn_guard.bootstrap_from_positions()"""

    def test_empty_positions(self):
        result = bootstrap_from_positions([])
        assert result["positions_seen"] == 0
        assert result["seeded"] == 0

    def test_seeds_entry_time_for_active_position(self):
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
        ]
        result = bootstrap_from_positions(positions)
        assert result["seeded"] == 1
        assert _key("BTCUSDT", "LONG") in result["keys"]

    def test_skips_zero_qty(self):
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0},
        ]
        result = bootstrap_from_positions(positions)
        assert result["seeded"] == 0

    def test_derives_side_from_qty_sign(self):
        positions = [
            {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": -0.5},
        ]
        result = bootstrap_from_positions(positions)
        assert result["seeded"] == 1
        assert _key("ETHUSDT", "SHORT") in result["keys"]

    def test_min_hold_active_after_bootstrap(self):
        """After bootstrap, safety-critical exits bypass min_hold; normal exits are blocked."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
        ]
        bootstrap_from_positions(positions)
        cfg = ChurnConfig(min_hold_seconds=120, crisis_override=True)

        # REGIME_FLIP is safety-critical → bypasses min_hold (Doctrine Law #7)
        ok, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="REGIME_FLIP", config=cfg,
        )
        assert ok is True

        # Non-safety exit (e.g. TIME_STOP) should still be blocked by min_hold
        ok2, reason2 = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="TIME_STOP", config=cfg,
        )
        assert ok2 is False
        assert "min_hold" in reason2.lower()

    def test_crisis_override_bypasses_bootstrap_hold(self):
        """CRISIS_OVERRIDE should bypass min_hold even after bootstrap."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
        ]
        bootstrap_from_positions(positions)
        cfg = ChurnConfig(min_hold_seconds=120, crisis_override=True)
        ok, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="CRISIS_OVERRIDE", config=cfg,
        )
        assert ok is True

    def test_multiple_positions(self):
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "positionAmt": -0.5},
            {"symbol": "SOLUSDT", "positionSide": "LONG", "positionAmt": 10},
        ]
        result = bootstrap_from_positions(positions)
        assert result["seeded"] == 3

    def test_idempotent(self):
        """Calling bootstrap twice doesn't overwrite existing entry times."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
        ]
        r1 = bootstrap_from_positions(positions)
        assert r1["seeded"] == 1
        r2 = bootstrap_from_positions(positions)
        assert r2["seeded"] == 0  # not re-seeded

    def test_fill_log_scan(self):
        """Bootstrap reads fill log to find last entry timestamp."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write a fill from 1 hour ago
            ts_old = time.time() - 3600
            fill = {
                "event_type": "order_fill",
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "executedQty": 0.01,
                "ts": f"2025-01-01T00:00:00",
            }
            f.write(json.dumps(fill) + "\n")
            f.flush()
            path = f.name
        try:
            positions = [
                {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
            ]
            result = bootstrap_from_positions(positions, fill_log_path=path)
            assert result["seeded"] == 1
        finally:
            os.unlink(path)

    def test_qty_field_variant(self):
        """Handles both 'positionAmt' and 'qty' field names."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.05},
        ]
        result = bootstrap_from_positions(positions)
        assert result["seeded"] == 1


# ── E2.3: cancel_all_open_orders ──────────────────────────────────────────

class TestCancelAllOpenOrders:
    """exchange_utils.cancel_all_open_orders()"""

    @patch("execution.exchange_utils.is_dry_run", return_value=True)
    def test_dry_run_skips(self, _mock):
        from execution.exchange_utils import cancel_all_open_orders
        result = cancel_all_open_orders(symbols=["BTCUSDT"])
        assert result["cancelled"] == {}
        assert result["skipped"] == 0

    @patch("execution.exchange_utils.get_um_client")
    @patch("execution.exchange_utils.is_dry_run", return_value=False)
    def test_cancels_per_symbol(self, _dry, mock_client_fn):
        from execution.exchange_utils import cancel_all_open_orders
        mock_client = MagicMock()
        mock_client.is_stub = False
        # First symbol has open orders, second doesn't
        mock_client.cancel_open_orders.side_effect = [
            [{"orderId": 123}],
            [],
        ]
        mock_client_fn.return_value = mock_client

        result = cancel_all_open_orders(symbols=["BTCUSDT", "ETHUSDT"])
        assert "BTCUSDT" in result["cancelled"]
        assert result["skipped"] == 1
        assert mock_client.cancel_open_orders.call_count == 2

    @patch("execution.exchange_utils.get_um_client")
    @patch("execution.exchange_utils.is_dry_run", return_value=False)
    def test_handles_no_orders_error(self, _dry, mock_client_fn):
        from execution.exchange_utils import cancel_all_open_orders
        mock_client = MagicMock()
        mock_client.is_stub = False
        mock_client.cancel_open_orders.side_effect = Exception("code=-2011, msg=Unknown order")
        mock_client_fn.return_value = mock_client

        result = cancel_all_open_orders(symbols=["BTCUSDT"])
        assert result["errors"] == {}
        assert result["skipped"] == 1  # -2011 is treated as skip

    @patch("execution.exchange_utils.get_um_client")
    @patch("execution.exchange_utils.is_dry_run", return_value=False)
    def test_handles_real_error(self, _dry, mock_client_fn):
        from execution.exchange_utils import cancel_all_open_orders
        mock_client = MagicMock()
        mock_client.is_stub = False
        mock_client.cancel_open_orders.side_effect = Exception("network timeout")
        mock_client_fn.return_value = mock_client

        result = cancel_all_open_orders(symbols=["BTCUSDT"])
        assert "BTCUSDT" in result["errors"]
        assert "network timeout" in result["errors"]["BTCUSDT"]

    @patch("execution.exchange_utils.get_um_client", return_value=None)
    @patch("execution.exchange_utils.is_dry_run", return_value=False)
    def test_no_client(self, _dry, _client):
        from execution.exchange_utils import cancel_all_open_orders
        result = cancel_all_open_orders(symbols=["BTCUSDT"])
        assert result["cancelled"] == {}


# ── E2.4: process_fill_for_quality signature ──────────────────────────────

class TestProcessFillForQualitySignature:
    """Verify that minotaur_integration.process_fill_for_quality accepts
    the unpacked keyword args that executor_live.py now passes."""

    def test_accepts_keyword_args(self):
        from execution.minotaur_integration import process_fill_for_quality
        # Should not raise TypeError
        result = process_fill_for_quality(
            symbol="BTCUSDT",
            side="LONG",
            fill_price=40000.0,
            model_price=40010.0,
            fill_qty=0.01,
            target_qty=0.01,
            used_twap=False,
        )
        assert result is not None

    def test_returns_quality_stats(self):
        from execution.minotaur_integration import process_fill_for_quality
        result = process_fill_for_quality(
            symbol="ETHUSDT",
            side="SHORT",
            fill_price=2500.0,
            model_price=2495.0,
            fill_qty=0.5,
            target_qty=0.5,
            used_twap=True,
        )
        assert hasattr(result, "symbol")
        assert result.symbol == "ETHUSDT"
