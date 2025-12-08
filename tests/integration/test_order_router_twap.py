"""
Tests for TWAP (Time-Weighted Average Price) execution in order router.

PATCHSET v7.4_C1 — TWAP Execution for Large Orders
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from execution.runtime_config import TWAPConfig, get_twap_config
from execution.order_router import (
    should_use_twap,
    split_twap_slices,
    ChildOrderResult,
    TWAPResult,
    _route_twap,
    _twap_result_to_exchange_response,
    _build_twap_metrics,
)


# ---------------------------------------------------------------------------
# TWAPConfig Tests
# ---------------------------------------------------------------------------

class TestTWAPConfig:
    """Tests for TWAPConfig dataclass and loading."""

    def test_twap_config_defaults(self):
        """Default TWAPConfig has sane values."""
        cfg = TWAPConfig()
        assert cfg.enabled is False
        assert cfg.min_notional_usd == 0.0
        assert cfg.slices == 1
        assert cfg.interval_seconds == 0.0

    def test_twap_config_custom(self):
        """Custom TWAPConfig values are stored correctly."""
        cfg = TWAPConfig(
            enabled=True,
            min_notional_usd=500.0,
            slices=4,
            interval_seconds=10.0,
        )
        assert cfg.enabled is True
        assert cfg.min_notional_usd == 500.0
        assert cfg.slices == 4
        assert cfg.interval_seconds == 10.0

    def test_get_twap_config_from_dict(self):
        """get_twap_config parses config dict correctly."""
        cfg_dict = {
            "router": {
                "twap": {
                    "enabled": True,
                    "min_notional_usd": 1000.0,
                    "slices": 5,
                    "interval_seconds": 15.0,
                }
            }
        }
        cfg = get_twap_config(cfg_dict)
        assert cfg.enabled is True
        assert cfg.min_notional_usd == 1000.0
        assert cfg.slices == 5
        assert cfg.interval_seconds == 15.0

    def test_get_twap_config_missing_router(self):
        """get_twap_config returns defaults when router section is missing."""
        cfg = get_twap_config({})
        assert cfg.enabled is False
        assert cfg.slices == 1

    def test_get_twap_config_missing_twap(self):
        """get_twap_config returns defaults when twap section is missing."""
        cfg = get_twap_config({"router": {}})
        assert cfg.enabled is False
        assert cfg.slices == 1

    def test_get_twap_config_clamps_slices(self):
        """get_twap_config clamps slices to at least 1."""
        cfg_dict = {
            "router": {
                "twap": {
                    "enabled": True,
                    "slices": 0,
                }
            }
        }
        cfg = get_twap_config(cfg_dict)
        assert cfg.slices == 1

    def test_get_twap_config_clamps_negative_values(self):
        """get_twap_config clamps negative values to 0."""
        cfg_dict = {
            "router": {
                "twap": {
                    "enabled": True,
                    "min_notional_usd": -100.0,
                    "slices": -5,
                    "interval_seconds": -10.0,
                }
            }
        }
        cfg = get_twap_config(cfg_dict)
        assert cfg.min_notional_usd == 0.0
        assert cfg.slices == 1
        assert cfg.interval_seconds == 0.0


# ---------------------------------------------------------------------------
# should_use_twap Tests
# ---------------------------------------------------------------------------

class TestShouldUseTWAP:
    """Tests for TWAP decision logic."""

    def test_below_threshold_no_twap(self):
        """Orders below min_notional_usd don't use TWAP."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4)
        assert should_use_twap(gross_usd=300.0, twap_cfg=cfg) is False

    def test_above_threshold_uses_twap(self):
        """Orders above min_notional_usd use TWAP."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4)
        assert should_use_twap(gross_usd=1000.0, twap_cfg=cfg) is True

    def test_at_threshold_uses_twap(self):
        """Orders at exactly min_notional_usd use TWAP."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4)
        assert should_use_twap(gross_usd=500.0, twap_cfg=cfg) is True

    def test_disabled_no_twap(self):
        """TWAP disabled means no TWAP even for large orders."""
        cfg = TWAPConfig(enabled=False, min_notional_usd=500.0, slices=4)
        assert should_use_twap(gross_usd=10000.0, twap_cfg=cfg) is False

    def test_single_slice_no_twap(self):
        """Single slice means no TWAP (pointless to split)."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=1)
        assert should_use_twap(gross_usd=10000.0, twap_cfg=cfg) is False

    def test_zero_gross_no_twap(self):
        """Zero gross USD doesn't use TWAP when min_notional > 0."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=100.0, slices=4)
        assert should_use_twap(gross_usd=0.0, twap_cfg=cfg) is False


# ---------------------------------------------------------------------------
# split_twap_slices Tests
# ---------------------------------------------------------------------------

class TestSplitTWAPSlices:
    """Tests for TWAP quantity splitting."""

    def test_equal_split(self):
        """Quantities are split equally across slices."""
        quantities = split_twap_slices(total_qty=100.0, slices=4)
        assert len(quantities) == 4
        assert sum(quantities) == pytest.approx(100.0)
        assert all(q == pytest.approx(25.0) for q in quantities)

    def test_single_slice(self):
        """Single slice returns full quantity."""
        quantities = split_twap_slices(total_qty=100.0, slices=1)
        assert len(quantities) == 1
        assert quantities[0] == pytest.approx(100.0)

    def test_rounding_preserved(self):
        """Total quantity is preserved despite rounding."""
        quantities = split_twap_slices(total_qty=100.0, slices=3)
        assert len(quantities) == 3
        assert sum(quantities) == pytest.approx(100.0, abs=1e-10)

    def test_zero_qty_returns_empty(self):
        """Zero quantity returns empty list."""
        quantities = split_twap_slices(total_qty=0.0, slices=4)
        assert quantities == []

    def test_negative_qty_returns_empty(self):
        """Negative quantity returns empty list."""
        quantities = split_twap_slices(total_qty=-100.0, slices=4)
        assert quantities == []

    def test_zero_slices_returns_empty(self):
        """Zero slices returns empty list."""
        quantities = split_twap_slices(total_qty=100.0, slices=0)
        assert quantities == []

    def test_min_notional_reduces_slices(self):
        """Slices are reduced if min notional not met."""
        # 100 qty at price 10 = 1000 USD total
        # 4 slices = 250 USD each
        # min_slice_notional = 300 means we can only do 3 slices (333 USD each)
        quantities = split_twap_slices(
            total_qty=100.0,
            slices=4,
            min_slice_notional=300.0,
            price=10.0,
        )
        assert len(quantities) == 3
        assert sum(quantities) == pytest.approx(100.0)

    def test_min_notional_forces_single_slice(self):
        """Very high min notional forces single slice."""
        # 100 qty at price 10 = 1000 USD total
        # min_slice_notional = 900 means we can only do 1 slice
        quantities = split_twap_slices(
            total_qty=100.0,
            slices=4,
            min_slice_notional=900.0,
            price=10.0,
        )
        assert len(quantities) == 1
        assert quantities[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ChildOrderResult and TWAPResult Tests
# ---------------------------------------------------------------------------

class TestTWAPDatastructures:
    """Tests for TWAP result datastructures."""

    def test_child_order_result(self):
        """ChildOrderResult stores slice execution data."""
        child = ChildOrderResult(
            slice_index=0,
            slice_count=4,
            slice_qty=25.0,
            order_id="12345",
            status="FILLED",
            filled_qty=25.0,
            avg_price=100.0,
            is_maker=True,
            slippage_bps=1.5,
            latency_ms=50.0,
        )
        assert child.slice_index == 0
        assert child.slice_count == 4
        assert child.order_id == "12345"
        assert child.is_maker is True

    def test_twap_result_defaults(self):
        """TWAPResult has correct defaults."""
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=4,
            interval_seconds=10.0,
        )
        assert result.children == []
        assert result.total_filled_qty == 0.0
        assert result.weighted_avg_price is None
        assert result.success is True
        assert result.error is None


# ---------------------------------------------------------------------------
# TWAP Result Conversion Tests
# ---------------------------------------------------------------------------

class TestTWAPResultConversion:
    """Tests for TWAP result to exchange response conversion."""

    def test_twap_result_to_exchange_response(self):
        """TWAPResult converts to exchange response format."""
        children = [
            ChildOrderResult(
                slice_index=i,
                slice_count=2,
                slice_qty=50.0,
                order_id=f"order_{i}",
                status="FILLED",
                filled_qty=50.0,
                avg_price=100.0 + i,
                is_maker=i == 0,
                slippage_bps=1.0,
                latency_ms=50.0,
            )
            for i in range(2)
        ]
        
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=2,
            interval_seconds=10.0,
            children=children,
            total_filled_qty=100.0,
            weighted_avg_price=100.5,
            avg_slippage_bps=1.0,
            maker_count=1,
            taker_count=1,
        )
        
        response = _twap_result_to_exchange_response(result)
        
        assert response["accepted"] is True
        assert response["status"] == "FILLED"
        assert response["price"] == 100.5
        assert response["qty"] == 100.0
        assert response["raw"]["twap"] is True
        assert len(response["raw"]["children"]) == 2
        assert response["router_meta"]["execution_style"] == "twap"

    def test_twap_result_partial_fill(self):
        """Partial TWAP fill is correctly reported."""
        children = [
            ChildOrderResult(
                slice_index=0,
                slice_count=2,
                slice_qty=50.0,
                order_id="order_0",
                status="FILLED",
                filled_qty=50.0,
                avg_price=100.0,
                is_maker=True,
                slippage_bps=1.0,
                latency_ms=50.0,
            ),
            ChildOrderResult(
                slice_index=1,
                slice_count=2,
                slice_qty=50.0,
                order_id="",
                status="FAILED",
                filled_qty=0.0,
                avg_price=None,
                is_maker=False,
                slippage_bps=0.0,
                latency_ms=None,
            ),
        ]
        
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=2,
            interval_seconds=10.0,
            children=children,
            total_filled_qty=50.0,
            weighted_avg_price=100.0,
            success=False,
            error="slice_1_failed",
        )
        
        response = _twap_result_to_exchange_response(result)
        
        assert response["status"] == "PARTIALLY_FILLED"
        assert response["qty"] == 50.0


# ---------------------------------------------------------------------------
# TWAP Metrics Tests
# ---------------------------------------------------------------------------

class TestBuildTWAPMetrics:
    """Tests for TWAP metrics building."""

    def test_build_twap_metrics(self):
        """TWAP metrics are built correctly."""
        children = [
            ChildOrderResult(
                slice_index=i,
                slice_count=4,
                slice_qty=25.0,
                order_id=f"order_{i}",
                status="FILLED",
                filled_qty=25.0,
                avg_price=100.0,
                is_maker=True,
                slippage_bps=1.0,
                latency_ms=50.0,
            )
            for i in range(4)
        ]
        
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=4,
            interval_seconds=10.0,
            children=children,
            total_filled_qty=100.0,
            weighted_avg_price=100.0,
            avg_slippage_bps=1.0,
            total_latency_ms=200.0,
            maker_count=4,
            taker_count=0,
        )
        
        metrics = _build_twap_metrics(
            twap_result=result,
            attempt_id="test_123",
            timing={},
            mark_px=100.0,
            submit_px=100.0,
            retry_count=0,
            router_ctx={},
        )
        
        assert metrics["execution_style"] == "twap"
        assert metrics["route"] == "twap"
        assert metrics["twap"]["slices"] == 4
        assert metrics["twap"]["slices_executed"] == 4
        assert metrics["twap"]["maker_count"] == 4
        assert metrics["twap"]["taker_count"] == 0
        assert len(metrics["twap"]["children"]) == 4


# ---------------------------------------------------------------------------
# _route_twap Integration Tests (with mocks)
# ---------------------------------------------------------------------------

class TestRouteTWAP:
    """Integration tests for _route_twap with mocked exchange."""

    @pytest.fixture
    def mock_route_order(self):
        """Mock route_order to avoid actual exchange calls."""
        with patch("execution.order_router.route_order") as mock:
            mock.return_value = {
                "accepted": True,
                "order_id": "mock_order_123",
                "status": "FILLED",
                "price": 100.0,
                "qty": 25.0,
                "raw": {"orderId": "mock_order_123"},
                "router_meta": {"is_maker_final": True},
            }
            yield mock

    @pytest.fixture
    def mock_log_event(self):
        """Mock log_event to avoid file I/O."""
        with patch("execution.order_router.log_event") as mock:
            yield mock

    def test_route_twap_executes_slices(self, mock_route_order, mock_log_event):
        """_route_twap executes all slices."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4, interval_seconds=0.0)
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 100.0,
            "price": 100.0,
        }
        
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        result = _route_twap(
            intent=intent,
            risk_ctx={},
            twap_cfg=cfg,
            dry_run=False,
            sleep_func=mock_sleep,
        )
        
        assert result.execution_style == "twap"
        assert result.slices == 4
        assert len(result.children) == 4
        assert mock_route_order.call_count == 4
        # No sleep with interval_seconds=0
        assert len(sleep_calls) == 0

    def test_route_twap_sleeps_between_slices(self, mock_route_order, mock_log_event):
        """_route_twap sleeps between slices when interval > 0."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4, interval_seconds=5.0)
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 100.0,
            "price": 100.0,
        }
        
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        result = _route_twap(
            intent=intent,
            risk_ctx={},
            twap_cfg=cfg,
            dry_run=False,
            sleep_func=mock_sleep,
        )
        
        # Should sleep N-1 times (3 times for 4 slices)
        assert len(sleep_calls) == 3
        assert all(s == 5.0 for s in sleep_calls)

    def test_route_twap_aggregates_fills(self, mock_route_order, mock_log_event):
        """_route_twap correctly aggregates fill data."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4, interval_seconds=0.0)
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 100.0,
            "price": 100.0,
            "mark_price": 100.0,
        }
        
        result = _route_twap(
            intent=intent,
            risk_ctx={},
            twap_cfg=cfg,
            dry_run=False,
            sleep_func=lambda x: None,
        )
        
        # Each slice fills 25.0 qty
        assert result.total_filled_qty == pytest.approx(100.0)
        assert result.success is True

    def test_route_twap_handles_slice_failure(self, mock_log_event):
        """_route_twap handles slice failures gracefully."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=500.0, slices=4, interval_seconds=0.0)
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 100.0,
            "price": 100.0,
        }
        
        call_count = [0]
        def failing_route_order(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                raise Exception("Simulated failure")
            return {
                "accepted": True,
                "order_id": f"order_{call_count[0]}",
                "status": "FILLED",
                "price": 100.0,
                "qty": 25.0,
                "raw": {},
                "router_meta": {"is_maker_final": True},
            }
        
        with patch("execution.order_router.route_order", side_effect=failing_route_order):
            result = _route_twap(
                intent=intent,
                risk_ctx={},
                twap_cfg=cfg,
                dry_run=False,
                sleep_func=lambda x: None,
            )
        
        assert result.success is False
        assert "Simulated failure" in result.error
        assert len(result.children) == 4
        assert result.children[2].status == "FAILED"


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------

class TestConfigOffNoTWAP:
    """Tests that disabled config prevents TWAP."""

    def test_disabled_config_never_twaps(self):
        """When enabled=False, should_use_twap always returns False."""
        cfg = TWAPConfig(enabled=False, min_notional_usd=100.0, slices=4)
        
        # Even with huge notional
        assert should_use_twap(gross_usd=1000000.0, twap_cfg=cfg) is False

    def test_single_slice_never_twaps(self):
        """When slices=1, should_use_twap always returns False."""
        cfg = TWAPConfig(enabled=True, min_notional_usd=100.0, slices=1)
        
        assert should_use_twap(gross_usd=1000000.0, twap_cfg=cfg) is False


# ---------------------------------------------------------------------------
# Execution Style Tagging Tests
# ---------------------------------------------------------------------------

class TestExecutionStyleTagging:
    """Tests for execution_style field in metrics."""

    def test_twap_metrics_tagged(self):
        """TWAP execution metrics have execution_style='twap'."""
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=4,
            interval_seconds=10.0,
        )
        
        metrics = _build_twap_metrics(
            twap_result=result,
            attempt_id="test",
            timing={},
            mark_px=100.0,
            submit_px=100.0,
            retry_count=0,
            router_ctx={},
        )
        
        assert metrics["execution_style"] == "twap"

    def test_twap_response_tagged(self):
        """TWAP exchange response has execution_style in router_meta."""
        result = TWAPResult(
            parent_symbol="BTCUSDT",
            parent_side="BUY",
            parent_qty=100.0,
            parent_gross_usd=10000.0,
            execution_style="twap",
            slices=4,
            interval_seconds=10.0,
        )
        
        response = _twap_result_to_exchange_response(result)
        
        assert response["router_meta"]["execution_style"] == "twap"
