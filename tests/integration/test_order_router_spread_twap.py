"""
Tests for spread-aware TWAP in order_router.py (v7.5_B1)
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict


# ===========================================================================
# Tests: Spread-Aware TWAP Slicing
# ===========================================================================

class TestSpreadAwareTWAP:
    """Test spread-aware TWAP execution."""
    
    @patch("execution.order_router.get_market_microstructure")
    @patch("execution.order_router.route_order")
    def test_normal_spread_executes_all_slices(
        self,
        mock_route_order,
        mock_microstructure,
    ):
        """All slices should execute when spread is normal."""
        from execution.order_router import _route_twap
        from execution.runtime_config import TWAPConfig
        
        # Mock normal spread
        mock_microstructure.return_value = (
            99.9, 100.1, 2.0,  # best_bid, best_ask, spread_bps (2 bps is normal)
            [(99.9, 10.0)],   # bids
            [(100.1, 10.0)],  # asks
        )
        
        # Mock successful order execution
        mock_route_order.return_value = {
            "accepted": True,
            "order_id": "12345",
            "status": "FILLED",
            "price": 100.0,
            "qty": 0.25,
            "raw": {},
            "router_meta": {"is_maker_final": True},
        }
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 1.0,
            "price": 100.0,
        }
        
        risk_ctx = {"price": 100.0}
        twap_cfg = TWAPConfig(
            enabled=True,
            min_notional_usd=100.0,
            slices=4,
            interval_seconds=0.01,  # Fast for testing
        )
        
        result = _route_twap(
            intent,
            risk_ctx,
            twap_cfg,
            dry_run=False,
            sleep_func=lambda x: None,
        )
        
        # All 4 slices should execute
        assert len(result.children) == 4
        assert result.success is True

    @patch("execution.order_router.get_market_microstructure")
    @patch("execution.order_router.route_order")
    @patch("execution.order_router._get_slippage_config")
    @patch("execution.order_router.get_bucket_for_symbol")
    def test_wide_spread_skips_slices(
        self,
        mock_get_bucket,
        mock_slippage_cfg,
        mock_route_order,
        mock_microstructure,
    ):
        """Slices should be skipped when spread is too wide."""
        from execution.order_router import _route_twap
        from execution.runtime_config import TWAPConfig
        from execution.liquidity_model import LiquidityBucketConfig
        from execution.slippage_model import SlippageConfig
        
        # Mock bucket with 5 bps max spread
        mock_get_bucket.return_value = LiquidityBucketConfig(
            name="A_HIGH",
            max_spread_bps=5.0,
            default_maker_bias=0.8,
        )
        
        # Mock slippage config with 1.5x pause factor
        mock_slippage_cfg.return_value = SlippageConfig(
            enabled=True,
            spread_pause_factor=1.5,  # pause threshold = 5 * 1.5 = 7.5 bps
            depth_levels=5,
        )
        
        # First call: wide spread (10 bps > 7.5 threshold)
        # Second call: normal spread
        # etc.
        spread_sequence = [
            (99.5, 100.5, 10.0, [], []),  # Wide - skip
            (99.9, 100.1, 2.0, [], []),   # Normal - execute
            (99.5, 100.5, 10.0, [], []),  # Wide - skip
            (99.9, 100.1, 2.0, [], []),   # Normal - execute
        ]
        mock_microstructure.side_effect = spread_sequence
        
        # Mock successful order execution
        mock_route_order.return_value = {
            "accepted": True,
            "order_id": "12345",
            "status": "FILLED",
            "price": 100.0,
            "qty": 0.25,
            "raw": {},
            "router_meta": {"is_maker_final": True},
        }
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 1.0,
            "price": 100.0,
        }
        
        risk_ctx = {"price": 100.0}
        twap_cfg = TWAPConfig(
            enabled=True,
            min_notional_usd=100.0,
            slices=4,
            interval_seconds=0.01,
        )
        
        result = _route_twap(
            intent,
            risk_ctx,
            twap_cfg,
            dry_run=False,
            sleep_func=lambda x: None,
        )
        
        # Only 2 slices executed (2 were skipped due to wide spread)
        assert len(result.children) == 2


# ===========================================================================
# Tests: Maker Bias by Liquidity Bucket
# ===========================================================================

class TestMakerBiasByBucket:
    """Test maker/taker bias based on liquidity bucket."""
    
    def test_high_liquidity_bucket_has_high_maker_bias(self):
        """A_HIGH bucket should have 80% maker bias."""
        from execution.liquidity_model import get_default_maker_bias, reload_liquidity_model
        
        reload_liquidity_model()
        
        bias = get_default_maker_bias("BTCUSDT")
        assert bias == 0.8

    def test_medium_liquidity_bucket_has_balanced_bias(self):
        """B_MEDIUM bucket should have 60% maker bias."""
        from execution.liquidity_model import get_default_maker_bias, reload_liquidity_model
        
        reload_liquidity_model()
        
        bias = get_default_maker_bias("SOLUSDT")
        assert bias == 0.6

    def test_low_liquidity_bucket_has_low_maker_bias(self):
        """C_LOW bucket should have 40% maker bias."""
        from execution.liquidity_model import get_default_maker_bias, reload_liquidity_model
        
        reload_liquidity_model()
        
        bias = get_default_maker_bias("WIFUSDT")
        assert bias == 0.4


# ===========================================================================
# Tests: Slippage Observation Recording
# ===========================================================================

class TestSlippageObservationRecording:
    """Test slippage observation recording in router."""
    
    def test_record_slippage_creates_observation(self):
        """_record_slippage should create valid observation."""
        from execution.order_router import _record_slippage
        from unittest.mock import patch
        
        with patch("execution.order_router.record_slippage_observation") as mock_record:
            with patch("execution.order_router._SLIPPAGE_AVAILABLE", True):
                with patch("execution.order_router.compute_realized_slippage_bps", return_value=5.0):
                    with patch("execution.order_router.estimate_expected_slippage_bps", return_value=3.0):
                        _record_slippage(
                            symbol="BTCUSDT",
                            side="BUY",
                            notional_usd=10000.0,
                            fill_price=100.5,
                            mid_price=100.0,
                            spread_bps=2.0,
                            is_maker=False,
                            depth=[(100.1, 10.0)],
                        )
                        
                        # Should have called record with observation
                        assert mock_record.called


# ===========================================================================
# Tests: get_market_microstructure
# ===========================================================================

class TestGetMarketMicrostructure:
    """Test market microstructure fetching."""
    
    @patch("execution.order_router.ex.get_orderbook")
    def test_returns_spread_and_depth(self, mock_orderbook):
        """Should return spread and depth from orderbook."""
        from execution.order_router import get_market_microstructure
        
        mock_orderbook.return_value = {
            "bids": [["99.9", "10.0"], ["99.8", "20.0"]],
            "asks": [["100.1", "10.0"], ["100.2", "20.0"]],
        }
        
        best_bid, best_ask, spread_bps, bids, asks = get_market_microstructure(
            "BTCUSDT", depth_levels=5
        )
        
        assert best_bid == 99.9
        assert best_ask == 100.1
        assert spread_bps == pytest.approx(20.0, rel=0.1)  # 0.2 / 100 * 10000
        assert len(bids) == 2
        assert len(asks) == 2

    @patch("execution.order_router.ex.get_orderbook")
    def test_handles_empty_orderbook(self, mock_orderbook):
        """Should handle empty orderbook gracefully."""
        from execution.order_router import get_market_microstructure
        
        mock_orderbook.return_value = {"bids": [], "asks": []}
        
        best_bid, best_ask, spread_bps, bids, asks = get_market_microstructure(
            "BTCUSDT", depth_levels=5
        )
        
        assert best_bid == 0.0
        assert best_ask == 0.0
        assert spread_bps == 0.0

    @patch("execution.order_router.ex.get_orderbook")
    def test_handles_exception(self, mock_orderbook):
        """Should handle orderbook fetch exception."""
        from execution.order_router import get_market_microstructure
        
        mock_orderbook.side_effect = Exception("API error")
        
        best_bid, best_ask, spread_bps, bids, asks = get_market_microstructure(
            "BTCUSDT", depth_levels=5
        )
        
        assert best_bid == 0.0
        assert best_ask == 0.0


# ===========================================================================
# Tests: TWAP Logging with Spread Info
# ===========================================================================

class TestTWAPLogging:
    """Test TWAP event logging includes spread info."""
    
    @patch("execution.order_router.get_market_microstructure")
    @patch("execution.order_router.route_order")
    @patch("execution.order_router.log_event")
    def test_twap_start_logs_liquidity_bucket(
        self,
        mock_log_event,
        mock_route_order,
        mock_microstructure,
    ):
        """TWAP start event should include liquidity bucket."""
        from execution.order_router import _route_twap
        from execution.runtime_config import TWAPConfig
        
        mock_microstructure.return_value = (99.9, 100.1, 2.0, [], [])
        mock_route_order.return_value = {
            "accepted": True,
            "order_id": "12345",
            "status": "FILLED",
            "price": 100.0,
            "qty": 0.5,
            "raw": {},
            "router_meta": {"is_maker_final": True},
        }
        
        intent = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 1.0,
            "price": 100.0,
        }
        
        twap_cfg = TWAPConfig(
            enabled=True,
            min_notional_usd=100.0,
            slices=2,
            interval_seconds=0.01,
        )
        
        _route_twap(intent, {}, twap_cfg, dry_run=False, sleep_func=lambda x: None)
        
        # Check log_event was called
        assert mock_log_event.called
