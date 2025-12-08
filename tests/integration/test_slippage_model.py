"""
Tests for slippage_model.py (v7.5_B1)
"""

import json
import math
import pytest
import tempfile
import time
from pathlib import Path


# ===========================================================================
# Tests: SlippageObservation
# ===========================================================================

class TestSlippageObservation:
    """Test SlippageObservation dataclass."""
    
    def test_basic_instantiation(self):
        """Should create observation with required fields."""
        from execution.slippage_model import SlippageObservation
        
        obs = SlippageObservation(
            symbol="BTCUSDT",
            side="BUY",
            notional_usd=10000.0,
            expected_bps=2.0,
            realized_bps=3.5,
            spread_bps=1.0,
            maker=False,
        )
        
        assert obs.symbol == "BTCUSDT"
        assert obs.side == "BUY"
        assert obs.notional_usd == 10000.0
        assert obs.expected_bps == 2.0
        assert obs.realized_bps == 3.5
        assert obs.spread_bps == 1.0
        assert obs.maker is False

    def test_timestamp_defaults_to_now(self):
        """Timestamp should default to current time."""
        from execution.slippage_model import SlippageObservation
        
        before = time.time()
        obs = SlippageObservation(
            symbol="BTCUSDT",
            side="BUY",
            notional_usd=1000.0,
            expected_bps=1.0,
            realized_bps=1.5,
            spread_bps=0.5,
            maker=True,
        )
        after = time.time()
        
        assert before <= obs.timestamp <= after


# ===========================================================================
# Tests: estimate_expected_slippage_bps
# ===========================================================================

class TestEstimateExpectedSlippage:
    """Test expected slippage estimation from order book."""
    
    def test_zero_qty_returns_zero(self):
        """Zero quantity should return zero slippage."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        depth = [(100.0, 10.0), (101.0, 10.0)]
        slippage = estimate_expected_slippage_bps("BUY", 0.0, depth, 100.0)
        
        assert slippage == 0.0

    def test_empty_depth_returns_zero(self):
        """Empty depth should return zero slippage."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        slippage = estimate_expected_slippage_bps("BUY", 10.0, [], 100.0)
        
        assert slippage == 0.0

    def test_buy_slippage_is_positive_when_crossing_book(self):
        """BUY crossing book should have positive slippage."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        # Asks: 101, 102, 103 (ascending)
        depth = [(101.0, 5.0), (102.0, 5.0), (103.0, 5.0)]
        mid_price = 100.0
        qty = 10.0  # Will consume first two levels
        
        slippage = estimate_expected_slippage_bps("BUY", qty, depth, mid_price)
        
        # VWAP = (101*5 + 102*5) / 10 = 101.5
        # Slippage = (101.5 - 100) / 100 * 10000 = 150 bps
        assert slippage > 0
        assert abs(slippage - 150.0) < 1.0

    def test_sell_slippage_is_positive_when_crossing_book(self):
        """SELL crossing book should have positive slippage."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        # Bids: 99, 98, 97 (descending)
        depth = [(99.0, 5.0), (98.0, 5.0), (97.0, 5.0)]
        mid_price = 100.0
        qty = 10.0  # Will consume first two levels
        
        slippage = estimate_expected_slippage_bps("SELL", qty, depth, mid_price)
        
        # VWAP = (99*5 + 98*5) / 10 = 98.5
        # Slippage = (100 - 98.5) / 100 * 10000 = 150 bps
        assert slippage > 0
        assert abs(slippage - 150.0) < 1.0

    def test_small_qty_has_minimal_slippage(self):
        """Small quantity should have minimal slippage."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        depth = [(100.1, 100.0), (100.2, 100.0)]
        mid_price = 100.0
        qty = 1.0  # Small qty, stays in first level
        
        slippage = estimate_expected_slippage_bps("BUY", qty, depth, mid_price)
        
        # VWAP = 100.1 (all from first level)
        # Slippage = (100.1 - 100) / 100 * 10000 = 10 bps
        assert slippage == pytest.approx(10.0, rel=0.01)

    def test_insufficient_depth_returns_capped_slippage(self):
        """Should cap slippage when depth insufficient."""
        from execution.slippage_model import estimate_expected_slippage_bps
        
        depth = [(101.0, 1.0)]  # Only 1 unit available
        mid_price = 100.0
        qty = 100.0  # Much larger than depth
        
        slippage = estimate_expected_slippage_bps("BUY", qty, depth, mid_price)
        
        # Should be capped at 50 bps (or partial fill value)
        assert slippage <= 100.0


# ===========================================================================
# Tests: compute_realized_slippage_bps
# ===========================================================================

class TestComputeRealizedSlippage:
    """Test realized slippage calculation."""
    
    def test_buy_positive_when_paid_more(self):
        """BUY slippage positive when fill price > mid."""
        from execution.slippage_model import compute_realized_slippage_bps
        
        slippage = compute_realized_slippage_bps("BUY", 101.0, 100.0)
        
        # (101 - 100) / 100 * 10000 = 100 bps
        assert slippage == pytest.approx(100.0, rel=0.01)

    def test_buy_negative_when_paid_less(self):
        """BUY slippage negative when fill price < mid."""
        from execution.slippage_model import compute_realized_slippage_bps
        
        slippage = compute_realized_slippage_bps("BUY", 99.0, 100.0)
        
        assert slippage == pytest.approx(-100.0, rel=0.01)

    def test_sell_positive_when_received_less(self):
        """SELL slippage positive when fill price < mid."""
        from execution.slippage_model import compute_realized_slippage_bps
        
        slippage = compute_realized_slippage_bps("SELL", 99.0, 100.0)
        
        # (100 - 99) / 100 * 10000 = 100 bps
        assert slippage == pytest.approx(100.0, rel=0.01)

    def test_sell_negative_when_received_more(self):
        """SELL slippage negative when fill price > mid."""
        from execution.slippage_model import compute_realized_slippage_bps
        
        slippage = compute_realized_slippage_bps("SELL", 101.0, 100.0)
        
        assert slippage == pytest.approx(-100.0, rel=0.01)

    def test_zero_mid_price_returns_zero(self):
        """Zero mid price should return zero slippage."""
        from execution.slippage_model import compute_realized_slippage_bps
        
        slippage = compute_realized_slippage_bps("BUY", 100.0, 0.0)
        
        assert slippage == 0.0


# ===========================================================================
# Tests: compute_spread_bps
# ===========================================================================

class TestComputeSpreadBps:
    """Test spread calculation."""
    
    def test_normal_spread(self):
        """Should compute spread correctly."""
        from execution.slippage_model import compute_spread_bps
        
        spread = compute_spread_bps(99.5, 100.5)
        
        # mid = 100, spread = 1 / 100 * 10000 = 100 bps
        assert spread == pytest.approx(100.0, rel=0.01)

    def test_tight_spread(self):
        """Should compute tight spread."""
        from execution.slippage_model import compute_spread_bps
        
        spread = compute_spread_bps(99.99, 100.01)
        
        # mid = 100, spread = 0.02 / 100 * 10000 = 2 bps
        assert spread == pytest.approx(2.0, rel=0.01)

    def test_zero_bid_returns_zero(self):
        """Zero bid should return zero spread."""
        from execution.slippage_model import compute_spread_bps
        
        spread = compute_spread_bps(0.0, 100.0)
        
        assert spread == 0.0


# ===========================================================================
# Tests: EWMA Update
# ===========================================================================

class TestEWMAUpdate:
    """Test EWMA slippage updates."""
    
    def test_first_observation_initializes(self):
        """First observation should initialize stats."""
        from execution.slippage_model import (
            update_slippage_ewma,
            SlippageObservation,
        )
        
        obs = SlippageObservation(
            symbol="BTCUSDT",
            side="BUY",
            notional_usd=1000.0,
            expected_bps=5.0,
            realized_bps=8.0,
            spread_bps=2.0,
            maker=False,
        )
        
        stats = update_slippage_ewma(None, obs, halflife_trades=50)
        
        assert stats.ewma_expected_bps == 5.0
        assert stats.ewma_realized_bps == 8.0
        assert stats.trade_count == 1

    def test_ewma_updates_smoothly(self):
        """EWMA should update smoothly with new observations."""
        from execution.slippage_model import (
            update_slippage_ewma,
            SlippageObservation,
            SlippageStats,
        )
        
        prev = SlippageStats(
            ewma_expected_bps=5.0,
            ewma_realized_bps=5.0,
            trade_count=10,
            last_obs_ts=time.time() - 100,
        )
        
        obs = SlippageObservation(
            symbol="BTCUSDT",
            side="BUY",
            notional_usd=1000.0,
            expected_bps=10.0,  # Higher than prev
            realized_bps=10.0,
            spread_bps=2.0,
            maker=False,
        )
        
        stats = update_slippage_ewma(prev, obs, halflife_trades=50)
        
        # EWMA should be between prev and obs
        assert 5.0 < stats.ewma_expected_bps < 10.0
        assert 5.0 < stats.ewma_realized_bps < 10.0
        assert stats.trade_count == 11

    def test_ewma_alpha_calculation(self):
        """EWMA alpha should be calculated correctly."""
        from execution.slippage_model import compute_ewma_alpha
        
        # halflife=1 means alpha=0.5
        alpha = compute_ewma_alpha(1)
        assert alpha == pytest.approx(0.5, rel=0.01)
        
        # halflife=50 means alpha ≈ 0.0138
        alpha = compute_ewma_alpha(50)
        assert alpha == pytest.approx(1 - 0.5**(1/50), rel=0.01)


# ===========================================================================
# Tests: SlippageMetricsStore
# ===========================================================================

class TestSlippageMetricsStore:
    """Test SlippageMetricsStore class."""
    
    def test_get_stats_returns_none_for_unknown(self):
        """Should return None for unknown symbol."""
        from execution.slippage_model import SlippageMetricsStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SlippageMetricsStore(Path(tmpdir))
            
            stats = store.get_stats("UNKNOWNUSDT")
            
            assert stats is None

    def test_update_creates_stats(self):
        """Update should create stats for new symbol."""
        from execution.slippage_model import SlippageMetricsStore, SlippageObservation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SlippageMetricsStore(Path(tmpdir))
            
            obs = SlippageObservation(
                symbol="BTCUSDT",
                side="BUY",
                notional_usd=1000.0,
                expected_bps=5.0,
                realized_bps=6.0,
                spread_bps=2.0,
                maker=False,
            )
            
            stats = store.update(obs)
            
            assert stats.ewma_expected_bps == 5.0
            assert stats.ewma_realized_bps == 6.0
            assert stats.trade_count == 1

    def test_get_all_stats(self):
        """get_all_stats should return all tracked symbols."""
        from execution.slippage_model import SlippageMetricsStore, SlippageObservation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SlippageMetricsStore(Path(tmpdir))
            
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                obs = SlippageObservation(
                    symbol=symbol,
                    side="BUY",
                    notional_usd=1000.0,
                    expected_bps=5.0,
                    realized_bps=6.0,
                    spread_bps=2.0,
                    maker=False,
                )
                store.update(obs)
            
            all_stats = store.get_all_stats()
            
            assert "BTCUSDT" in all_stats
            assert "ETHUSDT" in all_stats


# ===========================================================================
# Tests: SlippageConfig
# ===========================================================================

class TestSlippageConfig:
    """Test SlippageConfig loading."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from execution.slippage_model import SlippageConfig
        
        cfg = SlippageConfig()
        
        assert cfg.enabled is True
        assert cfg.ewma_halflife_trades == 50
        assert cfg.max_expected_slippage_bps == 15.0
        assert cfg.depth_levels == 5
        assert cfg.spread_pause_factor == 1.5


# ===========================================================================
# Tests: build_slippage_snapshot
# ===========================================================================

class TestBuildSlippageSnapshot:
    """Test build_slippage_snapshot function."""
    
    def test_returns_per_symbol_dict(self):
        """Should return dict with per_symbol slippage stats."""
        from execution.slippage_model import (
            build_slippage_snapshot,
            SlippageMetricsStore,
            SlippageObservation,
        )
        import execution.slippage_model as sm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store with some data
            store = SlippageMetricsStore(Path(tmpdir))
            obs = SlippageObservation(
                symbol="BTCUSDT",
                side="BUY",
                notional_usd=1000.0,
                expected_bps=5.0,
                realized_bps=6.0,
                spread_bps=2.0,
                maker=False,
            )
            store.update(obs)
            
            # Replace global store temporarily
            old_store = sm._SLIPPAGE_STORE
            sm._SLIPPAGE_STORE = store
            
            try:
                snapshot = build_slippage_snapshot()
                
                assert "updated_ts" in snapshot
                assert "per_symbol" in snapshot
                assert "BTCUSDT" in snapshot["per_symbol"]
            finally:
                sm._SLIPPAGE_STORE = old_store
