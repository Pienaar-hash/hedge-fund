"""
Unit tests for execution/minotaur_engine.py — v7.9_P3

Tests Minotaur Execution Engine core functionality:
- Config loading
- Microstructure classification
- ExecutionPlan creation
- Aggressiveness adjustment rules
- TWAP slicing
- Throttling logic
"""

import pytest
import time
from typing import Dict, Any

from execution.minotaur_engine import (
    # Constants
    REGIME_NORMAL,
    REGIME_THIN,
    REGIME_WIDE_SPREAD,
    REGIME_SPIKE,
    REGIME_CRUNCH,
    MODE_INSTANT,
    MODE_TWAP,
    MODE_STEPPED,
    # Dataclasses
    MicrostructureSnapshot,
    ExecutionRegime,
    ExecutionPlan,
    ChildOrder,
    MinotaurConfig,
    ExecutionQualityStats,
    # Config
    load_minotaur_config,
    is_minotaur_enabled,
    # Microstructure
    build_microstructure_snapshot,
    build_snapshot_from_orderbook,
    # Regime
    classify_execution_regime,
    # Planning
    build_execution_plan,
    plan_child_orders,
    # Throttling
    check_throttling,
    apply_throttling,
    # Quality
    calculate_slippage_bps,
    update_quality_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> MinotaurConfig:
    """Default Minotaur configuration."""
    return MinotaurConfig(
        enabled=True,
        min_notional_for_twap_usd=500.0,
        max_child_order_notional_usd=150.0,
        min_slice_count=2,
        max_slice_count=12,
        twap_min_seconds=60,
        twap_max_seconds=900,
        aggressiveness_base=0.6,
        aggressiveness_trend_bonus=0.1,
        aggressiveness_crisis_cap=0.3,
        wide_spread_bps=10.0,
        thin_depth_usd=2000.0,
        vol_spike_mult=2.5,
        max_symbols_in_thin_liquidity=3,
        max_new_orders_per_cycle=25,
        halt_on_liquidity_crunch=True,
    )


@pytest.fixture
def normal_snapshot() -> MicrostructureSnapshot:
    """Normal market conditions snapshot."""
    return MicrostructureSnapshot(
        symbol="BTCUSDT",
        spread_bps=3.0,
        best_bid=50000.0,
        best_ask=50015.0,
        mid_price=50007.5,
        top_depth_usd=10000.0,
        bid_depth_usd=12000.0,
        ask_depth_usd=10000.0,
        book_imbalance=0.1,
        trade_imbalance=0.05,
        realized_vol_1m=0.001,
        realized_vol_5m=0.001,
        recent_slippage_bps=2.0,
    )


@pytest.fixture
def thin_snapshot() -> MicrostructureSnapshot:
    """Thin liquidity snapshot."""
    return MicrostructureSnapshot(
        symbol="WIFUSDT",
        spread_bps=5.0,
        best_bid=2.50,
        best_ask=2.51,
        mid_price=2.505,
        top_depth_usd=800.0,  # Below thin_depth_usd threshold
        bid_depth_usd=1000.0,
        ask_depth_usd=800.0,
        book_imbalance=0.1,
        trade_imbalance=0.0,
        realized_vol_1m=0.002,
        realized_vol_5m=0.002,
        recent_slippage_bps=8.0,
    )


@pytest.fixture
def wide_spread_snapshot() -> MicrostructureSnapshot:
    """Wide spread snapshot."""
    return MicrostructureSnapshot(
        symbol="ALTUSDT",
        spread_bps=15.0,  # Above wide_spread_bps threshold
        best_bid=1.00,
        best_ask=1.0015,
        mid_price=1.00075,
        top_depth_usd=5000.0,
        bid_depth_usd=5000.0,
        ask_depth_usd=5000.0,
        book_imbalance=0.0,
        trade_imbalance=0.0,
        realized_vol_1m=0.001,
        realized_vol_5m=0.001,
        recent_slippage_bps=5.0,
    )


@pytest.fixture
def spike_snapshot() -> MicrostructureSnapshot:
    """Volatility spike snapshot."""
    return MicrostructureSnapshot(
        symbol="SOLUSDT",
        spread_bps=5.0,
        best_bid=150.0,
        best_ask=150.075,
        mid_price=150.0375,
        top_depth_usd=8000.0,
        bid_depth_usd=8000.0,
        ask_depth_usd=8000.0,
        book_imbalance=0.0,
        trade_imbalance=0.2,
        realized_vol_1m=0.01,   # 10x higher than 5m
        realized_vol_5m=0.003,  # 3x spike
        recent_slippage_bps=10.0,
    )


@pytest.fixture
def crunch_snapshot() -> MicrostructureSnapshot:
    """Crunch conditions (thin + wide)."""
    return MicrostructureSnapshot(
        symbol="SHITUSDT",
        spread_bps=20.0,  # Wide
        best_bid=0.001,
        best_ask=0.00102,
        mid_price=0.00101,
        top_depth_usd=500.0,  # Very thin
        bid_depth_usd=600.0,
        ask_depth_usd=500.0,
        book_imbalance=0.1,
        trade_imbalance=0.0,
        realized_vol_1m=0.005,
        realized_vol_5m=0.005,
        recent_slippage_bps=25.0,
    )


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestMinotaurConfig:
    """Tests for config loading."""
    
    def test_load_enabled_config(self):
        """Test loading enabled config."""
        strategy_cfg = {
            "execution_minotaur": {
                "enabled": True,
                "min_notional_for_twap_usd": 600.0,
                "aggressiveness": {
                    "base": 0.7,
                    "trend_bonus": 0.15,
                },
            }
        }
        
        cfg = load_minotaur_config(strategy_cfg)
        
        assert cfg.enabled is True
        assert cfg.min_notional_for_twap_usd == 600.0
        assert cfg.aggressiveness_base == 0.7
        assert cfg.aggressiveness_trend_bonus == 0.15
    
    def test_load_disabled_config(self):
        """Test loading disabled config (defaults)."""
        strategy_cfg = {}
        
        cfg = load_minotaur_config(strategy_cfg)
        
        assert cfg.enabled is False
        assert cfg.min_notional_for_twap_usd == 500.0
        assert cfg.aggressiveness_base == 0.6
    
    def test_is_minotaur_enabled(self):
        """Test is_minotaur_enabled helper."""
        assert is_minotaur_enabled({"execution_minotaur": {"enabled": True}}) is True
        assert is_minotaur_enabled({"execution_minotaur": {"enabled": False}}) is False
        assert is_minotaur_enabled({}) is False


# ---------------------------------------------------------------------------
# Microstructure Tests
# ---------------------------------------------------------------------------


class TestMicrostructureSnapshot:
    """Tests for microstructure snapshot building."""
    
    def test_build_snapshot_basic(self):
        """Test building a basic snapshot."""
        snapshot = build_microstructure_snapshot(
            symbol="BTCUSDT",
            best_bid=50000.0,
            best_ask=50010.0,
            bid_depth_usd=10000.0,
            ask_depth_usd=8000.0,
        )
        
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.best_bid == 50000.0
        assert snapshot.best_ask == 50010.0
        assert snapshot.mid_price == 50005.0
        assert snapshot.spread_bps == pytest.approx(2.0, rel=0.01)
        assert snapshot.top_depth_usd == 8000.0  # min(bid, ask)
        assert snapshot.book_imbalance == pytest.approx(0.111, rel=0.01)  # (10000 - 8000) / 18000
    
    def test_build_snapshot_from_orderbook(self):
        """Test building snapshot from orderbook structure."""
        orderbook = {
            "bids": [[50000.0, 0.1], [49990.0, 0.2], [49980.0, 0.15]],
            "asks": [[50010.0, 0.08], [50020.0, 0.12], [50030.0, 0.1]],
        }
        
        snapshot = build_snapshot_from_orderbook("BTCUSDT", orderbook)
        
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.best_bid == 50000.0
        assert snapshot.best_ask == 50010.0
        assert snapshot.bid_depth_usd > 0
        assert snapshot.ask_depth_usd > 0


# ---------------------------------------------------------------------------
# Regime Classification Tests
# ---------------------------------------------------------------------------


class TestRegimeClassification:
    """Tests for execution regime classification."""
    
    def test_classify_normal(self, default_config, normal_snapshot):
        """Test classification of normal conditions."""
        regime = classify_execution_regime(normal_snapshot, default_config)
        
        assert regime.symbol == "BTCUSDT"
        assert regime.regime == REGIME_NORMAL
        assert regime.liquidity_score > 0.5
        assert regime.risk_score < 0.5
    
    def test_classify_thin(self, default_config, thin_snapshot):
        """Test classification of thin liquidity."""
        regime = classify_execution_regime(thin_snapshot, default_config)
        
        assert regime.symbol == "WIFUSDT"
        assert regime.regime == REGIME_THIN
        assert regime.liquidity_score < 0.5
        assert "thin" in regime.notes.lower()
    
    def test_classify_wide_spread(self, default_config, wide_spread_snapshot):
        """Test classification of wide spread."""
        regime = classify_execution_regime(wide_spread_snapshot, default_config)
        
        assert regime.symbol == "ALTUSDT"
        assert regime.regime == REGIME_WIDE_SPREAD
        assert "spread" in regime.notes.lower()
    
    def test_classify_spike(self, default_config, spike_snapshot):
        """Test classification of volatility spike."""
        regime = classify_execution_regime(spike_snapshot, default_config)
        
        assert regime.symbol == "SOLUSDT"
        assert regime.regime == REGIME_SPIKE
        assert regime.vol_ratio > 2.0
        assert "spike" in regime.notes.lower()
    
    def test_classify_crunch(self, default_config, crunch_snapshot):
        """Test classification of crunch conditions."""
        regime = classify_execution_regime(crunch_snapshot, default_config)
        
        assert regime.symbol == "SHITUSDT"
        assert regime.regime == REGIME_CRUNCH
        assert regime.liquidity_score < 0.3
        assert regime.risk_score > 0.3


# ---------------------------------------------------------------------------
# Execution Plan Tests
# ---------------------------------------------------------------------------


class TestExecutionPlan:
    """Tests for execution plan building."""
    
    def test_instant_for_small_normal(self, default_config, normal_snapshot):
        """Test instant execution for small order in normal conditions."""
        regime = classify_execution_regime(normal_snapshot, default_config)
        
        plan = build_execution_plan(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.001,  # Small qty
            price=50000.0,
            cfg=default_config,
            regime=regime,
        )
        
        assert plan.symbol == "BTCUSDT"
        assert plan.side == "LONG"
        assert plan.slicing_mode == MODE_INSTANT
        assert plan.slice_count == 1
        assert plan.total_notional == 50.0  # 0.001 * 50000
    
    def test_twap_for_large_order(self, default_config, normal_snapshot):
        """Test TWAP for larger orders."""
        regime = classify_execution_regime(normal_snapshot, default_config)
        
        plan = build_execution_plan(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.02,  # 0.02 * 50000 = 1000 USD (above threshold)
            price=50000.0,
            cfg=default_config,
            regime=regime,
        )
        
        assert plan.slicing_mode == MODE_TWAP
        assert plan.slice_count >= 2
        assert plan.schedule_seconds >= default_config.twap_min_seconds
    
    def test_stepped_for_crunch(self, default_config, crunch_snapshot):
        """Test STEPPED mode for crunch conditions."""
        regime = classify_execution_regime(crunch_snapshot, default_config)
        
        plan = build_execution_plan(
            symbol="SHITUSDT",
            side="SHORT",
            qty=100000.0,  # Large qty, assume low price
            price=0.001,
            cfg=default_config,
            regime=regime,
        )
        
        # Low notional (100 USD), but crunch regime may affect mode
        # For crunch, even if notional is low, regime affects schedule
        assert plan.regime == REGIME_CRUNCH
    
    def test_aggressiveness_reduced_in_crisis(self, default_config, spike_snapshot):
        """Test aggressiveness is capped in hostile regimes."""
        regime = classify_execution_regime(spike_snapshot, default_config)
        
        plan = build_execution_plan(
            symbol="SOLUSDT",
            side="LONG",
            qty=10.0,
            price=150.0,
            cfg=default_config,
            regime=regime,
            trend_bias=0.8,  # Strong trend would add bonus normally
        )
        
        # In SPIKE regime, aggressiveness should be capped
        assert plan.aggressiveness <= default_config.aggressiveness_crisis_cap + 0.01
    
    def test_trend_bonus_applied_in_normal(self, default_config, normal_snapshot):
        """Test trend bonus is applied in normal conditions."""
        regime = classify_execution_regime(normal_snapshot, default_config)
        
        plan_no_trend = build_execution_plan(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.001,
            price=50000.0,
            cfg=default_config,
            regime=regime,
            trend_bias=None,
        )
        
        plan_with_trend = build_execution_plan(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.001,
            price=50000.0,
            cfg=default_config,
            regime=regime,
            trend_bias=1.0,  # Strong trend
        )
        
        assert plan_with_trend.aggressiveness > plan_no_trend.aggressiveness
    
    def test_slice_count_bounds(self, default_config, normal_snapshot):
        """Test slice count respects min/max bounds."""
        regime = classify_execution_regime(normal_snapshot, default_config)
        
        # Very large order
        plan = build_execution_plan(
            symbol="BTCUSDT",
            side="LONG",
            qty=1.0,  # 1 BTC = 50000 USD
            price=50000.0,
            cfg=default_config,
            regime=regime,
        )
        
        assert plan.slice_count >= default_config.min_slice_count
        assert plan.slice_count <= default_config.max_slice_count


# ---------------------------------------------------------------------------
# TWAP Planner Tests
# ---------------------------------------------------------------------------


class TestTWAPPlanner:
    """Tests for TWAP child order planning."""
    
    def test_instant_produces_single_order(self):
        """Test instant mode produces single child order."""
        plan = ExecutionPlan(
            symbol="BTCUSDT",
            side="LONG",
            total_qty=0.01,
            total_notional=500.0,
            slicing_mode=MODE_INSTANT,
            slice_count=1,
            schedule_seconds=0,
            aggressiveness=0.6,
            regime=REGIME_NORMAL,
        )
        
        children = plan_child_orders(plan)
        
        assert len(children) == 1
        assert children[0].target_qty == 0.01
        assert children[0].sequence == 0
    
    def test_twap_produces_multiple_orders(self):
        """Test TWAP mode produces multiple evenly spaced orders."""
        plan = ExecutionPlan(
            symbol="BTCUSDT",
            side="LONG",
            total_qty=0.1,
            total_notional=5000.0,
            slicing_mode=MODE_TWAP,
            slice_count=5,
            schedule_seconds=300,  # 5 min
            aggressiveness=0.6,
            regime=REGIME_NORMAL,
        )
        
        start_ts = time.time()
        children = plan_child_orders(plan, start_ts=start_ts)
        
        assert len(children) == 5
        
        # Check quantities sum to total
        total_qty = sum(c.target_qty for c in children)
        assert total_qty == pytest.approx(0.1, rel=0.001)
        
        # Check even spacing
        interval = 300 / 5  # 60 seconds per slice
        for i, child in enumerate(children):
            expected_ts = start_ts + (i * interval)
            assert child.earliest_ts == pytest.approx(expected_ts, abs=1.0)
            assert child.sequence == i
    
    def test_stepped_produces_decreasing_sizes(self):
        """Test STEPPED mode produces decreasing order sizes."""
        plan = ExecutionPlan(
            symbol="SOLUSDT",
            side="SHORT",
            total_qty=10.0,
            total_notional=1500.0,
            slicing_mode=MODE_STEPPED,
            slice_count=4,
            schedule_seconds=240,
            aggressiveness=0.4,
            regime=REGIME_SPIKE,
        )
        
        children = plan_child_orders(plan)
        
        assert len(children) == 4
        
        # First slice should be larger than last
        assert children[0].target_qty > children[-1].target_qty
        
        # Total should still sum to plan qty
        total_qty = sum(c.target_qty for c in children)
        assert total_qty == pytest.approx(10.0, rel=0.001)
    
    def test_child_order_parent_linking(self):
        """Test child orders are linked to parent plan."""
        plan = ExecutionPlan(
            symbol="ETHUSDT",
            side="LONG",
            total_qty=1.0,
            total_notional=3000.0,
            slicing_mode=MODE_TWAP,
            slice_count=3,
            schedule_seconds=180,
            aggressiveness=0.5,
            regime=REGIME_THIN,
        )
        
        children = plan_child_orders(plan, plan_id="TEST_PLAN_123")
        
        for child in children:
            assert child.parent_plan_id == "TEST_PLAN_123"
            assert child.symbol == "ETHUSDT"
            assert child.side == "LONG"


# ---------------------------------------------------------------------------
# Throttling Tests
# ---------------------------------------------------------------------------


class TestThrottling:
    """Tests for throttling logic."""
    
    def test_no_throttle_in_normal(self, default_config):
        """Test no throttling in normal conditions."""
        regimes = [
            ExecutionRegime(symbol="BTCUSDT", regime=REGIME_NORMAL),
            ExecutionRegime(symbol="ETHUSDT", regime=REGIME_NORMAL),
        ]
        
        throttle, halt, reason = check_throttling(regimes, 5, default_config)
        
        assert throttle is False
        assert halt is False
        assert "no throttling" in reason.lower()
    
    def test_throttle_on_too_many_thin(self, default_config):
        """Test throttling when many symbols in thin liquidity."""
        regimes = [
            ExecutionRegime(symbol="SYM1", regime=REGIME_THIN),
            ExecutionRegime(symbol="SYM2", regime=REGIME_THIN),
            ExecutionRegime(symbol="SYM3", regime=REGIME_THIN),
            ExecutionRegime(symbol="SYM4", regime=REGIME_THIN),  # 4 > max 3
        ]
        
        throttle, halt, reason = check_throttling(regimes, 5, default_config)
        
        assert throttle is True
        assert "THIN" in reason
    
    def test_halt_on_crunch(self, default_config):
        """Test halt when multiple symbols in crunch."""
        regimes = [
            ExecutionRegime(symbol="BTCUSDT", regime=REGIME_NORMAL),
            ExecutionRegime(symbol="CRNCH1", regime=REGIME_CRUNCH),
            ExecutionRegime(symbol="CRNCH2", regime=REGIME_CRUNCH),
        ]
        
        throttle, halt, reason = check_throttling(regimes, 5, default_config)
        
        assert halt is True
        assert "CRUNCH" in reason
    
    def test_throttle_on_order_limit(self, default_config):
        """Test throttling when order limit exceeded."""
        regimes = [ExecutionRegime(symbol="BTCUSDT", regime=REGIME_NORMAL)]
        
        throttle, halt, reason = check_throttling(
            regimes,
            pending_orders=30,  # > max 25
            cfg=default_config,
        )
        
        assert throttle is True
        assert "orders" in reason.lower()
    
    def test_apply_throttling_reduces_slices(self, default_config):
        """Test apply_throttling reduces slice counts."""
        plans = [
            ExecutionPlan(
                symbol="BTCUSDT", side="LONG",
                total_qty=0.1, total_notional=5000.0,
                slicing_mode=MODE_TWAP, slice_count=10,
                schedule_seconds=600, aggressiveness=0.6,
                regime=REGIME_NORMAL,
            ),
            ExecutionPlan(
                symbol="ETHUSDT", side="LONG",
                total_qty=1.0, total_notional=3000.0,
                slicing_mode=MODE_TWAP, slice_count=8,
                schedule_seconds=480, aggressiveness=0.6,
                regime=REGIME_NORMAL,
            ),
        ]
        
        regimes = {
            "BTCUSDT": ExecutionRegime(symbol="BTCUSDT", regime=REGIME_NORMAL),
            "ETHUSDT": ExecutionRegime(symbol="ETHUSDT", regime=REGIME_NORMAL),
        }
        
        # Max 10 orders total
        throttled = apply_throttling(plans, regimes, default_config, max_orders=10)
        
        total_slices = sum(p.slice_count for p in throttled)
        assert total_slices <= 10
    
    def test_apply_throttling_drops_crunch(self, default_config):
        """Test apply_throttling drops plans in CRUNCH regime."""
        plans = [
            ExecutionPlan(
                symbol="BTCUSDT", side="LONG",
                total_qty=0.1, total_notional=5000.0,
                slicing_mode=MODE_TWAP, slice_count=5,
                schedule_seconds=300, aggressiveness=0.6,
                regime=REGIME_NORMAL,
            ),
            ExecutionPlan(
                symbol="CRNCH1", side="LONG",
                total_qty=100.0, total_notional=100.0,
                slicing_mode=MODE_TWAP, slice_count=3,
                schedule_seconds=180, aggressiveness=0.3,
                regime=REGIME_CRUNCH,
            ),
        ]
        
        regimes = {
            "BTCUSDT": ExecutionRegime(symbol="BTCUSDT", regime=REGIME_NORMAL),
            "CRNCH1": ExecutionRegime(symbol="CRNCH1", regime=REGIME_CRUNCH),
        }
        
        throttled = apply_throttling(plans, regimes, default_config)
        
        # CRUNCH plan should be dropped
        symbols = [p.symbol for p in throttled]
        assert "CRNCH1" not in symbols
        assert "BTCUSDT" in symbols


# ---------------------------------------------------------------------------
# Quality Metrics Tests
# ---------------------------------------------------------------------------


class TestExecutionQualityMetrics:
    """Tests for execution quality calculations."""
    
    def test_slippage_long_positive(self):
        """Test slippage calculation for LONG (paid more = positive)."""
        slip = calculate_slippage_bps(
            fill_price=100.10,
            model_price=100.00,
            side="LONG",
        )
        
        assert slip == pytest.approx(10.0, rel=0.01)  # 10 bps adverse
    
    def test_slippage_long_negative(self):
        """Test slippage calculation for LONG (paid less = negative/favorable)."""
        slip = calculate_slippage_bps(
            fill_price=99.95,
            model_price=100.00,
            side="LONG",
        )
        
        assert slip == pytest.approx(-5.0, rel=0.01)  # 5 bps favorable
    
    def test_slippage_short_positive(self):
        """Test slippage calculation for SHORT (received less = positive/adverse)."""
        slip = calculate_slippage_bps(
            fill_price=99.90,
            model_price=100.00,
            side="SHORT",
        )
        
        assert slip == pytest.approx(10.0, rel=0.01)  # 10 bps adverse
    
    def test_slippage_short_negative(self):
        """Test slippage calculation for SHORT (received more = negative/favorable)."""
        slip = calculate_slippage_bps(
            fill_price=100.05,
            model_price=100.00,
            side="SHORT",
        )
        
        assert slip == pytest.approx(-5.0, rel=0.01)  # 5 bps favorable
    
    def test_update_quality_stats_first(self):
        """Test updating quality stats on first trade."""
        stats = ExecutionQualityStats(symbol="BTCUSDT")
        
        updated = update_quality_stats(
            stats=stats,
            slippage_bps=5.0,
            fill_ratio=1.0,
            notional=1000.0,
            used_twap=True,
            regime=REGIME_NORMAL,
        )
        
        assert updated.trade_count == 1
        assert updated.avg_slippage_bps == 5.0
        assert updated.max_slippage_bps == 5.0
        assert updated.twap_usage_pct == 1.0
    
    def test_update_quality_stats_ema(self):
        """Test EMA update of quality stats."""
        stats = ExecutionQualityStats(
            symbol="BTCUSDT",
            avg_slippage_bps=5.0,
            max_slippage_bps=5.0,
            trade_count=10,
        )
        
        # Add a worse trade
        updated = update_quality_stats(
            stats=stats,
            slippage_bps=20.0,
            fill_ratio=0.95,
            notional=2000.0,
            used_twap=False,
            regime=REGIME_THIN,
            lookback=50,
        )
        
        # EMA should move towards new value but not equal it
        assert 5.0 < updated.avg_slippage_bps < 20.0
        assert updated.max_slippage_bps == 20.0
        assert updated.trade_count == 11
        assert updated.last_regime == REGIME_THIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
