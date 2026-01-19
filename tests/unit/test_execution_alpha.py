"""
Unit tests for Execution Alpha Engine — v7.9_P4

Tests:
- Config loading
- Model price resolution
- Alpha computation (sign correctness for buy/sell)
- Penalty multiplier function
- Head attribution math
- Alert generation
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from execution.execution_alpha import (
    # Config
    ExecutionAlphaConfig,
    load_execution_alpha_config,
    is_execution_alpha_enabled,
    # Dataclasses
    AlphaSample,
    SymbolAlphaStats,
    HeadAlphaStats,
    RegimeBreakdown,
    # Model price
    resolve_model_price,
    # Alpha computation
    compute_alpha,
    create_alpha_sample,
    # Aggregation
    AlphaAggregator,
    # Penalties
    compute_penalty_multiplier,
    compute_symbol_multipliers,
    compute_head_multipliers,
    # Alerts
    check_and_generate_alerts,
)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestExecutionAlphaConfig:
    """Tests for config loading and defaults."""
    
    def test_default_config_disabled(self):
        """Default config has execution alpha disabled."""
        cfg = ExecutionAlphaConfig()
        assert cfg.enabled is False
        assert cfg.model_price_source == "router_model"
        assert cfg.lookback_trades == 500
    
    def test_load_config_from_dict(self):
        """Load config from strategy_config dict."""
        strategy_cfg = {
            "execution_alpha": {
                "enabled": True,
                "model_price_source": "mid",
                "lookback_trades": 200,
                "penalty": {
                    "symbol_drag_bps_soft": 10.0,
                    "symbol_drag_bps_hard": 25.0,
                },
                "alerts": {
                    "enabled": False,
                    "tail_slippage_bps": 40.0,
                },
            }
        }
        
        cfg = load_execution_alpha_config(strategy_cfg)
        
        assert cfg.enabled is True
        assert cfg.model_price_source == "mid"
        assert cfg.lookback_trades == 200
        assert cfg.symbol_drag_bps_soft == 10.0
        assert cfg.symbol_drag_bps_hard == 25.0
        assert cfg.alerts_enabled is False
        assert cfg.tail_slippage_bps == 40.0
    
    def test_load_config_missing_section(self):
        """Missing execution_alpha section returns defaults."""
        strategy_cfg = {"other_section": {}}
        cfg = load_execution_alpha_config(strategy_cfg)
        
        assert cfg.enabled is False
        assert cfg.lookback_trades == 500
    
    def test_is_enabled_false_by_default(self):
        """is_execution_alpha_enabled returns False by default."""
        assert is_execution_alpha_enabled({}) is False


# ---------------------------------------------------------------------------
# Model Price Resolution Tests
# ---------------------------------------------------------------------------


class TestModelPriceResolution:
    """Tests for model price resolution logic."""
    
    def test_resolve_from_expected_fill_price(self):
        """Resolves from expected_fill_price field when available."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50100.0,
            "expected_fill_price": 50000.0,
        }
        cfg = ExecutionAlphaConfig(use_expected_fill_price_field=True)
        
        model = resolve_model_price(fill, None, cfg)
        
        assert model == 50000.0
    
    def test_resolve_from_router_model_price(self):
        """Resolves from router_model_price field."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50100.0,
            "router_model_price": 50050.0,
        }
        cfg = ExecutionAlphaConfig(use_expected_fill_price_field=True)
        
        model = resolve_model_price(fill, None, cfg)
        
        assert model == 50050.0
    
    def test_resolve_from_mid_price(self):
        """Resolves from quote mid when source is 'mid'."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50100.0,
        }
        quotes = {
            "BTCUSDT": {"bid": 49990.0, "ask": 50010.0}
        }
        cfg = ExecutionAlphaConfig(
            use_expected_fill_price_field=False,
            model_price_source="mid"
        )
        
        model = resolve_model_price(fill, quotes, cfg)
        
        assert model == 50000.0  # (49990 + 50010) / 2
    
    def test_resolve_from_bid_ask_side_buy(self):
        """Resolves from ask for BUY orders when source is 'bid_ask_side'."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50100.0,
        }
        quotes = {
            "BTCUSDT": {"bid": 49990.0, "ask": 50010.0}
        }
        cfg = ExecutionAlphaConfig(
            use_expected_fill_price_field=False,
            model_price_source="bid_ask_side"
        )
        
        model = resolve_model_price(fill, quotes, cfg)
        
        assert model == 50010.0  # ask for BUY
    
    def test_resolve_from_bid_ask_side_sell(self):
        """Resolves from bid for SELL orders when source is 'bid_ask_side'."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "price": 49980.0,
        }
        quotes = {
            "BTCUSDT": {"bid": 49990.0, "ask": 50010.0}
        }
        cfg = ExecutionAlphaConfig(
            use_expected_fill_price_field=False,
            model_price_source="bid_ask_side"
        )
        
        model = resolve_model_price(fill, quotes, cfg)
        
        assert model == 49990.0  # bid for SELL
    
    def test_resolve_returns_none_when_unavailable(self):
        """Returns None when model price cannot be resolved."""
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50100.0,
        }
        cfg = ExecutionAlphaConfig(
            use_expected_fill_price_field=False,
            model_price_source="mid"
        )
        
        model = resolve_model_price(fill, None, cfg)
        
        assert model is None


# ---------------------------------------------------------------------------
# Alpha Computation Tests
# ---------------------------------------------------------------------------


class TestAlphaComputation:
    """Tests for alpha calculation with correct sign handling."""
    
    def test_buy_below_model_positive_alpha(self):
        """BUY below model price = positive alpha (we saved money)."""
        # Buy at 49900 when model was 50000 = saved $100 per unit
        alpha_usd, alpha_bps, drag_bps = compute_alpha(
            fill_price=49900.0,
            model_price=50000.0,
            qty=1.0,
            side="BUY"
        )
        
        assert alpha_usd > 0  # Positive = good
        assert alpha_usd == pytest.approx(100.0)  # Saved $100
        assert alpha_bps > 0  # Positive = good
        assert alpha_bps == pytest.approx(20.0)  # (50000-49900)/50000 * 10000 = 20 bps
        assert drag_bps == 0.0  # No drag for positive alpha
    
    def test_buy_above_model_negative_alpha(self):
        """BUY above model price = negative alpha (we overpaid)."""
        # Buy at 50100 when model was 50000 = overpaid $100 per unit
        alpha_usd, alpha_bps, drag_bps = compute_alpha(
            fill_price=50100.0,
            model_price=50000.0,
            qty=1.0,
            side="BUY"
        )
        
        assert alpha_usd < 0  # Negative = bad
        assert alpha_usd == pytest.approx(-100.0)
        assert alpha_bps < 0
        assert alpha_bps == pytest.approx(-20.0)  # -20 bps
        assert drag_bps == pytest.approx(20.0)  # Drag = -alpha_bps
    
    def test_sell_above_model_positive_alpha(self):
        """SELL above model price = positive alpha (we got more)."""
        # Sell at 50100 when model was 50000 = got $100 more per unit
        alpha_usd, alpha_bps, drag_bps = compute_alpha(
            fill_price=50100.0,
            model_price=50000.0,
            qty=1.0,
            side="SELL"
        )
        
        assert alpha_usd > 0  # Positive = good
        assert alpha_usd == pytest.approx(100.0)
        assert alpha_bps > 0
        assert drag_bps == 0.0
    
    def test_sell_below_model_negative_alpha(self):
        """SELL below model price = negative alpha (we got less)."""
        # Sell at 49900 when model was 50000 = got $100 less per unit
        alpha_usd, alpha_bps, drag_bps = compute_alpha(
            fill_price=49900.0,
            model_price=50000.0,
            qty=1.0,
            side="SELL"
        )
        
        assert alpha_usd < 0  # Negative = bad
        assert alpha_usd == pytest.approx(-100.0)
        assert alpha_bps < 0
        assert drag_bps > 0  # Has drag
    
    def test_alpha_scales_with_quantity(self):
        """Alpha USD scales with quantity."""
        alpha_usd_1, _, _ = compute_alpha(49900.0, 50000.0, 1.0, "BUY")
        alpha_usd_10, _, _ = compute_alpha(49900.0, 50000.0, 10.0, "BUY")
        
        assert alpha_usd_10 == pytest.approx(alpha_usd_1 * 10)
    
    def test_create_alpha_sample(self):
        """create_alpha_sample creates correct AlphaSample."""
        fill = {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "qty": 2.5,
            "price": 2990.0,
            "ts": 1234567890.0,
        }
        
        sample = create_alpha_sample(
            fill=fill,
            model_price=3000.0,
            regime="THIN",
            head_contributions={"TREND": 0.6, "ZSCORE": 0.4},
        )
        
        assert sample.symbol == "ETHUSDT"
        assert sample.side == "BUY"
        assert sample.qty == 2.5
        assert sample.fill_price == 2990.0
        assert sample.model_price == 3000.0
        assert sample.alpha_usd > 0  # Bought below model
        assert sample.regime == "THIN"
        assert sample.head_contributions == {"TREND": 0.6, "ZSCORE": 0.4}


# ---------------------------------------------------------------------------
# Aggregator Tests
# ---------------------------------------------------------------------------


class TestAlphaAggregator:
    """Tests for rolling alpha aggregation."""
    
    def test_add_sample_updates_symbol_stats(self):
        """Adding samples updates symbol statistics."""
        agg = AlphaAggregator(lookback_trades=100)
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="BTCUSDT",
            side="BUY",
            qty=0.01,
            fill_price=50100.0,
            model_price=50000.0,
            alpha_usd=-1.0,
            alpha_bps=-20.0,
            drag_bps=20.0,
            regime="NORMAL",
            head_contributions={"TREND": 1.0},
        )
        
        agg.add_sample(sample)
        
        stats = agg.get_symbol_stats("BTCUSDT")
        assert stats is not None
        assert stats.samples == 1
        assert stats.cum_alpha_usd == -1.0
        assert stats.avg_alpha_bps == -20.0
    
    def test_add_sample_updates_head_stats(self):
        """Adding samples updates head statistics."""
        agg = AlphaAggregator(lookback_trades=100)
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="BTCUSDT",
            side="BUY",
            qty=0.01,
            fill_price=50100.0,
            model_price=50000.0,
            alpha_usd=-2.0,
            alpha_bps=-20.0,
            drag_bps=20.0,
            regime="NORMAL",
            head_contributions={"TREND": 0.6, "ZSCORE": 0.4},
        )
        
        agg.add_sample(sample)
        
        # Check TREND head (60% attribution)
        trend_stats = agg.get_head_stats("TREND")
        assert trend_stats is not None
        assert trend_stats.samples == 1
        assert trend_stats.cum_alpha_usd == pytest.approx(-1.2)  # -2.0 * 0.6
        
        # Check ZSCORE head (40% attribution)
        zscore_stats = agg.get_head_stats("ZSCORE")
        assert zscore_stats is not None
        assert zscore_stats.cum_alpha_usd == pytest.approx(-0.8)  # -2.0 * 0.4
    
    def test_regime_breakdown(self):
        """Regime breakdown is computed correctly."""
        agg = AlphaAggregator(lookback_trades=100)
        
        # Add samples in different regimes
        for i in range(5):
            agg.add_sample(AlphaSample(
                ts=1234567890.0 + i,
                symbol="SOLUSDT",
                side="BUY",
                qty=1.0,
                fill_price=150.0,
                model_price=149.0,
                alpha_usd=-1.0,
                alpha_bps=-66.0,
                drag_bps=66.0,
                regime="NORMAL",
                head_contributions={},
            ))
        
        for i in range(3):
            agg.add_sample(AlphaSample(
                ts=1234567895.0 + i,
                symbol="SOLUSDT",
                side="BUY",
                qty=1.0,
                fill_price=152.0,
                model_price=149.0,
                alpha_usd=-3.0,
                alpha_bps=-200.0,
                drag_bps=200.0,
                regime="THIN",
                head_contributions={},
            ))
        
        stats = agg.get_symbol_stats("SOLUSDT")
        assert stats is not None
        
        # Check regime breakdown
        assert "NORMAL" in stats.regime_breakdown
        assert "THIN" in stats.regime_breakdown
        assert stats.regime_breakdown["NORMAL"].samples == 5
        assert stats.regime_breakdown["THIN"].samples == 3
    
    def test_lookback_limit(self):
        """Aggregator respects lookback limit."""
        agg = AlphaAggregator(lookback_trades=5)
        
        # Add 10 samples
        for i in range(10):
            agg.add_sample(AlphaSample(
                ts=float(i),
                symbol="BTCUSDT",
                side="BUY",
                qty=1.0,
                fill_price=50000.0 + i,
                model_price=50000.0,
                alpha_usd=-float(i),
                alpha_bps=-float(i),
                drag_bps=float(i),
                regime="NORMAL",
                head_contributions={},
            ))
        
        stats = agg.get_symbol_stats("BTCUSDT")
        assert stats.samples == 5  # Only last 5


# ---------------------------------------------------------------------------
# Penalty Multiplier Tests
# ---------------------------------------------------------------------------


class TestPenaltyMultiplier:
    """Tests for penalty multiplier calculation."""
    
    def test_below_soft_returns_one(self):
        """Drag below soft threshold returns 1.0."""
        mult = compute_penalty_multiplier(
            drag_bps=5.0,
            soft_threshold=8.0,
            hard_threshold=20.0,
            min_multiplier=0.6,
        )
        
        assert mult == 1.0
    
    def test_above_hard_returns_min(self):
        """Drag above hard threshold returns min_multiplier."""
        mult = compute_penalty_multiplier(
            drag_bps=25.0,
            soft_threshold=8.0,
            hard_threshold=20.0,
            min_multiplier=0.6,
        )
        
        assert mult == 0.6
    
    def test_at_soft_returns_one(self):
        """Drag exactly at soft threshold returns 1.0."""
        mult = compute_penalty_multiplier(
            drag_bps=8.0,
            soft_threshold=8.0,
            hard_threshold=20.0,
            min_multiplier=0.6,
        )
        
        assert mult == 1.0
    
    def test_at_hard_returns_min(self):
        """Drag exactly at hard threshold returns min_multiplier."""
        mult = compute_penalty_multiplier(
            drag_bps=20.0,
            soft_threshold=8.0,
            hard_threshold=20.0,
            min_multiplier=0.6,
        )
        
        assert mult == 0.6
    
    def test_linear_interpolation(self):
        """Between soft and hard uses linear interpolation."""
        # Midpoint between 8 and 20 = 14
        mult = compute_penalty_multiplier(
            drag_bps=14.0,
            soft_threshold=8.0,
            hard_threshold=20.0,
            min_multiplier=0.6,
        )
        
        # k = (14 - 8) / (20 - 8) = 0.5
        # mult = 1.0 - 0.5 * (1.0 - 0.6) = 1.0 - 0.2 = 0.8
        assert mult == pytest.approx(0.8)
    
    def test_compute_symbol_multipliers(self):
        """compute_symbol_multipliers respects min_samples."""
        cfg = ExecutionAlphaConfig(
            min_samples_for_penalty=10,
            symbol_drag_bps_soft=8.0,
            symbol_drag_bps_hard=20.0,
            symbol_min_multiplier=0.6,
        )
        
        stats = {
            "BTCUSDT": SymbolAlphaStats(
                symbol="BTCUSDT",
                samples=5,  # Below min
                avg_drag_bps=15.0,
            ),
            "ETHUSDT": SymbolAlphaStats(
                symbol="ETHUSDT",
                samples=50,  # Above min
                avg_drag_bps=15.0,
            ),
        }
        
        mults = compute_symbol_multipliers(stats, cfg)
        
        assert mults["BTCUSDT"] == 1.0  # Not enough samples
        assert mults["ETHUSDT"] < 1.0  # Has penalty


# ---------------------------------------------------------------------------
# Alert Generation Tests
# ---------------------------------------------------------------------------


class TestAlertGeneration:
    """Tests for alert generation."""
    
    def test_tail_slippage_alert(self):
        """Generates alert for extreme negative alpha."""
        cfg = ExecutionAlphaConfig(
            alerts_enabled=True,
            tail_slippage_bps=30.0,
        )
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="DOGEUSDT",
            side="BUY",
            qty=100.0,
            fill_price=0.11,
            model_price=0.10,
            alpha_usd=-10.0,
            alpha_bps=-45.0,  # Worse than -30
            drag_bps=45.0,
            regime="THIN",
            head_contributions={"EMERGENT_ALPHA": 1.0},
        )
        
        alerts = check_and_generate_alerts(sample, None, cfg)
        
        assert len(alerts) >= 1
        assert any(a["event"] == "EXEC_ALPHA_TAIL" for a in alerts)
        
        tail_alert = next(a for a in alerts if a["event"] == "EXEC_ALPHA_TAIL")
        assert tail_alert["symbol"] == "DOGEUSDT"
        assert tail_alert["alpha_bps"] == -45.0
    
    def test_no_alert_below_threshold(self):
        """No alert when alpha is within threshold."""
        cfg = ExecutionAlphaConfig(
            alerts_enabled=True,
            tail_slippage_bps=30.0,
        )
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="DOGEUSDT",
            side="BUY",
            qty=100.0,
            fill_price=0.101,
            model_price=0.10,
            alpha_usd=-1.0,
            alpha_bps=-10.0,  # Within threshold
            drag_bps=10.0,
            regime="NORMAL",
            head_contributions={},
        )
        
        alerts = check_and_generate_alerts(sample, None, cfg)
        
        tail_alerts = [a for a in alerts if a["event"] == "EXEC_ALPHA_TAIL"]
        assert len(tail_alerts) == 0
    
    def test_drag_high_alert(self):
        """Generates alert for high rolling drag."""
        cfg = ExecutionAlphaConfig(
            alerts_enabled=True,
            drag_bps_over_rolling=12.0,
        )
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="BTCUSDT",
            side="BUY",
            qty=0.01,
            fill_price=50100.0,
            model_price=50000.0,
            alpha_usd=-1.0,
            alpha_bps=-20.0,
            drag_bps=20.0,
            regime="NORMAL",
            head_contributions={},
        )
        
        symbol_stats = SymbolAlphaStats(
            symbol="BTCUSDT",
            samples=100,
            avg_drag_bps=15.0,  # Above 12
        )
        
        alerts = check_and_generate_alerts(sample, symbol_stats, cfg)
        
        assert len(alerts) >= 1
        assert any(a["event"] == "EXEC_DRAG_HIGH" for a in alerts)
    
    def test_alerts_disabled(self):
        """No alerts when disabled."""
        cfg = ExecutionAlphaConfig(alerts_enabled=False)
        
        sample = AlphaSample(
            ts=1234567890.0,
            symbol="BTCUSDT",
            side="BUY",
            qty=0.01,
            fill_price=60000.0,
            model_price=50000.0,
            alpha_usd=-100.0,
            alpha_bps=-2000.0,  # Extreme
            drag_bps=2000.0,
            regime="CRUNCH",
            head_contributions={},
        )
        
        alerts = check_and_generate_alerts(sample, None, cfg)
        
        assert len(alerts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
