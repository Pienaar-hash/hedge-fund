"""
Tests for v7.5_C2 factor PnL attribution.

Verifies:
- PnL attribution by factor weight
- Total PnL matches sum of factor PnL
- Zero-weight factors receive zero PnL
- Edge cases: empty trades, missing factors
"""
import pytest
import numpy as np


class TestPnlAttribution:
    """Test PnL attribution to factors."""
    
    def test_pnl_sum_equals_total(self):
        """Sum of factor PnL should equal total PnL."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"trend": 0.8, "carry": 0.2},
            ),
            TradeRecord(
                symbol="ETHUSDT",
                direction="LONG",
                realized_pnl_usd=-50.0,
                factor_components={"trend": 0.6, "carry": 0.4},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        
        total_from_factors = sum(result.by_factor.values())
        assert np.isclose(total_from_factors, result.total_pnl_usd, atol=0.01)
    
    def test_single_factor_gets_all_pnl(self):
        """If only one factor has non-zero weight, it gets all PnL."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"trend": 1.0, "carry": 0.0},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        
        assert result.by_factor["trend"] == 100.0
        assert result.by_factor["carry"] == 0.0
    
    def test_equal_weights_split_pnl_equally(self):
        """Equal factor weights should split PnL equally."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"a": 0.5, "b": 0.5},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a", "b"])
        
        assert np.isclose(result.by_factor["a"], 50.0)
        assert np.isclose(result.by_factor["b"], 50.0)
    
    def test_proportional_attribution(self):
        """PnL should be attributed proportionally to absolute weights."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"a": 0.8, "b": 0.2},  # 80% / 20%
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a", "b"])
        
        assert np.isclose(result.by_factor["a"], 80.0)
        assert np.isclose(result.by_factor["b"], 20.0)
    
    def test_negative_factor_values_use_absolute(self):
        """Negative factor values should use absolute value for weighting."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"a": 0.6, "b": -0.4},  # abs: 0.6 + 0.4 = 1.0
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a", "b"])
        
        # 0.6 / 1.0 = 60%, 0.4 / 1.0 = 40%
        assert np.isclose(result.by_factor["a"], 60.0)
        assert np.isclose(result.by_factor["b"], 40.0)


class TestMultipleTrades:
    """Test attribution across multiple trades."""
    
    def test_aggregates_across_trades(self):
        """PnL attribution should aggregate across trades."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=100.0,
                factor_components={"trend": 1.0, "carry": 0.0},
            ),
            TradeRecord(
                symbol="ETHUSDT",
                direction="LONG",
                realized_pnl_usd=50.0,
                factor_components={"trend": 0.0, "carry": 1.0},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        
        assert result.by_factor["trend"] == 100.0
        assert result.by_factor["carry"] == 50.0
        assert result.total_pnl_usd == 150.0
    
    def test_handles_losses(self):
        """Handles negative PnL (losses) correctly."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=-100.0,
                factor_components={"trend": 1.0, "carry": 0.0},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        
        assert result.by_factor["trend"] == -100.0
        assert result.by_factor["carry"] == 0.0
        assert result.total_pnl_usd == -100.0
    
    def test_mixed_profit_loss(self):
        """Handles mixed profits and losses."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=200.0,
                factor_components={"trend": 1.0},
            ),
            TradeRecord(
                symbol="ETHUSDT",
                direction="LONG",
                realized_pnl_usd=-150.0,
                factor_components={"trend": 1.0},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend"])
        
        assert result.by_factor["trend"] == 50.0  # 200 - 150
        assert result.total_pnl_usd == 50.0


class TestEdgeCases:
    """Test edge cases for PnL attribution."""
    
    def test_empty_trades_returns_zeros(self):
        """Empty trades list should return zero PnL."""
        from execution.factor_pnl_attribution import compute_factor_pnl_snapshot
        
        result = compute_factor_pnl_snapshot([], ["trend", "carry"])
        
        assert result.total_pnl_usd == 0.0
        assert result.by_factor.get("trend", 0.0) == 0.0
        assert result.trade_count == 0
    
    def test_missing_factor_components_distributes_equally(self):
        """Missing factor components should distribute PnL equally."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=90.0,
                factor_components={},  # No factor info
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a", "b", "c"])
        
        # Should distribute equally: 90 / 3 = 30 each
        assert np.isclose(result.by_factor["a"], 30.0)
        assert np.isclose(result.by_factor["b"], 30.0)
        assert np.isclose(result.by_factor["c"], 30.0)
    
    def test_all_zero_weights_distributes_equally(self):
        """All zero factor weights should distribute PnL equally."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord(
                symbol="BTCUSDT",
                direction="LONG",
                realized_pnl_usd=60.0,
                factor_components={"a": 0.0, "b": 0.0},
            ),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a", "b"])
        
        assert np.isclose(result.by_factor["a"], 30.0)
        assert np.isclose(result.by_factor["b"], 30.0)
    
    def test_trade_count_correct(self):
        """Trade count should be correct."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 10.0, {"a": 1.0}),
            TradeRecord("ETH", "LONG", 20.0, {"a": 1.0}),
            TradeRecord("SOL", "LONG", 30.0, {"a": 1.0}),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["a"])
        
        assert result.trade_count == 3
    
    def test_window_days_preserved(self):
        """Window days should be preserved in result."""
        from execution.factor_pnl_attribution import compute_factor_pnl_snapshot
        
        result = compute_factor_pnl_snapshot([], ["a"], window_days=7)
        
        assert result.window_days == 7


class TestToDict:
    """Test to_dict serialization."""
    
    def test_to_dict_contains_required_fields(self):
        """to_dict should contain all required fields."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 100.0, {"trend": 0.8, "carry": 0.2}),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        d = result.to_dict()
        
        assert "by_factor" in d
        assert "pct_by_factor" in d
        assert "total_pnl_usd" in d
        assert "window_days" in d
        assert "trade_count" in d
        assert "updated_ts" in d
    
    def test_pct_by_factor_sums_to_100(self):
        """Percentage by factor should sum to ~100%."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 100.0, {"trend": 0.8, "carry": 0.2}),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        d = result.to_dict()
        
        pct_sum = sum(d["pct_by_factor"].values())
        assert np.isclose(pct_sum, 100.0, atol=0.1)
    
    def test_to_dict_serializable(self):
        """to_dict should produce JSON-serializable output."""
        import json
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 100.0, {"trend": 0.8}),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend"])
        
        # Should not raise
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)
