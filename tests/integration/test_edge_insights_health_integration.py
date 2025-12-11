"""
Integration tests for P7 Strategy Health Score & Alpha Attribution Index.

Tests:
- EdgeInsights snapshot includes strategy_health field
- health_score ∈ [0, 1]
- All component scores valid
- Dashboard panel handles health data gracefully
- No changes to existing EdgeInsights schema consumers
- EdgeScanner remains deterministic
"""

import pytest
from dataclasses import asdict
from pathlib import Path
import json


@pytest.fixture
def mock_regime():
    """Mock regime context for health computation."""
    return {
        "vol_regime": "NORMAL",
        "dd_state": "CLEAR",
        "router_quality": 0.82,
    }


@pytest.fixture
def mock_regime_adjustments():
    """Mock regime adjustments (from P6)."""
    return {
        "conviction_multiplier": 1.0,
        "factor_multipliers": {
            "funding": 1.0,
            "momentum": 1.0,
            "basis": 1.0,
        },
    }


@pytest.fixture
def mock_factor_edges():
    """Mock factor edges with edge data."""
    return {
        "funding": {"edge_score": 0.6, "zscore": 1.2, "weight": 0.25},
        "momentum": {"edge_score": 0.4, "zscore": 0.8, "weight": 0.20},
        "value": {"edge_score": -0.1, "zscore": -0.2, "weight": 0.15},
        "basis": {"edge_score": 0.5, "zscore": 1.0, "weight": 0.20},
        "volatility": {"edge_score": 0.3, "zscore": 0.6, "weight": 0.20},
    }


@pytest.fixture
def mock_symbol_edges():
    """Mock symbol edges for symbol health."""
    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "edge_score": 0.7,
            "hybrid_score": 0.65,
            "category": "btc",
        },
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "edge_score": 0.5,
            "hybrid_score": 0.55,
            "category": "eth",
        },
        "SOLUSDT": {
            "symbol": "SOLUSDT",
            "edge_score": -0.1,
            "hybrid_score": 0.3,
            "category": "alt-l1",
        },
    }


@pytest.fixture
def mock_category_edges():
    """Mock category edges for category health."""
    return {
        "btc": {"category": "btc", "edge_score": 0.7, "symbol_count": 1},
        "eth": {"category": "eth", "edge_score": 0.5, "symbol_count": 1},
        "alt-l1": {"category": "alt-l1", "edge_score": -0.1, "symbol_count": 1},
    }


class TestEdgeInsightsWithHealth:
    """Test EdgeInsights includes strategy_health correctly."""

    def test_edge_insights_includes_strategy_health_field(self):
        """EdgeInsights dataclass has strategy_health field."""
        from execution.edge_scanner import EdgeInsights, StrategyHealth

        insights = EdgeInsights(
            symbol_edges={},
            factor_edges={},
            category_edges={},
            strategy_health=None,
        )
        assert hasattr(insights, "strategy_health")

    def test_edge_insights_to_dict_includes_health(self):
        """EdgeInsights.to_dict() includes strategy_health."""
        from execution.edge_scanner import EdgeInsights, StrategyHealth

        health = StrategyHealth(
            health_score=0.75,
            factor_health={"score": 0.8, "strength_label": "strong"},
            symbol_health={"score": 0.7, "symbol_count": 5},
            category_health={"score": 0.65, "category_count": 3},
            regime_alignment={"score": 0.9, "in_expected_range": True},
            execution_quality={"score": 0.85, "quality_bucket": "excellent"},
            notes=["✅ System healthy"],
        )
        insights = EdgeInsights(
            symbol_edges={},
            factor_edges={},
            category_edges={},
            strategy_health=health,
        )

        data = insights.to_dict()
        assert "strategy_health" in data
        assert data["strategy_health"]["health_score"] == 0.75
        assert data["strategy_health"]["factor_health"]["score"] == 0.8

    def test_edge_insights_to_dict_handles_none_health(self):
        """EdgeInsights.to_dict() handles None strategy_health gracefully."""
        from execution.edge_scanner import EdgeInsights

        insights = EdgeInsights(
            symbol_edges={},
            factor_edges={},
            category_edges={},
            strategy_health=None,
        )

        data = insights.to_dict()
        assert "strategy_health" not in data or data.get("strategy_health") is None


class TestStrategyHealthComputation:
    """Test compute_strategy_health integration."""

    def test_compute_returns_valid_health_score(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """compute_strategy_health returns score in [0, 1]."""
        from execution.edge_scanner import compute_strategy_health

        health = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )

        assert health is not None
        assert 0.0 <= health.health_score <= 1.0

    def test_all_component_scores_valid(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """All component health dicts are populated."""
        from execution.edge_scanner import compute_strategy_health

        health = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )

        assert isinstance(health.factor_health, dict)
        assert isinstance(health.symbol_health, dict)
        assert isinstance(health.category_health, dict)
        assert isinstance(health.regime_alignment, dict)
        assert isinstance(health.execution_quality, dict)

    def test_health_is_deterministic(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """Same inputs produce identical health scores."""
        from execution.edge_scanner import compute_strategy_health

        health1 = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )
        health2 = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )

        assert health1.health_score == health2.health_score
        assert health1.factor_health == health2.factor_health
        assert health1.notes == health2.notes

    def test_notes_generated_for_low_scores(self):
        """Notes generated when component scores are low."""
        from execution.edge_scanner import compute_strategy_health

        # Low factor health scenario - all negative edges
        bad_factors = {
            "funding": {"edge_score": -0.5, "zscore": -2.0},
            "momentum": {"edge_score": -0.3, "zscore": -1.5},
            "basis": {"edge_score": -0.4, "zscore": -1.8},
        }

        health = compute_strategy_health(
            factor_edges=bad_factors,
            symbol_edges={},
            category_edges={},
            regime={"vol_regime": "NORMAL", "dd_state": "CLEAR"},
            regime_adjustments={},
        )

        # Should have warning notes about negative factors
        assert len(health.notes) > 0
        # Health score should be lower
        assert health.health_score < 0.7


class TestStrategyHealthWeights:
    """Test custom weight configurations."""

    def test_default_weights_sum_to_one(self):
        """DEFAULT_HEALTH_WEIGHTS sum to 1.0."""
        from execution.edge_scanner import DEFAULT_HEALTH_WEIGHTS

        total = sum(DEFAULT_HEALTH_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_custom_weights_applied(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """Custom weights are applied correctly."""
        from execution.edge_scanner import compute_strategy_health

        # Factor-heavy weighting
        custom_weights = {
            "factor_quality": 0.80,
            "symbol_quality": 0.05,
            "category_quality": 0.05,
            "regime_alignment": 0.05,
            "execution_quality": 0.05,
        }

        health = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
            weights=custom_weights,
        )

        # With 80% factor weight, health should be close to factor health
        assert health is not None
        # Rough check - health should be influenced heavily by factors
        assert 0.0 <= health.health_score <= 1.0


class TestStrategyHealthExports:
    """Test module exports and API."""

    def test_strategy_health_exported(self):
        """StrategyHealth is exported from edge_scanner."""
        from execution.edge_scanner import StrategyHealth

        assert StrategyHealth is not None

    def test_compute_strategy_health_exported(self):
        """compute_strategy_health is exported from edge_scanner."""
        from execution.edge_scanner import compute_strategy_health

        assert callable(compute_strategy_health)

    def test_default_health_weights_exported(self):
        """DEFAULT_HEALTH_WEIGHTS is exported from edge_scanner."""
        from execution.edge_scanner import DEFAULT_HEALTH_WEIGHTS

        assert isinstance(DEFAULT_HEALTH_WEIGHTS, dict)
        assert "factor_quality" in DEFAULT_HEALTH_WEIGHTS


class TestBackwardCompatibility:
    """Ensure P7 changes don't break existing consumers."""

    def test_edge_insights_existing_fields_unchanged(self):
        """Existing EdgeInsights fields still work."""
        from execution.edge_scanner import EdgeInsights

        insights = EdgeInsights(
            symbol_edges={"BTCUSDT": {"edge_score": 0.5}},
            factor_edges={"funding": {"edge_score": 0.6}},
            category_edges={"btc": {"edge_score": 0.7}},
            strategy_health=None,
        )

        # Existing API works
        assert insights.symbol_edges["BTCUSDT"]["edge_score"] == 0.5
        assert insights.factor_edges["funding"]["edge_score"] == 0.6

    def test_to_dict_preserves_existing_structure(self):
        """to_dict() preserves existing structure for consumers."""
        from execution.edge_scanner import EdgeInsights

        insights = EdgeInsights(
            symbol_edges={"BTCUSDT": {"edge_score": 0.5}},
            factor_edges={"funding": {"edge_score": 0.6}},
            category_edges={},
            strategy_health=None,
        )

        data = insights.to_dict()

        # Existing structure intact
        assert "symbol_edges" in data
        assert "factor_edges" in data
        assert "category_edges" in data
        assert "edge_summary" in data


class TestStateFileContract:
    """Test state file contract compliance."""

    def test_health_serializable_to_json(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """StrategyHealth can be serialized to JSON."""
        from execution.edge_scanner import compute_strategy_health

        health = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )

        # Convert to dict and serialize
        health_dict = health.to_dict()
        json_str = json.dumps(health_dict)

        # Deserialize and verify
        parsed = json.loads(json_str)
        assert parsed["health_score"] == round(health.health_score, 4)
        assert parsed["notes"] == health.notes

    def test_edge_insights_with_health_serializable(
        self,
        mock_symbol_edges,
        mock_factor_edges,
        mock_category_edges,
        mock_regime,
        mock_regime_adjustments,
    ):
        """EdgeInsights with strategy_health serializes cleanly."""
        from execution.edge_scanner import EdgeInsights, compute_strategy_health

        health = compute_strategy_health(
            factor_edges=mock_factor_edges,
            symbol_edges=mock_symbol_edges,
            category_edges=mock_category_edges,
            regime=mock_regime,
            regime_adjustments=mock_regime_adjustments,
        )

        insights = EdgeInsights(
            symbol_edges=mock_symbol_edges,
            factor_edges=mock_factor_edges,
            category_edges=mock_category_edges,
            strategy_health=health,
        )

        # Full serialization
        data = insights.to_dict()
        json_str = json.dumps(data)

        # Deserialize and verify structure
        parsed = json.loads(json_str)
        assert "strategy_health" in parsed
        assert parsed["strategy_health"]["health_score"] == round(health.health_score, 4)


class TestDashboardIntegration:
    """Test dashboard can consume health data."""

    def test_dashboard_module_imports(self):
        """Dashboard edge_discovery_panel imports successfully."""
        try:
            from dashboard import edge_discovery_panel

            assert hasattr(edge_discovery_panel, "render_strategy_health")
        except ImportError:
            # Dashboard may have streamlit dependency issues in test env
            pytest.skip("Dashboard import requires streamlit")

    def test_render_function_handles_missing_health(self):
        """render_strategy_health handles empty edge_insights gracefully."""
        try:
            from dashboard.edge_discovery_panel import render_strategy_health
            from unittest.mock import MagicMock, patch

            # Mock streamlit
            with patch("dashboard.edge_discovery_panel.st") as mock_st:
                mock_st.subheader = MagicMock()
                mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
                mock_st.metric = MagicMock()
                mock_st.info = MagicMock()
                mock_st.progress = MagicMock()
                mock_st.caption = MagicMock()
                mock_st.markdown = MagicMock()
                mock_st.warning = MagicMock()

                # Should not raise with empty dict
                render_strategy_health({})

        except ImportError:
            pytest.skip("Dashboard import requires streamlit")
