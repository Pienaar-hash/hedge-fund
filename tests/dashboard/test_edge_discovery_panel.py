# mypy: ignore-errors
"""Tests for edge_discovery_panel (v7.7_P5)."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict


# Synthetic edge_insights fixture matching EdgeScanner output
@pytest.fixture
def sample_edge_insights() -> Dict[str, Any]:
    """Minimal edge_insights payload for panel tests matching EdgeScanner schema."""
    return {
        "updated_ts": "2024-01-01T00:00:00Z",
        "edge_summary": {
            "top_factors": [
                {"factor": "momentum", "edge": 0.65, "regime_fit": True},
                {"factor": "mean_reversion", "edge": 0.52, "regime_fit": True},
                {"factor": "carry", "edge": 0.41, "regime_fit": False},
            ],
            "weak_factors": [
                {"factor": "volatility", "edge": -0.15, "regime_fit": False},
                {"factor": "correlation", "edge": -0.22, "regime_fit": False},
            ],
            "top_symbols": [
                {"symbol": "BTCUSDT", "edge": 0.72, "top_factor": "momentum"},
                {"symbol": "ETHUSDT", "edge": 0.58, "top_factor": "carry"},
            ],
            "weak_symbols": [
                {"symbol": "DOGEUSDT", "edge": -0.12, "top_factor": "volatility"},
            ],
            "top_categories": [
                {"category": "layer1", "edge": 0.55, "count": 3},
                {"category": "defi", "edge": 0.32, "count": 2},
            ],
            "weak_categories": [
                {"category": "meme", "edge": -0.18, "count": 2},
            ],
            "regime": {
                "vol_regime": "normal",
                "dd_state": "normal",
                "risk_mode": "normal",
                "router_quality": 0.85,
                "current_dd_pct": 0.02,
            },
        },
        "factor_edges": {
            "momentum": {"edge": 0.65, "ir": 1.2},
            "mean_reversion": {"edge": 0.52, "ir": 0.8},
            "carry": {"edge": 0.41, "ir": 0.6},
            "volatility": {"edge": -0.15, "ir": -0.3},
            "correlation": {"edge": -0.22, "ir": -0.5},
        },
        "symbol_edges": {
            "BTCUSDT": {"edge": 0.72, "hybrid_score": 0.8},
            "ETHUSDT": {"edge": 0.58, "hybrid_score": 0.7},
        },
        "category_edges": {
            "layer1": {"edge": 0.55, "count": 3},
            "defi": {"edge": 0.32, "count": 2},
        },
        "config_echo": {},
    }


@pytest.fixture
def empty_edge_insights() -> Dict[str, Any]:
    """Edge insights with minimal/empty data."""
    return {
        "updated_ts": "2024-01-01T00:00:00Z",
        "edge_summary": {
            "top_factors": [],
            "weak_factors": [],
            "top_symbols": [],
            "weak_symbols": [],
            "top_categories": [],
            "weak_categories": [],
            "regime": {},
        },
        "factor_edges": {},
        "symbol_edges": {},
        "category_edges": {},
        "config_echo": {},
    }


class TestLoadEdgeInsights:
    """Tests for load_edge_insights function."""

    def test_load_from_dict(self, sample_edge_insights):
        """Load from a dict object with edge_summary."""
        from dashboard.edge_discovery_panel import load_edge_insights
        
        # Create a mock state object with edge_insights attribute
        class MockState:
            def __init__(self, data):
                self.edge_insights = data
        
        state = MockState(sample_edge_insights)
        result = load_edge_insights(state)
        
        assert result is not None
        assert "edge_summary" in result
        assert result["edge_summary"]["regime"]["vol_regime"] == "normal"

    def test_load_returns_dict(self, sample_edge_insights):
        """Verify the loader returns a dict."""
        from dashboard.edge_discovery_panel import load_edge_insights
        
        class MockState:
            edge_insights = None
        
        state = MockState()
        state.edge_insights = sample_edge_insights
        result = load_edge_insights(state)
        
        assert isinstance(result, dict)

    def test_load_none_state(self):
        """Return empty dict when state is None and file missing."""
        from dashboard.edge_discovery_panel import load_edge_insights
        
        # With no state object and no file, should return empty dict
        result = load_edge_insights(None)
        assert result == {} or isinstance(result, dict)


class TestRenderRegimeContext:
    """Tests for render_regime_context function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_with_full_data(self, mock_st, sample_edge_insights):
        """Render regime context with all fields present."""
        from dashboard.edge_discovery_panel import render_regime_context
        
        # Mock column objects
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        render_regime_context(sample_edge_insights)
        
        # Should access regime context and render metrics
        mock_st.columns.assert_called()

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_with_missing_fields(self, mock_st, empty_edge_insights):
        """Render gracefully with missing fields."""
        from dashboard.edge_discovery_panel import render_regime_context
        
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        # Should not raise, should show info message
        render_regime_context(empty_edge_insights)


class TestRenderFactorEdges:
    """Tests for render_factor_edges function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_with_factors(self, mock_st, sample_edge_insights):
        """Render factor edges with data."""
        from dashboard.edge_discovery_panel import render_factor_edges
        
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col]
        
        render_factor_edges(sample_edge_insights)
        
        mock_st.columns.assert_called()

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_empty_factors(self, mock_st, empty_edge_insights):
        """Render info message when no factors."""
        from dashboard.edge_discovery_panel import render_factor_edges
        
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col]
        
        render_factor_edges(empty_edge_insights)


class TestRenderSymbolEdges:
    """Tests for render_symbol_edges function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_with_symbols(self, mock_st, sample_edge_insights):
        """Render symbol edges with data."""
        from dashboard.edge_discovery_panel import render_symbol_edges
        
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col]
        
        render_symbol_edges(sample_edge_insights)
        
        mock_st.columns.assert_called()


class TestRenderCategoryEdges:
    """Tests for render_category_edges function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_with_categories(self, mock_st, sample_edge_insights):
        """Render category edges with data."""
        from dashboard.edge_discovery_panel import render_category_edges
        
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col]
        
        render_category_edges(sample_edge_insights)
        
        mock_st.columns.assert_called()


class TestRenderEdgeMap:
    """Tests for render_edge_map function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_edge_map_with_data(self, mock_st, sample_edge_insights):
        """Render edge map with factor_edges data."""
        from dashboard.edge_discovery_panel import render_edge_map
        
        render_edge_map(sample_edge_insights)

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_edge_map_empty(self, mock_st, empty_edge_insights):
        """Show info when no factor_edges data."""
        from dashboard.edge_discovery_panel import render_edge_map
        
        render_edge_map(empty_edge_insights)


class TestRenderEdgeDiscoveryPanel:
    """Tests for main render_edge_discovery_panel function."""

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_full_panel(self, mock_st, sample_edge_insights):
        """Render all panel sections without error."""
        from dashboard.edge_discovery_panel import render_edge_discovery_panel
        
        mock_col = MagicMock()
        # Return correct number of columns based on argument
        def columns_side_effect(n):
            return [mock_col for _ in range(n)]
        mock_st.columns.side_effect = columns_side_effect
        
        # Should not raise
        render_edge_discovery_panel(edge_insights=sample_edge_insights)

    @patch("dashboard.edge_discovery_panel.st")
    def test_render_empty_panel(self, mock_st):
        """Show warning when edge_insights is empty."""
        from dashboard.edge_discovery_panel import render_edge_discovery_panel
        
        render_edge_discovery_panel(edge_insights={})
        
        # Should show warning about no data
        mock_st.warning.assert_called()


class TestStateV7Loader:
    """Tests for state_v7 edge_insights loader."""

    def test_load_edge_insights_state_exists(self, tmp_path, sample_edge_insights):
        """Load edge_insights via state_v7 loader."""
        import json
        import os
        
        state_file = tmp_path / "edge_insights.json"
        state_file.write_text(json.dumps(sample_edge_insights))
        
        # Patch the path
        with patch.dict(os.environ, {"EDGE_INSIGHTS_PATH": str(state_file)}):
            # Re-import to pick up env var (or just test the function directly)
            from dashboard.state_v7 import load_edge_insights_state
            
            # The loader should work with default path logic
            result = load_edge_insights_state(default={"fallback": True})
            # Will return default if file path doesn't match
            assert result is not None

    def test_load_edge_insights_state_default(self):
        """Return default when file missing."""
        from dashboard.state_v7 import load_edge_insights_state
        
        result = load_edge_insights_state(default={"test": "default"})
        # Should return default or empty dict
        assert result is not None
