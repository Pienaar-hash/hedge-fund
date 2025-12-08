"""
Tests for Dashboard Overview Panel (v7)

Verifies:
- overview_panel loads without errors
- missing fields handled gracefully
- risk_mode displayed
- regime heatmap renders
- router gauge renders
- portfolio exposure renders
- positions table renders
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
def _make_sample_state() -> dict[str, Any]:
    """Create sample state for testing."""
    return {
        "risk_snapshot": {
            "nav": 100000.0,
            "exposure": 50000.0,
            "margin_used": 12500.0,
            "dd_frac": 0.05,
            "daily_loss_frac": 0.02,
            "risk_mode": "OK",
            "risk_mode_reason": "All systems nominal",
            "risk_mode_score": 0.15,
            "nav_age_s": 5.0,
        },
        "regimes": {
            "atr_regime": 1,
            "dd_regime": 0,
            "atr_value": 0.025,
            "dd_value": 0.05,
            "regime_label": "Normal/Low",
            "ts": 1700000000,
        },
        "router_health": {
            "fill_rate": 0.92,
            "maker_ratio": 0.85,
            "orders_1h": 15,
            "fills_1h": 14,
            "rejects_1h": 1,
            "avg_latency_ms": 45.0,
            "degraded": False,
        },
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "size": 0.5,
                "entry_price": 42000.0,
                "mark_price": 43000.0,
                "unrealized_pnl": 500.0,
                "pnl_pct": 2.38,
            },
            {
                "symbol": "ETHUSDT",
                "side": "short",
                "size": 5.0,
                "entry_price": 2200.0,
                "mark_price": 2150.0,
                "unrealized_pnl": 250.0,
                "pnl_pct": 2.27,
            },
        ],
    }


def _make_empty_state() -> dict[str, Any]:
    """Create empty state for edge case testing."""
    return {}


def _make_partial_state() -> dict[str, Any]:
    """Create partial state missing some fields."""
    return {
        "risk_snapshot": {
            "nav": 50000.0,
            # Missing exposure, margin_used, etc.
        },
        # Missing regimes
        # Missing router_health
        "positions": [],
    }


# ---------------------------------------------------------------------------
# Mock Streamlit for testing
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.divider")
@patch("streamlit.subheader")
@patch("streamlit.header")
@patch("streamlit.caption")
@patch("streamlit.dataframe")
@patch("streamlit.progress")
def test_overview_panel_loads_without_errors(
    mock_progress: MagicMock,
    mock_dataframe: MagicMock,
    mock_caption: MagicMock,
    mock_header: MagicMock,
    mock_subheader: MagicMock,
    mock_divider: MagicMock,
    mock_warning: MagicMock,
    mock_info: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """overview loads without errors."""
    from dashboard.overview_panel import render_overview_panel
    
    # Setup mock columns to return proper column context managers
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    # Return different length lists based on call count
    mock_columns.side_effect = lambda n: [mock_col for _ in range(n)] if isinstance(n, int) else [mock_col, mock_col]
    
    state = _make_sample_state()
    
    # Should not raise
    render_overview_panel(state)
    
    # Verify header was called
    mock_header.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.divider")
@patch("streamlit.subheader")
@patch("streamlit.header")
@patch("streamlit.caption")
@patch("streamlit.dataframe")
@patch("streamlit.progress")
def test_overview_panel_handles_missing_fields(
    mock_progress: MagicMock,
    mock_dataframe: MagicMock,
    mock_caption: MagicMock,
    mock_header: MagicMock,
    mock_subheader: MagicMock,
    mock_divider: MagicMock,
    mock_warning: MagicMock,
    mock_info: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """missing fields handled gracefully."""
    from dashboard.overview_panel import render_overview_panel
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.side_effect = lambda n: [mock_col for _ in range(n)] if isinstance(n, int) else [mock_col, mock_col]
    
    # Empty state should not crash
    render_overview_panel(_make_empty_state())
    
    # Partial state should not crash
    render_overview_panel(_make_partial_state())


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.divider")
@patch("streamlit.subheader")
@patch("streamlit.header")
@patch("streamlit.caption")
@patch("streamlit.dataframe")
@patch("streamlit.progress")
def test_risk_mode_displayed(
    mock_progress: MagicMock,
    mock_dataframe: MagicMock,
    mock_caption: MagicMock,
    mock_header: MagicMock,
    mock_subheader: MagicMock,
    mock_divider: MagicMock,
    mock_warning: MagicMock,
    mock_info: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """risk_mode displayed."""
    from dashboard.overview_panel import render_overview_panel
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.side_effect = lambda n: [mock_col for _ in range(n)] if isinstance(n, int) else [mock_col, mock_col]
    
    state = _make_sample_state()
    render_overview_panel(state)
    
    # Risk mode should be rendered via risk_health_card
    # We can verify markdown was called with risk-related content
    assert mock_markdown.called or mock_header.called


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.divider")
@patch("streamlit.header")
@patch("streamlit.caption")
@patch("streamlit.dataframe")
@patch("streamlit.progress")
def test_regime_heatmap_renders(
    mock_progress: MagicMock,
    mock_dataframe: MagicMock,
    mock_caption: MagicMock,
    mock_header: MagicMock,
    mock_divider: MagicMock,
    mock_warning: MagicMock,
    mock_info: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """regime heatmap renders."""
    from dashboard.overview_panel import render_overview_panel
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.side_effect = lambda n: [mock_col for _ in range(n)] if isinstance(n, int) else [mock_col, mock_col]
    
    state = _make_sample_state()
    render_overview_panel(state)
    
    # Should have called markdown (used by regime panel)
    assert mock_markdown.called


# ---------------------------------------------------------------------------
# Router Gauge Tests
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.warning")
@patch("streamlit.progress")
def test_router_gauge_renders(
    mock_progress: MagicMock,
    mock_warning: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Router gauge renders with valid data."""
    from dashboard.router_gauge import render_router_gauge
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    # Router gauge now uses [1, 2] columns layout
    mock_columns.side_effect = lambda spec: [mock_col for _ in range(len(spec) if isinstance(spec, list) else spec)]
    
    router_state = {
        "router_health_score": 0.85,
        "maker_ratio": 0.80,
        "fallback_ratio": 0.10,
        "reject_ratio": 0.05,
        "avg_slippage_bps": 3.0,
    }
    
    render_router_gauge(router_state)
    
    # Should render header and metrics
    assert mock_markdown.called


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.warning")
@patch("streamlit.progress")
def test_router_gauge_handles_empty_state(
    mock_progress: MagicMock,
    mock_warning: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Router gauge handles empty state gracefully."""
    from dashboard.router_gauge import render_router_gauge
    
    render_router_gauge({})
    
    # Should show warning
    mock_warning.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.warning")
@patch("streamlit.progress")
def test_router_gauge_degraded_status(
    mock_progress: MagicMock,
    mock_warning: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Router gauge shows degraded status correctly."""
    from dashboard.router_gauge import render_router_gauge, _get_health_status
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
    
    degraded_state = {
        "fill_rate": 0.50,
        "degraded": True,
    }
    
    status, color = _get_health_status(degraded_state)
    assert status == "DEGRADED"
    assert color == "#d94a4a"


def test_router_health_status_thresholds() -> None:
    """Router health status thresholds are correct."""
    from dashboard.router_gauge import _get_health_status
    
    # Healthy
    status, _ = _get_health_status({"fill_rate": 0.95, "degraded": False})
    assert status == "HEALTHY"
    
    # Marginal
    status, _ = _get_health_status({"fill_rate": 0.75, "degraded": False})
    assert status == "MARGINAL"
    
    # Poor
    status, _ = _get_health_status({"fill_rate": 0.50, "degraded": False})
    assert status == "POOR"
    
    # Degraded overrides fill rate
    status, _ = _get_health_status({"fill_rate": 0.95, "degraded": True})
    assert status == "DEGRADED"
    
    # Unknown
    status, _ = _get_health_status({})
    assert status == "UNKNOWN"


def test_fill_rate_color_thresholds() -> None:
    """Fill rate color thresholds are correct."""
    from dashboard.router_gauge import _get_fill_rate_color, COLOR_OK, COLOR_WARN, COLOR_DEGRADED
    
    assert _get_fill_rate_color(0.95) == COLOR_OK
    assert _get_fill_rate_color(0.90) == COLOR_OK
    assert _get_fill_rate_color(0.85) == COLOR_WARN
    assert _get_fill_rate_color(0.70) == COLOR_WARN
    assert _get_fill_rate_color(0.60) == COLOR_DEGRADED
    assert _get_fill_rate_color(0.30) == COLOR_DEGRADED


# ---------------------------------------------------------------------------
# Portfolio Exposure Tests
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
def test_portfolio_exposure_renders(
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Portfolio exposure summary renders correctly."""
    from dashboard.overview_panel import render_portfolio_exposure
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col, mock_col]
    
    state = _make_sample_state()
    render_portfolio_exposure(state)
    
    # Should render header
    mock_markdown.assert_called()
    # Should render 3 metrics
    assert mock_metric.call_count >= 3


@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
def test_portfolio_exposure_handles_zero_nav(
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Portfolio exposure handles zero NAV gracefully."""
    from dashboard.overview_panel import render_portfolio_exposure
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col, mock_col]
    
    state = {"risk_snapshot": {"nav": 0, "exposure": 0, "margin_used": 0}}
    
    # Should not raise
    render_portfolio_exposure(state)


# ---------------------------------------------------------------------------
# Open Positions Tests
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.dataframe")
@patch("streamlit.info")
def test_open_positions_renders(
    mock_info: MagicMock,
    mock_dataframe: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Open positions table renders correctly."""
    from dashboard.overview_panel import render_open_positions
    
    state = _make_sample_state()
    render_open_positions(state)
    
    # Should render header and dataframe
    mock_markdown.assert_called()
    mock_dataframe.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.dataframe")
@patch("streamlit.info")
def test_open_positions_handles_empty(
    mock_info: MagicMock,
    mock_dataframe: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Open positions handles empty list gracefully."""
    from dashboard.overview_panel import render_open_positions
    
    state = {"positions": []}
    render_open_positions(state)
    
    # Should show info message
    mock_info.assert_called_with("No open positions")


@patch("streamlit.markdown")
@patch("streamlit.dataframe")
@patch("streamlit.info")
def test_open_positions_handles_missing_positions(
    mock_info: MagicMock,
    mock_dataframe: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Open positions handles missing positions key gracefully."""
    from dashboard.overview_panel import render_open_positions
    
    render_open_positions({})
    
    # Should show info message
    mock_info.assert_called_with("No open positions")


# ---------------------------------------------------------------------------
# Equity Curve Section Tests
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.info")
def test_equity_curve_section_renders(
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Equity curve section renders."""
    from dashboard.overview_panel import render_equity_curve_section
    
    with patch("dashboard.overview_panel.load_equity_state") as mock_load:
        mock_load.return_value = {}
        render_equity_curve_section({})
    
    # Should render header and info
    mock_markdown.assert_called()
    mock_info.assert_called()


# ---------------------------------------------------------------------------
# Risk Mode Color Tests
# ---------------------------------------------------------------------------
def test_risk_mode_colors() -> None:
    """Risk mode colors are correct."""
    from dashboard.overview_panel import (
        _get_risk_mode_color,
        COLOR_OK,
        COLOR_WARN,
        COLOR_DEFENSIVE,
        COLOR_HALTED,
        COLOR_NEUTRAL,
    )
    
    assert _get_risk_mode_color("OK") == COLOR_OK
    assert _get_risk_mode_color("ok") == COLOR_OK  # Case insensitive
    assert _get_risk_mode_color("WARN") == COLOR_WARN
    assert _get_risk_mode_color("DEFENSIVE") == COLOR_DEFENSIVE
    assert _get_risk_mode_color("HALTED") == COLOR_HALTED
    assert _get_risk_mode_color("UNKNOWN") == COLOR_NEUTRAL
    assert _get_risk_mode_color("") == COLOR_NEUTRAL
    assert _get_risk_mode_color(None) == COLOR_NEUTRAL  # type: ignore


# ---------------------------------------------------------------------------
# Compact Overview Tests
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.metric")
@patch("streamlit.columns")
@patch("streamlit.info")
@patch("streamlit.warning")
@patch("streamlit.caption")
def test_overview_compact_renders(
    mock_caption: MagicMock,
    mock_warning: MagicMock,
    mock_info: MagicMock,
    mock_columns: MagicMock,
    mock_metric: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Compact overview renders correctly."""
    from dashboard.overview_panel import render_overview_compact
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col]
    
    state = _make_sample_state()
    render_overview_compact(state)
    
    # Should call columns for side-by-side layout
    mock_columns.assert_called()
