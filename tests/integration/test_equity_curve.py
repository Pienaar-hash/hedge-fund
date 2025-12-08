"""
Tests for Equity Curve (v7)

Verifies:
- mock PnL history → correct curve
- drawdown calculation correctness
- fallback on missing files
- export_equity_series() functionality
- load_equity_series() functionality
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
def _make_pnl_records() -> list[dict[str, Any]]:
    """Create sample PnL records for testing."""
    base_ts = time.time() - 86400  # 1 day ago
    return [
        {"_ts": base_ts, "realized_pnl": 100.0, "symbol": "BTCUSDT"},
        {"_ts": base_ts + 3600, "realized_pnl": -50.0, "symbol": "BTCUSDT"},
        {"_ts": base_ts + 7200, "realized_pnl": 200.0, "symbol": "ETHUSDT"},
        {"_ts": base_ts + 10800, "realized_pnl": -30.0, "symbol": "BTCUSDT"},
        {"_ts": base_ts + 14400, "realized_pnl": 80.0, "symbol": "BTCUSDT"},
        {"_ts": base_ts + 18000, "realized_pnl": -100.0, "symbol": "ETHUSDT"},
        {"_ts": base_ts + 21600, "realized_pnl": 150.0, "symbol": "BTCUSDT"},
    ]


def _make_equity_state() -> dict[str, Any]:
    """Create sample equity state for testing."""
    return {
        "timestamps": [1700000000, 1700003600, 1700007200, 1700010800],
        "equity": [100.0, 50.0, 250.0, 220.0],
        "pnl": [100.0, -50.0, 200.0, -30.0],
        "drawdown": [0.0, 0.5, 0.0, 0.12],
        "rolling_pnl": [100.0, 50.0, 250.0, 220.0],
        "ts": time.time(),
        "window_days": 30,
        "initial_equity": 0.0,
        "record_count": 4,
        "total_pnl": 220.0,
        "mean_pnl": 55.0,
        "std_pnl": 106.07,
        "max_drawdown": 0.5,
        "win_rate": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests for _compute_equity_series
# ---------------------------------------------------------------------------
def test_compute_equity_series_empty() -> None:
    """Empty records return empty series."""
    from execution.pnl_tracker import _compute_equity_series
    
    result = _compute_equity_series([])
    
    assert result["timestamps"] == []
    assert result["equity"] == []
    assert result["pnl"] == []
    assert result["drawdown"] == []


def test_compute_equity_series_single_record() -> None:
    """Single record produces correct series."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [{"_ts": 1700000000, "realized_pnl": 100.0}]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert len(result["timestamps"]) == 1
    assert result["equity"] == [100.0]
    assert result["pnl"] == [100.0]
    assert result["drawdown"] == [0.0]  # At peak, no drawdown


def test_compute_equity_series_cumulative_pnl() -> None:
    """PnL accumulates correctly."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},
        {"_ts": 1700003600, "realized_pnl": 50.0},
        {"_ts": 1700007200, "realized_pnl": -30.0},
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert result["equity"] == [100.0, 150.0, 120.0]
    assert result["pnl"] == [100.0, 50.0, -30.0]


def test_compute_equity_series_with_initial_equity() -> None:
    """Initial equity is respected."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [{"_ts": 1700000000, "realized_pnl": 100.0}]
    result = _compute_equity_series(records, initial_equity=1000.0)
    
    assert result["equity"] == [1100.0]


def test_compute_equity_series_drawdown_calculation() -> None:
    """Drawdown is calculated correctly."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},  # equity=100, peak=100, dd=0
        {"_ts": 1700003600, "realized_pnl": 100.0},  # equity=200, peak=200, dd=0
        {"_ts": 1700007200, "realized_pnl": -50.0},  # equity=150, peak=200, dd=0.25
        {"_ts": 1700010800, "realized_pnl": 100.0},  # equity=250, peak=250, dd=0
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert result["drawdown"][0] == 0.0  # At peak
    assert result["drawdown"][1] == 0.0  # New peak
    assert abs(result["drawdown"][2] - 0.25) < 0.01  # 25% drawdown
    assert result["drawdown"][3] == 0.0  # New peak


def test_compute_equity_series_sorts_by_timestamp() -> None:
    """Records are sorted by timestamp."""
    from execution.pnl_tracker import _compute_equity_series
    
    # Out of order records
    records = [
        {"_ts": 1700007200, "realized_pnl": 50.0},
        {"_ts": 1700000000, "realized_pnl": 100.0},
        {"_ts": 1700003600, "realized_pnl": -30.0},
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    # Should be sorted: 100, -30, 50 → cumulative: 100, 70, 120
    assert result["timestamps"] == [1700000000, 1700003600, 1700007200]
    assert result["equity"] == [100.0, 70.0, 120.0]


def test_compute_equity_series_handles_none_pnl() -> None:
    """Records with None PnL are treated as 0."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},
        {"_ts": 1700003600, "realized_pnl": None},
        {"_ts": 1700007200, "realized_pnl": 50.0},
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert result["equity"] == [100.0, 100.0, 150.0]


def test_compute_equity_series_skips_missing_timestamp() -> None:
    """Records without timestamp are skipped."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},
        {"realized_pnl": 50.0},  # No timestamp
        {"_ts": 1700003600, "realized_pnl": 30.0},
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert len(result["timestamps"]) == 2
    assert result["equity"] == [100.0, 130.0]


# ---------------------------------------------------------------------------
# Tests for _compute_rolling_returns
# ---------------------------------------------------------------------------
def test_compute_rolling_returns_empty() -> None:
    """Empty list returns empty."""
    from execution.pnl_tracker import _compute_rolling_returns
    
    assert _compute_rolling_returns([]) == []


def test_compute_rolling_returns_single() -> None:
    """Single value returns itself."""
    from execution.pnl_tracker import _compute_rolling_returns
    
    result = _compute_rolling_returns([100.0], window=5)
    assert result == [100.0]


def test_compute_rolling_returns_window() -> None:
    """Rolling window sums correctly."""
    from execution.pnl_tracker import _compute_rolling_returns
    
    pnl = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _compute_rolling_returns(pnl, window=3)
    
    # Window=3: [10], [10+20], [10+20+30], [20+30+40], [30+40+50]
    assert result == [10.0, 30.0, 60.0, 90.0, 120.0]


def test_compute_rolling_returns_zero_window() -> None:
    """Zero window returns empty."""
    from execution.pnl_tracker import _compute_rolling_returns
    
    assert _compute_rolling_returns([1, 2, 3], window=0) == []


# ---------------------------------------------------------------------------
# Tests for export_equity_series
# ---------------------------------------------------------------------------
def test_export_equity_series_creates_file() -> None:
    """export_equity_series creates equity.json."""
    from execution.pnl_tracker import export_equity_series, STATE_DIR, EQUITY_PATH
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.pnl_tracker.STATE_DIR", tmp_path), \
             patch("execution.pnl_tracker.EQUITY_PATH", tmp_path / "equity.json"), \
             patch("execution.pnl_tracker._recent_executed") as mock_recent:
            
            mock_recent.return_value = _make_pnl_records()
            
            result = export_equity_series(window_days=7)
            
            # File should be created
            assert (tmp_path / "equity.json").exists()
            
            # Result should have expected keys
            assert "timestamps" in result
            assert "equity" in result
            assert "pnl" in result
            assert "drawdown" in result
            assert "rolling_pnl" in result
            assert "total_pnl" in result
            assert "max_drawdown" in result


def test_export_equity_series_empty_records() -> None:
    """export_equity_series handles empty records."""
    from execution.pnl_tracker import export_equity_series
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.pnl_tracker.STATE_DIR", tmp_path), \
             patch("execution.pnl_tracker.EQUITY_PATH", tmp_path / "equity.json"), \
             patch("execution.pnl_tracker._recent_executed") as mock_recent:
            
            mock_recent.return_value = []
            
            result = export_equity_series()
            
            assert result["timestamps"] == []
            assert result["total_pnl"] == 0.0
            assert result["max_drawdown"] == 0.0


def test_export_equity_series_computes_stats() -> None:
    """export_equity_series computes summary statistics."""
    from execution.pnl_tracker import export_equity_series
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.pnl_tracker.STATE_DIR", tmp_path), \
             patch("execution.pnl_tracker.EQUITY_PATH", tmp_path / "equity.json"), \
             patch("execution.pnl_tracker._recent_executed") as mock_recent:
            
            # 3 wins, 2 losses
            records = [
                {"_ts": 1700000000, "realized_pnl": 100.0},
                {"_ts": 1700003600, "realized_pnl": -50.0},
                {"_ts": 1700007200, "realized_pnl": 75.0},
                {"_ts": 1700010800, "realized_pnl": -25.0},
                {"_ts": 1700014400, "realized_pnl": 50.0},
            ]
            mock_recent.return_value = records
            
            result = export_equity_series()
            
            assert result["total_pnl"] == 150.0  # 100-50+75-25+50
            assert result["record_count"] == 5
            assert result["win_rate"] == 0.6  # 3/5


# ---------------------------------------------------------------------------
# Tests for load_equity_series
# ---------------------------------------------------------------------------
def test_load_equity_series_missing_file() -> None:
    """load_equity_series returns empty dict for missing file."""
    from execution.pnl_tracker import load_equity_series
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.pnl_tracker.EQUITY_PATH", tmp_path / "nonexistent.json"):
            result = load_equity_series()
            assert result == {}


def test_load_equity_series_valid_file() -> None:
    """load_equity_series loads valid file."""
    from execution.pnl_tracker import load_equity_series
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        equity_file = tmp_path / "equity.json"
        
        test_data = {"timestamps": [1, 2, 3], "equity": [100, 200, 300]}
        equity_file.write_text(json.dumps(test_data))
        
        with patch("execution.pnl_tracker.EQUITY_PATH", equity_file):
            result = load_equity_series()
            assert result == test_data


def test_load_equity_series_invalid_json() -> None:
    """load_equity_series handles invalid JSON."""
    from execution.pnl_tracker import load_equity_series
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        equity_file = tmp_path / "equity.json"
        equity_file.write_text("not valid json{{{")
        
        with patch("execution.pnl_tracker.EQUITY_PATH", equity_file):
            result = load_equity_series()
            assert result == {}


# ---------------------------------------------------------------------------
# Tests for dashboard equity_panel
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.columns")
def test_equity_stats_renders(
    mock_columns: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Equity stats render correctly."""
    from dashboard.equity_panel import render_equity_stats
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
    
    equity_data = _make_equity_state()
    render_equity_stats(equity_data)
    
    # Should render markdown
    assert mock_markdown.called


@patch("streamlit.markdown")
@patch("streamlit.info")
@patch("streamlit.plotly_chart")
def test_equity_curve_renders_with_data(
    mock_plotly: MagicMock,
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Equity curve renders with valid data."""
    from dashboard.equity_panel import render_equity_curve
    
    equity_data = _make_equity_state()
    render_equity_curve(equity_data)
    
    # Should call plotly_chart (if plotly available)
    # or info if no data


@patch("streamlit.markdown")
@patch("streamlit.info")
def test_equity_curve_handles_empty_data(
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Equity curve shows info message for empty data."""
    from dashboard.equity_panel import render_equity_curve
    
    render_equity_curve({})
    
    mock_info.assert_called_with("No equity data available")


@patch("streamlit.markdown")
@patch("streamlit.info")
def test_equity_panel_handles_missing_state(
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Equity panel handles missing state gracefully."""
    from dashboard.equity_panel import load_equity_state
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("dashboard.equity_panel.EQUITY_PATH", tmp_path / "nonexistent.json"):
            result = load_equity_state()
            assert result == {}


@patch("streamlit.markdown")
@patch("streamlit.warning")
@patch("streamlit.info")
def test_equity_panel_full_render(
    mock_info: MagicMock,
    mock_warning: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Full equity panel renders without error."""
    from dashboard.equity_panel import render_equity_panel
    
    # With no data - should show warning
    with patch("dashboard.equity_panel.load_equity_state") as mock_load:
        mock_load.return_value = {}
        render_equity_panel()
        mock_warning.assert_called()


# ---------------------------------------------------------------------------
# Tests for drawdown calculation edge cases
# ---------------------------------------------------------------------------
def test_drawdown_never_negative() -> None:
    """Drawdown should never be negative."""
    from execution.pnl_tracker import _compute_equity_series
    
    # All positive PnL - should have zero drawdown throughout
    records = [
        {"_ts": 1700000000 + i * 3600, "realized_pnl": 100.0}
        for i in range(10)
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert all(dd >= 0 for dd in result["drawdown"])
    assert all(dd == 0 for dd in result["drawdown"])  # Always at new peak


def test_drawdown_max_is_one() -> None:
    """Drawdown should not exceed 1 (100%)."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},  # equity=100
        {"_ts": 1700003600, "realized_pnl": -99.0},  # equity=1, dd=0.99
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert all(dd <= 1.0 for dd in result["drawdown"])


def test_drawdown_recovery() -> None:
    """Drawdown resets when equity makes new high."""
    from execution.pnl_tracker import _compute_equity_series
    
    records = [
        {"_ts": 1700000000, "realized_pnl": 100.0},  # equity=100, peak=100
        {"_ts": 1700003600, "realized_pnl": -50.0},  # equity=50, peak=100, dd=0.5
        {"_ts": 1700007200, "realized_pnl": 60.0},   # equity=110, peak=110, dd=0
    ]
    result = _compute_equity_series(records, initial_equity=0.0)
    
    assert result["drawdown"][0] == 0.0
    assert result["drawdown"][1] == 0.5
    assert result["drawdown"][2] == 0.0  # Recovery to new peak


# ---------------------------------------------------------------------------
# Tests for overview panel integration
# ---------------------------------------------------------------------------
@patch("streamlit.markdown")
@patch("streamlit.info")
def test_overview_equity_section_renders(
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Overview equity section renders."""
    from dashboard.overview_panel import render_equity_curve_section
    
    with patch("dashboard.overview_panel.load_equity_state") as mock_load:
        mock_load.return_value = {}
        render_equity_curve_section({})
        mock_markdown.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.info")
@patch("streamlit.plotly_chart")
def test_overview_equity_section_with_data(
    mock_plotly: MagicMock,
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Overview equity section renders with data."""
    from dashboard.overview_panel import render_equity_curve_section
    
    state = {"equity": _make_equity_state()}
    render_equity_curve_section(state)
    
    mock_markdown.assert_called()
