"""
Tests for Router Health Gauge (v7)

Verifies:
- correct calculation of maker_ratio
- correct health_score derivation
- graceful handling of zero denominator
- dashboard rendering
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
def _make_router_events() -> list[dict[str, Any]]:
    """Create sample router events for testing."""
    return [
        {"order_type": "MAKER", "is_maker": True, "slippage_bps": 2.0},
        {"order_type": "MAKER", "is_maker": True, "slippage_bps": 3.0},
        {"order_type": "TAKER", "is_taker": True, "slippage_bps": 8.0, "fallback": True},
        {"order_type": "MAKER", "is_maker": True, "slippage_bps": 1.0},
        {"order_type": "TAKER", "is_taker": True, "slippage_bps": 5.0},
    ]


def _make_router_state() -> dict[str, Any]:
    """Create sample router state for testing."""
    return {
        "router_health_score": 0.75,
        "maker_ratio": 0.60,
        "fallback_ratio": 0.20,
        "reject_ratio": 0.05,
        "avg_slippage_bps": 3.8,
        "order_count": 5,
    }


# ---------------------------------------------------------------------------
# Tests for compute_maker_ratio
# ---------------------------------------------------------------------------
def test_maker_ratio_normal() -> None:
    """Maker ratio calculated correctly."""
    from execution.router_metrics import compute_maker_ratio
    
    # 3 makers, 2 takers = 60%
    assert compute_maker_ratio(3, 2) == 0.6
    
    # 5 makers, 5 takers = 50%
    assert compute_maker_ratio(5, 5) == 0.5
    
    # All makers
    assert compute_maker_ratio(10, 0) == 1.0
    
    # All takers
    assert compute_maker_ratio(0, 10) == 0.0


def test_maker_ratio_zero_denominator() -> None:
    """Maker ratio handles zero denominator gracefully."""
    from execution.router_metrics import compute_maker_ratio
    
    # No orders at all
    assert compute_maker_ratio(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Tests for compute_fallback_ratio
# ---------------------------------------------------------------------------
def test_fallback_ratio_normal() -> None:
    """Fallback ratio calculated correctly."""
    from execution.router_metrics import compute_fallback_ratio
    
    # 2 fallbacks out of 10
    assert compute_fallback_ratio(2, 10) == 0.2
    
    # No fallbacks
    assert compute_fallback_ratio(0, 10) == 0.0
    
    # All fallbacks
    assert compute_fallback_ratio(10, 10) == 1.0


def test_fallback_ratio_zero_denominator() -> None:
    """Fallback ratio handles zero denominator gracefully."""
    from execution.router_metrics import compute_fallback_ratio
    
    assert compute_fallback_ratio(0, 0) == 0.0
    assert compute_fallback_ratio(5, 0) == 0.0


# ---------------------------------------------------------------------------
# Tests for compute_reject_ratio
# ---------------------------------------------------------------------------
def test_reject_ratio_normal() -> None:
    """Reject ratio calculated correctly."""
    from execution.router_metrics import compute_reject_ratio
    
    assert compute_reject_ratio(1, 10) == 0.1
    assert compute_reject_ratio(0, 10) == 0.0


def test_reject_ratio_zero_denominator() -> None:
    """Reject ratio handles zero denominator gracefully."""
    from execution.router_metrics import compute_reject_ratio
    
    assert compute_reject_ratio(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Tests for compute_slippage_penalty
# ---------------------------------------------------------------------------
def test_slippage_penalty_thresholds() -> None:
    """Slippage penalty follows threshold rules."""
    from execution.router_metrics import compute_slippage_penalty
    
    # Low slippage (0-5 bps) = no penalty
    assert compute_slippage_penalty(0.0) == 0.0
    assert compute_slippage_penalty(5.0) == 0.0
    
    # Medium slippage (5-15 bps) = linear penalty
    penalty_10 = compute_slippage_penalty(10.0)
    assert 0.0 < penalty_10 < 0.1
    
    # High slippage (15+ bps) = higher penalty
    penalty_20 = compute_slippage_penalty(20.0)
    assert penalty_20 >= 0.1


def test_slippage_penalty_capped() -> None:
    """Slippage penalty is capped at reasonable maximum."""
    from execution.router_metrics import compute_slippage_penalty
    
    # Even very high slippage should be capped
    penalty = compute_slippage_penalty(100.0)
    assert penalty <= 0.31  # Allow for floating point


# ---------------------------------------------------------------------------
# Tests for compute_reject_penalty
# ---------------------------------------------------------------------------
def test_reject_penalty_thresholds() -> None:
    """Reject penalty follows threshold rules."""
    from execution.router_metrics import compute_reject_penalty
    
    # No rejects = no penalty
    assert compute_reject_penalty(0.0) == 0.0
    
    # Some rejects = proportional penalty
    assert compute_reject_penalty(0.1) > 0.0
    assert compute_reject_penalty(0.2) > compute_reject_penalty(0.1)


def test_reject_penalty_capped() -> None:
    """Reject penalty is capped at 0.2."""
    from execution.router_metrics import compute_reject_penalty
    
    # High reject ratio should be capped
    assert compute_reject_penalty(1.0) <= 0.2
    assert compute_reject_penalty(0.5) <= 0.2


# ---------------------------------------------------------------------------
# Tests for compute_router_health_score
# ---------------------------------------------------------------------------
def test_health_score_perfect() -> None:
    """Perfect health score with all makers, no penalties."""
    from execution.router_metrics import compute_router_health_score
    
    score = compute_router_health_score(
        maker_ratio=1.0,
        fallback_ratio=0.0,
        avg_slippage_bps=0.0,
        reject_ratio=0.0,
    )
    assert score == 1.0


def test_health_score_worst() -> None:
    """Worst health score with all takers and max penalties."""
    from execution.router_metrics import compute_router_health_score
    
    score = compute_router_health_score(
        maker_ratio=0.0,
        fallback_ratio=1.0,
        avg_slippage_bps=100.0,
        reject_ratio=1.0,
    )
    assert score == 0.0


def test_health_score_moderate() -> None:
    """Moderate health score calculation."""
    from execution.router_metrics import compute_router_health_score
    
    score = compute_router_health_score(
        maker_ratio=0.7,
        fallback_ratio=0.2,
        avg_slippage_bps=5.0,
        reject_ratio=0.05,
    )
    # Should be between 0 and 1
    assert 0.0 < score < 1.0
    # With 70% maker and small penalties, should be decent
    assert score > 0.5


def test_health_score_clamped() -> None:
    """Health score is clamped to [0, 1]."""
    from execution.router_metrics import compute_router_health_score
    
    # Even with invalid inputs, should be clamped
    score = compute_router_health_score(
        maker_ratio=1.5,  # Invalid but should handle
        fallback_ratio=0.0,
        avg_slippage_bps=0.0,
        reject_ratio=0.0,
    )
    assert score <= 1.0


def test_health_score_formula() -> None:
    """Health score follows formula: base - penalties."""
    from execution.router_metrics import compute_router_health_score
    
    # With known values, verify formula
    # base = 0.8, fallback_penalty = 0.1 * 0.3 = 0.03
    # slippage_penalty ~= 0, reject_penalty ~= 0
    score = compute_router_health_score(
        maker_ratio=0.8,
        fallback_ratio=0.1,
        avg_slippage_bps=3.0,  # Below threshold
        reject_ratio=0.0,
    )
    # Should be around 0.8 - 0.03 = 0.77
    assert abs(score - 0.77) < 0.02


# ---------------------------------------------------------------------------
# Tests for build_router_health_snapshot
# ---------------------------------------------------------------------------
def test_build_snapshot_empty() -> None:
    """Snapshot handles empty events."""
    from execution.router_metrics import build_router_health_snapshot
    
    snapshot = build_router_health_snapshot(router_events=[])
    
    assert snapshot["router_health_score"] == 0.0
    assert snapshot["maker_ratio"] == 0.0
    assert snapshot["fallback_ratio"] == 0.0
    assert snapshot["order_count"] == 0


def test_build_snapshot_with_events() -> None:
    """Snapshot correctly processes events."""
    from execution.router_metrics import build_router_health_snapshot
    
    events = _make_router_events()
    snapshot = build_router_health_snapshot(router_events=events)
    
    # Should have 5 orders
    assert snapshot["order_count"] == 5
    
    # 3 makers, 2 takers = 60% maker ratio
    assert snapshot["maker_ratio"] == 0.6
    
    # 1 fallback out of 5 = 20%
    assert snapshot["fallback_ratio"] == 0.2
    
    # Health score should be computed
    assert 0.0 <= snapshot["router_health_score"] <= 1.0


def test_build_snapshot_slippage() -> None:
    """Snapshot correctly calculates average slippage."""
    from execution.router_metrics import build_router_health_snapshot
    
    events = _make_router_events()
    snapshot = build_router_health_snapshot(router_events=events)
    
    # Average of [2.0, 3.0, 8.0, 1.0, 5.0] = 3.8
    assert abs(snapshot["avg_slippage_bps"] - 3.8) < 0.01


# ---------------------------------------------------------------------------
# Tests for dashboard router_gauge
# ---------------------------------------------------------------------------
def test_health_score_color_thresholds() -> None:
    """Health score colors follow thresholds."""
    from dashboard.router_gauge import _get_health_score_color, COLOR_OK, COLOR_WARN, COLOR_DEGRADED
    
    assert _get_health_score_color(0.90) == COLOR_OK
    assert _get_health_score_color(0.80) == COLOR_OK
    assert _get_health_score_color(0.70) == COLOR_WARN
    assert _get_health_score_color(0.50) == COLOR_WARN
    assert _get_health_score_color(0.40) == COLOR_DEGRADED
    assert _get_health_score_color(0.0) == COLOR_DEGRADED


def test_health_status_from_score() -> None:
    """Health status derived from health score."""
    from dashboard.router_gauge import _get_health_status
    
    # With health score
    status, _ = _get_health_status({"router_health_score": 0.85})
    assert status == "HEALTHY"
    
    status, _ = _get_health_status({"router_health_score": 0.60})
    assert status == "MARGINAL"
    
    status, _ = _get_health_status({"router_health_score": 0.30})
    assert status == "POOR"


def test_health_status_empty_state() -> None:
    """Health status handles empty state."""
    from dashboard.router_gauge import _get_health_status, COLOR_NEUTRAL
    
    status, color = _get_health_status({})
    assert status == "UNKNOWN"
    assert color == COLOR_NEUTRAL
    
    status, color = _get_health_status(None)  # type: ignore
    assert status == "UNKNOWN"


@patch("streamlit.markdown")
@patch("streamlit.warning")
@patch("streamlit.columns")
def test_router_gauge_renders(
    mock_columns: MagicMock,
    mock_warning: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Router gauge renders without errors."""
    from dashboard.router_gauge import render_router_gauge
    
    mock_col = MagicMock()
    mock_col.__enter__ = MagicMock(return_value=mock_col)
    mock_col.__exit__ = MagicMock(return_value=False)
    mock_columns.return_value = [mock_col, mock_col]
    
    router_state = _make_router_state()
    render_router_gauge(router_state)
    
    mock_markdown.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.warning")
def test_router_gauge_handles_empty(
    mock_warning: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Router gauge handles empty state."""
    from dashboard.router_gauge import render_router_gauge
    
    render_router_gauge({})
    mock_warning.assert_called()


@patch("streamlit.markdown")
@patch("streamlit.warning")
def test_router_circle_gauge_renders(
    mock_warning: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Circle gauge renders with SVG."""
    from dashboard.router_gauge import render_router_circle_gauge
    
    router_state = _make_router_state()
    render_router_circle_gauge(router_state)
    
    # Should render markdown with SVG
    mock_markdown.assert_called()
    call_args = mock_markdown.call_args[0][0]
    assert "<svg" in call_args


@patch("streamlit.markdown")
def test_router_gauge_compact_renders(
    mock_markdown: MagicMock,
) -> None:
    """Compact gauge renders."""
    from dashboard.router_gauge import render_router_gauge_compact
    
    router_state = _make_router_state()
    render_router_gauge_compact(router_state)
    
    mock_markdown.assert_called()


# ---------------------------------------------------------------------------
# Tests for state_publish integration
# ---------------------------------------------------------------------------
def test_write_router_health_adds_score() -> None:
    """write_router_health_state adds health score if missing."""
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.state_publish._write_state_file") as mock_write:
            from execution.state_publish import write_router_health_state
            
            payload = {
                "maker_ratio": 0.8,
                "fallback_ratio": 0.1,
                "avg_slippage_bps": 3.0,
                "reject_ratio": 0.0,
            }
            
            write_router_health_state(payload, tmp_path)
            
            # Should have called _write_state_file
            mock_write.assert_called_once()
            written_payload = mock_write.call_args[0][1]
            
            # Should have added router_health_score
            assert "router_health_score" in written_payload
            assert 0.0 <= written_payload["router_health_score"] <= 1.0


def test_write_router_health_preserves_existing_score() -> None:
    """write_router_health_state preserves existing health score."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch("execution.state_publish._write_state_file") as mock_write:
            from execution.state_publish import write_router_health_state
            
            payload = {
                "router_health_score": 0.99,  # Pre-existing
                "maker_ratio": 0.5,
            }
            
            write_router_health_state(payload, tmp_path)
            
            written_payload = mock_write.call_args[0][1]
            
            # Should preserve existing score
            assert written_payload["router_health_score"] == 0.99
