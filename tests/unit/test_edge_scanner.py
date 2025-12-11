"""Unit tests for execution/edge_scanner.py (v7.7_P4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from execution.edge_scanner import (
    EdgeInsights,
    EdgeScannerConfig,
    EdgeSummary,
    build_edge_insights_snapshot,
    compute_category_edges,
    compute_factor_edges,
    compute_symbol_edges,
    extract_regime_context,
    load_edge_insights,
    load_factor_diagnostics,
    load_risk_snapshot,
    load_router_health,
    load_symbol_scores,
    write_edge_insights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_factor_diagnostics() -> Dict[str, Any]:
    """Sample factor diagnostics state for testing."""
    return {
        "updated_ts": "2024-01-01T00:00:00Z",
        "weights": {
            "smoothed": {
                "trend": 0.20,
                "carry": 0.15,
                "rv_momentum": 0.18,
                "router_quality": 0.12,
                "expectancy": 0.25,
                "vol_regime": 0.10,
            }
        },
        "factor_ir": {
            "trend": 0.35,
            "carry": 0.10,
            "rv_momentum": 0.45,
            "router_quality": 0.05,
            "expectancy": 0.55,
            "vol_regime": -0.10,
        },
        "pnl_attribution": {
            "trend": 0.002,
            "carry": 0.001,
            "rv_momentum": 0.003,
            "router_quality": 0.0005,
            "expectancy": 0.004,
            "vol_regime": -0.001,
        },
        "factor_volatilities": {
            "trend": 0.15,
            "carry": 0.10,
            "rv_momentum": 0.20,
            "router_quality": 0.08,
            "expectancy": 0.18,
            "vol_regime": 0.12,
        },
    }


@pytest.fixture
def sample_symbol_scores() -> Dict[str, Any]:
    """Sample symbol scores state for testing."""
    return {
        "updated_ts": "2024-01-01T00:00:00Z",
        "global": {
            "vol_regime": "normal",
        },
        "per_symbol": {
            "BTCUSDT": {
                "hybrid_score": 0.75,
                "conviction": 0.8,
                "recent_pnl": 0.02,
                "category": "L1_BTC_ETH",
                "direction": "LONG",
            },
            "ETHUSDT": {
                "hybrid_score": 0.65,
                "conviction": 0.7,
                "recent_pnl": 0.01,
                "category": "L1_BTC_ETH",
                "direction": "LONG",
            },
            "SOLUSDT": {
                "hybrid_score": 0.55,
                "conviction": 0.5,
                "recent_pnl": -0.005,
                "category": "L1_ALT",
                "direction": "SHORT",
            },
            "DOGEUSDT": {
                "hybrid_score": 0.30,
                "conviction": 0.3,
                "recent_pnl": -0.02,
                "category": "MEME",
                "direction": "NEUTRAL",
            },
            "SHIBUSDT": {
                "hybrid_score": 0.25,
                "conviction": 0.2,
                "recent_pnl": -0.03,
                "category": "MEME",
                "direction": "NEUTRAL",
            },
        },
    }


@pytest.fixture
def sample_router_health() -> Dict[str, Any]:
    """Sample router health state for testing."""
    return {
        "global": {
            "quality_score": 0.85,
            "avg_slippage_bps": 2.5,
            "maker_rate": 0.65,
        },
    }


@pytest.fixture
def sample_risk_snapshot() -> Dict[str, Any]:
    """Sample risk snapshot state for testing."""
    return {
        "dd_state": "normal",
        "risk_mode": "normal",
        "current_dd_pct": 0.02,
        "portfolio_var": 0.05,
        "portfolio_cvar": 0.08,
    }


@pytest.fixture
def sample_category_momentum() -> Dict[str, Any]:
    """Sample category momentum data for testing."""
    return {
        "category_stats": {
            "L1_BTC_ETH": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "ir": 0.45,
                "momentum_score": 0.6,
                "total_pnl": 0.03,
                "mean_return": 0.01,
                "volatility": 0.02,
            },
            "L1_ALT": {
                "symbols": ["SOLUSDT", "AVAXUSDT"],
                "ir": 0.20,
                "momentum_score": 0.3,
                "total_pnl": 0.01,
                "mean_return": 0.005,
                "volatility": 0.03,
            },
            "MEME": {
                "symbols": ["DOGEUSDT", "SHIBUSDT"],
                "ir": -0.15,
                "momentum_score": -0.2,
                "total_pnl": -0.05,
                "mean_return": -0.01,
                "volatility": 0.05,
            },
        },
    }


@pytest.fixture
def default_config() -> EdgeScannerConfig:
    """Default edge scanner config."""
    return EdgeScannerConfig()


# ---------------------------------------------------------------------------
# Data Structure Tests
# ---------------------------------------------------------------------------


def test_edge_summary_dataclass():
    """EdgeSummary stores expected fields."""
    summary = EdgeSummary(
        top_factors=[{"factor": "expectancy", "edge_score": 0.8}],
        weak_factors=[{"factor": "vol_regime", "edge_score": -0.5}],
        top_symbols=[{"symbol": "BTCUSDT", "edge_score": 0.7}],
        weak_symbols=[{"symbol": "SHIBUSDT", "edge_score": -0.6}],
        top_categories=[{"category": "L1_BTC_ETH", "edge_score": 0.6}],
        weak_categories=[{"category": "MEME", "edge_score": -0.4}],
        regime={"vol_regime": "normal", "dd_state": "normal"},
    )

    assert len(summary.top_factors) == 1
    assert summary.top_factors[0]["factor"] == "expectancy"
    assert summary.regime["vol_regime"] == "normal"


def test_edge_summary_to_dict():
    """EdgeSummary.to_dict() produces correct structure."""
    summary = EdgeSummary(
        top_factors=[{"factor": "trend"}],
        regime={"vol_regime": "high"},
    )

    d = summary.to_dict()
    assert "top_factors" in d
    assert "weak_factors" in d
    assert "regime" in d
    assert d["regime"]["vol_regime"] == "high"


def test_edge_insights_dataclass():
    """EdgeInsights stores expected fields."""
    insights = EdgeInsights(
        updated_ts="2024-01-01T00:00:00Z",
        edge_summary=EdgeSummary(),
        factor_edges={"trend": {"ir": 0.3}},
        symbol_edges={"BTCUSDT": {"hybrid_score": 0.7}},
        category_edges={"L1_BTC_ETH": {"momentum": 0.5}},
        config_echo={"top_n": 3},
    )

    assert insights.updated_ts == "2024-01-01T00:00:00Z"
    assert "trend" in insights.factor_edges
    assert "BTCUSDT" in insights.symbol_edges


def test_edge_insights_to_dict():
    """EdgeInsights.to_dict() produces correct JSON-serializable structure."""
    insights = EdgeInsights(
        updated_ts="2024-01-01T00:00:00Z",
        factor_edges={"trend": {"ir": 0.3}},
    )

    d = insights.to_dict()
    assert "updated_ts" in d
    assert "edge_summary" in d
    assert "factor_edges" in d
    assert d["factor_edges"]["trend"]["ir"] == 0.3


def test_edge_scanner_config_defaults():
    """EdgeScannerConfig has sensible defaults."""
    cfg = EdgeScannerConfig()

    assert cfg.enabled is True
    assert cfg.top_n == 3
    assert cfg.factor_ir_threshold == 0.1
    assert cfg.symbol_score_threshold == 0.3


# ---------------------------------------------------------------------------
# State Loader Tests
# ---------------------------------------------------------------------------


def test_load_factor_diagnostics_missing_file():
    """Returns empty dict for missing file."""
    result = load_factor_diagnostics("/nonexistent/path.json")
    assert result == {}


def test_load_symbol_scores_missing_file():
    """Returns empty dict for missing file."""
    result = load_symbol_scores("/nonexistent/path.json")
    assert result == {}


def test_load_router_health_missing_file():
    """Returns empty dict for missing file."""
    result = load_router_health("/nonexistent/path.json")
    assert result == {}


def test_load_risk_snapshot_missing_file():
    """Returns empty dict for missing file."""
    result = load_risk_snapshot("/nonexistent/path.json")
    assert result == {}


def test_load_factor_diagnostics_from_file(sample_factor_diagnostics):
    """Loads factor diagnostics from temp file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_factor_diagnostics, f)
        temp_path = f.name

    result = load_factor_diagnostics(temp_path)

    assert "weights" in result
    assert "factor_ir" in result
    Path(temp_path).unlink()


def test_load_invalid_json():
    """Returns empty dict for invalid JSON."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("{invalid json")
        temp_path = f.name

    result = load_factor_diagnostics(temp_path)
    assert result == {}
    Path(temp_path).unlink()


# ---------------------------------------------------------------------------
# Factor Edge Computation Tests
# ---------------------------------------------------------------------------


def test_compute_factor_edges_basic(sample_factor_diagnostics, default_config):
    """Computes factor edges from diagnostics."""
    edges, top, weak = compute_factor_edges(sample_factor_diagnostics, default_config)

    # Should have edges for all factors
    assert "trend" in edges
    assert "expectancy" in edges
    assert "vol_regime" in edges

    # Each factor should have edge_score
    for f, data in edges.items():
        assert "edge_score" in data
        assert "ir" in data
        assert "weight" in data
        assert "pnl_contrib" in data

    # Top 3 factors should be sorted by edge_score descending
    assert len(top) == 3
    assert top[0]["edge_score"] >= top[1]["edge_score"]

    # Weak factors should be sorted worst-first
    assert len(weak) == 3


def test_compute_factor_edges_empty_input(default_config):
    """Returns empty results for empty input."""
    edges, top, weak = compute_factor_edges({}, default_config)

    assert edges == {}
    assert top == []
    assert weak == []


def test_compute_factor_edges_ranking(sample_factor_diagnostics, default_config):
    """Expectancy should rank higher than vol_regime based on IR/PnL."""
    edges, top, weak = compute_factor_edges(sample_factor_diagnostics, default_config)

    # expectancy has highest IR (0.55) and PnL (0.004)
    # vol_regime has lowest IR (-0.10) and PnL (-0.001)
    expectancy_score = edges["expectancy"]["edge_score"]
    vol_regime_score = edges["vol_regime"]["edge_score"]

    assert expectancy_score > vol_regime_score


# ---------------------------------------------------------------------------
# Symbol Edge Computation Tests
# ---------------------------------------------------------------------------


def test_compute_symbol_edges_basic(sample_symbol_scores, default_config):
    """Computes symbol edges from scores."""
    edges, top, weak = compute_symbol_edges(sample_symbol_scores, default_config)

    # Should have edges for all symbols
    assert "BTCUSDT" in edges
    assert "ETHUSDT" in edges
    assert "DOGEUSDT" in edges

    # Each symbol should have edge_score
    for sym, data in edges.items():
        assert "edge_score" in data
        assert "hybrid_score" in data
        assert "conviction" in data

    # Top 3 symbols
    assert len(top) == 3
    assert top[0]["edge_score"] >= top[1]["edge_score"]


def test_compute_symbol_edges_empty_input(default_config):
    """Returns empty results for empty input."""
    edges, top, weak = compute_symbol_edges({}, default_config)

    assert edges == {}
    assert top == []
    assert weak == []


def test_compute_symbol_edges_ranking(sample_symbol_scores, default_config):
    """BTCUSDT should rank higher than SHIBUSDT based on scores/PnL."""
    edges, top, weak = compute_symbol_edges(sample_symbol_scores, default_config)

    btc_score = edges["BTCUSDT"]["edge_score"]
    shib_score = edges["SHIBUSDT"]["edge_score"]

    assert btc_score > shib_score


# ---------------------------------------------------------------------------
# Category Edge Computation Tests
# ---------------------------------------------------------------------------


def test_compute_category_edges_basic(sample_category_momentum, default_config):
    """Computes category edges from momentum data."""
    edges, top, weak = compute_category_edges(sample_category_momentum, default_config)

    # Should have edges for all categories
    assert "L1_BTC_ETH" in edges
    assert "L1_ALT" in edges
    assert "MEME" in edges

    # Each category should have edge_score
    for cat, data in edges.items():
        assert "edge_score" in data
        assert "ir" in data
        assert "momentum" in data
        assert "sharpe_proxy" in data


def test_compute_category_edges_empty_input(default_config):
    """Returns empty results for empty input."""
    edges, top, weak = compute_category_edges({}, default_config)

    assert edges == {}
    assert top == []
    assert weak == []


def test_compute_category_edges_ranking(sample_category_momentum, default_config):
    """L1_BTC_ETH should rank higher than MEME based on IR/momentum."""
    edges, top, weak = compute_category_edges(sample_category_momentum, default_config)

    l1_score = edges["L1_BTC_ETH"]["edge_score"]
    meme_score = edges["MEME"]["edge_score"]

    assert l1_score > meme_score


# ---------------------------------------------------------------------------
# Regime Context Tests
# ---------------------------------------------------------------------------


def test_extract_regime_context(
    sample_risk_snapshot, sample_router_health, sample_symbol_scores
):
    """Extracts regime context from state surfaces."""
    regime = extract_regime_context(
        sample_risk_snapshot, sample_router_health, sample_symbol_scores
    )

    assert regime["dd_state"] == "normal"
    assert regime["risk_mode"] == "normal"
    assert regime["vol_regime"] == "normal"
    assert regime["router_quality"] == 0.85
    assert regime["current_dd_pct"] == 0.02


def test_extract_regime_context_empty_inputs():
    """Handles empty inputs gracefully."""
    regime = extract_regime_context({}, {}, {})

    assert regime["dd_state"] == "normal"  # default
    assert regime["risk_mode"] == "normal"  # default
    assert regime["router_quality"] == 0.8  # default


# ---------------------------------------------------------------------------
# Snapshot Builder Tests
# ---------------------------------------------------------------------------


def test_build_edge_insights_snapshot(
    sample_factor_diagnostics,
    sample_symbol_scores,
    sample_router_health,
    sample_risk_snapshot,
    sample_category_momentum,
):
    """Builds complete edge insights snapshot."""
    snapshot = build_edge_insights_snapshot(
        factor_diagnostics=sample_factor_diagnostics,
        symbol_scores=sample_symbol_scores,
        router_health=sample_router_health,
        risk_snapshot=sample_risk_snapshot,
        category_momentum=sample_category_momentum,
    )

    assert isinstance(snapshot, EdgeInsights)
    assert snapshot.updated_ts != ""
    assert len(snapshot.factor_edges) > 0
    assert len(snapshot.symbol_edges) > 0
    assert len(snapshot.category_edges) > 0
    assert "top_factors" in snapshot.edge_summary.to_dict()


def test_build_edge_insights_snapshot_empty_inputs():
    """Handles empty inputs gracefully."""
    snapshot = build_edge_insights_snapshot(
        factor_diagnostics={},
        symbol_scores={},
        router_health={},
        risk_snapshot={},
        category_momentum={},
    )

    assert isinstance(snapshot, EdgeInsights)
    assert snapshot.factor_edges == {}
    assert snapshot.symbol_edges == {}
    assert snapshot.category_edges == {}


def test_build_edge_insights_snapshot_config_echo(
    sample_factor_diagnostics,
    sample_symbol_scores,
):
    """Config echo captures parameters for reproducibility."""
    config = EdgeScannerConfig(top_n=5, factor_ir_threshold=0.2)

    snapshot = build_edge_insights_snapshot(
        factor_diagnostics=sample_factor_diagnostics,
        symbol_scores=sample_symbol_scores,
        config=config,
    )

    assert snapshot.config_echo["top_n"] == 5
    assert snapshot.config_echo["factor_ir_threshold"] == 0.2


# ---------------------------------------------------------------------------
# Writer Tests
# ---------------------------------------------------------------------------


def test_write_edge_insights(sample_factor_diagnostics, sample_symbol_scores):
    """Writes edge insights to file."""
    snapshot = build_edge_insights_snapshot(
        factor_diagnostics=sample_factor_diagnostics,
        symbol_scores=sample_symbol_scores,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "edge_insights.json"
        write_edge_insights(snapshot, path)

        assert path.exists()

        # Verify contents
        loaded = json.loads(path.read_text())
        assert "updated_ts" in loaded
        assert "edge_summary" in loaded
        assert "factor_edges" in loaded


def test_load_edge_insights():
    """Loads edge insights from file."""
    data = {
        "updated_ts": "2024-01-01T00:00:00Z",
        "edge_summary": {"top_factors": []},
        "factor_edges": {"trend": {"ir": 0.3}},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        temp_path = f.name

    loaded = load_edge_insights(temp_path)

    assert loaded["updated_ts"] == "2024-01-01T00:00:00Z"
    assert "factor_edges" in loaded
    Path(temp_path).unlink()


def test_load_edge_insights_missing_file():
    """Returns empty dict for missing file."""
    result = load_edge_insights("/nonexistent/path.json")
    assert result == {}


# ---------------------------------------------------------------------------
# JSON Schema Compliance Tests
# ---------------------------------------------------------------------------


def test_snapshot_json_serializable(
    sample_factor_diagnostics,
    sample_symbol_scores,
    sample_category_momentum,
):
    """Snapshot is fully JSON serializable."""
    snapshot = build_edge_insights_snapshot(
        factor_diagnostics=sample_factor_diagnostics,
        symbol_scores=sample_symbol_scores,
        category_momentum=sample_category_momentum,
    )

    # Should not raise
    json_str = json.dumps(snapshot.to_dict())
    assert json_str is not None

    # Round-trip
    loaded = json.loads(json_str)
    assert loaded["updated_ts"] == snapshot.updated_ts


def test_snapshot_required_fields(sample_factor_diagnostics, sample_symbol_scores):
    """Snapshot contains all required fields per manifest."""
    snapshot = build_edge_insights_snapshot(
        factor_diagnostics=sample_factor_diagnostics,
        symbol_scores=sample_symbol_scores,
    )

    d = snapshot.to_dict()

    # Required fields per v7_manifest.json
    assert "updated_ts" in d
    assert "edge_summary" in d
    assert "factor_edges" in d
    assert "symbol_edges" in d
    assert "category_edges" in d
    assert "config_echo" in d


def test_edge_summary_required_fields(sample_factor_diagnostics):
    """EdgeSummary contains all required fields."""
    edges, top, weak = compute_factor_edges(sample_factor_diagnostics, EdgeScannerConfig())

    summary = EdgeSummary(
        top_factors=top,
        weak_factors=weak,
        regime={"vol_regime": "normal"},
    )

    d = summary.to_dict()

    assert "top_factors" in d
    assert "weak_factors" in d
    assert "top_symbols" in d
    assert "weak_symbols" in d
    assert "top_categories" in d
    assert "weak_categories" in d
    assert "regime" in d
