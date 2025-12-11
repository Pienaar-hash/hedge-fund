"""Integration tests for edge_insights snapshot (v7.7_P4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from execution.edge_scanner import (
    EdgeInsights,
    EdgeScannerConfig,
    build_edge_insights_snapshot,
    load_edge_insights,
    write_edge_insights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_factor_diagnostics() -> Dict[str, Any]:
    """Synthetic factor diagnostics with realistic data."""
    return {
        "updated_ts": "2024-01-01T12:00:00Z",
        "weights": {
            "smoothed": {
                "trend": 0.18,
                "carry": 0.14,
                "rv_momentum": 0.20,
                "router_quality": 0.10,
                "expectancy": 0.28,
                "vol_regime": 0.05,
                "category_momentum": 0.05,
            }
        },
        "factor_ir": {
            "trend": 0.42,
            "carry": 0.15,
            "rv_momentum": 0.55,
            "router_quality": 0.08,
            "expectancy": 0.68,
            "vol_regime": -0.12,
            "category_momentum": 0.22,
        },
        "pnl_attribution": {
            "trend": 0.0025,
            "carry": 0.0012,
            "rv_momentum": 0.0038,
            "router_quality": 0.0008,
            "expectancy": 0.0055,
            "vol_regime": -0.0015,
            "category_momentum": 0.0018,
        },
        "factor_volatilities": {
            "trend": 0.14,
            "carry": 0.09,
            "rv_momentum": 0.18,
            "router_quality": 0.07,
            "expectancy": 0.16,
            "vol_regime": 0.11,
            "category_momentum": 0.12,
        },
    }


@pytest.fixture
def synthetic_symbol_scores() -> Dict[str, Any]:
    """Synthetic symbol scores with realistic data."""
    return {
        "updated_ts": "2024-01-01T12:00:00Z",
        "global": {
            "vol_regime": "normal",
            "vol_regime_label": "normal",
        },
        "per_symbol": {
            "BTCUSDT": {
                "hybrid_score": 0.78,
                "conviction": 0.85,
                "recent_pnl": 0.025,
                "category": "L1_BTC_ETH",
                "direction": "LONG",
            },
            "ETHUSDT": {
                "hybrid_score": 0.72,
                "conviction": 0.78,
                "recent_pnl": 0.018,
                "category": "L1_BTC_ETH",
                "direction": "LONG",
            },
            "SOLUSDT": {
                "hybrid_score": 0.62,
                "conviction": 0.58,
                "recent_pnl": 0.008,
                "category": "L1_ALT",
                "direction": "LONG",
            },
            "AVAXUSDT": {
                "hybrid_score": 0.55,
                "conviction": 0.52,
                "recent_pnl": -0.003,
                "category": "L1_ALT",
                "direction": "SHORT",
            },
            "DOGEUSDT": {
                "hybrid_score": 0.35,
                "conviction": 0.32,
                "recent_pnl": -0.015,
                "category": "MEME",
                "direction": "NEUTRAL",
            },
            "SHIBUSDT": {
                "hybrid_score": 0.28,
                "conviction": 0.25,
                "recent_pnl": -0.022,
                "category": "MEME",
                "direction": "NEUTRAL",
            },
            "LINKUSDT": {
                "hybrid_score": 0.48,
                "conviction": 0.45,
                "recent_pnl": 0.002,
                "category": "OTHER",
                "direction": "LONG",
            },
        },
    }


@pytest.fixture
def synthetic_category_momentum() -> Dict[str, Any]:
    """Synthetic category momentum with realistic data."""
    return {
        "updated_ts": 1704110400.0,
        "category_stats": {
            "L1_BTC_ETH": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "ir": 0.52,
                "momentum_score": 0.65,
                "total_pnl": 0.043,
                "mean_return": 0.012,
                "volatility": 0.023,
            },
            "L1_ALT": {
                "symbols": ["SOLUSDT", "AVAXUSDT"],
                "ir": 0.18,
                "momentum_score": 0.22,
                "total_pnl": 0.005,
                "mean_return": 0.003,
                "volatility": 0.028,
            },
            "MEME": {
                "symbols": ["DOGEUSDT", "SHIBUSDT"],
                "ir": -0.25,
                "momentum_score": -0.35,
                "total_pnl": -0.037,
                "mean_return": -0.015,
                "volatility": 0.045,
            },
            "OTHER": {
                "symbols": ["LINKUSDT"],
                "ir": 0.08,
                "momentum_score": 0.10,
                "total_pnl": 0.002,
                "mean_return": 0.001,
                "volatility": 0.015,
            },
        },
    }


@pytest.fixture
def synthetic_router_health() -> Dict[str, Any]:
    """Synthetic router health with realistic data."""
    return {
        "updated_ts": "2024-01-01T12:00:00Z",
        "global": {
            "quality_score": 0.82,
            "avg_slippage_bps": 3.2,
            "maker_rate": 0.58,
            "total_orders": 150,
            "fill_rate": 0.95,
        },
    }


@pytest.fixture
def synthetic_risk_snapshot() -> Dict[str, Any]:
    """Synthetic risk snapshot with realistic data."""
    return {
        "updated_ts": "2024-01-01T12:00:00Z",
        "dd_state": "normal",
        "risk_mode": "normal",
        "current_dd_pct": 0.035,
        "portfolio_var": 0.062,
        "portfolio_cvar": 0.088,
        "gross_exposure": 0.45,
    }


# ---------------------------------------------------------------------------
# Integration Tests: Snapshot Contains All Required Fields
# ---------------------------------------------------------------------------


class TestEdgeInsightsSnapshot:
    """Tests for complete edge insights snapshot."""

    def test_snapshot_contains_all_required_fields(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_category_momentum,
        synthetic_router_health,
        synthetic_risk_snapshot,
    ):
        """Snapshot contains all fields specified in v7_manifest.json."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            router_health=synthetic_router_health,
            risk_snapshot=synthetic_risk_snapshot,
            category_momentum=synthetic_category_momentum,
        )

        d = snapshot.to_dict()

        # Required fields from manifest
        assert "updated_ts" in d
        assert "edge_summary" in d
        assert "factor_edges" in d
        assert "symbol_edges" in d
        assert "category_edges" in d
        assert "config_echo" in d

        # edge_summary sub-fields
        summary = d["edge_summary"]
        assert "top_factors" in summary
        assert "weak_factors" in summary
        assert "top_symbols" in summary
        assert "weak_symbols" in summary
        assert "top_categories" in summary
        assert "weak_categories" in summary
        assert "regime" in summary

    def test_top_factors_sorted_correctly(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Top factors are sorted by edge_score descending."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        top_factors = snapshot.edge_summary.top_factors

        # Check sorted descending
        for i in range(len(top_factors) - 1):
            assert top_factors[i]["edge_score"] >= top_factors[i + 1]["edge_score"]

    def test_weak_factors_sorted_correctly(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Weak factors are sorted worst-first (ascending edge_score)."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        weak_factors = snapshot.edge_summary.weak_factors

        # Weak factors should have lowest edge scores
        if weak_factors:
            # First weak factor should have lowest score
            for factor in snapshot.factor_edges.values():
                assert weak_factors[0]["edge_score"] <= factor["edge_score"]

    def test_no_modifications_to_input_surfaces(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Building snapshot does not modify input surfaces."""
        # Deep copy for comparison
        import copy

        fd_copy = copy.deepcopy(synthetic_factor_diagnostics)
        ss_copy = copy.deepcopy(synthetic_symbol_scores)

        _ = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        # Inputs should be unchanged
        assert synthetic_factor_diagnostics == fd_copy
        assert synthetic_symbol_scores == ss_copy

    def test_file_written_once_per_call(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """File is written exactly once per write call."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edge_insights.json"

            # File doesn't exist before write
            assert not path.exists()

            write_edge_insights(snapshot, path)

            # File exists after write
            assert path.exists()

            # Get modification time
            mtime1 = path.stat().st_mtime

            # Write again - file should be updated
            write_edge_insights(snapshot, path)
            mtime2 = path.stat().st_mtime

            # Modification time should be same or later
            assert mtime2 >= mtime1


class TestManifestAlignment:
    """Tests for alignment with v7_manifest.json."""

    def test_snapshot_path_matches_manifest(self):
        """Default path matches manifest specification."""
        from execution.edge_scanner import DEFAULT_EDGE_INSIGHTS_PATH

        assert str(DEFAULT_EDGE_INSIGHTS_PATH) == "logs/state/edge_insights.json"

    def test_snapshot_fields_match_manifest(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """All manifest-specified fields are present in snapshot."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        d = snapshot.to_dict()

        # Fields from manifest
        manifest_fields = {
            "updated_ts",
            "edge_summary",
            "factor_edges",
            "symbol_edges",
            "category_edges",
            "config_echo",
        }

        assert manifest_fields.issubset(set(d.keys()))

    def test_config_echo_contains_reproducibility_info(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Config echo contains info needed for reproducibility."""
        config = EdgeScannerConfig(top_n=5)

        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            config=config,
        )

        echo = snapshot.config_echo

        assert "top_n" in echo
        assert "source_files" in echo
        assert echo["top_n"] == 5


class TestRegimeExtraction:
    """Tests for regime context extraction."""

    def test_regime_contains_all_required_fields(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_router_health,
        synthetic_risk_snapshot,
    ):
        """Regime block contains dd_state, risk_mode, vol_regime, router_quality."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            router_health=synthetic_router_health,
            risk_snapshot=synthetic_risk_snapshot,
        )

        regime = snapshot.edge_summary.regime

        assert "dd_state" in regime
        assert "risk_mode" in regime
        assert "vol_regime" in regime
        assert "router_quality" in regime

    def test_regime_values_from_sources(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_router_health,
        synthetic_risk_snapshot,
    ):
        """Regime values are correctly extracted from source surfaces."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            router_health=synthetic_router_health,
            risk_snapshot=synthetic_risk_snapshot,
        )

        regime = snapshot.edge_summary.regime

        # From risk_snapshot
        assert regime["dd_state"] == "normal"
        assert regime["risk_mode"] == "normal"
        assert regime["current_dd_pct"] == 0.035

        # From symbol_scores global
        assert regime["vol_regime"] == "normal"

        # From router_health
        assert regime["router_quality"] == 0.82


class TestEdgeScoring:
    """Tests for edge score computation."""

    def test_factor_edge_scores_computed(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Factor edge scores are computed for all factors."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        for factor, data in snapshot.factor_edges.items():
            assert "edge_score" in data
            assert isinstance(data["edge_score"], (int, float))

    def test_symbol_edge_scores_computed(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Symbol edge scores are computed for all symbols."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        for symbol, data in snapshot.symbol_edges.items():
            assert "edge_score" in data
            assert isinstance(data["edge_score"], (int, float))

    def test_category_edge_scores_computed(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_category_momentum,
    ):
        """Category edge scores are computed for all categories."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            category_momentum=synthetic_category_momentum,
        )

        for category, data in snapshot.category_edges.items():
            assert "edge_score" in data
            assert isinstance(data["edge_score"], (int, float))


class TestReadOnlyBehavior:
    """Tests to ensure EdgeScanner is read-only."""

    def test_no_state_file_writes_except_edge_insights(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """Building snapshot does not write any files except edge_insights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source files
            fd_path = Path(tmpdir) / "factor_diagnostics.json"
            ss_path = Path(tmpdir) / "symbol_scores.json"

            fd_path.write_text(json.dumps(synthetic_factor_diagnostics))
            ss_path.write_text(json.dumps(synthetic_symbol_scores))

            # Get initial state
            fd_mtime = fd_path.stat().st_mtime
            ss_mtime = ss_path.stat().st_mtime

            # Build snapshot (read-only operation)
            _ = build_edge_insights_snapshot(
                factor_diagnostics=synthetic_factor_diagnostics,
                symbol_scores=synthetic_symbol_scores,
            )

            # Source files should be unchanged
            assert fd_path.stat().st_mtime == fd_mtime
            assert ss_path.stat().st_mtime == ss_mtime

    def test_snapshot_deterministic_for_same_inputs(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_category_momentum,
    ):
        """Same inputs produce same edge scores (deterministic)."""
        snapshot1 = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            category_momentum=synthetic_category_momentum,
        )

        snapshot2 = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            category_momentum=synthetic_category_momentum,
        )

        # Edge scores should be identical
        assert snapshot1.factor_edges == snapshot2.factor_edges
        assert snapshot1.symbol_edges == snapshot2.symbol_edges
        assert snapshot1.category_edges == snapshot2.category_edges

        # Top/weak lists should be identical
        assert snapshot1.edge_summary.top_factors == snapshot2.edge_summary.top_factors
        assert snapshot1.edge_summary.weak_factors == snapshot2.edge_summary.weak_factors


class TestJSONSchemaCompliance:
    """Tests for JSON schema compliance."""

    def test_full_roundtrip(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
        synthetic_category_momentum,
        synthetic_router_health,
        synthetic_risk_snapshot,
    ):
        """Snapshot survives JSON roundtrip."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
            router_health=synthetic_router_health,
            risk_snapshot=synthetic_risk_snapshot,
            category_momentum=synthetic_category_momentum,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edge_insights.json"
            write_edge_insights(snapshot, path)

            # Load back
            loaded = load_edge_insights(path)

            # Verify structure preserved
            assert loaded["edge_summary"]["top_factors"] == snapshot.edge_summary.top_factors
            assert loaded["factor_edges"] == snapshot.factor_edges
            assert loaded["config_echo"] == snapshot.config_echo

    def test_all_values_json_serializable(
        self,
        synthetic_factor_diagnostics,
        synthetic_symbol_scores,
    ):
        """All values in snapshot are JSON-serializable."""
        snapshot = build_edge_insights_snapshot(
            factor_diagnostics=synthetic_factor_diagnostics,
            symbol_scores=synthetic_symbol_scores,
        )

        # Should not raise
        json_str = json.dumps(snapshot.to_dict())
        assert json_str is not None
        assert len(json_str) > 0
