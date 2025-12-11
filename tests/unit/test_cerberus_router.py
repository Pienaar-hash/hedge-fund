"""
Unit tests for Cerberus Multi-Strategy Portfolio Router — v7.8_P8
"""
import tempfile
from pathlib import Path

import pytest

from execution.cerberus_router import (
    STRATEGY_HEADS,
    CerberusConfig,
    CerberusSignals,
    CerberusState,
    HeadMetrics,
    StrategyHeadState,
    compute_decay_component,
    compute_edge_component,
    compute_head_signal_score,
    compute_health_component,
    compute_meta_component,
    compute_regime_multiplier,
    compute_rv_component,
    compute_trend_direction,
    compute_universe_component,
    extract_cerberus_signals,
    get_cerberus_all_multipliers,
    get_cerberus_alpha_router_adjustment,
    get_cerberus_conviction_multiplier,
    get_cerberus_crossfire_multiplier,
    get_cerberus_factor_weight_overlay,
    get_cerberus_head_multiplier,
    get_cerberus_prospector_multiplier,
    get_cerberus_universe_category_multiplier,
    load_cerberus_config,
    load_cerberus_state,
    normalize_multipliers,
    run_cerberus_step,
    update_head_multiplier,
    write_cerberus_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> CerberusConfig:
    """Create default Cerberus config."""
    return CerberusConfig()


@pytest.fixture
def enabled_config() -> CerberusConfig:
    """Create enabled Cerberus config."""
    return CerberusConfig(enabled=True)


@pytest.fixture
def sample_signals() -> CerberusSignals:
    """Create sample signals for testing."""
    return CerberusSignals(
        regime="TREND_UP",
        regime_probs={"TREND_UP": 0.7, "MEAN_REVERT": 0.2, "CHOPPY": 0.1},
        avg_symbol_survival=0.85,
        avg_factor_survival=0.80,
        overall_alpha_health=0.75,
        meta_active=True,
        meta_factor_weights={"trend": 1.1, "carry": 0.95},
        meta_conviction_strength=1.05,
        meta_category_overlays={"MEME": 1.2, "L1_MAJOR": 1.0},
        strategy_health_score=0.72,
        top_factors=["trend", "momentum"],
        weak_factors=["value"],
        factor_edges={"trend": 0.65, "carry": 0.45},
        category_edges={"MEME": 0.15, "L1_MAJOR": 0.05},
        crossfire_active=True,
        avg_pair_edge=0.58,
        pairs_eligible=3,
        prospector_active=True,
        candidates_count=5,
        avg_candidate_score=0.62,
        universe_size=8,
        effective_max_size=12,
        category_diversity=4,
    )


@pytest.fixture
def sample_state() -> CerberusState:
    """Create sample Cerberus state."""
    heads = {}
    for head in STRATEGY_HEADS:
        heads[head] = HeadMetrics(
            multiplier=1.0,
            ema_score=0.5,
            signal_score=0.5,
            regime_component=1.0,
            decay_component=1.0,
            meta_component=1.0,
            edge_component=0.5,
            health_component=0.5,
            trend_direction="flat",
            last_update_ts="2025-01-01T00:00:00+00:00",
            sample_count=10,
        )

    return CerberusState(
        updated_ts="2025-01-01T00:00:00+00:00",
        cycle_count=10,
        head_state=StrategyHeadState(heads=heads, mean_multiplier=1.0, normalized=True),
        regime="TREND_UP",
        regime_probs={"TREND_UP": 0.7},
        overall_health=0.72,
        avg_decay_survival=0.85,
        meta_scheduler_active=True,
        notes=["Test note"],
        errors=[],
        meta={"config_version": "v7.8_P8"},
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestCerberusConfig:
    """Tests for CerberusConfig."""

    def test_default_config(self) -> None:
        """Test default config creation."""
        cfg = CerberusConfig()
        assert cfg.enabled is False
        assert cfg.learning_rate == 0.05
        assert cfg.bounds == {"min": 0.25, "max": 2.0}
        assert len(cfg.strategy_heads) == 6
        assert cfg.sentinel_x_integration is True
        assert cfg.decay_integration is True
        assert cfg.meta_scheduler_integration is True

    def test_config_post_init_adds_missing_heads(self) -> None:
        """Test that post_init adds missing strategy heads."""
        cfg = CerberusConfig(strategy_heads={"TREND": 1.5})
        assert "TREND" in cfg.strategy_heads
        assert "MEAN_REVERT" in cfg.strategy_heads
        assert cfg.strategy_heads["TREND"] == 1.5
        assert cfg.strategy_heads["MEAN_REVERT"] == 1.0

    def test_config_clamps_bounds(self) -> None:
        """Test that bounds are clamped to valid range."""
        cfg = CerberusConfig(bounds={"min": 0.01, "max": 5.0})
        assert cfg.bounds["min"] >= 0.10
        assert cfg.bounds["max"] <= 3.0

    def test_config_clamps_learning_rate(self) -> None:
        """Test that learning rate is clamped."""
        cfg = CerberusConfig(learning_rate=0.0001)
        assert cfg.learning_rate >= 0.001

        cfg2 = CerberusConfig(learning_rate=1.0)
        assert cfg2.learning_rate <= 0.5

    def test_load_config_from_dict(self) -> None:
        """Test loading config from strategy_config dict."""
        strategy_config = {
            "cerberus_router": {
                "enabled": True,
                "learning_rate": 0.10,
                "strategy_heads": {"TREND": 1.5, "MEAN_REVERT": 0.8},
            }
        }
        cfg = load_cerberus_config(strategy_config=strategy_config)
        assert cfg.enabled is True
        assert cfg.learning_rate == 0.10
        assert cfg.strategy_heads["TREND"] == 1.5

    def test_load_config_missing_block(self) -> None:
        """Test loading config when cerberus_router block is missing."""
        cfg = load_cerberus_config(strategy_config={})
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------


class TestCerberusState:
    """Tests for state management."""

    def test_head_metrics_to_dict(self) -> None:
        """Test HeadMetrics serialization."""
        metrics = HeadMetrics(
            multiplier=1.25,
            ema_score=0.65,
            signal_score=0.70,
            trend_direction="up",
        )
        d = metrics.to_dict()
        assert d["multiplier"] == 1.25
        assert d["ema_score"] == 0.65
        assert d["trend_direction"] == "up"

    def test_strategy_head_state_to_dict(self) -> None:
        """Test StrategyHeadState serialization."""
        heads = {"TREND": HeadMetrics(multiplier=1.5)}
        state = StrategyHeadState(heads=heads, mean_multiplier=1.0)
        d = state.to_dict()
        assert "heads" in d
        assert "TREND" in d["heads"]
        assert d["mean_multiplier"] == 1.0

    def test_cerberus_state_to_dict(self, sample_state: CerberusState) -> None:
        """Test full state serialization."""
        d = sample_state.to_dict()
        assert d["updated_ts"] == "2025-01-01T00:00:00+00:00"
        assert d["cycle_count"] == 10
        assert "head_state" in d
        assert d["regime"] == "TREND_UP"

    def test_state_round_trip(self, sample_state: CerberusState) -> None:
        """Test state save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cerberus_state.json"
            assert write_cerberus_state(sample_state, path)

            loaded = load_cerberus_state(path)
            assert loaded is not None
            assert loaded.cycle_count == sample_state.cycle_count
            assert loaded.regime == sample_state.regime

    def test_load_state_missing_file(self) -> None:
        """Test loading state from missing file."""
        state = load_cerberus_state("/nonexistent/path.json")
        assert state is None


# ---------------------------------------------------------------------------
# Signal Extraction Tests
# ---------------------------------------------------------------------------


class TestSignalExtraction:
    """Tests for signal extraction."""

    def test_extract_empty_signals(self) -> None:
        """Test extraction with empty inputs."""
        signals = extract_cerberus_signals(
            sentinel_x={},
            alpha_decay={},
            meta_scheduler={},
            edge_insights={},
            cross_pair={},
            alpha_miner={},
            universe_optimizer={},
        )
        assert signals.regime == "UNKNOWN"
        assert signals.avg_symbol_survival == 1.0
        assert signals.meta_active is False

    def test_extract_sentinel_x_signals(self) -> None:
        """Test extraction of Sentinel-X signals."""
        sentinel_x = {
            "primary_regime": "BREAKOUT",
            "smoothed_probs": {"BREAKOUT": 0.6, "TREND_UP": 0.3},
        }
        signals = extract_cerberus_signals(
            sentinel_x=sentinel_x,
            alpha_decay={},
            meta_scheduler={},
            edge_insights={},
            cross_pair={},
            alpha_miner={},
            universe_optimizer={},
        )
        assert signals.regime == "BREAKOUT"
        assert signals.regime_probs["BREAKOUT"] == 0.6

    def test_extract_alpha_decay_signals(self) -> None:
        """Test extraction of alpha decay signals."""
        alpha_decay = {
            "avg_symbol_survival": 0.75,
            "avg_factor_survival": 0.80,
            "overall_alpha_health": 0.65,
        }
        signals = extract_cerberus_signals(
            sentinel_x={},
            alpha_decay=alpha_decay,
            meta_scheduler={},
            edge_insights={},
            cross_pair={},
            alpha_miner={},
            universe_optimizer={},
        )
        assert signals.avg_symbol_survival == 0.75
        assert signals.avg_factor_survival == 0.80
        assert signals.overall_alpha_health == 0.65

    def test_extract_crossfire_signals(self) -> None:
        """Test extraction of Crossfire signals."""
        cross_pair = {
            "pairs_eligible": 5,
            "pair_edges": {
                "BTCUSDT-ETHUSDT": {"edge_score": 0.7},
                "SOLUSDT-AVAXUSDT": {"edge_score": 0.5},
            },
        }
        signals = extract_cerberus_signals(
            sentinel_x={},
            alpha_decay={},
            meta_scheduler={},
            edge_insights={},
            cross_pair=cross_pair,
            alpha_miner={},
            universe_optimizer={},
        )
        assert signals.crossfire_active is True
        assert signals.pairs_eligible == 5
        assert signals.avg_pair_edge == 0.6  # (0.7 + 0.5) / 2


# ---------------------------------------------------------------------------
# Signal Scoring Tests
# ---------------------------------------------------------------------------


class TestSignalScoring:
    """Tests for head signal scoring."""

    def test_compute_regime_multiplier(self, enabled_config: CerberusConfig) -> None:
        """Test regime multiplier computation."""
        mult = compute_regime_multiplier("TREND", "TREND_UP", enabled_config)
        assert mult == 1.3  # From default config

        mult_mr = compute_regime_multiplier("MEAN_REVERT", "TREND_UP", enabled_config)
        assert mult_mr == 0.7

    def test_compute_regime_multiplier_unknown_regime(
        self, enabled_config: CerberusConfig
    ) -> None:
        """Test regime multiplier for unknown regime."""
        mult = compute_regime_multiplier("TREND", "UNKNOWN", enabled_config)
        assert mult == 1.0

    def test_compute_decay_component_trend(self, sample_signals: CerberusSignals) -> None:
        """Test decay component for TREND head."""
        comp = compute_decay_component("TREND", sample_signals)
        assert comp == sample_signals.avg_factor_survival

    def test_compute_decay_component_vol_harvest(
        self, sample_signals: CerberusSignals
    ) -> None:
        """Test decay component for VOL_HARVEST (neutral)."""
        comp = compute_decay_component("VOL_HARVEST", sample_signals)
        assert comp == 1.0

    def test_compute_meta_component_inactive(self) -> None:
        """Test meta component when meta-scheduler is inactive."""
        signals = CerberusSignals(meta_active=False)
        comp = compute_meta_component("TREND", signals)
        assert comp == 1.0

    def test_compute_edge_component_rv(self, sample_signals: CerberusSignals) -> None:
        """Test edge component for RELATIVE_VALUE."""
        comp = compute_edge_component("RELATIVE_VALUE", sample_signals)
        assert comp == sample_signals.avg_pair_edge

    def test_compute_health_component(self, sample_signals: CerberusSignals) -> None:
        """Test health component."""
        comp = compute_health_component(sample_signals)
        assert comp == sample_signals.strategy_health_score

    def test_compute_universe_component(self, sample_signals: CerberusSignals) -> None:
        """Test universe component."""
        comp = compute_universe_component("TREND", sample_signals)
        # utilization = 8/12 = 0.667, diversity_bonus = 4*0.05 = 0.2, clamped to 1.0
        expected = min(1.0, 8 / 12 + 0.2)
        assert abs(comp - expected) < 0.01

    def test_compute_rv_component_non_rv(self, sample_signals: CerberusSignals) -> None:
        """Test RV component for non-RV head."""
        comp = compute_rv_component("TREND", sample_signals)
        assert comp == 0.5

    def test_compute_head_signal_score(
        self, enabled_config: CerberusConfig, sample_signals: CerberusSignals
    ) -> None:
        """Test full signal score computation."""
        score, components = compute_head_signal_score(
            "TREND", sample_signals, enabled_config
        )
        assert 0.0 <= score <= 1.0
        assert "regime_component" in components
        assert "decay_component" in components


# ---------------------------------------------------------------------------
# Multiplier Update Tests
# ---------------------------------------------------------------------------


class TestMultiplierUpdate:
    """Tests for multiplier update logic."""

    def test_update_head_multiplier(self) -> None:
        """Test basic multiplier update."""
        bounds = {"min": 0.25, "max": 2.0}
        new_mult = update_head_multiplier(
            prev_multiplier=1.0,
            signal_score=0.75,
            learning_rate=0.10,
            bounds=bounds,
        )
        # target = 0.75 * 2 = 1.5
        # new = 1.0 + 0.10 * (1.5 - 1.0) = 1.05
        assert abs(new_mult - 1.05) < 0.001

    def test_update_head_multiplier_clamping_max(self) -> None:
        """Test multiplier clamping at max bound."""
        bounds = {"min": 0.25, "max": 1.5}
        new_mult = update_head_multiplier(
            prev_multiplier=1.4,
            signal_score=1.0,  # target = 2.0
            learning_rate=0.50,  # aggressive
            bounds=bounds,
        )
        assert new_mult == 1.5

    def test_update_head_multiplier_clamping_min(self) -> None:
        """Test multiplier clamping at min bound."""
        bounds = {"min": 0.5, "max": 2.0}
        new_mult = update_head_multiplier(
            prev_multiplier=0.6,
            signal_score=0.0,  # target = 0.0
            learning_rate=0.50,
            bounds=bounds,
        )
        assert new_mult == 0.5

    def test_normalize_multipliers(self) -> None:
        """Test multiplier normalization to mean = 1.0."""
        multipliers = {"A": 1.5, "B": 0.5, "C": 1.0}
        normalized = normalize_multipliers(multipliers)

        mean = sum(normalized.values()) / len(normalized)
        assert abs(mean - 1.0) < 0.001

    def test_normalize_multipliers_empty(self) -> None:
        """Test normalization with empty dict."""
        result = normalize_multipliers({})
        assert result == {}

    def test_compute_trend_direction_up(self) -> None:
        """Test trend direction detection - up."""
        direction = compute_trend_direction(0.7, 0.5, threshold=0.05)
        assert direction == "up"

    def test_compute_trend_direction_down(self) -> None:
        """Test trend direction detection - down."""
        direction = compute_trend_direction(0.3, 0.5, threshold=0.05)
        assert direction == "down"

    def test_compute_trend_direction_flat(self) -> None:
        """Test trend direction detection - flat."""
        direction = compute_trend_direction(0.52, 0.5, threshold=0.05)
        assert direction == "flat"


# ---------------------------------------------------------------------------
# Main Step Function Tests
# ---------------------------------------------------------------------------


class TestRunCerberusStep:
    """Tests for the main step function."""

    def test_run_step_disabled(self, default_config: CerberusConfig) -> None:
        """Test step with disabled config."""
        state = run_cerberus_step(default_config)
        assert state.updated_ts != ""
        assert "Cerberus disabled" in state.notes[0]

    def test_run_step_enabled_cold_start(
        self, enabled_config: CerberusConfig
    ) -> None:
        """Test step with enabled config, no previous state."""
        state = run_cerberus_step(enabled_config)
        assert state.cycle_count == 1
        assert len(state.head_state.heads) == 6
        # All multipliers should be normalized (mean ~1.0)
        mean = sum(h.multiplier for h in state.head_state.heads.values()) / 6
        assert abs(mean - 1.0) < 0.01

    def test_run_step_with_signals(
        self, enabled_config: CerberusConfig, sample_state: CerberusState
    ) -> None:
        """Test step with full signals."""
        sentinel_x = {"primary_regime": "TREND_UP", "smoothed_probs": {"TREND_UP": 0.7}}
        alpha_decay = {"avg_symbol_survival": 0.85}
        edge_insights = {"strategy_health": {"health_score": 0.72}}

        state = run_cerberus_step(
            enabled_config,
            prev_state=sample_state,
            sentinel_x=sentinel_x,
            alpha_decay=alpha_decay,
            edge_insights=edge_insights,
        )

        assert state.cycle_count == 11
        assert state.regime == "TREND_UP"
        assert state.avg_decay_survival == 0.85


# ---------------------------------------------------------------------------
# Integration Helper Tests
# ---------------------------------------------------------------------------


class TestIntegrationHelpers:
    """Tests for integration helper functions."""

    def test_get_head_multiplier(self, sample_state: CerberusState) -> None:
        """Test getting single head multiplier."""
        mult = get_cerberus_head_multiplier("TREND", sample_state)
        assert mult == 1.0

    def test_get_head_multiplier_disabled(self) -> None:
        """Test getting multiplier when disabled."""
        cfg = CerberusConfig(enabled=False)
        mult = get_cerberus_head_multiplier("TREND", None, cfg)
        assert mult == 1.0

    def test_get_all_multipliers(self, sample_state: CerberusState) -> None:
        """Test getting all multipliers."""
        mults = get_cerberus_all_multipliers(sample_state)
        assert len(mults) == 6
        assert all(m == 1.0 for m in mults.values())

    def test_get_factor_weight_overlay_trend(
        self, sample_state: CerberusState
    ) -> None:
        """Test factor weight overlay for trend factor."""
        overlay = get_cerberus_factor_weight_overlay("trend", sample_state)
        assert overlay == 1.0  # Maps to TREND head

    def test_get_factor_weight_overlay_rv(
        self, sample_state: CerberusState
    ) -> None:
        """Test factor weight overlay for rv_momentum."""
        overlay = get_cerberus_factor_weight_overlay("rv_momentum", sample_state)
        assert overlay == 1.0  # Maps to RELATIVE_VALUE head

    def test_get_conviction_multiplier(self, sample_state: CerberusState) -> None:
        """Test conviction multiplier (avg of TREND and MEAN_REVERT)."""
        mult = get_cerberus_conviction_multiplier(sample_state)
        assert mult == 1.0  # Both are 1.0

    def test_get_universe_category_multiplier(
        self, sample_state: CerberusState
    ) -> None:
        """Test universe category multiplier."""
        mult = get_cerberus_universe_category_multiplier("MEME", sample_state)
        assert mult == 1.0  # Uses CATEGORY head

    def test_get_alpha_router_adjustment(
        self, sample_state: CerberusState
    ) -> None:
        """Test alpha router adjustment."""
        adj = get_cerberus_alpha_router_adjustment(sample_state)
        # All heads at 1.0, weighted average = 1.0
        assert abs(adj - 1.0) < 0.01

    def test_get_crossfire_multiplier(self, sample_state: CerberusState) -> None:
        """Test crossfire multiplier."""
        mult = get_cerberus_crossfire_multiplier(sample_state)
        assert mult == 1.0  # RELATIVE_VALUE head

    def test_get_prospector_multiplier(self, sample_state: CerberusState) -> None:
        """Test prospector multiplier."""
        mult = get_cerberus_prospector_multiplier(sample_state)
        assert mult == 1.0  # EMERGENT_ALPHA head


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_run_step_with_none_inputs(
        self, enabled_config: CerberusConfig
    ) -> None:
        """Test step handles None inputs gracefully."""
        state = run_cerberus_step(
            enabled_config,
            sentinel_x=None,
            alpha_decay=None,
        )
        assert state.cycle_count == 1
        assert state.regime == "UNKNOWN"

    def test_state_with_missing_heads(self) -> None:
        """Test state handles missing heads gracefully."""
        state = CerberusState()
        # StrategyHeadState post_init should add missing heads
        assert len(state.head_state.heads) == 6

    def test_get_multiplier_missing_head(
        self, sample_state: CerberusState
    ) -> None:
        """Test getting multiplier for non-existent head."""
        mult = get_cerberus_head_multiplier("NONEXISTENT", sample_state)
        assert mult == 1.0

    def test_normalize_zero_multipliers(self) -> None:
        """Test normalization with zero values."""
        multipliers = {"A": 0.0, "B": 0.0, "C": 0.0}
        result = normalize_multipliers(multipliers)
        # Should return as-is when mean is 0
        assert result == multipliers
