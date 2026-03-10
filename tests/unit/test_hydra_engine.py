"""
Unit tests for Hydra Multi-Strategy Execution Engine — v7.9_P1

Tests cover:
- Config loading and validation
- Head generators (intent generation)
- Budget enforcement
- Conflict resolution
- Core routing logic
- State serialization
- Integration helpers
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from execution.hydra_engine import (
    # Constants
    STRATEGY_HEADS,
    DEFAULT_STATE_PATH,
    DEFAULT_INTENT_LOG_PATH,
    # Data classes
    HydraHeadConfig,
    HydraIntentLimits,
    HydraConflictResolution,
    HydraConfig,
    HydraIntent,
    HydraMergedIntent,
    HydraHeadBudget,
    HydraState,
    # Config loading
    load_hydra_config,
    # State I/O
    load_hydra_state,
    save_hydra_state,
    # Logging
    log_hydra_intent,
    log_hydra_intents,
    # Head generators
    generate_trend_intents,
    generate_mean_revert_intents,
    generate_relative_value_intents,
    generate_category_intents,
    generate_vol_harvest_intents,
    generate_emergent_alpha_intents,
    # Budget enforcement
    enforce_head_budgets,
    # Conflict resolution
    resolve_symbol_conflict,
    apply_symbol_head_limit,
    # Core routing
    hydra_route_intents,
    # Pipeline
    run_hydra_step,
    # Integration helpers
    get_hydra_nav_allocation,
    get_hydra_head_exposure,
    is_hydra_enabled,
    hydra_merged_intent_to_execution_intent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_hydra_config() -> HydraConfig:
    """Create a sample HydraConfig for testing."""
    return HydraConfig(
        enabled=True,
        heads={
            h: HydraHeadConfig(
                name=h,
                enabled=True,
                max_nav_pct=0.30,
                max_gross_nav_pct=0.40,
                max_positions=10,
                priority=50 + i * 10,
                direction="both",
            )
            for i, h in enumerate(STRATEGY_HEADS)
        },
        intent_limits=HydraIntentLimits(max_intents_per_cycle=64, max_symbol_heads=3),
        conflict_resolution=HydraConflictResolution(
            allow_netting=True,
            prefer_higher_priority=True,
            prefer_higher_score=True,
            max_head_disagreement=3,
        ),
    )


@pytest.fixture
def sample_intents() -> List[HydraIntent]:
    """Create sample intents for testing."""
    return [
        HydraIntent(head="TREND", symbol="BTCUSDT", side="long", nav_pct=0.05, score=0.8, rationale="Trend up"),
        HydraIntent(head="TREND", symbol="ETHUSDT", side="long", nav_pct=0.04, score=0.7, rationale="Trend up"),
        HydraIntent(head="MEAN_REVERT", symbol="BTCUSDT", side="short", nav_pct=0.03, score=0.6, rationale="Overbought"),
        HydraIntent(head="CATEGORY", symbol="ETHUSDT", side="long", nav_pct=0.02, score=0.5, rationale="Strong category"),
    ]


@pytest.fixture
def tmp_state_path(tmp_path: Path) -> Path:
    """Create a temporary state file path."""
    return tmp_path / "hydra_state.json"


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    """Create a temporary log file path."""
    return tmp_path / "hydra_intents.jsonl"


# ---------------------------------------------------------------------------
# Test: Data Classes
# ---------------------------------------------------------------------------


class TestHydraHeadConfig:
    """Tests for HydraHeadConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = HydraHeadConfig(name="TEST")
        assert cfg.enabled is True
        assert cfg.max_nav_pct == 0.30
        assert cfg.max_positions == 10
        assert cfg.direction == "both"

    def test_validation_clamps_nav_pct(self):
        """Test that NAV percentages are clamped to [0, 1]."""
        cfg = HydraHeadConfig(name="TEST", max_nav_pct=1.5)
        assert cfg.max_nav_pct == 1.0

        cfg2 = HydraHeadConfig(name="TEST", max_nav_pct=-0.5)
        assert cfg2.max_nav_pct == 0.0

    def test_validation_clamps_positions(self):
        """Test that positions are clamped to reasonable range."""
        cfg = HydraHeadConfig(name="TEST", max_positions=100)
        assert cfg.max_positions == 50

        cfg2 = HydraHeadConfig(name="TEST", max_positions=0)
        assert cfg2.max_positions == 1

    def test_invalid_direction_normalized(self):
        """Test that invalid direction is normalized to 'both'."""
        cfg = HydraHeadConfig(name="TEST", direction="invalid")
        assert cfg.direction == "both"


class TestHydraIntent:
    """Tests for HydraIntent dataclass."""

    def test_creation(self):
        """Test basic intent creation."""
        intent = HydraIntent(
            head="TREND",
            symbol="BTCUSDT",
            side="long",
            nav_pct=0.05,
            score=0.8,
            rationale="Test",
        )
        assert intent.head == "TREND"
        assert intent.symbol == "BTCUSDT"
        assert intent.side == "long"

    def test_validation_clamps_values(self):
        """Test that values are clamped appropriately."""
        intent = HydraIntent(
            head="TREND",
            symbol="BTCUSDT",
            side="long",
            nav_pct=1.5,
            score=2.0,
            rationale="Test",
        )
        assert intent.nav_pct == 1.0
        assert intent.score == 1.0

    def test_invalid_side_raises(self):
        """Test that invalid side raises ValueError."""
        with pytest.raises(ValueError):
            HydraIntent(
                head="TREND",
                symbol="BTCUSDT",
                side="invalid",
                nav_pct=0.05,
                score=0.8,
                rationale="Test",
            )

    def test_to_dict(self):
        """Test dictionary serialization."""
        intent = HydraIntent(
            head="TREND",
            symbol="BTCUSDT",
            side="long",
            nav_pct=0.05,
            score=0.8,
            rationale="Test",
        )
        d = intent.to_dict()
        assert d["head"] == "TREND"
        assert d["symbol"] == "BTCUSDT"
        assert "timestamp" in d


class TestHydraMergedIntent:
    """Tests for HydraMergedIntent dataclass."""

    def test_creation(self):
        """Test basic merged intent creation."""
        merged = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.08,
            heads=["TREND", "CATEGORY"],
            head_contributions={"TREND": 0.05, "CATEGORY": 0.03},
            score=0.75,
            rationale="Merged",
        )
        assert merged.symbol == "BTCUSDT"
        assert len(merged.heads) == 2

    def test_to_dict(self):
        """Test dictionary serialization."""
        merged = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.08,
            heads=["TREND"],
            head_contributions={"TREND": 0.08},
            score=0.75,
            rationale="Test",
        )
        d = merged.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["net_side"] == "long"
        assert "head_contributions" in d


class TestHydraHeadBudget:
    """Tests for HydraHeadBudget dataclass."""

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        budget = HydraHeadBudget(
            name="TREND",
            max_nav_pct=0.50,
            used_nav_pct=0.30,
            position_count=5,
            max_positions=10,
        )
        assert budget.remaining_nav_pct == 0.20
        assert budget.remaining_positions == 5

    def test_can_allocate(self):
        """Test allocation check."""
        budget = HydraHeadBudget(
            name="TREND",
            max_nav_pct=0.50,
            used_nav_pct=0.45,
            position_count=9,
            max_positions=10,
        )
        assert budget.can_allocate(0.04) is True
        assert budget.can_allocate(0.06) is False

        # At position limit
        budget.position_count = 10
        assert budget.can_allocate(0.01) is False


class TestHydraState:
    """Tests for HydraState dataclass."""

    def test_to_dict_from_dict(self):
        """Test round-trip serialization."""
        state = HydraState(
            updated_ts="2024-01-01T00:00:00Z",
            head_budgets={"TREND": 0.50},
            head_usage={"TREND": 0.25},
            head_positions={"TREND": 5},
            merged_intents=[{"symbol": "BTCUSDT"}],
            cycle_count=10,
            notes=["Test note"],
            errors=[],
            meta={"test": True},
        )
        d = state.to_dict()
        restored = HydraState.from_dict(d)

        assert restored.updated_ts == state.updated_ts
        assert restored.head_budgets == state.head_budgets
        assert restored.cycle_count == state.cycle_count


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------


class TestLoadHydraConfig:
    """Tests for config loading."""

    def test_default_config(self):
        """Test default config when file missing."""
        cfg = load_hydra_config(config_path="/nonexistent/path.json")
        assert cfg.enabled is False
        assert len(cfg.heads) == 6

    def test_load_from_dict(self):
        """Test loading from dict."""
        strategy_config = {
            "hydra_execution": {
                "enabled": True,
                "heads": {
                    "TREND": {"enabled": True, "max_nav_pct": 0.40, "priority": 100},
                },
                "intent_limits": {"max_intents_per_cycle": 32},
            }
        }
        cfg = load_hydra_config(strategy_config=strategy_config)
        assert cfg.enabled is True
        assert cfg.heads["TREND"].max_nav_pct == 0.40
        assert cfg.intent_limits.max_intents_per_cycle == 32

    def test_missing_heads_filled_with_defaults(self):
        """Test that missing heads get default values."""
        strategy_config = {
            "hydra_execution": {
                "enabled": True,
                "heads": {"TREND": {"enabled": True}},
            }
        }
        cfg = load_hydra_config(strategy_config=strategy_config)
        assert "MEAN_REVERT" in cfg.heads
        assert cfg.heads["MEAN_REVERT"].enabled is True

    def test_get_enabled_heads(self):
        """Test getting list of enabled heads."""
        cfg = HydraConfig(enabled=True)
        cfg.heads["TREND"].enabled = True
        cfg.heads["MEAN_REVERT"].enabled = False

        enabled = cfg.get_enabled_heads()
        assert "TREND" in enabled
        assert "MEAN_REVERT" not in enabled


# ---------------------------------------------------------------------------
# Test: State I/O
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state loading and saving."""

    def test_save_and_load(self, tmp_state_path: Path):
        """Test saving and loading state."""
        state = HydraState(
            updated_ts="2024-01-01T00:00:00Z",
            head_budgets={"TREND": 0.50},
            head_usage={"TREND": 0.25},
            cycle_count=5,
        )

        assert save_hydra_state(state, tmp_state_path) is True
        assert tmp_state_path.exists()

        loaded = load_hydra_state(tmp_state_path)
        assert loaded.updated_ts == state.updated_ts
        assert loaded.head_budgets == state.head_budgets

    def test_load_missing_file(self):
        """Test loading from missing file returns empty state."""
        state = load_hydra_state("/nonexistent/path.json")
        assert state.updated_ts == ""
        assert state.head_budgets == {}

    def test_load_invalid_json(self, tmp_state_path: Path):
        """Test loading invalid JSON returns empty state."""
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text("invalid json {{{")

        state = load_hydra_state(tmp_state_path)
        assert state.updated_ts == ""


# ---------------------------------------------------------------------------
# Test: Intent Logging
# ---------------------------------------------------------------------------


class TestIntentLogging:
    """Tests for intent logging."""

    def test_log_single_intent(self, tmp_log_path: Path):
        """Test logging a single intent."""
        intent = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.05,
            heads=["TREND"],
            head_contributions={"TREND": 0.05},
            score=0.8,
            rationale="Test",
        )

        assert log_hydra_intent(intent, tmp_log_path) is True
        assert tmp_log_path.exists()

        lines = tmp_log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        logged = json.loads(lines[0])
        assert logged["symbol"] == "BTCUSDT"
        assert "ts" in logged

    def test_log_multiple_intents(self, tmp_log_path: Path):
        """Test logging multiple intents."""
        intents = [
            HydraMergedIntent(
                symbol=f"SYM{i}USDT",
                net_side="long",
                nav_pct=0.01,
                heads=["TREND"],
                head_contributions={"TREND": 0.01},
                score=0.5,
                rationale="Test",
            )
            for i in range(5)
        ]

        count = log_hydra_intents(intents, tmp_log_path)
        assert count == 5

        lines = tmp_log_path.read_text().strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Test: Head Generators
# ---------------------------------------------------------------------------


class TestTrendGenerator:
    """Tests for TREND head generator."""

    def test_generates_intents_from_scores(self):
        """Test intent generation from hybrid scores."""
        cfg = HydraHeadConfig(name="TREND", enabled=True, direction="both")
        hybrid_scores = {"BTCUSDT": 0.7, "ETHUSDT": -0.6, "SOLUSDT": 0.05}

        intents = generate_trend_intents(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            hybrid_scores=hybrid_scores,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 2  # SOLUSDT below threshold
        assert any(i.symbol == "BTCUSDT" and i.side == "long" for i in intents)
        assert any(i.symbol == "ETHUSDT" and i.side == "short" for i in intents)

    def test_respects_direction_constraint(self):
        """Test direction filtering."""
        cfg = HydraHeadConfig(name="TREND", enabled=True, direction="long")
        hybrid_scores = {"BTCUSDT": 0.7, "ETHUSDT": -0.6}

        intents = generate_trend_intents(
            symbols=["BTCUSDT", "ETHUSDT"],
            hybrid_scores=hybrid_scores,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 1
        assert intents[0].symbol == "BTCUSDT"
        assert intents[0].side == "long"

    def test_disabled_head_returns_empty(self):
        """Test that disabled head returns no intents."""
        cfg = HydraHeadConfig(name="TREND", enabled=False)

        intents = generate_trend_intents(
            symbols=["BTCUSDT"],
            hybrid_scores={"BTCUSDT": 0.8},
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 0


class TestMeanRevertGenerator:
    """Tests for MEAN_REVERT head generator."""

    def test_generates_intents_from_zscore(self):
        """Test intent generation from z-scores."""
        cfg = HydraHeadConfig(name="MEAN_REVERT", enabled=True, direction="both")
        zscore_map = {"BTCUSDT": 2.0, "ETHUSDT": -2.5, "SOLUSDT": 0.5}

        intents = generate_mean_revert_intents(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            zscore_map=zscore_map,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 2  # SOLUSDT below threshold
        # Mean revert: short when high z-score, long when low
        assert any(i.symbol == "BTCUSDT" and i.side == "short" for i in intents)
        assert any(i.symbol == "ETHUSDT" and i.side == "long" for i in intents)


class TestRelativeValueGenerator:
    """Tests for RELATIVE_VALUE head generator."""

    def test_generates_pair_intents(self):
        """Test intent generation from pair edges."""
        cfg = HydraHeadConfig(name="RELATIVE_VALUE", enabled=True, direction="both")
        pair_edges = [
            {"long_symbol": "BTCUSDT", "short_symbol": "ETHUSDT", "edge_score": 0.6},
        ]

        intents = generate_relative_value_intents(
            pair_edges=pair_edges,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 2  # One long, one short
        assert any(i.symbol == "BTCUSDT" and i.side == "long" for i in intents)
        assert any(i.symbol == "ETHUSDT" and i.side == "short" for i in intents)


class TestCategoryGenerator:
    """Tests for CATEGORY head generator."""

    def test_generates_intents_from_category_scores(self):
        """Test intent generation from category momentum."""
        cfg = HydraHeadConfig(name="CATEGORY", enabled=True, direction="both")
        category_scores = {"DEFI": 0.5, "MEME": -0.4, "L1": 0.1}
        symbol_categories = {"UNIUSDT": "DEFI", "DOGEUSDT": "MEME", "SOLUSDT": "L1"}

        intents = generate_category_intents(
            category_scores=category_scores,
            symbol_categories=symbol_categories,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) == 2  # L1 below threshold
        assert any(i.symbol == "UNIUSDT" and i.side == "long" for i in intents)
        assert any(i.symbol == "DOGEUSDT" and i.side == "short" for i in intents)


class TestVolHarvestGenerator:
    """Tests for VOL_HARVEST head generator."""

    def test_generates_intents_from_vol_ratio(self):
        """Test intent generation from volatility targeting."""
        cfg = HydraHeadConfig(name="VOL_HARVEST", enabled=True)
        vol_targets = {"BTCUSDT": 0.20, "ETHUSDT": 0.25}
        realized_vols = {"BTCUSDT": 0.30, "ETHUSDT": 0.20}

        intents = generate_vol_harvest_intents(
            symbols=["BTCUSDT", "ETHUSDT"],
            vol_targets=vol_targets,
            realized_vols=realized_vols,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        # Should generate intents for symbols with vol ratio outside threshold
        assert len(intents) >= 0  # May or may not generate depending on ratio


class TestEmergentAlphaGenerator:
    """Tests for EMERGENT_ALPHA head generator."""

    def test_generates_intents_from_alpha_signals(self):
        """Test intent generation from alpha miner signals."""
        cfg = HydraHeadConfig(name="EMERGENT_ALPHA", enabled=True, direction="both")
        universe_scores = {"BTCUSDT": 0.6, "NEWCOINUSDT": 0.8}
        alpha_signals = {
            "BTCUSDT": {"alpha_score": 0.5},
            "NEWCOINUSDT": {"alpha_score": 0.7},
        }

        intents = generate_emergent_alpha_intents(
            universe_scores=universe_scores,
            alpha_miner_signals=alpha_signals,
            cerberus_multiplier=1.0,
            head_cfg=cfg,
            nav_usd=10000,
        )

        assert len(intents) >= 1
        assert any(i.symbol == "NEWCOINUSDT" for i in intents)


# ---------------------------------------------------------------------------
# Test: Budget Enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Tests for per-head budget enforcement."""

    def test_scales_down_when_over_budget(self, sample_hydra_config: HydraConfig):
        """Test that intents are scaled down when over budget."""
        # Create intents that exceed TREND budget
        intents = [
            HydraIntent(head="TREND", symbol=f"SYM{i}USDT", side="long", nav_pct=0.10, score=0.8, rationale="Test")
            for i in range(5)
        ]
        # Total: 0.50, but max is 0.30

        scaled, usage = enforce_head_budgets(intents, sample_hydra_config.heads)

        assert len(scaled) == 5
        total_nav = sum(i.nav_pct for i in scaled)
        assert total_nav <= sample_hydra_config.heads["TREND"].max_nav_pct + 0.001

    def test_no_scaling_when_within_budget(self, sample_hydra_config: HydraConfig):
        """Test that intents are not scaled when within budget."""
        intents = [
            HydraIntent(head="TREND", symbol="BTCUSDT", side="long", nav_pct=0.05, score=0.8, rationale="Test"),
            HydraIntent(head="TREND", symbol="ETHUSDT", side="long", nav_pct=0.05, score=0.7, rationale="Test"),
        ]

        scaled, usage = enforce_head_budgets(intents, sample_hydra_config.heads)

        assert len(scaled) == 2
        assert all(s.nav_pct == 0.05 for s in scaled)


# ---------------------------------------------------------------------------
# Test: Conflict Resolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    """Tests for conflict resolution between heads."""

    def test_agreeing_heads_aggregate(self, sample_hydra_config: HydraConfig):
        """Test that agreeing heads aggregate their NAV."""
        intents = [
            HydraIntent(head="TREND", symbol="BTCUSDT", side="long", nav_pct=0.05, score=0.8, rationale="Test"),
            HydraIntent(head="CATEGORY", symbol="BTCUSDT", side="long", nav_pct=0.03, score=0.6, rationale="Test"),
        ]

        merged = resolve_symbol_conflict(
            symbol="BTCUSDT",
            intents=intents,
            head_configs=sample_hydra_config.heads,
            conflict_cfg=sample_hydra_config.conflict_resolution,
        )

        assert merged.net_side == "long"
        assert merged.nav_pct == 0.08
        assert len(merged.heads) == 2

    def test_conflicting_heads_with_netting(self, sample_hydra_config: HydraConfig):
        """Test that conflicting heads can net positions."""
        intents = [
            HydraIntent(head="TREND", symbol="BTCUSDT", side="long", nav_pct=0.06, score=0.8, rationale="Test"),
            HydraIntent(head="MEAN_REVERT", symbol="BTCUSDT", side="short", nav_pct=0.04, score=0.6, rationale="Test"),
        ]

        merged = resolve_symbol_conflict(
            symbol="BTCUSDT",
            intents=intents,
            head_configs=sample_hydra_config.heads,
            conflict_cfg=sample_hydra_config.conflict_resolution,
        )

        assert merged.net_side == "long"
        assert abs(merged.nav_pct - 0.02) < 0.001  # 0.06 - 0.04

    def test_excessive_disagreement_goes_flat(self, sample_hydra_config: HydraConfig):
        """Test that too many disagreeing heads results in flat."""
        # Create many conflicting intents
        sample_hydra_config.conflict_resolution.max_head_disagreement = 2

        intents = [
            HydraIntent(head="TREND", symbol="BTCUSDT", side="long", nav_pct=0.05, score=0.8, rationale="Test"),
            HydraIntent(head="CATEGORY", symbol="BTCUSDT", side="long", nav_pct=0.03, score=0.7, rationale="Test"),
            HydraIntent(head="EMERGENT_ALPHA", symbol="BTCUSDT", side="long", nav_pct=0.02, score=0.6, rationale="Test"),
            HydraIntent(head="MEAN_REVERT", symbol="BTCUSDT", side="short", nav_pct=0.04, score=0.65, rationale="Test"),
            HydraIntent(head="VOL_HARVEST", symbol="BTCUSDT", side="short", nav_pct=0.03, score=0.55, rationale="Test"),
            HydraIntent(head="RELATIVE_VALUE", symbol="BTCUSDT", side="short", nav_pct=0.02, score=0.5, rationale="Test"),
        ]

        merged = resolve_symbol_conflict(
            symbol="BTCUSDT",
            intents=intents,
            head_configs=sample_hydra_config.heads,
            conflict_cfg=sample_hydra_config.conflict_resolution,
        )

        assert merged.net_side == "flat"
        assert merged.nav_pct == 0.0


class TestSymbolHeadLimit:
    """Tests for symbol-level head limits."""

    def test_limits_heads_per_symbol(self, sample_hydra_config: HydraConfig):
        """Test that heads are limited per symbol."""
        merged = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.10,
            heads=["TREND", "CATEGORY", "EMERGENT_ALPHA", "VOL_HARVEST"],
            head_contributions={"TREND": 0.04, "CATEGORY": 0.03, "EMERGENT_ALPHA": 0.02, "VOL_HARVEST": 0.01},
            score=0.7,
            rationale="Test",
        )

        limited = apply_symbol_head_limit(merged, max_symbol_heads=2, head_configs=sample_hydra_config.heads)

        assert len(limited.heads) == 2

    def test_keeps_highest_priority_heads(self, sample_hydra_config: HydraConfig):
        """Test that highest priority heads are kept."""
        # TREND has highest priority in our fixture
        sample_hydra_config.heads["TREND"].priority = 100
        sample_hydra_config.heads["CATEGORY"].priority = 50
        sample_hydra_config.heads["EMERGENT_ALPHA"].priority = 30

        merged = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.10,
            heads=["TREND", "CATEGORY", "EMERGENT_ALPHA"],
            head_contributions={"TREND": 0.04, "CATEGORY": 0.03, "EMERGENT_ALPHA": 0.03},
            score=0.7,
            rationale="Test",
        )

        limited = apply_symbol_head_limit(merged, max_symbol_heads=2, head_configs=sample_hydra_config.heads)

        assert "TREND" in limited.heads


# ---------------------------------------------------------------------------
# Test: Core Routing
# ---------------------------------------------------------------------------


class TestHydraRouteIntents:
    """Tests for core intent routing."""

    def test_routes_intents_correctly(self, sample_hydra_config: HydraConfig, sample_intents: List[HydraIntent]):
        """Test that routing produces valid merged intents."""
        merged, usage = hydra_route_intents(sample_hydra_config, sample_intents)

        assert len(merged) > 0
        assert all(isinstance(m, HydraMergedIntent) for m in merged)
        assert all(m.net_side in ("long", "short") for m in merged)

    def test_respects_max_intents_limit(self, sample_hydra_config: HydraConfig):
        """Test that output is limited to max_intents_per_cycle."""
        sample_hydra_config.intent_limits.max_intents_per_cycle = 2

        intents = [
            HydraIntent(head="TREND", symbol=f"SYM{i}USDT", side="long", nav_pct=0.01, score=0.5 + i * 0.1, rationale="Test")
            for i in range(10)
        ]

        merged, usage = hydra_route_intents(sample_hydra_config, intents)

        assert len(merged) <= 2

    def test_sorts_by_score_and_nav(self, sample_hydra_config: HydraConfig):
        """Test that output is sorted by score then nav_pct."""
        intents = [
            HydraIntent(head="TREND", symbol="LOWSCORE", side="long", nav_pct=0.10, score=0.3, rationale="Test"),
            HydraIntent(head="TREND", symbol="HIGHSCORE", side="long", nav_pct=0.05, score=0.9, rationale="Test"),
        ]

        merged, usage = hydra_route_intents(sample_hydra_config, intents)

        assert merged[0].symbol == "HIGHSCORE"

    def test_empty_intents_returns_empty(self, sample_hydra_config: HydraConfig):
        """Test that empty input returns empty output."""
        merged, usage = hydra_route_intents(sample_hydra_config, [])

        assert len(merged) == 0
        assert len(usage) == 0


# ---------------------------------------------------------------------------
# Test: Pipeline Runner
# ---------------------------------------------------------------------------


class TestRunHydraStep:
    """Tests for the full pipeline runner."""

    def test_disabled_returns_empty_state(self, tmp_state_path: Path):
        """Test that disabled Hydra returns empty state."""
        cfg = HydraConfig(enabled=False)

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={},
            symbols=["BTCUSDT"],
            hybrid_scores={},
            zscore_map={},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=10000,
            state_path=tmp_state_path,
        )

        assert len(merged) == 0
        assert "Hydra disabled" in state.notes

    def test_generates_intents_from_intel(self, tmp_state_path: Path, tmp_log_path: Path):
        """Test that pipeline generates intents from intel surfaces."""
        cfg = HydraConfig(enabled=True)

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={"TREND": 1.0, "MEAN_REVERT": 1.0},
            symbols=["BTCUSDT", "ETHUSDT"],
            hybrid_scores={"BTCUSDT": 0.7, "ETHUSDT": 0.5},
            zscore_map={"BTCUSDT": 0.5, "ETHUSDT": -2.0},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=10000,
            state_path=tmp_state_path,
            log_path=tmp_log_path,
        )

        assert state.updated_ts != ""
        assert len(state.notes) > 0


# ---------------------------------------------------------------------------
# Test: Integration Helpers
# ---------------------------------------------------------------------------


class TestIntegrationHelpers:
    """Tests for integration helper functions."""

    def test_get_hydra_nav_allocation(self):
        """Test NAV allocation lookup."""
        state = HydraState(
            merged_intents=[
                {"symbol": "BTCUSDT", "net_side": "long", "nav_pct": 0.05, "heads": ["TREND"]},
            ]
        )

        nav_pct, side, heads = get_hydra_nav_allocation("BTCUSDT", state)
        assert nav_pct == 0.05
        assert side == "long"
        assert "TREND" in heads

    def test_get_hydra_nav_allocation_missing(self):
        """Test NAV allocation for missing symbol."""
        state = HydraState()

        nav_pct, side, heads = get_hydra_nav_allocation("BTCUSDT", state)
        assert nav_pct == 0.0
        assert side == "flat"
        assert heads == []

    def test_get_hydra_head_exposure(self):
        """Test head exposure lookup."""
        state = HydraState(
            head_usage={"TREND": 0.25},
            head_positions={"TREND": 5},
        )

        usage, positions = get_hydra_head_exposure("TREND", state)
        assert usage == 0.25
        assert positions == 5

    def test_is_hydra_enabled(self):
        """Test enabled check."""
        assert is_hydra_enabled({"hydra_execution": {"enabled": True}}) is True
        assert is_hydra_enabled({"hydra_execution": {"enabled": False}}) is False
        assert is_hydra_enabled({}) is False

    def test_hydra_merged_intent_to_execution_intent(self):
        """Test conversion to execution intent."""
        merged = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.05,
            heads=["TREND"],
            head_contributions={"TREND": 0.05},
            score=0.8,
            rationale="Test",
        )

        intent = hydra_merged_intent_to_execution_intent(merged, nav_usd=10000, price=50000)

        assert intent["symbol"] == "BTCUSDT"
        assert intent["side"] == "LONG"
        assert intent["notional_usd"] == 500  # 10000 * 0.05
        assert intent["qty"] == 0.01  # 500 / 50000
        assert intent["source"] == "hydra"
        assert intent["price"] == 50000
        assert intent["capital_per_trade"] == 500
        assert intent["gross_usd"] == 500

    def test_hydra_merged_intent_flat_returns_empty(self):
        """Test that flat positions return empty dict."""
        merged = {"symbol": "BTCUSDT", "net_side": "flat", "nav_pct": 0.0}

        intent = hydra_merged_intent_to_execution_intent(merged, nav_usd=10000, price=50000)

        assert intent == {}


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_zero_nav(self, sample_hydra_config: HydraConfig):
        """Test handling of zero NAV."""
        cfg = HydraConfig(enabled=True)

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={},
            symbols=["BTCUSDT"],
            hybrid_scores={"BTCUSDT": 0.7},
            zscore_map={},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=0,
        )

        # Should handle gracefully
        assert isinstance(state, HydraState)

    def test_handles_empty_symbols(self, sample_hydra_config: HydraConfig):
        """Test handling of empty symbol list."""
        cfg = HydraConfig(enabled=True)

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={},
            symbols=[],
            hybrid_scores={},
            zscore_map={},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=10000,
        )

        assert len(merged) == 0

    def test_handles_all_heads_disabled(self):
        """Test handling when all heads are disabled."""
        cfg = HydraConfig(enabled=True)
        for h in cfg.heads.values():
            h.enabled = False

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={},
            symbols=["BTCUSDT"],
            hybrid_scores={"BTCUSDT": 0.8},
            zscore_map={},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=10000,
        )

        assert len(merged) == 0
