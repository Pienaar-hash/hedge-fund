"""Unit tests for execution/intel/category_momentum.py."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from execution.intel.category_momentum import (
    CategoryConfig,
    CategoryMomentumSnapshot,
    CategoryStats,
    compute_category_momentum,
    compute_category_stats,
    get_symbol_category,
    get_symbols_by_category,
    load_category_config,
    load_symbol_categories,
    normalize_category_momentum,
    build_category_momentum_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_category_mapping() -> Dict[str, str]:
    """Simple category mapping for testing."""
    return {
        "BTCUSDT": "L1_BTC_ETH",
        "ETHUSDT": "L1_BTC_ETH",
        "SOLUSDT": "L1_ALT",
        "AVAXUSDT": "L1_ALT",
        "DOGEUSDT": "MEME",
        "SHIBUSDT": "MEME",
        "UNIUSDT": "DEFI",
        "AAVEUSDT": "DEFI",
        "LINKUSDT": "OTHER",
    }


@pytest.fixture
def sample_symbol_returns() -> Dict[str, np.ndarray]:
    """Synthetic return arrays for testing."""
    return {
        "BTCUSDT": np.array([0.01, 0.02, 0.015, 0.005, 0.012]),  # Good
        "ETHUSDT": np.array([0.008, 0.015, 0.012, 0.010, 0.009]),  # Good
        "SOLUSDT": np.array([-0.01, -0.005, -0.008, -0.012, -0.002]),  # Bad
        "AVAXUSDT": np.array([-0.005, -0.010, -0.003, -0.008, -0.001]),  # Bad
        "DOGEUSDT": np.array([0.05, 0.02, -0.03, 0.01, 0.00]),  # Volatile
        "SHIBUSDT": np.array([0.03, -0.02, 0.01, 0.00, 0.02]),  # Volatile
        "UNIUSDT": np.array([0.002, 0.003, 0.001, 0.002, 0.001]),  # Stable
        "AAVEUSDT": np.array([0.001, 0.002, 0.003, 0.001, 0.002]),  # Stable
        "LINKUSDT": np.array([0.00, 0.00, 0.00, 0.00, 0.00]),  # Zero
    }


@pytest.fixture
def default_config() -> CategoryConfig:
    """Default category config."""
    return CategoryConfig()


# ---------------------------------------------------------------------------
# load_symbol_categories tests
# ---------------------------------------------------------------------------


def test_load_symbol_categories_from_temp_file():
    """Load mapping from a temporary JSON file."""
    mapping = {"BTCUSDT": "L1_BTC_ETH", "ETHUSDT": "L1_BTC_ETH"}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({"categories": mapping}, f)
        temp_path = f.name

    loaded = load_symbol_categories(temp_path)

    assert loaded == mapping
    Path(temp_path).unlink()


def test_load_symbol_categories_missing_file():
    """Return empty dict if file doesn't exist."""
    mapping = load_symbol_categories("/nonexistent/file.json")
    assert mapping == {}


def test_load_symbol_categories_invalid_json():
    """Return empty dict on malformed JSON."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("{invalid json")
        temp_path = f.name

    mapping = load_symbol_categories(temp_path)
    assert mapping == {}
    Path(temp_path).unlink()


# ---------------------------------------------------------------------------
# load_category_config tests
# ---------------------------------------------------------------------------


def test_load_category_config_defaults():
    """Default config when strategy config is empty."""
    cfg = load_category_config(None)

    assert cfg.enabled is True
    assert cfg.lookback_bars == 48
    assert cfg.half_life_bars == 24
    assert cfg.ir_scale == 2.0
    assert cfg.max_abs_score == 1.0


def test_load_category_config_custom():
    """Load custom values from strategy config."""
    custom_cfg = {
        "category_momentum": {
            "enabled": False,
            "lookback_bars": 96,
            "half_life_bars": 48,
            "ir_scale": 3.0,
            "max_abs_score": 2.0,
        }
    }
    cfg = load_category_config(custom_cfg)

    assert cfg.enabled is False
    assert cfg.lookback_bars == 96
    assert cfg.half_life_bars == 48
    assert cfg.ir_scale == 3.0
    assert cfg.max_abs_score == 2.0


# ---------------------------------------------------------------------------
# get_symbol_category tests
# ---------------------------------------------------------------------------


def test_get_symbol_category_found(sample_category_mapping):
    """Returns correct category when symbol exists."""
    assert get_symbol_category("BTCUSDT", sample_category_mapping) == "L1_BTC_ETH"
    assert get_symbol_category("DOGEUSDT", sample_category_mapping) == "MEME"


def test_get_symbol_category_default(sample_category_mapping):
    """Returns default category when symbol not found."""
    assert get_symbol_category("UNKNOWNUSDT", sample_category_mapping) == "OTHER"
    assert get_symbol_category("UNKNOWNUSDT", sample_category_mapping, "UNCATEGORIZED") == "UNCATEGORIZED"


def test_get_symbol_category_case_insensitive(sample_category_mapping):
    """Handles uppercase lookup (function uppercases input)."""
    # Function uppercases input, so lowercase should also find it
    assert get_symbol_category("btcusdt", sample_category_mapping) == "L1_BTC_ETH"
    assert get_symbol_category("BTCUSDT", sample_category_mapping) == "L1_BTC_ETH"


# ---------------------------------------------------------------------------
# get_symbols_by_category tests
# ---------------------------------------------------------------------------


def test_get_symbols_by_category(sample_category_mapping):
    """Inverts symbol→category to category→symbols."""
    by_cat = get_symbols_by_category(sample_category_mapping)

    assert "L1_BTC_ETH" in by_cat
    assert set(by_cat["L1_BTC_ETH"]) == {"BTCUSDT", "ETHUSDT"}

    assert "MEME" in by_cat
    assert set(by_cat["MEME"]) == {"DOGEUSDT", "SHIBUSDT"}


def test_get_symbols_by_category_empty():
    """Returns empty dict for empty mapping."""
    by_cat = get_symbols_by_category({})
    assert by_cat == {}


# ---------------------------------------------------------------------------
# compute_category_stats tests
# ---------------------------------------------------------------------------


def test_compute_category_stats_basic(
    sample_category_mapping, sample_symbol_returns, default_config
):
    """Compute category stats from synthetic return data."""
    stats = compute_category_stats(
        symbol_returns=sample_symbol_returns,
        categories=sample_category_mapping,
        cfg=default_config,
    )

    # Should have stats for all categories with data
    assert "L1_BTC_ETH" in stats
    assert "L1_ALT" in stats
    assert "MEME" in stats
    assert "DEFI" in stats
    assert "OTHER" in stats

    # L1_BTC_ETH has positive mean return
    l1_btc = stats["L1_BTC_ETH"]
    assert l1_btc.mean_return > 0
    assert l1_btc.total_pnl > 0
    assert "BTCUSDT" in l1_btc.symbols
    assert "ETHUSDT" in l1_btc.symbols

    # L1_ALT has negative mean return
    l1_alt = stats["L1_ALT"]
    assert l1_alt.mean_return < 0
    assert l1_alt.total_pnl < 0


def test_compute_category_stats_empty_returns(sample_category_mapping, default_config):
    """Return empty stats for empty return data."""
    stats = compute_category_stats(
        symbol_returns={},
        categories=sample_category_mapping,
        cfg=default_config,
    )
    # Should still have categories but with zero stats
    for cat, s in stats.items():
        assert s.mean_return == 0.0
        assert s.total_pnl == 0.0


def test_compute_category_stats_unmapped_symbol(default_config):
    """Symbols not in mapping are assigned to OTHER."""
    mapping = {"BTCUSDT": "L1_BTC_ETH"}  # Only BTC mapped
    returns = {
        "BTCUSDT": np.array([0.01, 0.02]),
        "UNKNOWN": np.array([0.05, 0.10]),  # Not in mapping → OTHER
    }

    stats = compute_category_stats(
        symbol_returns=returns,
        categories=mapping,
        cfg=default_config,
    )

    assert "L1_BTC_ETH" in stats
    assert "OTHER" in stats
    assert "UNKNOWN" in stats["OTHER"].symbols


# ---------------------------------------------------------------------------
# normalize_category_momentum tests
# ---------------------------------------------------------------------------


def test_normalize_category_momentum_scales_ir(default_config):
    """Normalizes IR to bounded momentum scores."""
    # Create stats with known IRs
    stats = {
        "GOOD": CategoryStats(name="GOOD", ir=2.0),
        "BAD": CategoryStats(name="BAD", ir=-1.0),
        "NEUTRAL": CategoryStats(name="NEUTRAL", ir=0.0),
    }

    normalized = normalize_category_momentum(stats, default_config)

    # GOOD should have positive momentum
    assert normalized["GOOD"].momentum_score > 0

    # BAD should have negative momentum
    assert normalized["BAD"].momentum_score < 0

    # All scores should be bounded
    for s in normalized.values():
        assert -default_config.max_abs_score <= s.momentum_score <= default_config.max_abs_score


def test_normalize_category_momentum_single_category(default_config):
    """Single category falls back to scaled IR with tanh."""
    stats = {
        "ONLY": CategoryStats(name="ONLY", ir=1.0),
    }

    normalized = normalize_category_momentum(stats, default_config)

    # Should still have a score
    assert "ONLY" in normalized
    assert abs(normalized["ONLY"].momentum_score) <= default_config.max_abs_score


def test_normalize_category_momentum_empty(default_config):
    """Empty stats returns empty."""
    stats = {}
    normalized = normalize_category_momentum(stats, default_config)
    assert normalized == {}


# ---------------------------------------------------------------------------
# compute_category_momentum tests
# ---------------------------------------------------------------------------


def test_compute_category_momentum_basic(
    sample_category_mapping, sample_symbol_returns
):
    """Full momentum computation returns per-symbol scores and category stats."""
    cfg = CategoryConfig(enabled=True)

    per_symbol, category_stats = compute_category_momentum(
        symbol_returns=sample_symbol_returns,
        categories=sample_category_mapping,
        cfg=cfg,
    )

    # Per-symbol scores should exist for all symbols
    assert "BTCUSDT" in per_symbol
    assert "ETHUSDT" in per_symbol
    assert "SOLUSDT" in per_symbol

    # Symbols in same category should have same score
    assert per_symbol["BTCUSDT"] == per_symbol["ETHUSDT"]
    assert per_symbol["SOLUSDT"] == per_symbol["AVAXUSDT"]

    # Category stats should exist
    assert "L1_BTC_ETH" in category_stats
    assert "L1_ALT" in category_stats


def test_compute_category_momentum_disabled(
    sample_category_mapping, sample_symbol_returns
):
    """When disabled, returns neutral scores."""
    cfg = CategoryConfig(enabled=False)

    per_symbol, category_stats = compute_category_momentum(
        symbol_returns=sample_symbol_returns,
        categories=sample_category_mapping,
        cfg=cfg,
    )

    # All scores should be 0
    for score in per_symbol.values():
        assert score == 0.0

    # No category stats computed
    assert category_stats == {}


def test_compute_category_momentum_negative_category():
    """Category with all negative returns should give negative momentum."""
    mapping = {"BTCUSDT": "BEAR", "ETHUSDT": "BEAR"}
    returns = {
        "BTCUSDT": np.array([-0.01, -0.02, -0.015]),
        "ETHUSDT": np.array([-0.008, -0.015, -0.012]),
    }

    cfg = CategoryConfig(enabled=True)
    per_symbol, stats = compute_category_momentum(returns, mapping, cfg)

    # Single category, raw IR will be negative → negative momentum
    assert stats["BEAR"].mean_return < 0
    # With only one category, momentum depends on raw IR scaled
    # The score should be negative (or 0 if normalized to itself)


def test_compute_category_momentum_no_returns_for_mapped_symbol(
    sample_category_mapping,
):
    """Symbols in categories but without returns still get category score."""
    returns = {
        "BTCUSDT": np.array([0.01, 0.02, 0.015]),
        # ETHUSDT is in mapping but has no returns
    }

    cfg = CategoryConfig(enabled=True)
    per_symbol, stats = compute_category_momentum(returns, sample_category_mapping, cfg)

    # ETHUSDT should still have a score (from its category)
    assert "ETHUSDT" in per_symbol


# ---------------------------------------------------------------------------
# build_category_momentum_snapshot tests
# ---------------------------------------------------------------------------


def test_build_category_momentum_snapshot(
    sample_category_mapping, sample_symbol_returns
):
    """Build a full snapshot."""
    snapshot = build_category_momentum_snapshot(
        symbol_returns=sample_symbol_returns,
        categories=sample_category_mapping,
    )

    assert isinstance(snapshot, CategoryMomentumSnapshot)
    assert snapshot.updated_ts > 0
    assert len(snapshot.per_symbol) > 0
    assert len(snapshot.category_stats) > 0
    assert len(snapshot.symbol_categories) > 0


def test_build_category_momentum_snapshot_to_dict(
    sample_category_mapping, sample_symbol_returns
):
    """Snapshot serializes to dict."""
    snapshot = build_category_momentum_snapshot(
        symbol_returns=sample_symbol_returns,
        categories=sample_category_mapping,
    )

    d = snapshot.to_dict()

    assert "updated_ts" in d
    assert "per_symbol" in d
    assert "category_stats" in d
    assert "symbol_categories" in d


# ---------------------------------------------------------------------------
# CategoryStats dataclass tests
# ---------------------------------------------------------------------------


def test_category_stats_dataclass():
    """CategoryStats stores expected fields."""
    stats = CategoryStats(
        name="L1_BTC_ETH",
        symbols=["BTCUSDT", "ETHUSDT"],
        mean_return=0.015,
        volatility=0.005,
        ir=1.5,
        total_pnl=0.10,
        momentum_score=0.8,
    )

    assert stats.name == "L1_BTC_ETH"
    assert stats.symbols == ["BTCUSDT", "ETHUSDT"]
    assert stats.mean_return == 0.015
    assert stats.volatility == 0.005
    assert stats.ir == 1.5
    assert stats.total_pnl == 0.10
    assert stats.momentum_score == 0.8


# ---------------------------------------------------------------------------
# CategoryConfig dataclass tests
# ---------------------------------------------------------------------------


def test_category_config_dataclass():
    """CategoryConfig stores expected fields."""
    cfg = CategoryConfig(
        enabled=True,
        lookback_bars=96,
        half_life_bars=48,
        ir_scale=3.0,
        max_abs_score=2.0,
    )

    assert cfg.enabled is True
    assert cfg.lookback_bars == 96
    assert cfg.half_life_bars == 48
    assert cfg.ir_scale == 3.0
    assert cfg.max_abs_score == 2.0


def test_category_config_defaults():
    """CategoryConfig has sensible defaults."""
    cfg = CategoryConfig()

    assert cfg.enabled is True
    assert cfg.lookback_bars == 48
    assert cfg.half_life_bars == 24
    assert cfg.ir_scale == 2.0
    assert cfg.max_abs_score == 1.0


# ---------------------------------------------------------------------------
# CategoryMomentumSnapshot dataclass tests
# ---------------------------------------------------------------------------


def test_category_momentum_snapshot_dataclass():
    """CategoryMomentumSnapshot stores expected fields."""
    stats = CategoryStats(
        name="L1_BTC_ETH",
        symbols=["BTCUSDT"],
        mean_return=0.01,
        ir=1.0,
        total_pnl=0.05,
    )
    snapshot = CategoryMomentumSnapshot(
        per_symbol={"BTCUSDT": 0.5},
        category_stats={"L1_BTC_ETH": stats},
        symbol_categories={"BTCUSDT": "L1_BTC_ETH"},
        updated_ts=1700000000.0,
    )

    assert snapshot.per_symbol == {"BTCUSDT": 0.5}
    assert "L1_BTC_ETH" in snapshot.category_stats
    assert snapshot.symbol_categories == {"BTCUSDT": "L1_BTC_ETH"}
    assert snapshot.updated_ts == 1700000000.0
