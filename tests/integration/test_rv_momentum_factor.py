"""
Tests for v7.5_C1 — RV Momentum Factor Computation.

Tests:
- Pair relative momentum (BTC vs ETH)
- Basket relative momentum (L1 vs ALT, Meme vs Rest)
- Per-symbol RV score normalization
"""
from __future__ import annotations

import numpy as np
import pytest

from execution.rv_momentum import (
    RvConfig,
    RvSnapshot,
    RvSymbolScore,
    compute_pair_relative_momentum,
    compute_basket_relative_momentum,
    normalize_scores,
    build_rv_snapshot,
    get_rv_score,
)


class TestComputePairRelativeMomentum:
    """Test pair momentum computation."""

    def test_positive_spread_when_long_outperforms(self):
        """Positive spread when long symbol outperforms short."""
        returns = {
            "BTCUSDT": np.array([0.01, 0.02, 0.01, 0.015]),  # BTC up
            "ETHUSDT": np.array([0.005, 0.01, 0.005, 0.008]),  # ETH up less
        }
        
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", returns)
        
        assert spread > 0, "BTC outperforming ETH should give positive spread"

    def test_negative_spread_when_short_outperforms(self):
        """Negative spread when short symbol outperforms long."""
        returns = {
            "BTCUSDT": np.array([0.001, 0.002, 0.001]),  # BTC barely up
            "ETHUSDT": np.array([0.02, 0.03, 0.025]),  # ETH up more
        }
        
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", returns)
        
        assert spread < 0, "ETH outperforming BTC should give negative spread"

    def test_zero_spread_when_equal_performance(self):
        """Zero spread when symbols perform equally."""
        returns = {
            "BTCUSDT": np.array([0.01, 0.02, 0.01]),
            "ETHUSDT": np.array([0.01, 0.02, 0.01]),
        }
        
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", returns)
        
        assert abs(spread) < 1e-10, "Equal performance should give zero spread"

    def test_missing_symbol_uses_zeros(self):
        """Missing symbol is treated as zeros (returns non-zero spread)."""
        returns = {
            "BTCUSDT": np.array([0.01, 0.02]),
        }
        
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", returns)
        
        # BTC vs zeros = positive spread (BTC outperforming "nothing")
        assert spread > 0

    def test_both_symbols_missing_returns_zero(self):
        """Both symbols missing returns zero spread."""
        returns = {
            "SOLUSDT": np.array([0.01, 0.02]),
        }
        
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", returns)
        
        # Both default to zeros, so spread is zero
        assert spread == 0.0

    def test_empty_returns_dict(self):
        """Empty returns dict returns zero spread."""
        spread = compute_pair_relative_momentum("BTCUSDT", "ETHUSDT", {})
        
        assert spread == 0.0


class TestComputeBasketRelativeMomentum:
    """Test basket momentum computation."""

    def test_positive_when_group_a_outperforms(self):
        """Positive spread when group A outperforms group B."""
        returns = {
            "BTCUSDT": np.array([0.02, 0.02, 0.02]),
            "ETHUSDT": np.array([0.015, 0.015, 0.015]),
            "SOLUSDT": np.array([0.025, 0.025, 0.025]),  # L1 average = 0.02
            "LTCUSDT": np.array([0.005, 0.005, 0.005]),
            "LINKUSDT": np.array([0.01, 0.01, 0.01]),  # ALTS average = 0.0075
        }
        
        l1_basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        alts_basket = ["LTCUSDT", "LINKUSDT"]
        
        spread = compute_basket_relative_momentum(l1_basket, alts_basket, returns)
        
        assert spread > 0, "L1 outperforming ALTs should give positive spread"

    def test_negative_when_group_b_outperforms(self):
        """Negative spread when group B outperforms group A."""
        returns = {
            "BTCUSDT": np.array([0.001, 0.001]),
            "ETHUSDT": np.array([0.001, 0.001]),  # L1 weak
            "LTCUSDT": np.array([0.05, 0.05]),
            "LINKUSDT": np.array([0.05, 0.05]),  # ALTS strong
        }
        
        l1_basket = ["BTCUSDT", "ETHUSDT"]
        alts_basket = ["LTCUSDT", "LINKUSDT"]
        
        spread = compute_basket_relative_momentum(l1_basket, alts_basket, returns)
        
        assert spread < 0

    def test_empty_basket_returns_zero(self):
        """Empty basket returns zero spread."""
        returns = {
            "BTCUSDT": np.array([0.01, 0.02]),
        }
        
        spread = compute_basket_relative_momentum(["BTCUSDT"], [], returns)
        
        assert spread == 0.0

    def test_meme_vs_rest_calculation(self):
        """Meme vs rest basket calculation works."""
        returns = {
            "DOGEUSDT": np.array([0.05, 0.05, 0.05]),  # Meme strong
            "WIFUSDT": np.array([0.04, 0.04, 0.04]),
            "BTCUSDT": np.array([0.01, 0.01, 0.01]),  # Rest weak
            "ETHUSDT": np.array([0.01, 0.01, 0.01]),
        }
        
        meme = ["DOGEUSDT", "WIFUSDT"]
        rest = ["BTCUSDT", "ETHUSDT"]
        
        spread = compute_basket_relative_momentum(meme, rest, returns)
        
        assert spread > 0, "Meme outperforming rest should give positive spread"


class TestNormalizeScores:
    """Test score normalization."""

    def test_zscore_normalization(self):
        """Z-score normalization produces bounded output."""
        raw_scores = {
            "BTCUSDT": 0.5,
            "ETHUSDT": 0.2,
            "SOLUSDT": -0.1,
            "DOGEUSDT": -0.3,
        }
        
        normalized = normalize_scores(raw_scores, mode="zscore", max_abs=1.0)
        
        # All values should be bounded
        for sym, score in normalized.items():
            assert -1.0 <= score <= 1.0, f"{sym} score {score} out of bounds"
        
        # Order should be preserved
        assert normalized["BTCUSDT"] > normalized["ETHUSDT"]
        assert normalized["ETHUSDT"] > normalized["SOLUSDT"]
        assert normalized["SOLUSDT"] > normalized["DOGEUSDT"]

    def test_rank_normalization(self):
        """Rank normalization distributes evenly."""
        raw_scores = {
            "A": 100.0,
            "B": 50.0,
            "C": 10.0,
            "D": -5.0,
        }
        
        normalized = normalize_scores(raw_scores, mode="rank", max_abs=1.0)
        
        # Should span from -1 to +1
        assert normalized["D"] == -1.0  # Lowest rank
        assert normalized["A"] == 1.0   # Highest rank
        # Middle ranks interpolated
        assert -1.0 < normalized["B"] < 1.0
        assert -1.0 < normalized["C"] < 1.0

    def test_empty_scores_returns_empty(self):
        """Empty input returns empty output."""
        normalized = normalize_scores({}, mode="zscore", max_abs=1.0)
        
        assert normalized == {}

    def test_single_score_returns_zero(self):
        """Single score returns zero (no variance)."""
        raw_scores = {"BTCUSDT": 0.5}
        
        normalized = normalize_scores(raw_scores, mode="zscore", max_abs=1.0)
        
        assert normalized["BTCUSDT"] == 0.0

    def test_all_equal_scores_return_zero(self):
        """All equal scores return zero."""
        raw_scores = {"A": 0.5, "B": 0.5, "C": 0.5}
        
        normalized = normalize_scores(raw_scores, mode="zscore", max_abs=1.0)
        
        for score in normalized.values():
            assert score == 0.0

    def test_max_abs_scaling(self):
        """Scores are scaled to max_abs."""
        raw_scores = {"A": 100.0, "B": -100.0}
        
        normalized = normalize_scores(raw_scores, mode="rank", max_abs=0.5)
        
        assert max(normalized.values()) == 0.5
        assert min(normalized.values()) == -0.5


class TestBuildRvSnapshot:
    """Test full RV snapshot building."""

    @pytest.fixture
    def sample_returns(self):
        return {
            "BTCUSDT": np.array([0.02, 0.02, 0.02, 0.02]),  # Strong
            "ETHUSDT": np.array([0.01, 0.01, 0.01, 0.01]),
            "SOLUSDT": np.array([0.015, 0.015, 0.015, 0.015]),
            "LTCUSDT": np.array([0.005, 0.005, 0.005, 0.005]),  # Weak
            "LINKUSDT": np.array([0.008, 0.008, 0.008, 0.008]),
            "DOGEUSDT": np.array([0.03, 0.03, 0.03, 0.03]),  # Meme strong
            "WIFUSDT": np.array([0.025, 0.025, 0.025, 0.025]),
        }

    @pytest.fixture
    def sample_baskets_cfg(self):
        return {
            "pairs": {
                "btc_vs_eth": {"long": "BTCUSDT", "short": "ETHUSDT"}
            },
            "baskets": {
                "l1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "alts": ["LTCUSDT", "LINKUSDT"],
                "meme": ["DOGEUSDT", "WIFUSDT"],
            }
        }

    def test_builds_snapshot_with_all_fields(self, sample_returns, sample_baskets_cfg):
        """Snapshot contains all required fields."""
        cfg = RvConfig()
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        
        assert isinstance(snapshot, RvSnapshot)
        assert snapshot.updated_ts > 0
        assert isinstance(snapshot.btc_vs_eth_spread, float)
        assert isinstance(snapshot.l1_vs_alt_spread, float)
        assert isinstance(snapshot.meme_vs_rest_spread, float)
        assert isinstance(snapshot.per_symbol, dict)

    def test_btc_outperforming_gives_positive_btc_eth_spread(self, sample_returns, sample_baskets_cfg):
        """BTC outperforming ETH gives positive BTC vs ETH spread."""
        cfg = RvConfig()
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        
        # BTC return (0.02) > ETH return (0.01)
        assert snapshot.btc_vs_eth_spread > 0

    def test_l1_outperforming_gives_positive_l1_alt_spread(self, sample_returns, sample_baskets_cfg):
        """L1 outperforming ALTs gives positive L1 vs ALT spread."""
        cfg = RvConfig()
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        
        # L1 average (~0.015) > ALTS average (~0.0065)
        assert snapshot.l1_vs_alt_spread > 0

    def test_per_symbol_scores_normalized(self, sample_returns, sample_baskets_cfg):
        """Per-symbol scores are normalized to [-max_abs, max_abs]."""
        cfg = RvConfig(max_abs_score=1.0)
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        
        for sym, entry in snapshot.per_symbol.items():
            assert -1.0 <= entry.score <= 1.0, f"{sym} score out of bounds"

    def test_symbol_baskets_recorded(self, sample_returns, sample_baskets_cfg):
        """Symbol basket membership is recorded."""
        cfg = RvConfig()
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        
        assert "l1" in snapshot.per_symbol["BTCUSDT"].baskets
        assert "alts" in snapshot.per_symbol["LTCUSDT"].baskets
        assert "meme" in snapshot.per_symbol["DOGEUSDT"].baskets

    def test_to_dict_serializable(self, sample_returns, sample_baskets_cfg):
        """Snapshot converts to serializable dict."""
        cfg = RvConfig()
        
        snapshot = build_rv_snapshot(cfg, sample_returns, sample_baskets_cfg)
        d = snapshot.to_dict()
        
        assert "updated_ts" in d
        assert "per_symbol" in d
        assert "spreads" in d
        assert "btc_vs_eth" in d["spreads"]
        
        # Values should be rounded
        for sym_data in d["per_symbol"].values():
            assert isinstance(sym_data["score"], float)


class TestGetRvScore:
    """Test RV score retrieval."""

    def test_get_score_from_snapshot(self):
        """Get score from pre-computed snapshot."""
        snapshot = RvSnapshot(
            per_symbol={
                "BTCUSDT": RvSymbolScore(symbol="BTCUSDT", score=0.75, raw_score=0.5, baskets=["l1"]),
            },
            btc_vs_eth_spread=0.01,
            l1_vs_alt_spread=0.02,
            meme_vs_rest_spread=-0.01,
        )
        
        score = get_rv_score("BTCUSDT", snapshot)
        
        assert score == 0.75

    def test_missing_symbol_returns_zero(self):
        """Missing symbol returns 0.0."""
        snapshot = RvSnapshot(per_symbol={})
        
        score = get_rv_score("UNKNOWNUSDT", snapshot)
        
        assert score == 0.0

    def test_disabled_config_returns_zero(self):
        """Disabled config returns 0.0."""
        cfg = RvConfig(enabled=False)
        
        score = get_rv_score("BTCUSDT", rv_snapshot=None, cfg=cfg)
        
        assert score == 0.0
