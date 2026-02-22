"""Tests for Bayesian expectancy prior and blending (v7.9).

Verifies:
  - Prior produces non-neutral expectancy with zero episodes
  - Prior increases with higher regime confidence + good router health
  - Prior decreases with poor router health or extreme vol state
  - Blending returns prior when episode_count == 0
  - Blending converges to posterior as episodes grow
  - build_expectancy_snapshot with prior_inputs enriches immature symbols
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from execution.intel.expectancy_v6 import (
    compute_expectancy_prior,
    blend_expectancy,
    build_expectancy_snapshot,
    MIN_EXPECTANCY_TRADES,
)


class TestComputeExpectancyPrior:
    """Test prior expectancy from observable inputs."""

    def test_returns_non_null_expectancy(self):
        result = compute_expectancy_prior()
        assert result["expectancy"] is not None
        assert result["is_prior"] is True
        assert result["count"] == 0

    def test_trend_up_high_confidence_positive(self):
        result = compute_expectancy_prior(
            regime="TREND_UP",
            regime_confidence=0.9,
        )
        assert result["expectancy"] > 0, "TREND_UP with high confidence should be positive"

    def test_crisis_negative(self):
        result = compute_expectancy_prior(
            regime="CRISIS",
            regime_confidence=0.9,
        )
        assert result["expectancy"] < 0, "CRISIS should produce negative expectancy"

    def test_good_router_health_boosts(self):
        base = compute_expectancy_prior(router_health_score=0.5)
        good = compute_expectancy_prior(router_health_score=0.9)
        assert good["expectancy"] > base["expectancy"]

    def test_poor_router_health_reduces(self):
        base = compute_expectancy_prior(router_health_score=0.5)
        poor = compute_expectancy_prior(router_health_score=0.1)
        assert poor["expectancy"] < base["expectancy"]

    def test_high_vol_reduces(self):
        base = compute_expectancy_prior(vol_state="normal")
        high = compute_expectancy_prior(vol_state="high")
        assert high["expectancy"] < base["expectancy"]

    def test_crisis_vol_reduces_further(self):
        high = compute_expectancy_prior(vol_state="high")
        crisis = compute_expectancy_prior(vol_state="crisis")
        assert crisis["expectancy"] < high["expectancy"]

    def test_trend_and_carry_alignment_boosts(self):
        neutral = compute_expectancy_prior(trend_score=0.5, carry_score=0.5)
        aligned = compute_expectancy_prior(trend_score=0.8, carry_score=0.7)
        assert aligned["expectancy"] > neutral["expectancy"]

    def test_components_present(self):
        result = compute_expectancy_prior(regime="TREND_UP", regime_confidence=0.8)
        assert "prior_components" in result
        assert "regime_base" in result["prior_components"]
        assert "vol_adj" in result["prior_components"]
        assert "router_adj" in result["prior_components"]
        assert "trend_adj" in result["prior_components"]
        assert "carry_adj" in result["prior_components"]

    def test_prior_inputs_recorded(self):
        result = compute_expectancy_prior(regime="TREND_UP", regime_confidence=0.8)
        assert "prior_inputs" in result
        assert result["prior_inputs"]["regime"] == "TREND_UP"


class TestBlendExpectancy:
    """Test Bayesian shrinkage blending."""

    def test_zero_episodes_returns_prior(self):
        prior = compute_expectancy_prior(regime="TREND_UP", regime_confidence=0.9)
        posterior = {"expectancy": 0.0, "expectancy_per_risk": 0.0, "count": 0}
        result = blend_expectancy(prior, posterior, episode_count=0)
        assert result["expectancy"] == prior["expectancy"]

    def test_none_posterior_returns_prior(self):
        prior = compute_expectancy_prior(regime="TREND_UP", regime_confidence=0.9)
        posterior = {"expectancy": None}
        result = blend_expectancy(prior, posterior, episode_count=10)
        assert result["expectancy"] == prior["expectancy"]

    def test_maturity_converges_to_posterior(self):
        prior = compute_expectancy_prior(regime="CRISIS", regime_confidence=0.9)
        posterior = {
            "expectancy": 0.05,
            "expectancy_per_risk": 0.3,
            "count": 60,
            "hit_rate": 0.55,
            "is_mature": True,
        }
        result = blend_expectancy(prior, posterior, episode_count=60, maturity_n=60)
        # At full maturity, posterior dominates but prior retains minimum weight
        assert abs(result["expectancy"] - 0.05) < 0.02

    def test_partial_episodes_blends(self):
        prior = compute_expectancy_prior()
        posterior = {"expectancy": 0.10, "expectancy_per_risk": 0.5, "count": 15}
        result = blend_expectancy(prior, posterior, episode_count=15, maturity_n=60)
        # Should be between prior and posterior
        assert result["is_prior"] is False
        assert "blend_weights" in result
        assert result["blend_weights"]["prior"] > 0.3  # still weighted toward prior

    def test_blend_weights_recorded(self):
        prior = compute_expectancy_prior()
        posterior = {"expectancy": 0.10, "expectancy_per_risk": 0.5}
        result = blend_expectancy(prior, posterior, episode_count=30, maturity_n=60)
        assert "blend_weights" in result
        w = result["blend_weights"]
        assert abs(w["prior"] + w["posterior"] - 1.0) < 1e-6


class TestBuildExpectancySnapshotWithPrior:
    """Test build_expectancy_snapshot with prior_inputs."""

    def test_without_prior_backward_compatible(self):
        snapshot = build_expectancy_snapshot([], 48.0)
        assert "symbols" in snapshot
        assert snapshot["sample_count"] == 0

    def test_with_prior_enriches_empty_symbols(self):
        prior_inputs = {
            "regime": "TREND_UP",
            "regime_confidence": 0.8,
            "vol_state": "normal",
            "router_health_score": 0.7,
            "trend_score": 0.6,
            "carry_score": 0.55,
            "symbols": ["BTCUSDT", "ETHUSDT"],
        }
        snapshot = build_expectancy_snapshot([], 48.0, prior_inputs=prior_inputs)
        assert "BTCUSDT" in snapshot["symbols"]
        assert "ETHUSDT" in snapshot["symbols"]
        btc = snapshot["symbols"]["BTCUSDT"]
        assert btc["expectancy"] is not None
        assert btc.get("is_prior") is True

    def test_has_prior_flag_set(self):
        snapshot = build_expectancy_snapshot([], 48.0, prior_inputs={"regime": "TREND_UP"})
        assert snapshot.get("has_prior") is True

    def test_without_prior_no_flag(self):
        snapshot = build_expectancy_snapshot([], 48.0)
        assert "has_prior" not in snapshot


class TestScoreSymbolPriorScaling:
    """Verify score_symbol uses expectancy_per_risk for prior entries."""

    def test_prior_entry_uses_expectancy_per_risk(self):
        """Prior entry should use expectancy_per_risk (pre-scaled), not raw expectancy."""
        from execution.intel.symbol_score_v6 import score_symbol

        # Prior entry: small raw expectancy, larger per_risk
        prior_metrics = {
            "expectancy": {
                "expectancy": 0.08,            # raw prior: tiny → tanh(0.008)≈0.008
                "expectancy_per_risk": 0.80,    # pre-scaled: tanh(0.08)≈0.08
                "is_prior": True,
            },
            "router": {"maker_fill_rate": 0.7, "fallback_rate": 0.1},
        }
        result_prior = score_symbol("BTCUSDT", prior_metrics)
        exp_comp_prior = result_prior["components"]["expectancy"]

        # Non-prior entry with same raw expectancy
        posterior_metrics = {
            "expectancy": {
                "expectancy": 0.08,
                "expectancy_per_risk": 0.80,
                # no is_prior key → uses raw expectancy
            },
            "router": {"maker_fill_rate": 0.7, "fallback_rate": 0.1},
        }
        result_posterior = score_symbol("BTCUSDT", posterior_metrics)
        exp_comp_posterior = result_posterior["components"]["expectancy"]

        # Prior should produce a LARGER expectancy component because
        # it uses the pre-scaled expectancy_per_risk (0.80) vs raw (0.08)
        assert exp_comp_prior > exp_comp_posterior

    def test_non_prior_uses_raw_expectancy(self):
        """Non-prior (trade-derived) uses raw expectancy, not per_risk."""
        from execution.intel.symbol_score_v6 import score_symbol

        result = score_symbol("ETHUSDT", {
            "expectancy": {"expectancy": 5.0, "hit_rate": 0.6},
            "router": {},
        })
        # 5.0 is trade-scale → tanh(0.5) ≈ 0.462 → 0.5 + 0.5*0.462 ≈ 0.731
        exp = result["components"]["expectancy"]
        assert 0.65 < exp < 0.85  # well above 0.5
