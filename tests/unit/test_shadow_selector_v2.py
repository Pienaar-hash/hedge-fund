"""Tests for shadow_selector_v2 — Phase 5 research scaffold.

Covers:
  - classify_regime correctness for all symbols
  - Candidate A: Hydra preference band logic
  - Candidate B: positive-region routing + abstention
  - Candidate C: placeholder returns
  - evaluate_v2_shadow event structure and env gating
  - JSONL logging (append-only, fail-open)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ── Regime classification ────────────────────────────────────────────────────

class TestClassifyRegime:
    def test_btc_hydra_regime(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("BTCUSDT", 0.45) == "HYDRA_REGIME"

    def test_btc_legacy_regime(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("BTCUSDT", 0.55) == "LEGACY_REGIME"

    def test_btc_boundary_exact(self):
        from execution.shadow_selector_v2 import classify_regime
        # At boundary → legacy (>= threshold)
        assert classify_regime("BTCUSDT", 0.5236) == "LEGACY_REGIME"

    def test_btc_just_below_boundary(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("BTCUSDT", 0.5235) == "HYDRA_REGIME"

    def test_eth_three_regimes(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("ETHUSDT", 0.40) == "LEGACY_LOW"
        assert classify_regime("ETHUSDT", 0.45) == "HYDRA_REGIME"
        assert classify_regime("ETHUSDT", 0.50) == "LEGACY_HIGH"

    def test_eth_boundaries(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("ETHUSDT", 0.4291) == "HYDRA_REGIME"
        assert classify_regime("ETHUSDT", 0.4883) == "LEGACY_HIGH"

    def test_sol_legacy_only(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("SOLUSDT", 0.30) == "LEGACY_ONLY"
        assert classify_regime("SOLUSDT", 0.90) == "LEGACY_ONLY"

    def test_unknown_symbol(self):
        from execution.shadow_selector_v2 import classify_regime
        assert classify_regime("DOGEUSDT", 0.50) == "UNKNOWN"


# ── Candidate A: Hydra preference band ──────────────────────────────────────

class TestSelectorA:
    def test_btc_inside_band(self):
        from execution.shadow_selector_v2 import _selector_a
        r = _selector_a("BTCUSDT", 0.45, "legacy")
        assert r["v2_choice"] == "hydra"
        assert r["v2_abstain"] is False
        assert r["rule"] == "A_band_hit"

    def test_btc_outside_band_fallback(self):
        from execution.shadow_selector_v2 import _selector_a
        r = _selector_a("BTCUSDT", 0.55, "legacy")
        assert r["v2_choice"] == "legacy"  # falls back to ECS choice
        assert r["v2_abstain"] is False
        assert r["rule"] == "A_ecs_fallback"

    def test_sol_no_band(self):
        from execution.shadow_selector_v2 import _selector_a
        r = _selector_a("SOLUSDT", 0.40, "legacy")
        assert r["v2_choice"] == "legacy"  # no band → fallback
        assert r["rule"] == "A_ecs_fallback"

    def test_eth_inside_band(self):
        from execution.shadow_selector_v2 import _selector_a
        r = _selector_a("ETHUSDT", 0.46, "legacy")
        assert r["v2_choice"] == "hydra"
        assert r["rule"] == "A_band_hit"

    def test_band_boundary_inclusive(self):
        from execution.shadow_selector_v2 import _selector_a
        # Lower boundary
        r = _selector_a("BTCUSDT", 0.42, "legacy")
        assert r["v2_choice"] == "hydra"
        # Upper boundary
        r = _selector_a("BTCUSDT", 0.52, "legacy")
        assert r["v2_choice"] == "hydra"


# ── Candidate B: Positive-region routing ─────────────────────────────────────

class TestSelectorB:
    def test_btc_hydra_region(self):
        from execution.shadow_selector_v2 import _selector_b
        r = _selector_b("BTCUSDT", 0.45)
        assert r["v2_choice"] == "hydra"
        assert r["v2_abstain"] is False
        assert r["rule"] == "B_hydra_regime"

    def test_btc_legacy_region_abstain(self):
        from execution.shadow_selector_v2 import _selector_b
        r = _selector_b("BTCUSDT", 0.55)
        assert r["v2_choice"] == "none"
        assert r["v2_abstain"] is True
        assert r["rule"] == "B_abstain"

    def test_eth_hydra_middle(self):
        from execution.shadow_selector_v2 import _selector_b
        r = _selector_b("ETHUSDT", 0.45)
        assert r["v2_choice"] == "hydra"
        assert r["v2_abstain"] is False

    def test_eth_legacy_tails_abstain(self):
        from execution.shadow_selector_v2 import _selector_b
        # Low tail
        r = _selector_b("ETHUSDT", 0.40)
        assert r["v2_abstain"] is True
        # High tail
        r = _selector_b("ETHUSDT", 0.50)
        assert r["v2_abstain"] is True

    def test_sol_always_abstain(self):
        from execution.shadow_selector_v2 import _selector_b
        r = _selector_b("SOLUSDT", 0.40)
        assert r["v2_abstain"] is True
        assert r["rule"] == "B_abstain"


# ── Candidate C: Placeholder ─────────────────────────────────────────────────

class TestSelectorC:
    def test_always_inactive(self):
        from execution.shadow_selector_v2 import _selector_c
        r = _selector_c("BTCUSDT", 0.45, 0.55)
        assert r["v2_choice"] is None
        assert r["v2_abstain"] is None
        assert r["rule"] == "C_inactive"


# ── evaluate_v2_shadow integration ───────────────────────────────────────────

class TestEvaluateV2Shadow:
    def test_disabled_by_default(self):
        from execution.shadow_selector_v2 import evaluate_v2_shadow
        with mock.patch.dict(os.environ, {"SELECTOR_V2_SHADOW": "0"}):
            result = evaluate_v2_shadow(
                symbol="BTCUSDT", hydra_score=0.45, legacy_score=0.55,
                ecs_choice="legacy", cycle=1,
            )
        assert result is None

    def test_returns_none_without_hydra_score(self):
        from execution.shadow_selector_v2 import evaluate_v2_shadow
        with mock.patch.dict(os.environ, {"SELECTOR_V2_SHADOW": "1"}):
            result = evaluate_v2_shadow(
                symbol="BTCUSDT", hydra_score=None, legacy_score=0.55,
                ecs_choice="legacy", cycle=1,
            )
        assert result is None

    def test_full_event_structure(self):
        from execution.shadow_selector_v2 import evaluate_v2_shadow
        with mock.patch.dict(os.environ, {"SELECTOR_V2_SHADOW": "1"}):
            with mock.patch("execution.shadow_selector_v2._append_v2_event"):
                result = evaluate_v2_shadow(
                    symbol="BTCUSDT", hydra_score=0.45, legacy_score=0.55,
                    ecs_choice="legacy", merge_conflict=True, cycle=42,
                )
        assert result is not None
        # Four canonical columns present
        assert result["ecs_choice"] == "legacy"
        assert result["a_choice"] == "hydra"  # inside BTC band
        assert result["b_choice"] == "hydra"  # inside BTC Hydra regime
        assert result["b_abstain"] is False
        # Metadata
        assert result["schema"] == "selector_v2_shadow_v1"
        assert result["symbol"] == "BTCUSDT"
        assert result["hydra_score"] == 0.45
        assert result["legacy_score"] == 0.55
        assert result["hydra_regime_band"] == "HYDRA_REGIME"
        assert result["ecs_conflict"] is True
        assert result["cycle"] == 42
        # Score delta
        assert abs(result["score_delta"] - (-0.1)) < 1e-5

    def test_btc_legacy_region_produces_abstain(self):
        from execution.shadow_selector_v2 import evaluate_v2_shadow
        with mock.patch.dict(os.environ, {"SELECTOR_V2_SHADOW": "1"}):
            with mock.patch("execution.shadow_selector_v2._append_v2_event"):
                result = evaluate_v2_shadow(
                    symbol="BTCUSDT", hydra_score=0.55, legacy_score=0.60,
                    ecs_choice="legacy", cycle=1,
                )
        assert result is not None
        assert result["b_abstain"] is True
        assert result["b_choice"] == "none"
        assert result["a_choice"] == "legacy"  # A falls back to ECS

    def test_sol_all_abstain(self):
        from execution.shadow_selector_v2 import evaluate_v2_shadow
        with mock.patch.dict(os.environ, {"SELECTOR_V2_SHADOW": "1"}):
            with mock.patch("execution.shadow_selector_v2._append_v2_event"):
                result = evaluate_v2_shadow(
                    symbol="SOLUSDT", hydra_score=0.40, legacy_score=0.55,
                    ecs_choice="legacy", cycle=1,
                )
        assert result is not None
        assert result["b_abstain"] is True
        # A: no SOL bands → fallback to ECS
        assert result["a_choice"] == "legacy"


# ── JSONL logging ────────────────────────────────────────────────────────────

class TestLogging:
    def test_append_creates_file(self, tmp_path):
        from execution.shadow_selector_v2 import _append_v2_event, _V2_LOG_PATH
        log_path = tmp_path / "test_v2.jsonl"
        with mock.patch("execution.shadow_selector_v2._V2_LOG_PATH", log_path):
            _append_v2_event({"test": True, "ts": 1234})

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["test"] is True

    def test_append_is_additive(self, tmp_path):
        from execution.shadow_selector_v2 import _append_v2_event
        log_path = tmp_path / "test_v2.jsonl"
        with mock.patch("execution.shadow_selector_v2._V2_LOG_PATH", log_path):
            _append_v2_event({"n": 1})
            _append_v2_event({"n": 2})

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_append_fail_open(self, tmp_path):
        """Logging failure must not raise."""
        from execution.shadow_selector_v2 import _append_v2_event
        bad_path = tmp_path / "nonexistent" / "deep" / "path" / "log.jsonl"
        # This should create the directory tree
        with mock.patch("execution.shadow_selector_v2._V2_LOG_PATH", bad_path):
            _append_v2_event({"test": True})
        assert bad_path.exists()
