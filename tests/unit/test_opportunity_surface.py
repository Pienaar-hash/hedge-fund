"""Tests for execution/opportunity_surface.py (P5A + P5C).

Conservation invariants are tested as hard assertions (not approximate).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List

import pytest

from execution.opportunity_surface import (
    _band_key,
    _duration_bucket,
    _is_friction_dominated,
    _safe_share,
    build_composition_audit,
    build_full_surface,
    compute_duration_quality,
    compute_exit_class_pnl,
    compute_friction_burden,
    compute_regime_mismatch_cost,
    compute_symbol_drag,
    diagnose_band_drag,
    load_surface,
    save_surface,
    summarize_loss_drivers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _ep(
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    conviction: float = 0.45,
    gross_pnl: float = -1.0,
    fees: float = 0.5,
    net_pnl: float = -1.5,
    entry_notional: float = 1000.0,
    regime: str = "MEAN_REVERT",
    exit_reason: str = "THESIS_INVALIDATED",
    duration_hours: float = 3.0,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a minimal episode dict."""
    d: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "conviction_score": conviction,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "entry_notional": entry_notional,
        "regime_at_entry": regime,
        "exit_reason": exit_reason,
        "duration_hours": duration_hours,
    }
    d.update(extra)
    return d


def _make_pool(n: int = 20, **defaults: Any) -> List[Dict[str, Any]]:
    """Diverse pool of episodes for testing."""
    pool = []
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    sides = ["LONG", "SHORT"]
    regimes = ["MEAN_REVERT", "TREND_UP", "unknown"]
    exits = ["THESIS_INVALIDATED", "REGIME_CHANGE", "TAKE_PROFIT"]
    durations = [0.5, 3.0, 12.0, 30.0]

    for i in range(n):
        sym = symbols[i % len(symbols)]
        side = sides[i % len(sides)]
        regime = regimes[i % len(regimes)]
        exit_r = exits[i % len(exits)]
        dur = durations[i % len(durations)]
        # Vary PnL: some winners, mostly losers
        gross = 2.0 if i % 5 == 0 else -1.0
        fee = 0.5
        net = gross - fee
        conviction = 0.40 + (i % 6) * 0.05  # 0.40 .. 0.65
        pool.append(_ep(
            symbol=sym, side=side, conviction=conviction,
            gross_pnl=gross, fees=fee, net_pnl=net,
            entry_notional=1000.0, regime=regime,
            exit_reason=exit_r, duration_hours=dur,
            **defaults,
        ))
    return pool


# ── Helpers ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_band_key(self):
        assert _band_key(0.42, 0.05) == "0.40-0.45"
        assert _band_key(0.45, 0.05) == "0.45-0.50"
        assert _band_key(0.50, 0.05) == "0.50-0.55"

    def test_duration_bucket(self):
        assert _duration_bucket(0.5) == "lt_1h"
        assert _duration_bucket(3.0) == "1h_6h"
        assert _duration_bucket(12.0) == "6h_24h"
        assert _duration_bucket(30.0) == "gt_24h"
        assert _duration_bucket(0.0) == "lt_1h"

    def test_friction_dominated_true(self):
        assert _is_friction_dominated({"gross_pnl": 1.0, "net_pnl": -0.5})
        assert _is_friction_dominated({"gross_pnl": 0.01, "net_pnl": 0.0})

    def test_friction_dominated_false(self):
        assert not _is_friction_dominated({"gross_pnl": -1.0, "net_pnl": -1.5})
        assert not _is_friction_dominated({"gross_pnl": 1.0, "net_pnl": 0.5})
        assert not _is_friction_dominated({"gross_pnl": 0.0, "net_pnl": -0.5})

    def test_safe_share(self):
        assert _safe_share(-5.0, -10.0) == 0.5
        assert _safe_share(-3.0, -10.0) == pytest.approx(0.3)
        assert _safe_share(1.0, -10.0) == 0.0   # positive part
        assert _safe_share(-1.0, 5.0) == 0.0    # positive total


# ── P5A: Band Composition Audit ──────────────────────────────────────────

class TestBandCompositionAudit:
    def test_empty_episodes(self):
        audit = build_composition_audit([])
        assert audit == {}

    def test_single_band(self):
        eps = [_ep(conviction=0.42), _ep(conviction=0.43)]
        audit = build_composition_audit(eps)
        assert "0.40-0.45" in audit
        band = audit["0.40-0.45"]
        assert band["episode_count"] == 2

    def test_multiple_bands(self):
        eps = [_ep(conviction=0.42), _ep(conviction=0.52)]
        audit = build_composition_audit(eps)
        assert len(audit) == 2
        assert "0.40-0.45" in audit
        assert "0.50-0.55" in audit

    def test_symbol_breakdown_conservation(self):
        """Sum of symbol net_pnl == band net_pnl."""
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            sym_sum = sum(c["net_pnl"] for c in band["symbol_breakdown"].values())
            assert abs(sym_sum - band["net_pnl"]) < 1e-4, \
                f"Symbol conservation failed for {band_key}"

    def test_side_breakdown_conservation(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            side_sum = sum(c["net_pnl"] for c in band["side_breakdown"].values())
            assert abs(side_sum - band["net_pnl"]) < 1e-4, \
                f"Side conservation failed for {band_key}"

    def test_regime_breakdown_conservation(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            r_sum = sum(c["net_pnl"] for c in band["regime_breakdown"].values())
            assert abs(r_sum - band["net_pnl"]) < 1e-4, \
                f"Regime conservation failed for {band_key}"

    def test_exit_reason_breakdown_conservation(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            ex_sum = sum(c["net_pnl"] for c in band["exit_reason_breakdown"].values())
            assert abs(ex_sum - band["net_pnl"]) < 1e-4, \
                f"Exit reason conservation failed for {band_key}"

    def test_duration_breakdown_conservation(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            d_sum = sum(c["net_pnl"] for c in band["duration_breakdown"].values())
            assert abs(d_sum - band["net_pnl"]) < 1e-4, \
                f"Duration conservation failed for {band_key}"

    def test_friction_dominated_fields(self):
        # Mix: one friction-dominated, one not
        eps = [
            _ep(conviction=0.42, gross_pnl=1.0, fees=1.5, net_pnl=-0.5),  # friction
            _ep(conviction=0.43, gross_pnl=-2.0, fees=0.5, net_pnl=-2.5),  # not friction
        ]
        audit = build_composition_audit(eps)
        band = audit["0.40-0.45"]
        assert band["friction_dominated_count"] == 1
        assert band["friction_dominated_pnl"] == pytest.approx(-0.5)
        assert band["non_friction_pnl"] == pytest.approx(-2.5)
        # Conservation: friction + non_friction == total
        assert abs(band["friction_dominated_pnl"] + band["non_friction_pnl"]
                    - band["net_pnl"]) < 1e-6

    def test_top_dragger(self):
        eps = [
            _ep(symbol="BTCUSDT", conviction=0.42, net_pnl=-5.0, gross_pnl=-4.5, fees=0.5),
            _ep(symbol="ETHUSDT", conviction=0.43, net_pnl=-1.0, gross_pnl=-0.5, fees=0.5),
        ]
        audit = build_composition_audit(eps)
        band = audit["0.40-0.45"]
        assert band["top_dragger"] == "BTCUSDT"
        assert band["top_dragger_share"] > 0.5

    def test_skips_low_conviction(self):
        eps = [_ep(conviction=0.10)]  # Below default 0.20
        audit = build_composition_audit(eps)
        assert audit == {}

    def test_skips_missing_net_pnl(self):
        eps = [_ep(conviction=0.42)]
        eps[0]["net_pnl"] = None
        audit = build_composition_audit(eps)
        assert audit == {}

    def test_skips_zero_notional(self):
        eps = [_ep(conviction=0.42, entry_notional=0)]
        audit = build_composition_audit(eps)
        assert audit == {}

    def test_share_of_band_loss_present(self):
        pool = _make_pool(20)
        audit = build_composition_audit(pool)
        for band_key, band in audit.items():
            for sym, cohort in band["symbol_breakdown"].items():
                assert "share_of_band_loss" in cohort


# ── Diagnose Band Drag ───────────────────────────────────────────────────

class TestDiagnoseBandDrag:
    def test_returns_sorted_by_loss(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        drags = diagnose_band_drag(audit)
        # Sorted most negative first
        for i in range(len(drags) - 1):
            assert drags[i]["net_pnl"] <= drags[i + 1]["net_pnl"]

    def test_only_negative_entries(self):
        pool = _make_pool(30)
        audit = build_composition_audit(pool)
        drags = diagnose_band_drag(audit)
        for d in drags:
            assert d["net_pnl"] < 0

    def test_empty_audit(self):
        assert diagnose_band_drag({}) == []


# ── P5C: Friction Burden ─────────────────────────────────────────────────

class TestFrictionBurden:
    def test_basic(self):
        eps = [
            _ep(gross_pnl=1.0, fees=1.5, net_pnl=-0.5),   # friction
            _ep(gross_pnl=-2.0, fees=0.5, net_pnl=-2.5),   # not
            _ep(gross_pnl=0.5, fees=0.5, net_pnl=0.0),     # friction (net==0)
        ]
        result = compute_friction_burden(eps)
        assert result["total_episodes"] == 3
        assert result["friction_dominated_count"] == 2
        assert result["friction_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_per_symbol(self):
        eps = [
            _ep(symbol="BTC", gross_pnl=1.0, fees=1.5, net_pnl=-0.5),
            _ep(symbol="ETH", gross_pnl=-2.0, fees=0.5, net_pnl=-2.5),
        ]
        result = compute_friction_burden(eps)
        assert result["per_symbol"]["BTC"]["friction"] == 1
        assert result["per_symbol"]["ETH"]["friction"] == 0

    def test_empty(self):
        result = compute_friction_burden([])
        assert result["total_episodes"] == 0


# ── P5C: Exit Class PnL ─────────────────────────────────────────────────

class TestExitClassPnl:
    def test_conservation(self):
        """Sum of exit class net_pnl == total net_pnl."""
        pool = _make_pool(40)
        result = compute_exit_class_pnl(pool)
        total = sum(float(e["net_pnl"]) for e in pool)
        check = sum(v["net_pnl"] for v in result.values())
        assert abs(check - total) < 1e-4

    def test_all_reasons_present(self):
        pool = _make_pool(30)
        result = compute_exit_class_pnl(pool)
        for reason in ["THESIS_INVALIDATED", "REGIME_CHANGE", "TAKE_PROFIT"]:
            assert reason in result

    def test_share_of_total_loss(self):
        pool = _make_pool(30)
        result = compute_exit_class_pnl(pool)
        for reason, data in result.items():
            if data["net_pnl"] < 0:
                assert data["share_of_total_loss"] > 0

    def test_median_loss(self):
        eps = [
            _ep(exit_reason="TP", net_pnl=-1.0, gross_pnl=-0.5, fees=0.5),
            _ep(exit_reason="TP", net_pnl=-3.0, gross_pnl=-2.5, fees=0.5),
            _ep(exit_reason="TP", net_pnl=5.0, gross_pnl=5.5, fees=0.5),
        ]
        result = compute_exit_class_pnl(eps)
        # Median of losses [-1.0, -3.0] = -2.0
        assert result["TP"]["median_loss"] == pytest.approx(-2.0)


# ── P5C: Duration Quality ───────────────────────────────────────────────

class TestDurationQuality:
    def test_buckets(self):
        eps = [
            _ep(duration_hours=0.5, net_pnl=-1.0, entry_notional=100),
            _ep(duration_hours=3.0, net_pnl=-2.0, entry_notional=200),
            _ep(duration_hours=12.0, net_pnl=1.0, entry_notional=300),
            _ep(duration_hours=30.0, net_pnl=-0.5, entry_notional=400),
        ]
        result = compute_duration_quality(eps)
        assert "lt_1h" in result
        assert "1h_6h" in result
        assert "6h_24h" in result
        assert "gt_24h" in result
        assert result["lt_1h"]["n"] == 1
        assert result["6h_24h"]["net_pnl"] == pytest.approx(1.0)


# ── P5C: Symbol Drag ────────────────────────────────────────────────────

class TestSymbolDrag:
    def test_conservation(self):
        """Sum of symbol net_pnl == total net_pnl."""
        pool = _make_pool(40)
        result = compute_symbol_drag(pool)
        total = sum(float(e["net_pnl"]) for e in pool)
        check = sum(v["net_pnl"] for v in result.values())
        assert abs(check - total) < 1e-4

    def test_share_of_total_loss(self):
        pool = _make_pool(30)
        result = compute_symbol_drag(pool)
        for sym, data in result.items():
            if data["net_pnl"] < 0:
                assert data["share_of_total_loss"] > 0

    def test_all_symbols_present(self):
        pool = _make_pool(30)
        result = compute_symbol_drag(pool)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert "SOLUSDT" in result


# ── P5C: Regime Mismatch Cost ───────────────────────────────────────────

class TestRegimeMismatchCost:
    def test_conservation(self):
        pool = _make_pool(30)
        result = compute_regime_mismatch_cost(pool)
        total = sum(float(e["net_pnl"]) for e in pool)
        check = (result["regime_mismatch"]["net_pnl"]
                 + result["thesis_driven"]["net_pnl"]
                 + result["other"]["net_pnl"])
        assert abs(check - total) < 1e-4

    def test_identifies_mismatch(self):
        eps = [
            _ep(exit_reason="REGIME_CHANGE", net_pnl=-3.0, gross_pnl=-2.5, fees=0.5),
            _ep(exit_reason="THESIS_INVALIDATED", net_pnl=-1.0, gross_pnl=-0.5, fees=0.5),
        ]
        result = compute_regime_mismatch_cost(eps)
        assert result["regime_mismatch"]["n"] == 1
        assert result["thesis_driven"]["n"] == 1

    def test_frozen_definition(self):
        """Only REGIME_CHANGE counts as mismatch — no narrative inference."""
        eps = [
            _ep(exit_reason="REGIME_CHANGE", net_pnl=-1.0, gross_pnl=-0.5, fees=0.5),
            _ep(exit_reason="STOP_LOSS", net_pnl=-5.0, gross_pnl=-4.5, fees=0.5),
        ]
        result = compute_regime_mismatch_cost(eps)
        assert result["regime_mismatch"]["n"] == 1
        assert result["other"]["n"] == 1  # STOP_LOSS is "other", not mismatch


# ── P5C: Loss Driver Summary ────────────────────────────────────────────

class TestLossDriverSummary:
    def test_sorted_by_magnitude(self):
        pool = _make_pool(30)
        drivers = summarize_loss_drivers(pool)
        for i in range(len(drivers) - 1):
            assert drivers[i]["net_pnl"] <= drivers[i + 1]["net_pnl"] or \
                   (drivers[i]["net_pnl"] == drivers[i + 1]["net_pnl"] and
                    drivers[i]["coverage_pct"] >= drivers[i + 1]["coverage_pct"])

    def test_has_required_fields(self):
        pool = _make_pool(20)
        drivers = summarize_loss_drivers(pool)
        required = {"driver", "net_pnl", "gross_pnl", "fees",
                     "episode_count", "share_of_total_loss",
                     "median_loss", "coverage_pct"}
        for d in drivers:
            assert required <= set(d.keys())

    def test_coverage_pct(self):
        pool = _make_pool(20)
        drivers = summarize_loss_drivers(pool)
        # Coverage of symbol:BTCUSDT should be ~1/3
        btc = [d for d in drivers if d["driver"] == "symbol:BTCUSDT"]
        if btc:
            assert 0.2 <= btc[0]["coverage_pct"] <= 0.5

    def test_empty(self):
        assert summarize_loss_drivers([]) == []

    def test_friction_class_tag(self):
        eps = [
            _ep(gross_pnl=1.0, fees=1.5, net_pnl=-0.5),  # friction dominated
        ]
        drivers = summarize_loss_drivers(eps)
        tags = [d["driver"] for d in drivers]
        assert "class:friction_dominated" in tags


# ── Full Surface Build ───────────────────────────────────────────────────

class TestFullSurface:
    def test_builds_without_error(self):
        pool = _make_pool(30)
        surface = build_full_surface(pool)
        assert "meta" in surface
        assert "band_audit" in surface
        assert "friction_burden" in surface
        assert "exit_class_pnl" in surface
        assert "duration_quality" in surface
        assert "symbol_drag" in surface
        assert "regime_mismatch" in surface
        assert "loss_drivers" in surface

    def test_band_pool_conservation(self):
        """Band-level net_pnl sum == scored pool net_pnl."""
        pool = _make_pool(30)
        surface = build_full_surface(pool)
        band_sum = sum(b["net_pnl"] for b in surface["band_audit"].values())
        scored_net = surface["meta"]["total_net_pnl"]
        # All episodes in pool are scored (conviction >= 0.40, notional > 0)
        assert abs(band_sum - scored_net) < 1e-4

    def test_friction_conservation(self):
        """friction_dominated_pnl + non_friction_pnl == total net_pnl."""
        pool = _make_pool(30)
        surface = build_full_surface(pool)
        meta = surface["meta"]
        assert abs(meta["friction_dominated_pnl"] + meta["non_friction_pnl"]
                    - meta["total_net_pnl"]) < 1e-6

    def test_meta_counts(self):
        pool = _make_pool(30)
        surface = build_full_surface(pool)
        assert surface["meta"]["n_episodes_total"] == 30
        assert surface["meta"]["n_episodes_valid"] == 30

    def test_with_mixed_scored_unscored(self):
        """Some episodes below min_conviction are still in P5C totals."""
        pool = _make_pool(10)
        # Add unscored episodes
        pool.append(_ep(conviction=0.10, net_pnl=-2.0, gross_pnl=-1.5, fees=0.5))
        pool.append(_ep(conviction=0.0, net_pnl=-1.0, gross_pnl=-0.5, fees=0.5))
        surface = build_full_surface(pool)
        assert surface["meta"]["n_episodes_total"] == 12
        assert surface["meta"]["n_episodes_scored"] == 10  # 10 have conv >= 0.20
        # Symbol drag includes all 12
        total_drag = sum(v["net_pnl"] for v in surface["symbol_drag"].values())
        assert abs(total_drag - surface["meta"]["total_net_pnl"]) < 1e-4


# ── Persistence ──────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        pool = _make_pool(20)
        surface = build_full_surface(pool)
        path = str(tmp_path / "test_surface.json")
        save_surface(surface, path)
        loaded = load_surface(path)
        assert loaded is not None
        assert loaded["meta"]["n_episodes_total"] == 20

    def test_load_missing(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_surface(path) is None
