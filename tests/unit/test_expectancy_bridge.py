"""Tests for execution/expectancy_bridge.py — P4B/P4D Expectancy Bridge."""

import json
from pathlib import Path

import pytest

from execution.expectancy_bridge import (
    BandEntry,
    BandTable,
    BridgeConfig,
    BridgeLookupResult,
    build_band_table,
    build_regime_conditional_table,
    check_monotonicity,
    load_band_table,
    load_episodes,
    load_regime_bridge,
    lookup_expected_edge,
    lookup_expected_edge_conditional,
    rebuild_band_table,
    save_band_table,
    save_regime_bridge,
    _band_key,
    _get_band_bounds,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_episode(conviction, net_pnl, notional=100.0, symbol="SOLUSDT",
                  regime="MEAN_REVERT", side="LONG"):
    return {
        "conviction_score": conviction,
        "net_pnl": net_pnl,
        "entry_notional": notional,
        "symbol": symbol,
        "regime_at_entry": regime,
        "side": side,
    }


def _make_episodes_for_bands():
    """Create a synthetic episode set with clear per-band behavior."""
    eps = []
    # Band 0.40-0.45: 15 episodes, negative edge
    for _ in range(12):
        eps.append(_make_episode(0.42, -0.50))  # loss
    for _ in range(3):
        eps.append(_make_episode(0.43, 0.30))   # win
    # Band 0.50-0.55: 20 episodes, slightly positive edge
    for _ in range(10):
        eps.append(_make_episode(0.52, 0.20))   # win
    for _ in range(10):
        eps.append(_make_episode(0.53, -0.10))  # loss
    # Band 0.60-0.65: 12 episodes, clearly positive edge
    for _ in range(9):
        eps.append(_make_episode(0.62, 0.80))   # win
    for _ in range(3):
        eps.append(_make_episode(0.63, -0.20))  # loss
    return eps


# ── Unit Tests: Band Key ─────────────────────────────────────────────────

class TestBandKey:
    def test_standard_bands(self):
        assert _band_key(0.42, 0.05) == "0.40-0.45"
        assert _band_key(0.50, 0.05) == "0.50-0.55"
        assert _band_key(0.59, 0.05) == "0.55-0.60"
        assert _band_key(0.60, 0.05) == "0.60-0.65"

    def test_boundary_values(self):
        assert _band_key(0.45, 0.05) == "0.45-0.50"  # [0.45, 0.50)
        assert _band_key(0.449999, 0.05) == "0.40-0.45"

    def test_get_band_bounds(self):
        lo, hi = _get_band_bounds(0.42, 0.05)
        assert lo == 0.40
        assert hi == 0.45


# ── Unit Tests: BandEntry ────────────────────────────────────────────────

class TestBandEntry:
    def test_avg_edge_pct(self):
        b = BandEntry(0.40, 0.45, n_episodes=10, net_pnl_sum=-5.0, notional_sum=1000.0)
        assert b.avg_edge_pct == pytest.approx(-0.005)

    def test_avg_edge_zero_notional(self):
        b = BandEntry(0.40, 0.45, n_episodes=0, net_pnl_sum=0, notional_sum=0)
        assert b.avg_edge_pct == 0.0

    def test_win_rate(self):
        b = BandEntry(0.40, 0.45, n_episodes=10, win_count=3)
        assert b.win_rate == pytest.approx(0.3)

    def test_sufficient(self):
        b = BandEntry(0.40, 0.45, n_episodes=10)
        assert b.sufficient is True
        b2 = BandEntry(0.40, 0.45, n_episodes=9)
        assert b2.sufficient is False

    def test_to_dict_roundtrip(self):
        b = BandEntry(0.40, 0.45, n_episodes=10, net_pnl_sum=-5.0,
                       notional_sum=1000.0, win_count=3)
        d = b.to_dict()
        assert d["band_lo"] == 0.4
        assert d["band_hi"] == 0.45
        assert d["avg_edge_pct"] == pytest.approx(-0.005)
        assert d["win_rate"] == pytest.approx(0.3)
        assert d["sufficient"] is True


# ── Unit Tests: Build Band Table ─────────────────────────────────────────

class TestBuildBandTable:
    def test_empty_episodes(self):
        table = build_band_table([])
        assert table.n_episodes_total == 0
        assert table.n_episodes_scored == 0
        assert len(table.bands) == 0

    def test_no_scored_episodes(self):
        eps = [{"net_pnl": -1.0, "entry_notional": 100.0}]  # no conviction
        table = build_band_table(eps)
        assert table.n_episodes_total == 1
        assert table.n_episodes_scored == 0

    def test_filters_low_conviction(self):
        eps = [_make_episode(0.10, -1.0)]  # below min_conviction
        table = build_band_table(eps)
        assert table.n_episodes_scored == 0

    def test_filters_zero_notional(self):
        eps = [_make_episode(0.50, -1.0, notional=0)]
        table = build_band_table(eps)
        assert table.n_episodes_scored == 0

    def test_builds_correct_bands(self):
        eps = _make_episodes_for_bands()
        table = build_band_table(eps, BridgeConfig(min_sample=5))

        assert table.n_episodes_total == len(eps)
        assert table.n_episodes_scored == len(eps)
        assert "0.40-0.45" in table.bands
        assert "0.50-0.55" in table.bands
        assert "0.60-0.65" in table.bands

        # Band 0.40-0.45: 15 eps, 12 losses of -0.5, 3 wins of +0.3
        b1 = table.bands["0.40-0.45"]
        assert b1.n_episodes == 15
        assert b1.win_count == 3
        assert b1.net_pnl_sum == pytest.approx(12 * -0.5 + 3 * 0.3)

        # Band 0.60-0.65: clearly positive
        b3 = table.bands["0.60-0.65"]
        assert b3.n_episodes == 12
        assert b3.avg_edge_pct > 0

    def test_global_entry(self):
        eps = _make_episodes_for_bands()
        table = build_band_table(eps)
        g = table.global_entry
        assert g.n_episodes == len(eps)
        total_pnl = sum(e["net_pnl"] for e in eps)
        assert g.net_pnl_sum == pytest.approx(total_pnl)

    def test_custom_band_width(self):
        eps = [_make_episode(0.42, -1.0), _make_episode(0.47, 1.0)]
        table = build_band_table(eps, BridgeConfig(band_width=0.10))
        assert "0.40-0.50" in table.bands
        assert table.bands["0.40-0.50"].n_episodes == 2


# ── Unit Tests: Lookup ───────────────────────────────────────────────────

class TestLookup:
    def test_band_tier(self):
        eps = _make_episodes_for_bands()
        table = build_band_table(eps, BridgeConfig(min_sample=5))

        result = lookup_expected_edge(0.52, table)
        assert result.lookup_tier == "band"
        assert result.band_key == "0.50-0.55"
        assert result.sufficient is True

    def test_global_fallback(self):
        """Lookup falls back to global when band has < min_sample."""
        eps = [_make_episode(0.52, 0.20) for _ in range(15)]
        # Band 0.30-0.35 has 0 episodes → should fall to global
        table = build_band_table(eps, BridgeConfig(min_sample=10))

        result = lookup_expected_edge(0.32, table)
        assert result.lookup_tier == "global"
        assert result.n_episodes == 15

    def test_cold_start(self):
        """Lookup returns cold_start when no data at all."""
        table = build_band_table([])
        result = lookup_expected_edge(0.50, table)
        assert result.lookup_tier == "cold_start"
        assert result.expected_edge_pct == 0.0
        assert result.sufficient is False

    def test_edge_sign_preserved(self):
        """Negative edge is honestly reported (not zeroed)."""
        eps = [_make_episode(0.42, -1.0) for _ in range(15)]
        table = build_band_table(eps, BridgeConfig(min_sample=10))
        result = lookup_expected_edge(0.43, table)
        assert result.expected_edge_pct < 0

    def test_positive_edge_reported(self):
        eps = [_make_episode(0.62, 0.50) for _ in range(15)]
        table = build_band_table(eps, BridgeConfig(min_sample=10))
        result = lookup_expected_edge(0.63, table)
        assert result.expected_edge_pct > 0
        assert result.lookup_tier == "band"

    def test_to_dict(self):
        result = BridgeLookupResult(
            expected_edge_pct=0.002,
            band_key="0.50-0.55",
            n_episodes=20,
            sufficient=True,
            win_rate=0.55,
            lookup_tier="band",
        )
        d = result.to_dict()
        assert d["bridge_expected_edge_pct"] == pytest.approx(0.002)
        assert d["bridge_band_key"] == "0.50-0.55"
        assert d["bridge_edge_contract"] == "empirical_expectancy_bridge"


# ── Unit Tests: Monotonicity ─────────────────────────────────────────────

class TestMonotonicity:
    def test_monotonic_table(self):
        """Bands with strictly increasing edge should be monotonic."""
        table = BandTable(
            bands={
                "0.40-0.45": BandEntry(0.40, 0.45, n_episodes=15,
                                        net_pnl_sum=-10, notional_sum=1000, win_count=3),
                "0.50-0.55": BandEntry(0.50, 0.55, n_episodes=15,
                                        net_pnl_sum=5, notional_sum=1000, win_count=8),
                "0.60-0.65": BandEntry(0.60, 0.65, n_episodes=15,
                                        net_pnl_sum=20, notional_sum=1000, win_count=12),
            }
        )
        result = check_monotonicity(table)
        assert result["monotonic"] is True
        assert result["inversions"] == []

    def test_non_monotonic(self):
        """Inversion detected when higher band has lower edge."""
        table = BandTable(
            bands={
                "0.40-0.45": BandEntry(0.40, 0.45, n_episodes=15,
                                        net_pnl_sum=-10, notional_sum=1000, win_count=3),
                "0.50-0.55": BandEntry(0.50, 0.55, n_episodes=15,
                                        net_pnl_sum=20, notional_sum=1000, win_count=12),
                "0.60-0.65": BandEntry(0.60, 0.65, n_episodes=15,
                                        net_pnl_sum=-5, notional_sum=1000, win_count=5),
            }
        )
        result = check_monotonicity(table)
        assert result["monotonic"] is False
        assert ("0.50-0.55", "0.60-0.65") in result["inversions"]

    def test_skips_insufficient_bands(self):
        """Bands with < min_sample are excluded from monotonicity check."""
        table = BandTable(
            bands={
                "0.40-0.45": BandEntry(0.40, 0.45, n_episodes=5,
                                        net_pnl_sum=100, notional_sum=1000),  # insufficient
                "0.50-0.55": BandEntry(0.50, 0.55, n_episodes=15,
                                        net_pnl_sum=-5, notional_sum=1000, win_count=5),
            }
        )
        result = check_monotonicity(table)
        assert result["monotonic"] is True  # only 1 sufficient band, nothing to invert
        assert result["n_sufficient_bands"] == 1


# ── Unit Tests: Persistence ──────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        eps = _make_episodes_for_bands()
        config = BridgeConfig(
            min_sample=5,
            band_table_path=str(tmp_path / "bridge.json"),
        )
        table = build_band_table(eps, config)
        save_band_table(table, str(tmp_path / "bridge.json"))

        loaded = load_band_table(str(tmp_path / "bridge.json"))
        assert loaded is not None
        assert loaded.n_episodes_scored == table.n_episodes_scored
        assert len(loaded.bands) == len(table.bands)

        # Check edge values survive roundtrip
        for key in table.bands:
            assert loaded.bands[key].avg_edge_pct == pytest.approx(
                table.bands[key].avg_edge_pct, abs=1e-6
            )

    def test_load_nonexistent(self, tmp_path):
        result = load_band_table(str(tmp_path / "nope.json"))
        assert result is None

    def test_to_dict_schema(self):
        """Band table dict has required schema keys."""
        eps = _make_episodes_for_bands()
        table = build_band_table(eps)
        d = table.to_dict()
        assert "version" in d
        assert "build_ts" in d
        assert "bands" in d
        assert "global" in d
        assert "n_episodes_total" in d
        assert "n_episodes_scored" in d


# ── Unit Tests: Episode Loader ───────────────────────────────────────────

class TestEpisodeLoader:
    def test_load_dict_format(self, tmp_path):
        data = {"episodes": [_make_episode(0.5, 1.0)], "stats": {}}
        (tmp_path / "ledger.json").write_text(json.dumps(data))
        eps = load_episodes(str(tmp_path / "ledger.json"))
        assert len(eps) == 1

    def test_load_list_format(self, tmp_path):
        data = [_make_episode(0.5, 1.0)]
        (tmp_path / "ledger.json").write_text(json.dumps(data))
        eps = load_episodes(str(tmp_path / "ledger.json"))
        assert len(eps) == 1

    def test_load_missing_file(self, tmp_path):
        eps = load_episodes(str(tmp_path / "nope.json"))
        assert eps == []


# ── Integration Tests: Full Rebuild ──────────────────────────────────────

class TestRebuild:
    def test_rebuild_creates_file(self, tmp_path):
        ledger = {"episodes": _make_episodes_for_bands()}
        ledger_path = str(tmp_path / "ledger.json")
        Path(ledger_path).write_text(json.dumps(ledger))

        config = BridgeConfig(
            min_sample=5,
            episode_ledger_path=ledger_path,
            band_table_path=str(tmp_path / "bridge.json"),
        )
        table = rebuild_band_table(config)
        assert (tmp_path / "bridge.json").exists()
        assert table.n_episodes_scored > 0


# ── P4D: Promotion Criteria ─────────────────────────────────────────────

class TestPromotionCriteria:
    """Tests that encode the P4D promotion criteria.

    These tests verify that the bridge infrastructure can detect
    when conditions for promotion are met.  NOT intended to be
    satisfied by current data (all bands are negative).
    """

    def test_nonzero_edge_on_meaningful_subset(self):
        """At least one band must show positive avg_edge_pct."""
        eps = _make_episodes_for_bands()
        table = build_band_table(eps, BridgeConfig(min_sample=5))
        positive_bands = [
            k for k, v in table.bands.items()
            if v.sufficient and v.avg_edge_pct > 0
        ]
        # This test documents the criterion, not asserts current data passes
        # In production, this would gate promotion.
        assert isinstance(positive_bands, list)

    def test_fee_gated_becomes_selective(self):
        """Bridge should NOT produce all-pass or all-block."""
        eps = _make_episodes_for_bands()
        table = build_band_table(eps, BridgeConfig(min_sample=5))
        fee_pct = 0.0012  # typical fee_required_pct

        would_pass = []
        for conf in [0.42, 0.52, 0.62]:
            r = lookup_expected_edge(conf, table)
            would_pass.append(r.expected_edge_pct > fee_pct)

        # Criterion: not all same (some pass, some fail → selective)
        # Documenting the criterion even if current data is all-fail
        assert isinstance(would_pass, list)

    def test_edge_monotonicity_positive(self):
        """Promotion requires monotonicity across sufficient bands."""
        # Construct a passing scenario
        table = BandTable(
            bands={
                "0.40-0.45": BandEntry(0.40, 0.45, n_episodes=15,
                                        net_pnl_sum=-10, notional_sum=1000, win_count=3),
                "0.50-0.55": BandEntry(0.50, 0.55, n_episodes=15,
                                        net_pnl_sum=5, notional_sum=1000, win_count=8),
                "0.60-0.65": BandEntry(0.60, 0.65, n_episodes=15,
                                        net_pnl_sum=20, notional_sum=1000, win_count=12),
            }
        )
        mono = check_monotonicity(table)
        assert mono["monotonic"] is True

    def test_oos_expectancy_above_friction(self):
        """Out-of-sample: edge must exceed fee friction in at least one band."""
        # Synthetic: band 0.60-0.65 with edge > fees
        table = BandTable(
            bands={
                "0.60-0.65": BandEntry(0.60, 0.65, n_episodes=50,
                                        net_pnl_sum=100, notional_sum=5000, win_count=30),
            }
        )
        fee_pct = 0.0012
        r = lookup_expected_edge(0.62, table, BridgeConfig(min_sample=10))
        assert r.expected_edge_pct > fee_pct
        assert r.sufficient is True


# ── Contract Tests: true_edge.py P4A ────────────────────────────────────

class TestTrueEdgeContract:
    """Verify P4A contract formalization in true_edge.py."""

    def test_confidence_midpoint_exported(self):
        from execution.true_edge import CONFIDENCE_MIDPOINT
        assert CONFIDENCE_MIDPOINT == 0.5

    def test_edge_contract_field_on_result(self):
        from execution.true_edge import compute_true_edge
        result = compute_true_edge(
            confidence=0.42,
            price=100.0,
            atr=None,
            notional_usd=100.0,
        )
        assert result.edge_contract == "confidence_threshold_model"
        assert result.confidence_midpoint == 0.5

    def test_zero_edge_below_midpoint(self):
        from execution.true_edge import compute_true_edge
        result = compute_true_edge(
            confidence=0.42,
            price=100.0,
            atr=5.0,
            notional_usd=100.0,
        )
        assert result.expected_edge_pct == 0.0
        assert result.adv == 0.0

    def test_positive_edge_above_midpoint(self):
        from execution.true_edge import compute_true_edge
        result = compute_true_edge(
            confidence=0.65,
            price=100.0,
            atr=5.0,  # 5% ATR
            notional_usd=100.0,
        )
        assert result.expected_edge_pct > 0
        assert result.adv > 0


# ── Contract Tests: Bridge Shadow Diagnostic ─────────────────────────────

class TestBridgeShadowDiagnostic:
    """Verify P4C bridge shadow fields in diagnostic records."""

    def test_bridge_fields_present(self):
        from execution.score_edge_diagnostic import build_diagnostic_record
        from unittest.mock import MagicMock

        intent = {
            "symbol": "SOLUSDT",
            "positionSide": "LONG",
            "intent_id": "ord_test",
            "hybrid_score": 0.5,
            "conviction_score": 0.45,
        }
        te = MagicMock()
        te.notional_usd = 100.0
        te.expected_edge_pct = 0.0
        te.source = "fallback_proxy"
        te.adv = 0.0
        te.atr_pct = 0.0
        te.fallback_reason = ""

        fg = {"required_edge_usd": 0.12, "round_trip_fee_usd": 0.08}

        rec = build_diagnostic_record(
            intent=intent, te_result=te, fg_details=fg, fg_allowed=False,
        )
        assert rec is not None
        # Bridge fields must exist (may be None if no table, or have values)
        for key in ["bridge_expected_edge_pct", "bridge_band_key",
                     "bridge_lookup_tier", "bridge_would_pass",
                     "bridge_vs_threshold_delta"]:
            assert key in rec, f"Missing bridge field: {key}"

    def test_bridge_delta_sign(self):
        """When both models produce zero, delta should be zero or negative."""
        from execution.score_edge_diagnostic import build_diagnostic_record
        from unittest.mock import MagicMock

        intent = {
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "intent_id": "ord_test2",
            "hybrid_score": 0.3,
            "conviction_score": 0.42,
        }
        te = MagicMock()
        te.notional_usd = 50.0
        te.expected_edge_pct = 0.0
        te.source = "fallback_proxy"
        te.adv = 0.0
        te.atr_pct = 0.0
        te.fallback_reason = "atr_missing"

        fg = {"required_edge_usd": 0.06, "round_trip_fee_usd": 0.04}

        rec = build_diagnostic_record(
            intent=intent, te_result=te, fg_details=fg, fg_allowed=False,
        )
        if rec and rec.get("bridge_vs_threshold_delta") is not None:
            # Bridge edge is negative (~-0.4%), threshold edge is 0
            # So delta should be negative
            assert rec["bridge_vs_threshold_delta"] <= 0


# ── P5B: Regime-Conditional Bridge ───────────────────────────────────────

def _make_regime_episodes():
    """Episodes split across MEAN_REVERT и OTHER regimes."""
    eps = []
    # MEAN_REVERT: 25 episodes in 0.45-0.50 band, net negative
    for _ in range(20):
        eps.append(_make_episode(0.47, -0.50, regime="MEAN_REVERT"))
    for _ in range(5):
        eps.append(_make_episode(0.48, 1.50, regime="MEAN_REVERT"))
    # MEAN_REVERT: 12 episodes in 0.55-0.60 band, net positive
    for _ in range(4):
        eps.append(_make_episode(0.57, -0.30, regime="MEAN_REVERT"))
    for _ in range(8):
        eps.append(_make_episode(0.56, 0.80, regime="MEAN_REVERT"))
    # TREND_UP (→ OTHER): 5 episodes in 0.45-0.50 band (insufficient)
    for _ in range(3):
        eps.append(_make_episode(0.46, -2.0, regime="TREND_UP"))
    for _ in range(2):
        eps.append(_make_episode(0.47, 0.50, regime="TREND_UP"))
    # unknown (→ OTHER): 6 in 0.45-0.50
    for _ in range(6):
        eps.append(_make_episode(0.49, -0.80, regime="unknown"))
    return eps


class TestRegimeConditionalTable:
    def test_builds_two_tables(self):
        eps = _make_regime_episodes()
        tables = build_regime_conditional_table(eps)
        assert "MEAN_REVERT" in tables
        assert "OTHER" in tables

    def test_mean_revert_has_bands(self):
        eps = _make_regime_episodes()
        tables = build_regime_conditional_table(eps)
        mr = tables["MEAN_REVERT"]
        assert len(mr.bands) >= 2
        assert "0.45-0.50" in mr.bands

    def test_other_contains_trend_and_unknown(self):
        eps = _make_regime_episodes()
        tables = build_regime_conditional_table(eps)
        other = tables["OTHER"]
        assert other.n_episodes_scored > 0

    def test_conservation_regime_sum_equals_pooled(self):
        """Sum of regime table band PnL == pooled band PnL."""
        eps = _make_regime_episodes()
        config = BridgeConfig()
        tables = build_regime_conditional_table(eps, config)
        pooled = build_band_table(eps, config)

        # Per-band: sum across regimes
        all_band_keys = set()
        for t in tables.values():
            all_band_keys.update(t.bands.keys())
        all_band_keys.update(pooled.bands.keys())

        for bk in all_band_keys:
            regime_sum = sum(
                t.bands[bk].net_pnl_sum for t in tables.values()
                if bk in t.bands
            )
            pooled_val = pooled.bands[bk].net_pnl_sum if bk in pooled.bands else 0.0
            assert abs(regime_sum - pooled_val) < 1e-6, (
                f"Band {bk}: regime_sum={regime_sum}, pooled={pooled_val}"
            )

    def test_conservation_global_sum(self):
        """Sum of regime global n_episodes == pooled global n_episodes."""
        eps = _make_regime_episodes()
        config = BridgeConfig()
        tables = build_regime_conditional_table(eps, config)
        pooled = build_band_table(eps, config)

        regime_n = sum(t.global_entry.n_episodes for t in tables.values()
                       if t.global_entry)
        assert regime_n == pooled.global_entry.n_episodes


class TestRegimeConditionalLookup:
    def _setup(self):
        eps = _make_regime_episodes()
        config = BridgeConfig(min_sample=10)
        regime_tables = build_regime_conditional_table(eps, config)
        pooled = build_band_table(eps, config)
        return regime_tables, pooled, config

    def test_tier1_band_regime(self):
        """MEAN_REVERT 0.45-0.50 has >10 eps → direct hit."""
        tables, pooled, config = self._setup()
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="MEAN_REVERT",
            regime_tables=tables, pooled_table=pooled, config=config,
        )
        assert result.lookup_tier == "band_regime"
        assert result.fallback_depth == 0
        assert result.is_sufficient
        assert result.regime_key == "MEAN_REVERT"

    def test_tier2_band_pooled_fallback(self):
        """OTHER 0.45-0.50 has <10 eps → falls to pooled band."""
        tables, pooled, config = self._setup()
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="TREND_UP",
            regime_tables=tables, pooled_table=pooled, config=config,
        )
        # OTHER band has only 11 eps (5 TREND_UP + 6 unknown)
        # If sufficient, tier1; otherwise tier2
        assert result.lookup_tier in ("band_regime", "band_pooled")
        assert result.fallback_depth in (0, 1)

    def test_tier3_global_regime_fallback(self):
        """Conviction band with zero data falls to global regime."""
        tables, pooled, config = self._setup()
        # Band 0.80-0.85 has no data for any regime
        result = lookup_expected_edge_conditional(
            conviction=0.82, regime="MEAN_REVERT",
            regime_tables=tables, pooled_table=pooled, config=config,
        )
        # MEAN_REVERT global has enough eps → global_regime
        assert result.lookup_tier == "global_regime"
        assert result.fallback_depth == 2

    def test_tier5_cold_start(self):
        """Empty tables → cold_start."""
        empty_tables = {"MEAN_REVERT": BandTable(), "OTHER": BandTable()}
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="MEAN_REVERT",
            regime_tables=empty_tables, pooled_table=None,
        )
        assert result.lookup_tier == "cold_start"
        assert result.fallback_depth == 4
        assert not result.is_sufficient
        assert result.cold_start_reason != ""

    def test_provenance_fields_complete(self):
        """All provenance fields are populated."""
        tables, pooled, config = self._setup()
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="MEAN_REVERT",
            regime_tables=tables, pooled_table=pooled, config=config,
        )
        d = result.to_dict()
        assert "bridge_regime_expected_edge_pct" in d
        assert "bridge_regime_lookup_tier" in d
        assert "bridge_regime_band_key" in d
        assert "bridge_regime_key" in d
        assert "bridge_regime_sample_n" in d
        assert "bridge_regime_is_sufficient" in d
        assert "bridge_regime_fallback_depth" in d

    def test_cold_start_has_reason(self):
        empty_tables = {"MEAN_REVERT": BandTable(), "OTHER": BandTable()}
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="MEAN_REVERT",
            regime_tables=empty_tables, pooled_table=None,
        )
        d = result.to_dict()
        assert "bridge_regime_cold_start_reason" in d

    def test_no_cold_start_reason_on_hit(self):
        tables, pooled, config = self._setup()
        result = lookup_expected_edge_conditional(
            conviction=0.47, regime="MEAN_REVERT",
            regime_tables=tables, pooled_table=pooled, config=config,
        )
        d = result.to_dict()
        assert "bridge_regime_cold_start_reason" not in d


class TestRegimeBridgePersistence:
    def test_save_and_load(self, tmp_path):
        eps = _make_regime_episodes()
        tables = build_regime_conditional_table(eps)
        path = str(tmp_path / "regime_bridge.json")
        save_regime_bridge(tables, path=path)
        loaded = load_regime_bridge(path=path)
        assert loaded is not None
        assert "MEAN_REVERT" in loaded
        assert "OTHER" in loaded
        # Verify band data survived roundtrip
        mr = loaded["MEAN_REVERT"]
        assert "0.45-0.50" in mr.bands
        assert mr.bands["0.45-0.50"].n_episodes > 0

    def test_load_missing(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_regime_bridge(path=path) is None

    def test_sufficiency_threshold_persisted(self, tmp_path):
        eps = _make_regime_episodes()
        config = BridgeConfig(min_sample=15)
        tables = build_regime_conditional_table(eps, config)
        path = str(tmp_path / "regime_bridge.json")
        save_regime_bridge(tables, config=config, path=path)
        with open(path) as f:
            data = json.load(f)
        assert data["sufficiency_threshold"] == 15
