"""Tests for P6 Candidate 2 — Price-State / Region Surface."""

import pytest
import time

from execution.p6_price_state import (
    C2Config,
    DEFAULT_CONFIG,
    REGIONS,
    classify_region,
    compute_region_conviction,
    generate_price_state_signals,
)
from execution.p6_simple_rules import P6Signal

TS = 1700000000.0


# ── Config tests ─────────────────────────────────────────────────────────

class TestC2Config:
    def test_frozen(self):
        cfg = C2Config()
        with pytest.raises(AttributeError):
            cfg.range_pos_low = 0.3  # type: ignore[misc]

    def test_defaults(self):
        cfg = DEFAULT_CONFIG
        assert cfg.range_pos_low == 0.20
        assert cfg.range_pos_high == 0.80
        assert cfg.vol_compress_z == -1.0
        assert cfg.vol_expand_z == 1.0
        assert cfg.min_data_quality == 0.5

    def test_to_dict(self):
        d = DEFAULT_CONFIG.to_dict()
        assert isinstance(d, dict)
        assert d["range_pos_low"] == 0.20

    def test_conviction_map_present(self):
        cfg = DEFAULT_CONFIG
        assert cfg.conv_low_range == 0.58
        assert cfg.conv_high_range == 0.58
        assert cfg.conv_vol_compressed == 0.56
        assert cfg.conv_vol_expanded == 0.52
        assert cfg.conv_center == 0.50
        assert cfg.conv_rp_bump_k == 0.10
        assert cfg.conv_bump_cap == 0.08


# ── Region conviction tests ──────────────────────────────────────────────

class TestRegionConviction:
    def test_center_base(self):
        c = compute_region_conviction("center", {"range_position": 0.5}, DEFAULT_CONFIG)
        assert c == pytest.approx(0.50, abs=1e-6)

    def test_low_range_base(self):
        c = compute_region_conviction("low_range", {"range_position": 0.5}, DEFAULT_CONFIG)
        assert c == pytest.approx(0.58, abs=1e-6)

    def test_extreme_rp_adds_bump(self):
        c = compute_region_conviction("low_range", {"range_position": 0.0}, DEFAULT_CONFIG)
        # base=0.58 + bump from |0.0 - 0.5| * 0.10 = 0.05 → 0.63
        assert c > 0.58

    def test_bump_capped(self):
        c = compute_region_conviction("low_range", {"range_position": 0.0}, DEFAULT_CONFIG)
        # max bump = 0.08, base = 0.58 → max = 0.66
        assert c <= 0.58 + DEFAULT_CONFIG.conv_bump_cap + 1e-6

    def test_clipped_bounds(self):
        c = compute_region_conviction("high_range", {"range_position": 1.0}, DEFAULT_CONFIG)
        assert 0.0 <= c <= 1.0

    def test_deterministic(self):
        args = ("vol_compressed", {"range_position": 0.3}, DEFAULT_CONFIG)
        assert compute_region_conviction(*args) == compute_region_conviction(*args)


# ── Region classifier tests ─────────────────────────────────────────────

class TestClassifyRegion:
    """Region assignment is the critical contract. Tests cover every branch."""

    def test_low_range(self):
        assert classify_region({"range_position": 0.1, "vol_regime_z": 0.0}, DEFAULT_CONFIG) == "low_range"

    def test_low_range_boundary(self):
        """Boundary: exactly at threshold is NOT low_range (strict <)."""
        assert classify_region({"range_position": 0.2, "vol_regime_z": 0.0}, DEFAULT_CONFIG) != "low_range"

    def test_high_range(self):
        assert classify_region({"range_position": 0.9, "vol_regime_z": 0.0}, DEFAULT_CONFIG) == "high_range"

    def test_high_range_boundary(self):
        """Boundary: exactly at threshold is NOT high_range (strict >)."""
        assert classify_region({"range_position": 0.8, "vol_regime_z": 0.0}, DEFAULT_CONFIG) != "high_range"

    def test_vol_compressed(self):
        assert classify_region({"range_position": 0.5, "vol_regime_z": -1.5}, DEFAULT_CONFIG) == "vol_compressed"

    def test_vol_compressed_boundary(self):
        assert classify_region({"range_position": 0.5, "vol_regime_z": -1.0}, DEFAULT_CONFIG) != "vol_compressed"

    def test_vol_expanded(self):
        assert classify_region({"range_position": 0.5, "vol_regime_z": 1.5}, DEFAULT_CONFIG) == "vol_expanded"

    def test_vol_expanded_boundary(self):
        assert classify_region({"range_position": 0.5, "vol_regime_z": 1.0}, DEFAULT_CONFIG) != "vol_expanded"

    def test_center_fallback(self):
        assert classify_region({"range_position": 0.5, "vol_regime_z": 0.0}, DEFAULT_CONFIG) == "center"

    def test_mutually_exclusive(self):
        """Each feature point maps to exactly one region."""
        test_points = [
            {"range_position": 0.1, "vol_regime_z": -2.0},  # Could be low_range or vol_compressed
            {"range_position": 0.9, "vol_regime_z": 2.0},   # Could be high_range or vol_expanded
            {"range_position": 0.0, "vol_regime_z": 0.5},
            {"range_position": 1.0, "vol_regime_z": -0.5},
        ]
        for feat in test_points:
            region = classify_region(feat, DEFAULT_CONFIG)
            assert region in REGIONS, f"Unknown region: {region}"

    def test_exhaustive_coverage(self):
        """All points map to a valid region (no gaps)."""
        import itertools
        for rp, vz in itertools.product(
            [0.0, 0.1, 0.19, 0.2, 0.5, 0.8, 0.81, 0.9, 1.0],
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        ):
            feat = {"range_position": rp, "vol_regime_z": vz}
            region = classify_region(feat, DEFAULT_CONFIG)
            assert region in REGIONS, f"Gap at rp={rp}, vz={vz}"

    def test_precedence_low_range_before_vol_compressed(self):
        """low_range takes precedence over vol_compressed."""
        feat = {"range_position": 0.1, "vol_regime_z": -2.0}
        assert classify_region(feat, DEFAULT_CONFIG) == "low_range"

    def test_precedence_high_range_before_vol_expanded(self):
        """high_range takes precedence over vol_expanded."""
        feat = {"range_position": 0.9, "vol_regime_z": 2.0}
        assert classify_region(feat, DEFAULT_CONFIG) == "high_range"

    def test_missing_features_default_to_center(self):
        """Missing features default to center."""
        assert classify_region({}, DEFAULT_CONFIG) == "center"


# ── Signal generation tests ──────────────────────────────────────────────

class TestGeneratePriceStateSignals:
    def test_low_range_produces_normal_long(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "MEAN_REVERT", "BTCUSDT", ts=TS)
        normal = [s for s in sigs if s.candidate_id == "C2_REGION_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "LONG"
        assert normal[0].region == "low_range"

    def test_high_range_produces_normal_short(self):
        feat = {"range_position": 0.9, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
        normal = [s for s in sigs if s.candidate_id == "C2_REGION_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "SHORT"
        assert normal[0].region == "high_range"

    def test_vol_compressed_produces_normal_long(self):
        feat = {"range_position": 0.5, "vol_regime_z": -1.5, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "CHOPPY", "ETHUSDT", ts=TS)
        normal = [s for s in sigs if s.candidate_id == "C2_REGION_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "LONG"
        assert normal[0].region == "vol_compressed"

    def test_vol_expanded_produces_normal_short(self):
        feat = {"range_position": 0.5, "vol_regime_z": 1.5, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "BREAKOUT", "SOLUSDT", ts=TS)
        normal = [s for s in sigs if s.candidate_id == "C2_REGION_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "SHORT"
        assert normal[0].region == "vol_expanded"

    def test_center_produces_no_signal(self):
        feat = {"range_position": 0.5, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
        assert len(sigs) == 0

    def test_always_produces_pair_normal_inverted(self):
        """Non-center regions produce exactly 2 signals: normal + inverted."""
        for rp, vz in [(0.1, 0.0), (0.9, 0.0), (0.5, -1.5), (0.5, 1.5)]:
            feat = {"range_position": rp, "vol_regime_z": vz, "data_quality": 1.0}
            sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
            assert len(sigs) == 2
            ids = {s.candidate_id for s in sigs}
            assert ids == {"C2_REGION_NORMAL", "C2_REGION_INVERTED"}

    def test_polarity_flip_correctness(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "MEAN_REVERT", "BTCUSDT", ts=TS)
        normal = [s for s in sigs if s.polarity == "normal"][0]
        inverted = [s for s in sigs if s.polarity == "inverted"][0]
        assert normal.side != inverted.side

    def test_low_data_quality_blocks(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 0.3}
        sigs = generate_price_state_signals(feat, "MEAN_REVERT", "BTCUSDT", ts=TS)
        assert len(sigs) == 0

    def test_data_quality_at_threshold_passes(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 0.5}
        sigs = generate_price_state_signals(feat, "MEAN_REVERT", "BTCUSDT", ts=TS)
        assert len(sigs) == 2

    def test_feature_snapshot_contains_provenance(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
        for s in sigs:
            snap = s.feature_snapshot
            assert "range_position" in snap
            assert "vol_regime_z" in snap
            assert "data_quality" in snap
            assert snap["regime"] == "TREND_UP"
            assert snap["region"] == "low_range"
            assert snap["region_assigned"] == "low_range"
            # Pre-class continuous scores
            assert "region_score_vector" in snap
            rsv = snap["region_score_vector"]
            assert "low_range_dist" in rsv
            assert "high_range_dist" in rsv
            assert "vol_compress_dist" in rsv
            assert "vol_expand_dist" in rsv
            # Conviction provenance
            assert "conviction_proxy" in snap

    def test_signal_metadata_correct(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
        for s in sigs:
            assert s.candidate_family == "C2"
            assert s.symbol == "BTCUSDT"
            assert s.ts == TS
            assert s.rule_name == "region_low_range"
            assert s.region == "low_range"

    def test_conviction_proxy_set(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
        for s in sigs:
            assert s.conviction > 0.5  # low_range base=0.58 + bump from 0.1
            assert s.conviction <= 1.0

    def test_custom_config(self):
        # Wider range → center for 0.1
        wide = C2Config(range_pos_low=0.05, range_pos_high=0.95)
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", config=wide, ts=TS)
        # 0.1 > 0.05, so not low_range... check if center or vol?
        # range_pos=0.1 >= 0.05, range_pos < 0.95, vol_z=0 (not compressed/expanded) → center
        assert len(sigs) == 0

    def test_timestamp_auto_generated(self):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        before = time.time()
        sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT")
        after = time.time()
        for s in sigs:
            assert before <= s.ts <= after

    def test_all_active_regions_have_signal(self):
        """Every non-center region produces signals."""
        cases = [
            ({"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}, "low_range"),
            ({"range_position": 0.9, "vol_regime_z": 0.0, "data_quality": 1.0}, "high_range"),
            ({"range_position": 0.5, "vol_regime_z": -1.5, "data_quality": 1.0}, "vol_compressed"),
            ({"range_position": 0.5, "vol_regime_z": 1.5, "data_quality": 1.0}, "vol_expanded"),
        ]
        for feat, expected_region in cases:
            sigs = generate_price_state_signals(feat, "TREND_UP", "BTCUSDT", ts=TS)
            assert len(sigs) == 2, f"Expected 2 signals for {expected_region}, got {len(sigs)}"
            assert sigs[0].region == expected_region
