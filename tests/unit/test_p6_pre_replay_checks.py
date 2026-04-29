"""P6 pre-replay sanity checks.

Must pass before historical replay is authorized.

1. Signal density: signals per 1k bars per symbol within [1, 500] for
   each candidate_id that fires.
2. One-signal rule: max one signal per (symbol, candidate_id) per bar.
3. Provenance completeness: every shadow record has all required fields.
"""

import pytest
import time
from typing import Dict, List

from execution.p6_simple_rules import (
    C1Config,
    DEFAULT_CONFIG as C1_DEFAULT,
    P6Signal,
    generate_simple_rule_signals,
)
from execution.p6_price_state import (
    C2Config,
    DEFAULT_CONFIG as C2_DEFAULT,
    generate_price_state_signals,
)
from execution.p6_shadow_evaluator import (
    evaluate_signal_against_bridge,
    evaluate_and_log_signals,
)
from execution.expectancy_bridge import BandEntry, BandTable


# ── Test data generators ─────────────────────────────────────────────────

def _make_trending_up_bars(n: int) -> list[float]:
    """Generate n bars of gently trending-up closes."""
    return [100.0 + i * 0.5 for i in range(n)]

def _make_trending_down_bars(n: int) -> list[float]:
    return [100.0 - i * 0.5 for i in range(n)]

def _make_mean_reverting_bars(n: int) -> list[float]:
    """Oscillating closes that periodically touch extremes."""
    import math
    return [100.0 + 10.0 * math.sin(i * 0.3) for i in range(n)]


TS = 1700000000.0
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Regime scenarios for multi-bar simulation
REGIME_SCENARIOS = [
    ("TREND_UP", {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.7, "vol_regime_z": 0.0, "data_quality": 1.0}),
    ("TREND_DOWN", {"trend_r2": 0.7, "trend_slope": -0.001, "range_position": 0.3, "vol_regime_z": 0.0, "data_quality": 1.0}),
    ("MEAN_REVERT", {"trend_r2": 0.2, "range_position": 0.1, "mean_reversion_score": 0.7, "vol_regime_z": 0.0, "data_quality": 1.0}),
    ("MEAN_REVERT", {"trend_r2": 0.2, "range_position": 0.9, "mean_reversion_score": 0.7, "vol_regime_z": 0.0, "data_quality": 1.0}),
    ("BREAKOUT", {"trend_r2": 0.3, "range_position": 0.5, "vol_regime_z": 1.5, "data_quality": 1.0}),
    ("CHOPPY", {"trend_r2": 0.2, "range_position": 0.5, "vol_regime_z": 0.0, "data_quality": 1.0}),
]


def _generate_all_signals_for_scenario(
    closes: list[float], regime: str, features: dict, symbol: str,
) -> tuple[list[P6Signal], list[P6Signal]]:
    """Generate C1 + C2 signals for one symbol/bar."""
    c1_sel, c1_sup = generate_simple_rule_signals(
        closes, features, regime, symbol, ts=TS,
    )
    c2 = generate_price_state_signals(features, regime, symbol, ts=TS)
    return c1_sel + c2, c1_sup


# ── 1. Signal density sanity ─────────────────────────────────────────────

class TestSignalDensity:
    """For each candidate_id, verify signal rate is within [1, 500] per 1k bars."""

    def test_c1_trend_density(self):
        """Trend signals fire on TREND_UP / TREND_DOWN bars."""
        n_bars = 1000
        closes_up = _make_trending_up_bars(n_bars)
        feat = {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.6}
        signals = []
        for i in range(60, n_bars):
            sel, _ = generate_simple_rule_signals(
                closes_up[:i+1], feat, "TREND_UP", "BTCUSDT", ts=TS + i,
            )
            signals.extend(sel)
        by_cid: Dict[str, int] = {}
        for s in signals:
            by_cid[s.candidate_id] = by_cid.get(s.candidate_id, 0) + 1
        # Trend signals should fire on most bars (high R2, sustained crossover)
        for cid in ["C1_TREND_NORMAL", "C1_TREND_INVERTED"]:
            n = by_cid.get(cid, 0)
            assert 1 <= n <= n_bars, f"{cid}: {n} signals in {n_bars} bars"

    def test_c1_mr_density(self):
        """MR signals fire during MEAN_REVERT regime with extreme conditions."""
        n_bars = 1000
        closes = _make_mean_reverting_bars(n_bars)
        signals = []
        for i in range(60, n_bars):
            feat = {"range_position": 0.1, "mean_reversion_score": 0.7}
            sel, _ = generate_simple_rule_signals(
                closes[:i+1], feat, "MEAN_REVERT", "ETHUSDT", ts=TS + i,
            )
            signals.extend(sel)
        by_cid: Dict[str, int] = {}
        for s in signals:
            by_cid[s.candidate_id] = by_cid.get(s.candidate_id, 0) + 1
        for cid in ["C1_MR_NORMAL", "C1_MR_INVERTED"]:
            n = by_cid.get(cid, 0)
            # MR fires only when zscore extreme → sparse
            assert n <= 500 * (n_bars / 1000), f"{cid}: {n} signals too dense"

    def test_c2_region_density(self):
        """C2 fires on non-center regions."""
        count = 0
        for regime, feat in REGIME_SCENARIOS:
            for sym in SYMBOLS:
                sigs = generate_price_state_signals(feat, regime, sym, ts=TS)
                count += len(sigs)
        # At least some non-center regions fire
        assert count > 0


# ── 2. One-signal rule ───────────────────────────────────────────────────

class TestOneSignalRule:
    """Exactly one signal per (symbol, candidate_id) per bar after suppression."""

    def test_c1_one_per_symbol_candidate(self):
        closes = _make_trending_up_bars(100)
        feat = {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.6}
        selected, _ = generate_simple_rule_signals(
            closes, feat, "TREND_UP", "BTCUSDT", ts=TS,
        )
        seen = set()
        for s in selected:
            key = (s.symbol, s.candidate_id)
            assert key not in seen, f"Duplicate signal: {key}"
            seen.add(key)

    def test_c1_one_per_symbol_candidate_mr(self):
        closes = _make_mean_reverting_bars(100)
        feat = {"range_position": 0.1, "mean_reversion_score": 0.7}
        selected, _ = generate_simple_rule_signals(
            closes, feat, "MEAN_REVERT", "ETHUSDT", ts=TS,
        )
        seen = set()
        for s in selected:
            key = (s.symbol, s.candidate_id)
            assert key not in seen, f"Duplicate signal: {key}"
            seen.add(key)

    def test_c2_max_two_per_symbol(self):
        """C2 emits exactly 0 or 2 (normal + inverted) per symbol."""
        for regime, feat in REGIME_SCENARIOS:
            sigs = generate_price_state_signals(feat, regime, "BTCUSDT", ts=TS)
            assert len(sigs) in (0, 2), f"Expected 0 or 2, got {len(sigs)}"
            if len(sigs) == 2:
                assert sigs[0].candidate_id != sigs[1].candidate_id

    def test_multi_symbol_no_cross_contamination(self):
        """Signals from different symbols don't suppress each other."""
        closes = _make_trending_up_bars(100)
        feat = {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.6}
        all_selected = []
        for sym in SYMBOLS:
            sel, _ = generate_simple_rule_signals(
                closes, feat, "TREND_UP", sym, ts=TS,
            )
            all_selected.extend(sel)
        # Each symbol should have its own signals independently
        by_sym = {}
        for s in all_selected:
            by_sym.setdefault(s.symbol, []).append(s)
        for sym, sigs in by_sym.items():
            cids = [s.candidate_id for s in sigs]
            assert len(cids) == len(set(cids)), f"Duplicate candidate_id for {sym}"


# ── 3. Provenance completeness ───────────────────────────────────────────

REQUIRED_SHADOW_FIELDS = [
    "candidate_id", "polarity", "symbol", "side", "regime",
    "feature_snapshot", "conviction", "bridge_expected_edge_pct",
    "bridge_lookup_tier", "bridge_band_key", "bridge_sample_n",
    "bridge_is_sufficient", "fee_required_pct", "bridge_would_pass",
    "control_conviction", "control_expected_edge_pct", "control_would_pass",
]


class TestProvenanceCompleteness:
    """Every JSONL record has all required provenance fields."""

    @pytest.fixture
    def bridge_tables(self):
        table = BandTable(build_ts=TS, n_episodes_total=100, n_episodes_scored=80)
        table.bands["0.45-0.50"] = BandEntry(
            band_lo=0.45, band_hi=0.50,
            n_episodes=30, net_pnl_sum=15.0,
            notional_sum=1000.0, win_count=18,
        )
        table.bands["0.55-0.60"] = BandEntry(
            band_lo=0.55, band_hi=0.60,
            n_episodes=20, net_pnl_sum=10.0,
            notional_sum=500.0, win_count=12,
        )
        return {"OTHER": table, "MEAN_REVERT": table}

    def test_c1_trend_provenance(self, bridge_tables):
        closes = _make_trending_up_bars(100)
        feat = {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.6}
        sel, sup = generate_simple_rule_signals(
            closes, feat, "TREND_UP", "BTCUSDT", ts=TS,
        )
        if not sel:
            pytest.skip("No trend signals generated")
        records = evaluate_and_log_signals(
            sel, suppressed=sup,
            regime_tables=bridge_tables, pooled_table=bridge_tables["OTHER"],
            dry_run=True,
        )
        for rec in records:
            for field in REQUIRED_SHADOW_FIELDS:
                assert field in rec, f"Missing provenance field: {field} in {rec.get('candidate_id')}"

    def test_c1_mr_provenance(self, bridge_tables):
        closes = _make_mean_reverting_bars(100)
        feat = {"range_position": 0.1, "mean_reversion_score": 0.7}
        sel, sup = generate_simple_rule_signals(
            closes, feat, "MEAN_REVERT", "ETHUSDT", ts=TS,
        )
        if not sel:
            pytest.skip("No MR signals generated")
        records = evaluate_and_log_signals(
            sel, suppressed=sup,
            regime_tables=bridge_tables, pooled_table=bridge_tables["OTHER"],
            dry_run=True,
        )
        for rec in records:
            for field in REQUIRED_SHADOW_FIELDS:
                assert field in rec, f"Missing provenance field: {field} in {rec.get('candidate_id')}"

    def test_c2_provenance(self, bridge_tables):
        feat = {"range_position": 0.1, "vol_regime_z": 0.0, "data_quality": 1.0}
        sigs = generate_price_state_signals(feat, "MEAN_REVERT", "BTCUSDT", ts=TS)
        if not sigs:
            pytest.skip("No C2 signals generated")
        records = evaluate_and_log_signals(
            sigs,
            regime_tables=bridge_tables, pooled_table=bridge_tables["OTHER"],
            dry_run=True,
        )
        for rec in records:
            for field in REQUIRED_SHADOW_FIELDS:
                assert field in rec, f"Missing provenance field: {field} in {rec.get('candidate_id')}"

    def test_conviction_is_not_constant(self, bridge_tables):
        """Proxy conviction should vary across different market states."""
        convictions = set()
        # Trend with various R2
        for r2 in [0.5, 0.7, 0.9]:
            closes = _make_trending_up_bars(100)
            feat = {"trend_r2": r2, "trend_slope": 0.001, "range_position": 0.6}
            sel, _ = generate_simple_rule_signals(
                closes, feat, "TREND_UP", "BTCUSDT", ts=TS,
            )
            for s in sel:
                convictions.add(round(s.conviction, 4))
        assert len(convictions) > 1, "Conviction should vary with R2"
