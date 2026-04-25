"""
Tests for P6B.5 Historical Replay (execution/p6_replay.py)

Tests are organized by component:
  1. Kline helpers (iso_to_ms, find_bar_index, extract_features_at_bar)
  2. Single-episode replay (replay_episode)
  3. Summary statistics (compute_replay_summary)
  4. Fast-fail gates (apply_fast_fail_gates)
  5. Promotion gates (apply_promotion_gates)
  6. Spearman rho
  7. Run manifest
  8. CSV export
  9. Integration: run_replay with mock klines
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List


from execution.expectancy_bridge import BandEntry, BandTable, BridgeConfig
from execution.p6_price_state import C2Config
from execution.p6_replay import (
    ALL_CANDIDATE_IDS,
    WARMUP_BARS,
    _find_bar_index,
    _iso_to_ms,
    _snap_to_15m_open,
    _skip_record,
    _spearman_rho,
    apply_fast_fail_gates,
    apply_promotion_gates,
    build_run_manifest,
    compute_replay_summary,
    export_replay_tables_csv,
    extract_features_at_bar,
    replay_episode,
    run_replay,
)
from execution.p6_simple_rules import C1Config


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_klines(n: int, base_price: float = 100.0, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """Generate n synthetic 15m klines with 15m-aligned open_time."""
    # Align base time to a 15m boundary (900_000 ms)
    base_ms = (1700000000000 // 900_000) * 900_000
    klines = []
    for i in range(n):
        price = base_price + i * 0.1
        klines.append({
            "open_time": base_ms + i * 900_000,
            "open": price - 0.05,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0 + i * 10,
        })
    return klines


def _make_episode(
    episode_id: str = "EP_0001",
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    regime: str = "MEAN_REVERT",
    entry_ts_ms: int = 1700000000000 + 60 * 900_000,  # bar 60
    net_pnl: float = -0.5,
    gross_pnl: float = -0.3,
    fees: float = 0.2,
    entry_notional: float = 200.0,
    conviction_score: float = 0.45,
) -> Dict[str, Any]:
    """Build a synthetic episode dict."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(entry_ts_ms / 1000, tz=timezone.utc)
    return {
        "episode_id": episode_id,
        "symbol": symbol,
        "side": side,
        "entry_ts": dt.isoformat(),
        "exit_ts": dt.isoformat(),
        "regime_at_entry": regime,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "entry_notional": entry_notional,
        "conviction_score": conviction_score,
        "exit_reason": "THESIS_INVALIDATED",
    }


def _make_band_table(
    edge_pcts: Dict[str, float] = None,
    n_per_band: int = 20,
) -> BandTable:
    """Build a synthetic band table."""
    if edge_pcts is None:
        edge_pcts = {
            "0.45-0.50": -0.005,
            "0.50-0.55": -0.003,
            "0.55-0.60": 0.001,
        }
    bands = {}
    for key, edge in edge_pcts.items():
        parts = key.split("-")
        lo, hi = float(parts[0]), float(parts[1])
        notional = 10000.0
        bands[key] = BandEntry(
            band_lo=lo,
            band_hi=hi,
            n_episodes=n_per_band,
            net_pnl_sum=edge * notional,
            notional_sum=notional,
            win_count=int(n_per_band * 0.3),
        )
    global_entry = BandEntry(
        band_lo=0.2, band_hi=1.0,
        n_episodes=sum(b.n_episodes for b in bands.values()),
        net_pnl_sum=sum(b.net_pnl_sum for b in bands.values()),
        notional_sum=sum(b.notional_sum for b in bands.values()),
        win_count=sum(b.win_count for b in bands.values()),
    )
    return BandTable(bands=bands, global_entry=global_entry, n_episodes_scored=global_entry.n_episodes)


def _make_regime_tables() -> Dict[str, BandTable]:
    return {
        "MEAN_REVERT": _make_band_table(),
        "OTHER": _make_band_table(),
    }


# ── Tests: ISO/MS conversion ────────────────────────────────────────────

class TestIsoToMs:
    def test_basic_utc(self):
        ms = _iso_to_ms("2025-12-09T21:43:56.526778+00:00")
        assert isinstance(ms, int)
        assert ms > 0

    def test_roundtrip(self):
        from datetime import datetime, timezone
        dt = datetime(2025, 12, 9, 21, 43, 56, tzinfo=timezone.utc)
        ms = _iso_to_ms(dt.isoformat())
        assert abs(ms - int(dt.timestamp() * 1000)) < 1000


# ── Tests: Find bar index ───────────────────────────────────────────────

class TestFindBarIndex:
    def test_empty_klines(self):
        assert _find_bar_index([], 100) is None

    def test_exact_match(self):
        klines = _make_klines(10)
        target = klines[5]["open_time"]
        assert _find_bar_index(klines, target) == 5

    def test_between_bars(self):
        klines = _make_klines(10)
        # Target between bar 3 and bar 4 — snaps to bar 3's 15m window
        target = (klines[3]["open_time"] + klines[4]["open_time"]) // 2
        assert _find_bar_index(klines, target) == 3

    def test_before_first_bar(self):
        klines = _make_klines(10)
        assert _find_bar_index(klines, klines[0]["open_time"] - 900_000) is None

    def test_after_last_bar(self):
        klines = _make_klines(10)
        idx = _find_bar_index(klines, klines[-1]["open_time"] + 1000000)
        assert idx == len(klines) - 1

    def test_snap_to_15m(self):
        """Verify target is snapped to 15m bar open."""
        ts = 1700000000000 + 450_000  # 0.5 bars into a 15m bar
        snapped = _snap_to_15m_open(ts)
        assert snapped % 900_000 == 0


# ── Tests: Feature extraction ───────────────────────────────────────────

class TestExtractFeaturesAtBar:
    def test_returns_closes_and_features(self):
        klines = _make_klines(100)
        closes, features = extract_features_at_bar(klines, 80)
        assert len(closes) == WARMUP_BARS
        assert "trend_r2" in features
        assert "range_position" in features
        assert "vol_regime_z" in features
        assert "data_quality" in features

    def test_short_window(self):
        klines = _make_klines(10)
        closes, features = extract_features_at_bar(klines, 5)
        assert len(closes) == 6  # bars 0..5

    def test_at_start(self):
        klines = _make_klines(100)
        closes, features = extract_features_at_bar(klines, 0)
        assert len(closes) == 1


# ── Tests: Skip record ──────────────────────────────────────────────────

class TestSkipRecord:
    def test_structure(self):
        ep = _make_episode()
        rec = _skip_record(ep, "no_klines")
        assert rec["episode_id"] == "EP_0001"
        assert rec["skip_reason"] == "no_klines"
        assert rec["signal_generated"] is False


# ── Tests: Replay episode ───────────────────────────────────────────────

class TestReplayEpisode:
    def test_insufficient_warmup(self):
        klines = _make_klines(10)
        ep = _make_episode(entry_ts_ms=klines[5]["open_time"])
        records = replay_episode(
            ep, klines, bar_idx=5,
            c1_config=C1Config(), c2_config=C2Config(),
            regime_tables=_make_regime_tables(),
            pooled_table=_make_band_table(),
            bridge_config=BridgeConfig(),
        )
        assert len(records) == 1
        assert "insufficient_warmup" in records[0].get("skip_reason", "")

    def test_mean_revert_episode_generates_signals(self):
        """MEAN_REVERT episode with enough bars should produce C2 signals (and maybe C1 MR)."""
        klines = _make_klines(150)
        ep = _make_episode(
            regime="MEAN_REVERT",
            entry_ts_ms=klines[100]["open_time"],
        )
        records = replay_episode(
            ep, klines, bar_idx=100,
            c1_config=C1Config(), c2_config=C2Config(),
            regime_tables=_make_regime_tables(),
            pooled_table=_make_band_table(),
            bridge_config=BridgeConfig(),
        )
        # Should have at least C2 signals (normal + inverted)
        # C1 MR may or may not fire depending on zscore/range triggers
        assert len(records) >= 1
        # All signal records should have realized PnL
        for rec in records:
            if rec.get("signal_generated"):
                assert "realized_net_pnl" in rec
                assert "realized_net_edge_pct" in rec

    def test_unknown_regime_may_get_c2_signals(self):
        """Unknown regime: C1 won't fire, but C2 may if not in center region."""
        klines = _make_klines(150)
        ep = _make_episode(
            regime="unknown",
            entry_ts_ms=klines[100]["open_time"],
        )
        records = replay_episode(
            ep, klines, bar_idx=100,
            c1_config=C1Config(), c2_config=C2Config(),
            regime_tables=_make_regime_tables(),
            pooled_table=_make_band_table(),
            bridge_config=BridgeConfig(),
        )
        # C1 trend won't fire (not TREND_UP/DOWN), C1 MR won't fire (not MEAN_REVERT)
        # C2 will fire if region is not center
        c1_recs = [r for r in records if r.get("candidate_family") == "C1"]
        assert len(c1_recs) == 0  # No C1 signals for unknown regime

    def test_episode_pnl_attached(self):
        """Verify realized PnL fields are attached to signal records."""
        klines = _make_klines(150)
        ep = _make_episode(
            regime="MEAN_REVERT",
            net_pnl=-2.5,
            gross_pnl=-1.5,
            fees=1.0,
            entry_notional=500.0,
            entry_ts_ms=klines[100]["open_time"],
        )
        records = replay_episode(
            ep, klines, bar_idx=100,
            c1_config=C1Config(), c2_config=C2Config(),
            regime_tables=_make_regime_tables(),
            pooled_table=_make_band_table(),
            bridge_config=BridgeConfig(),
        )
        for rec in records:
            if rec.get("signal_generated"):
                assert rec["realized_net_pnl"] == -2.5
                assert rec["realized_fees"] == 1.0
                assert rec["entry_notional"] == 500.0
                assert abs(rec["realized_net_edge_pct"] - (-2.5 / 500.0)) < 1e-8


# ── Tests: Spearman rho ─────────────────────────────────────────────────

class TestSpearmanRho:
    def test_perfect_positive(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert rho is not None
        assert abs(rho - 1.0) < 1e-6

    def test_perfect_negative(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert rho is not None
        assert abs(rho - (-1.0)) < 1e-6

    def test_zero_correlation(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [3, 1, 4, 5, 2])
        assert rho is not None
        assert abs(rho) < 0.5

    def test_too_few_pairs(self):
        assert _spearman_rho([1, 2], [3, 4]) is None
        assert _spearman_rho([], []) is None

    def test_ties_handled(self):
        rho = _spearman_rho([1, 1, 2, 3], [1, 2, 3, 4])
        assert rho is not None
        assert -1 <= rho <= 1


# ── Tests: Summary stats ────────────────────────────────────────────────

def _make_signal_records(n: int = 50, candidate_id: str = "C2_REGION_NORMAL") -> List[Dict[str, Any]]:
    """Generate synthetic signal records for summary testing."""
    recs = []
    for i in range(n):
        conv = 0.5 + 0.01 * (i % 10)
        edge_pct = -0.003 + 0.001 * (i % 5)
        recs.append({
            "signal_generated": True,
            "candidate_id": candidate_id,
            "candidate_family": "C2",
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"][i % 3],
            "regime": "MEAN_REVERT" if i % 4 != 0 else "OTHER",
            "conviction": conv,
            "bridge_expected_edge_pct": edge_pct,
            "bridge_would_pass": edge_pct > 0.0012,
            "bridge_band_key": f"0.{50 + (i % 3) * 5:02d}-0.{55 + (i % 3) * 5:02d}",
            "bridge_lookup_tier": "band_regime",
            "control_expected_edge_pct": -0.004,
            "control_would_pass": False,
            "realized_net_edge_pct": edge_pct + 0.001,
            "realized_net_pnl": (edge_pct + 0.001) * 200,
            "realized_gross_pnl": (edge_pct + 0.002) * 200,
            "realized_fees": 0.2,
            "entry_notional": 200.0,
            "episode_id": f"EP_{i:04d}",
            "episode_regime": "MEAN_REVERT",
        })
    return recs


class TestComputeReplaySummary:
    def test_empty_records(self):
        s = compute_replay_summary([])
        assert s["global"]["n_signals"] == 0

    def test_single_candidate(self):
        recs = _make_signal_records(30, "C2_REGION_NORMAL")
        s = compute_replay_summary(recs)
        assert "C2_REGION_NORMAL" in s["per_candidate"]
        stats = s["per_candidate"]["C2_REGION_NORMAL"]
        assert stats["n_signals"] == 30
        assert 0 <= stats["pass_rate"] <= 1
        assert stats["spearman_rho"] is not None or stats["n_signals"] < 3

    def test_multiple_candidates(self):
        recs = (
            _make_signal_records(20, "C1_MR_NORMAL")
            + _make_signal_records(20, "C2_REGION_NORMAL")
        )
        s = compute_replay_summary(recs)
        assert len(s["per_candidate"]) == 2
        assert "C1_MR_NORMAL" in s["per_candidate"]
        assert "C2_REGION_NORMAL" in s["per_candidate"]

    def test_skip_records_counted(self):
        recs = [{"skip_reason": "no_klines", "candidate_id": "SKIPPED"}] * 5
        s = compute_replay_summary(recs)
        assert s["global"]["n_skipped"] == 5


# ── Tests: Fast-fail gates ──────────────────────────────────────────────

class TestFastFailGates:
    def _summary_with_stats(self, **overrides) -> Dict[str, Any]:
        """Build a summary with one candidate having specified stats."""
        base = {
            "n_signals": 100,
            "pass_rate": 0.30,
            "spearman_rho": 0.15,
            "max_symbol_share": 0.40,
            "max_symbol": "BTCUSDT",
            "mean_realized_edge_pct": 0.15,
            "band_edge_stats": {
                "0.50-0.55": {
                    "n": 30,
                    "n_realized": 30,
                    "mean_realized_edge_pct": 0.15,
                    "mean_expected_edge_pct": 0.13,
                    "n_pass": 10,
                },
            },
            "best_band": {
                "band_key": "0.50-0.55",
                "n_realized": 30,
                "mean_realized_edge_pct": 0.15,
            },
        }
        base.update(overrides)
        return {"per_candidate": {"C2_REGION_NORMAL": base}}

    def test_all_pass(self):
        s = self._summary_with_stats()
        v = apply_fast_fail_gates(s)
        assert v["C2_REGION_NORMAL"]["passed"] is True
        assert v["C2_REGION_NORMAL"]["n_fails"] == 0

    def test_fail_no_fee_clearing_band(self):
        s = self._summary_with_stats(
            band_edge_stats={
                "0.50-0.55": {
                    "n": 30, "n_realized": 30,
                    "mean_realized_edge_pct": 0.0005,  # below 0.12%
                    "mean_expected_edge_pct": 0.0005, "n_pass": 0,
                },
            }
        )
        v = apply_fast_fail_gates(s)
        assert v["C2_REGION_NORMAL"]["passed"] is False
        assert "no_fee_clearing_band" in v["C2_REGION_NORMAL"]["fails"]

    def test_fail_best_band_too_small(self):
        s = self._summary_with_stats(
            best_band={"band_key": "0.50-0.55", "n_realized": 5, "mean_realized_edge_pct": 0.01},
        )
        v = apply_fast_fail_gates(s)
        assert v["C2_REGION_NORMAL"]["passed"] is False
        assert any("best_band_n" in f for f in v["C2_REGION_NORMAL"]["fails"])

    def test_fail_spearman_negative(self):
        s = self._summary_with_stats(spearman_rho=-0.1)
        v = apply_fast_fail_gates(s)
        assert v["C2_REGION_NORMAL"]["passed"] is False
        assert any("spearman_rho" in f for f in v["C2_REGION_NORMAL"]["fails"])

    def test_fail_spearman_none(self):
        s = self._summary_with_stats(spearman_rho=None)
        v = apply_fast_fail_gates(s)
        assert any("spearman_rho" in f for f in v["C2_REGION_NORMAL"]["fails"])

    def test_fail_pass_rate_zero(self):
        s = self._summary_with_stats(pass_rate=0.0)
        v = apply_fast_fail_gates(s)
        assert any("pass_rate_zero" in f for f in v["C2_REGION_NORMAL"]["fails"])

    def test_fail_pass_rate_too_high(self):
        s = self._summary_with_stats(pass_rate=0.95)
        v = apply_fast_fail_gates(s)
        assert any("pass_rate_too_high" in f for f in v["C2_REGION_NORMAL"]["fails"])

    def test_fail_dominant_symbol_negative_edge(self):
        s = self._summary_with_stats(
            max_symbol_share=0.75,
            max_symbol="BTCUSDT",
            per_symbol_edge={
                "BTCUSDT": {
                    "n": 75,
                    "mean_realized_edge_pct": -0.001,
                    "mean_expected_edge_pct": -0.002,
                    "best_band": {"n_realized": 5, "mean_realized_edge_pct": -0.001},
                },
            },
        )
        v = apply_fast_fail_gates(s)
        assert any("dominant_symbol" in f for f in v["C2_REGION_NORMAL"]["fails"])


# ── Tests: Promotion gates ──────────────────────────────────────────────

class TestPromotionGates:
    def _base_summary(self) -> Dict[str, Any]:
        return {
            "per_candidate": {
                "C2_REGION_NORMAL": {
                    "pass_rate": 0.25,
                    "spearman_rho": 0.20,
                    "best_band": {
                        "band_key": "0.50-0.55",
                        "n_realized": 30,
                        "mean_realized_edge_pct": 0.20,
                    },
                }
            }
        }

    def test_promoted_when_all_gates_pass(self):
        ff = {"C2_REGION_NORMAL": {"passed": True, "fails": []}}
        v = apply_promotion_gates(self._base_summary(), ff)
        assert v["C2_REGION_NORMAL"]["promoted"] is True

    def test_not_promoted_if_fast_fail_failed(self):
        ff = {"C2_REGION_NORMAL": {"passed": False, "fails": ["no_fee_clearing_band"]}}
        v = apply_promotion_gates(self._base_summary(), ff)
        assert v["C2_REGION_NORMAL"]["promoted"] is False
        assert v["C2_REGION_NORMAL"]["reason"] == "failed_fast_fail"

    def test_not_promoted_low_rho(self):
        summary = self._base_summary()
        summary["per_candidate"]["C2_REGION_NORMAL"]["spearman_rho"] = 0.05
        ff = {"C2_REGION_NORMAL": {"passed": True, "fails": []}}
        v = apply_promotion_gates(summary, ff)
        assert v["C2_REGION_NORMAL"]["promoted"] is False
        assert any("spearman_rho" in g for g in v["C2_REGION_NORMAL"]["gate_fails"])

    def test_not_promoted_low_selectivity(self):
        summary = self._base_summary()
        summary["per_candidate"]["C2_REGION_NORMAL"]["pass_rate"] = 0.02
        ff = {"C2_REGION_NORMAL": {"passed": True, "fails": []}}
        v = apply_promotion_gates(summary, ff)
        assert v["C2_REGION_NORMAL"]["promoted"] is False
        assert any("selectivity" in g for g in v["C2_REGION_NORMAL"]["gate_fails"])


# ── Tests: Run manifest ─────────────────────────────────────────────────

class TestRunManifest:
    def test_structure(self):
        m = build_run_manifest()
        assert m["version"] == "p6b5_v1"
        assert "c1_config" in m
        assert "c2_config" in m
        assert "file_hashes" in m
        assert "fee_threshold_pct" in m
        assert m["warmup_bars"] == WARMUP_BARS

    def test_configs_frozen(self):
        m = build_run_manifest()
        assert m["c1_config"]["ema_fast_period"] == 15
        assert m["c1_config"]["conv_k1"] == 0.30
        assert m["c2_config"]["conv_low_range"] == 0.58

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            m = build_run_manifest()
            from execution.p6_replay import save_run_manifest
            save_run_manifest(m, path)
            loaded = json.loads(path.read_text())
            assert loaded["version"] == "p6b5_v1"


# ── Tests: CSV export ───────────────────────────────────────────────────

class TestCSVExport:
    def test_empty_summary(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tables.csv"
            export_replay_tables_csv({"per_candidate": {}}, path)
            content = path.read_text()
            assert "candidate_id" in content

    def test_with_data(self):
        recs = _make_signal_records(20, "C2_REGION_NORMAL")
        summary = compute_replay_summary(recs)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tables.csv"
            export_replay_tables_csv(summary, path)
            content = path.read_text()
            assert "C2_REGION_NORMAL" in content
            lines = content.strip().split("\n")
            assert len(lines) == 2  # header + 1 row


# ── Tests: Integration (run_replay with mock klines) ────────────────────

class TestRunReplayIntegration:
    def test_dry_run_with_mock_klines(self):
        """Integration: run_replay with synthetic klines and episodes."""
        klines = _make_klines(200)
        eps = [
            _make_episode("EP_1", "BTCUSDT", "LONG", "MEAN_REVERT",
                          entry_ts_ms=klines[100]["open_time"]),
            _make_episode("EP_2", "BTCUSDT", "SHORT", "TREND_UP",
                          entry_ts_ms=klines[120]["open_time"]),
            _make_episode("EP_3", "ETHUSDT", "LONG", "MEAN_REVERT",
                          entry_ts_ms=klines[130]["open_time"]),
        ]
        cache = {
            "BTCUSDT": klines,
            "ETHUSDT": klines,  # reuse same klines
            "SOLUSDT": klines,
        }
        records = run_replay(
            episodes=eps,
            kline_cache=cache,
            dry_run=True,
        )
        assert len(records) > 0
        # All records should have episode_id
        for rec in records:
            assert "episode_id" in rec or "skip_reason" in rec

    def test_out_of_universe_episodes_skipped(self):
        klines = _make_klines(200)
        eps = [
            _make_episode("EP_1", "BTCUSDT", entry_ts_ms=klines[100]["open_time"]),
            _make_episode("EP_2", "LINKUSDT", entry_ts_ms=klines[100]["open_time"]),
        ]
        cache = {"BTCUSDT": klines, "ETHUSDT": klines, "SOLUSDT": klines}
        records = run_replay(episodes=eps, kline_cache=cache, dry_run=True)
        skipped = [r for r in records if r.get("skip_reason") == "out_of_universe"]
        assert len(skipped) == 1
        assert skipped[0]["symbol"] == "LINKUSDT"

    def test_no_klines_skipped(self):
        eps = [_make_episode("EP_1", "BTCUSDT")]
        cache = {"BTCUSDT": [], "ETHUSDT": [], "SOLUSDT": []}
        records = run_replay(episodes=eps, kline_cache=cache, dry_run=True)
        assert any(r.get("skip_reason") == "no_klines" for r in records)

    def test_full_pipeline_dry_run(self):
        """Test compute_replay_summary + fast_fail + promotion on synthetic data."""
        recs = _make_signal_records(50, "C2_REGION_NORMAL")
        summary = compute_replay_summary(recs)
        ff = apply_fast_fail_gates(summary)
        promo = apply_promotion_gates(summary, ff)
        assert "C2_REGION_NORMAL" in ff
        assert "C2_REGION_NORMAL" in promo
        # Verify structure
        assert "passed" in ff["C2_REGION_NORMAL"]
        assert "promoted" in promo["C2_REGION_NORMAL"]


# ── Tests: Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_episode_zero_notional(self):
        klines = _make_klines(150)
        ep = _make_episode(entry_notional=0, entry_ts_ms=klines[100]["open_time"])
        records = replay_episode(
            ep, klines, bar_idx=80,
            c1_config=C1Config(), c2_config=C2Config(),
            regime_tables=_make_regime_tables(),
            pooled_table=_make_band_table(),
            bridge_config=BridgeConfig(),
        )
        for rec in records:
            if rec.get("signal_generated"):
                assert rec["realized_net_edge_pct"] == 0.0

    def test_spearman_with_constant_values(self):
        """Constant values → zero variance → None."""
        rho = _spearman_rho([0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4])
        assert rho is None

    def test_candidate_ids_constant(self):
        assert len(ALL_CANDIDATE_IDS) == 6
        assert "C1_TREND_NORMAL" in ALL_CANDIDATE_IDS
        assert "C2_REGION_INVERTED" in ALL_CANDIDATE_IDS
