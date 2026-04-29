"""Tests for P6 Shadow Evaluator — bridge integration + logging."""

import json
import pytest
from pathlib import Path

from execution.p6_simple_rules import P6Signal
from execution.p6_shadow_evaluator import (
    compute_shadow_summary,
    evaluate_and_log_signals,
    evaluate_signal_against_bridge,
)
from execution.expectancy_bridge import (
    BandEntry,
    BandTable,
)

TS = 1700000000.0


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_signal():
    return P6Signal(
        candidate_id="C1_TREND_NORMAL", candidate_family="C1",
        rule_name="trend_ema_crossover", symbol="BTCUSDT",
        side="LONG", polarity="normal", regime="TREND_UP",
        feature_snapshot={"ema_fast": 105.0, "ema_slow": 100.0, "trend_r2": 0.8},
        ts=TS,
    )

@pytest.fixture
def sample_mr_signal():
    return P6Signal(
        candidate_id="C1_MR_NORMAL", candidate_family="C1",
        rule_name="mr_zscore_range", symbol="ETHUSDT",
        side="LONG", polarity="normal", regime="MEAN_REVERT",
        feature_snapshot={"zscore": -2.0, "range_position": 0.1},
        ts=TS,
    )

@pytest.fixture
def sample_c2_signal():
    return P6Signal(
        candidate_id="C2_REGION_NORMAL", candidate_family="C2",
        rule_name="region_low_range", symbol="SOLUSDT",
        side="LONG", polarity="normal", regime="MEAN_REVERT",
        feature_snapshot={"range_position": 0.1, "vol_regime_z": 0.0},
        ts=TS, region="low_range",
    )

@pytest.fixture
def mock_regime_tables():
    """Build a minimal regime table for testing."""
    # Create a table with one band entry that has positive edge
    table = BandTable(
        build_ts=TS,
        n_episodes_total=100,
        n_episodes_scored=80,
    )
    table.bands["0.45-0.50"] = BandEntry(
        band_lo=0.45, band_hi=0.50,
        n_episodes=30, net_pnl_sum=15.0,
        notional_sum=1000.0, win_count=18,
    )
    return {"OTHER": table, "MEAN_REVERT": table}

@pytest.fixture
def mock_pooled_table():
    table = BandTable(
        build_ts=TS,
        n_episodes_total=200,
        n_episodes_scored=150,
    )
    table.bands["0.45-0.50"] = BandEntry(
        band_lo=0.45, band_hi=0.50,
        n_episodes=50, net_pnl_sum=25.0,
        notional_sum=2000.0, win_count=30,
    )
    return table


# ── evaluate_signal_against_bridge tests ─────────────────────────────────

class TestEvaluateSignalAgainstBridge:
    def test_no_bridge_returns_no_bridge_tier(self, sample_signal):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=None, pooled_table=None,
        )
        assert rec["bridge_lookup_tier"] == "no_bridge"
        assert rec["bridge_expected_edge_pct"] == 0.0
        assert rec["fee_pass"] is False

    def test_with_bridge_returns_edge(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        # Should have gone through bridge lookup
        assert rec["bridge_lookup_tier"] != "no_bridge"
        assert isinstance(rec["bridge_expected_edge_pct"], float)
        assert rec["bridge_regime_key"] == "OTHER"  # TREND_UP → OTHER

    def test_mr_regime_maps_to_mean_revert(self, sample_mr_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_mr_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["bridge_regime_key"] == "MEAN_REVERT"

    def test_provenance_fields_complete(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        required_fields = [
            "ts", "signal_ts", "candidate_id", "candidate_family", "rule_name",
            "symbol", "side", "polarity", "regime", "region",
            "selected_for_eval", "conviction", "bridge_regime_key",
            "bridge_expected_edge_pct", "bridge_lookup_tier", "bridge_band_key",
            "bridge_sample_n", "bridge_is_sufficient", "bridge_fallback_depth",
            "bridge_cold_start_reason", "bridge_would_pass",
            "fee_required_pct", "fee_pass",
            "control_conviction", "control_expected_edge_pct",
            "control_band_key", "control_would_pass",
            "feature_snapshot", "suppressed_alternatives",
        ]
        for f in required_fields:
            assert f in rec, f"Missing field: {f}"

    def test_conviction_uses_signal_proxy(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["conviction"] == round(sample_signal.conviction, 6)
        # Control always at 0.5
        assert rec["control_conviction"] == 0.5

    def test_fee_threshold_custom(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
            fee_threshold_pct=100.0,  # impossibly high
        )
        assert rec["fee_pass"] is False
        assert rec["bridge_would_pass"] is False
        assert rec["fee_required_pct"] == 100.0

    def test_fee_pass_when_edge_exceeds_threshold(self, sample_signal):
        """If bridge returns high edge, fee_pass should be True."""
        # Create a table with very high edge
        table = BandTable(build_ts=TS, n_episodes_total=100, n_episodes_scored=80)
        table.bands["0.45-0.50"] = BandEntry(
            band_lo=0.45, band_hi=0.50,
            n_episodes=30, net_pnl_sum=500.0,
            notional_sum=1000.0, win_count=25,
        )
        tables = {"OTHER": table, "MEAN_REVERT": table}
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=tables, pooled_table=table,
            fee_threshold_pct=0.01,  # very low threshold
        )
        # Edge should be positive and exceed 0.01%
        if rec["bridge_expected_edge_pct"] > 0.01:
            assert rec["fee_pass"] is True

    def test_c2_signal_carries_region(self, sample_c2_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_c2_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["region"] == "low_range"

    def test_feature_snapshot_preserved(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["feature_snapshot"] == sample_signal.feature_snapshot

    def test_control_conviction_always_half(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["control_conviction"] == 0.5
        assert "control_expected_edge_pct" in rec
        assert "control_band_key" in rec
        assert "control_would_pass" in rec

    def test_bridge_would_pass_present(self, sample_signal, mock_regime_tables, mock_pooled_table):
        rec = evaluate_signal_against_bridge(
            sample_signal, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert "bridge_would_pass" in rec
        assert isinstance(rec["bridge_would_pass"], bool)

    def test_high_conviction_signal(self, mock_regime_tables, mock_pooled_table):
        """Signal with high proxy conviction goes to different band."""
        sig = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, conviction=0.75, ts=TS,
        )
        rec = evaluate_signal_against_bridge(
            sig, regime_tables=mock_regime_tables, pooled_table=mock_pooled_table,
        )
        assert rec["conviction"] == 0.75
        # Control should still use 0.5
        assert rec["control_conviction"] == 0.5


# ── evaluate_and_log_signals tests ───────────────────────────────────────

class TestEvaluateAndLogSignals:
    def test_dry_run_returns_records_no_file(self, sample_signal, mock_regime_tables, mock_pooled_table):
        records = evaluate_and_log_signals(
            [sample_signal],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0]["candidate_id"] == "C1_TREND_NORMAL"

    def test_writes_jsonl(self, sample_signal, mock_regime_tables, mock_pooled_table, tmp_path):
        log_file = tmp_path / "test_shadow.jsonl"
        evaluate_and_log_signals(
            [sample_signal],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            log_path=log_file,
        )
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["candidate_id"] == "C1_TREND_NORMAL"

    def test_multiple_signals(self, sample_signal, sample_mr_signal, mock_regime_tables, mock_pooled_table, tmp_path):
        log_file = tmp_path / "test_multi.jsonl"
        records = evaluate_and_log_signals(
            [sample_signal, sample_mr_signal],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            log_path=log_file,
        )
        assert len(records) == 2
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_suppressed_alternatives_attached(self, sample_signal, mock_regime_tables, mock_pooled_table):
        suppressed_sig = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="mr_zscore_range", symbol="BTCUSDT",
            side="SHORT", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS, selected_for_eval=False,
        )
        records = evaluate_and_log_signals(
            [sample_signal],
            suppressed=[suppressed_sig],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            dry_run=True,
        )
        assert len(records[0]["suppressed_alternatives"]) == 1
        assert records[0]["suppressed_alternatives"][0]["rule_name"] == "mr_zscore_range"

    def test_auto_loads_tables_when_none(self, sample_signal, tmp_path):
        """When tables are None, auto-load (may return empty if files missing)."""
        log_file = tmp_path / "test_auto.jsonl"
        # This should not crash even if bridge files don't exist
        records = evaluate_and_log_signals(
            [sample_signal],
            regime_tables=None,
            pooled_table=None,
            log_path=log_file,
            dry_run=True,
        )
        assert len(records) == 1

    def test_empty_signals_list(self, mock_regime_tables, mock_pooled_table):
        records = evaluate_and_log_signals(
            [],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            dry_run=True,
        )
        assert records == []

    def test_write_failure_does_not_crash(self, sample_signal, mock_regime_tables, mock_pooled_table):
        """Write failures are caught and logged, not propagated."""
        bad_path = Path("/nonexistent/deeply/nested/file.jsonl")
        # This should not raise
        records = evaluate_and_log_signals(
            [sample_signal],
            regime_tables=mock_regime_tables,
            pooled_table=mock_pooled_table,
            log_path=bad_path,
        )
        assert len(records) == 1


# ── compute_shadow_summary tests ─────────────────────────────────────────

class TestComputeShadowSummary:
    def test_empty_records(self):
        summary = compute_shadow_summary([])
        assert summary["n_signals"] == 0
        assert summary["n_fee_pass"] == 0
        assert summary["by_candidate"] == {}

    def test_single_record(self):
        records = [{
            "candidate_id": "C1_TREND_NORMAL",
            "symbol": "BTCUSDT",
            "regime": "TREND_UP",
            "fee_pass": True,
        }]
        summary = compute_shadow_summary(records)
        assert summary["n_signals"] == 1
        assert summary["n_fee_pass"] == 1
        assert summary["by_candidate"]["C1_TREND_NORMAL"]["n_signals"] == 1
        assert summary["by_candidate"]["C1_TREND_NORMAL"]["n_fee_pass"] == 1

    def test_multiple_candidates(self):
        records = [
            {"candidate_id": "C1_TREND_NORMAL", "symbol": "BTCUSDT", "regime": "TREND_UP", "fee_pass": True},
            {"candidate_id": "C1_TREND_INVERTED", "symbol": "BTCUSDT", "regime": "TREND_UP", "fee_pass": False},
            {"candidate_id": "C2_REGION_NORMAL", "symbol": "ETHUSDT", "regime": "MEAN_REVERT", "fee_pass": True},
        ]
        summary = compute_shadow_summary(records)
        assert summary["n_signals"] == 3
        assert summary["n_fee_pass"] == 2
        assert len(summary["by_candidate"]) == 3
        assert summary["by_symbol"]["BTCUSDT"]["n_signals"] == 2
        assert summary["by_regime"]["TREND_UP"]["n_signals"] == 2

    def test_no_fee_pass(self):
        records = [
            {"candidate_id": "C1_MR_NORMAL", "symbol": "ETHUSDT", "regime": "MEAN_REVERT", "fee_pass": False},
            {"candidate_id": "C1_MR_INVERTED", "symbol": "ETHUSDT", "regime": "MEAN_REVERT", "fee_pass": False},
        ]
        summary = compute_shadow_summary(records)
        assert summary["n_fee_pass"] == 0
        assert summary["by_candidate"]["C1_MR_NORMAL"]["n_fee_pass"] == 0

    def test_summary_has_timestamp(self):
        summary = compute_shadow_summary([])
        assert "ts" in summary
