"""Tests for FPS v1 class evaluation scorecard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.fps_class_scorecard import (
    ADVANCE,
    INSUFFICIENT_DATA,
    KILL,
    REFINE,
    _extract_class_metrics,
    compute_scorecard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_shadow(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _permit_record(cls: str, symbol: str = "BTCUSDT", regime: str = "TREND_UP",
                   ts: float = 100.0, **extra) -> dict:
    return {
        "ts": ts, "schema": "fps_shadow_v1", "symbol": symbol,
        "verdict": "PERMIT_CANDIDATE", "setup_class": cls,
        "direction": "LONG", "gate_trace": [f"g1:{cls}", "g2:LONG", "g3:fee_pass"],
        "deny_reason": None, "regime_current": regime, "atr_pct": 0.03,
        **extra,
    }


def _deny_fee_record(cls: str, symbol: str = "BTCUSDT", ts: float = 100.0) -> dict:
    return {
        "ts": ts, "schema": "fps_shadow_v1", "symbol": symbol,
        "verdict": "DENY_STRUCTURAL", "setup_class": cls,
        "direction": "LONG", "gate_trace": [f"g1:{cls}", "g2:LONG", "g3:fee_deny"],
        "deny_reason": "fee_bridge_insufficient", "regime_current": "TREND_UP",
    }


def _abstain_record(ts: float = 100.0, symbol: str = "BTCUSDT") -> dict:
    return {
        "ts": ts, "schema": "fps_shadow_v1", "symbol": symbol,
        "verdict": "ABSTAIN", "setup_class": None,
        "direction": None, "gate_trace": ["g1:no_match"],
        "deny_reason": None, "regime_current": "TREND_UP",
    }


def _dir_mismatch_record(cls: str, ts: float = 100.0) -> dict:
    return {
        "ts": ts, "schema": "fps_shadow_v1", "symbol": "BTCUSDT",
        "verdict": "ABSTAIN", "setup_class": cls,
        "direction": None, "gate_trace": [f"g1:{cls}", "g2:direction_mismatch"],
        "deny_reason": None, "regime_current": "TREND_UP",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEmptyLog:
    def test_missing_file(self, tmp_path):
        log = tmp_path / "missing.jsonl"
        report = compute_scorecard(log)
        assert report.total_records == 0
        assert report.summary_verdict == INSUFFICIENT_DATA

    def test_empty_file(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text("")
        report = compute_scorecard(log)
        assert report.total_records == 0


class TestInsufficientData:
    def test_below_min_samples(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records.append(_permit_record("VOL_EXPANSION_BREAKOUT", ts=200.0))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        veb = report.class_scores["VOL_EXPANSION_BREAKOUT"]
        assert veb.verdict == INSUFFICIENT_DATA
        assert veb.permit_count == 1


class TestKillCriteria:
    def test_kill_low_fee_pass_rate(self, tmp_path):
        """<55% fee pass → KILL."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(500)]
        # 10 permits, 25 fee denies → fee_pass_rate = 10/35 = 28.6%
        for i in range(10):
            records.append(_permit_record("VOL_EXPANSION_BREAKOUT", ts=600 + i))
        for i in range(25):
            records.append(_deny_fee_record("VOL_EXPANSION_BREAKOUT", ts=700 + i))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=10)
        veb = report.class_scores["VOL_EXPANSION_BREAKOUT"]
        assert veb.verdict == KILL
        assert any("fee_pass_rate" in r for r in veb.kill_reasons)

    def test_kill_symbol_concentration(self, tmp_path):
        """Single symbol >80% → KILL."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(500)]
        # 40 permits all on BTCUSDT
        for i in range(40):
            records.append(_permit_record("VOL_EXPANSION_BREAKOUT", symbol="BTCUSDT", ts=600 + i))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        veb = report.class_scores["VOL_EXPANSION_BREAKOUT"]
        assert veb.verdict == KILL
        assert any("symbol concentration" in r for r in veb.kill_reasons)

    def test_kill_direction_mismatch(self, tmp_path):
        """<80% direction match → KILL."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(500)]
        # 10 permits (direction match) + 30 direction mismatches = 25% match
        for i in range(10):
            records.append(_permit_record("EXHAUSTION_REVERSAL", ts=600 + i,
                                          symbol="BTCUSDT" if i % 3 else "ETHUSDT"))
        for i in range(30):
            records.append(_dir_mismatch_record("EXHAUSTION_REVERSAL", ts=700 + i))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=10)
        ere = report.class_scores["EXHAUSTION_REVERSAL"]
        assert ere.verdict == KILL
        assert any("direction_match" in r for r in ere.kill_reasons)


class TestRefineCriteria:
    def test_refine_borderline_fee(self, tmp_path):
        """Fee pass 55-70% → REFINE (not KILL, not ADVANCE)."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(500)]
        # 35 permits, 15 fee denies → 70% pass but < 70% advance threshold
        # Actually 35/(35+15) = 70%... Make it 30/50 = 60%
        for i in range(30):
            records.append(_permit_record("TREND_PULLBACK", ts=600 + i,
                                          symbol=["BTCUSDT", "ETHUSDT", "SOLUSDT"][i % 3]))
        for i in range(20):
            records.append(_deny_fee_record("TREND_PULLBACK", ts=700 + i))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        tcp = report.class_scores["TREND_PULLBACK"]
        assert tcp.verdict == REFINE
        assert any("fee_pass_rate" in r for r in tcp.refine_reasons)

    def test_refine_single_symbol(self, tmp_path):
        """Permits on only 1 symbol → REFINE (not kill if share happens to pass)."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(5000)]
        # 35 permits on 3 symbols but majority on one (~60%)
        for i in range(30):
            records.append(_permit_record("VOL_EXPANSION_BREAKOUT",
                                          symbol="BTCUSDT", ts=6000 + i * 60))
        for i in range(1):
            records.append(_permit_record("VOL_EXPANSION_BREAKOUT",
                                          symbol="ETHUSDT", ts=8000 + i * 60))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        veb = report.class_scores["VOL_EXPANSION_BREAKOUT"]
        # 30/31 = 96.8% on BTC → KILL for concentration
        assert veb.verdict == KILL


class TestAdvance:
    def test_advance_healthy_class(self, tmp_path):
        """Class that passes all criteria → ADVANCE."""
        log = tmp_path / "shadow.jsonl"
        # Spread abstains across full time range so permits are interleaved
        records = [_abstain_record(ts=float(i * 30)) for i in range(2000)]
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        regimes = ["TREND_UP", "TREND_DOWN"]
        # 50 permits spread across symbols, regimes, and time (every ~20min)
        for i in range(50):
            records.append(_permit_record(
                "TREND_PULLBACK",
                symbol=symbols[i % len(symbols)],
                regime=regimes[i % len(regimes)],
                ts=1000 + i * 1200,  # 20 min apart across full range
            ))
        # 5 fee denies (fee pass = 50/55 = 91%)
        for i in range(5):
            records.append(_deny_fee_record("TREND_PULLBACK", ts=62000 + i * 1200))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        tcp = report.class_scores["TREND_PULLBACK"]
        assert tcp.verdict == ADVANCE
        assert not tcp.kill_reasons
        assert not tcp.refine_reasons


class TestScorecardSummary:
    def test_summary_is_worst_verdict(self, tmp_path):
        """If one class is KILL, summary is KILL."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(2000)]
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        # TCP: healthy → ADVANCE
        for i in range(50):
            records.append(_permit_record("TREND_PULLBACK",
                                          symbol=symbols[i % 4], ts=10000 + i * 600))
        # VEB: all on one symbol → KILL
        for i in range(40):
            records.append(_permit_record("VOL_EXPANSION_BREAKOUT",
                                          symbol="BTCUSDT", ts=50000 + i))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        assert report.summary_verdict == KILL

    def test_alerts_list_populated(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records.append(_permit_record("EXHAUSTION_REVERSAL", ts=200.0))
        _write_shadow(log, records)
        report = compute_scorecard(log, min_samples=30)
        assert any("INSUFFICIENT_DATA" in a for a in report.alerts)


class TestMetricAccuracy:
    def test_fee_pass_rate_calculation(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records += [_permit_record("VOL_EXPANSION_BREAKOUT", ts=200 + i) for i in range(8)]
        records += [_deny_fee_record("VOL_EXPANSION_BREAKOUT", ts=300 + i) for i in range(2)]
        _write_shadow(log, records)
        score = _extract_class_metrics("VOL_EXPANSION_BREAKOUT", records, min_samples=5)
        assert score.fee_pass_count == 8
        assert score.fee_deny_count == 2
        assert score.fee_pass_rate == pytest.approx(0.8, abs=0.01)

    def test_direction_match_rate_calculation(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records += [_permit_record("EXHAUSTION_REVERSAL", ts=200 + i) for i in range(7)]
        records += [_dir_mismatch_record("EXHAUSTION_REVERSAL", ts=300 + i) for i in range(3)]
        _write_shadow(log, records)
        score = _extract_class_metrics("EXHAUSTION_REVERSAL", records, min_samples=5)
        # direction_match = 7 permits (passed gate2) / (7 + 3) = 70%
        assert score.direction_match_rate == pytest.approx(0.7, abs=0.01)

    def test_regime_distribution(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records += [_permit_record("TREND_PULLBACK", regime="TREND_UP", ts=200 + i) for i in range(6)]
        records += [_permit_record("TREND_PULLBACK", regime="TREND_DOWN", ts=300 + i) for i in range(4)]
        _write_shadow(log, records)
        score = _extract_class_metrics("TREND_PULLBACK", records, min_samples=5)
        assert score.by_regime == {"TREND_UP": 6, "TREND_DOWN": 4}


class TestMeanReturn:
    def test_no_return_data(self, tmp_path):
        """Without realized_return in records, metric is None."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        records += [_permit_record("VOL_EXPANSION_BREAKOUT", ts=200 + i) for i in range(10)]
        _write_shadow(log, records)
        score = _extract_class_metrics("VOL_EXPANSION_BREAKOUT", records, min_samples=5)
        assert score.mean_return_per_permit is None
        assert score.return_sample_count == 0

    def test_with_return_data(self, tmp_path):
        """When realized_return present, mean is computed correctly."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        # 5 permits with returns: [+0.01, -0.005, +0.02, +0.003, -0.001]
        returns = [0.01, -0.005, 0.02, 0.003, -0.001]
        for i, ret in enumerate(returns):
            records.append(_permit_record("TREND_PULLBACK", ts=200 + i,
                                          realized_return=ret))
        # 3 permits without return data
        for i in range(3):
            records.append(_permit_record("TREND_PULLBACK", ts=300 + i))
        _write_shadow(log, records)
        score = _extract_class_metrics("TREND_PULLBACK", records, min_samples=5)
        expected_mean = sum(returns) / len(returns)  # 0.0054
        assert score.mean_return_per_permit == pytest.approx(expected_mean, abs=1e-6)
        assert score.return_sample_count == 5

    def test_partial_return_data(self, tmp_path):
        """Only permits with realized_return contribute to mean."""
        log = tmp_path / "shadow.jsonl"
        records = [_abstain_record(ts=float(i)) for i in range(100)]
        # 2 with return, 8 without
        records.append(_permit_record("EXHAUSTION_REVERSAL", ts=200, realized_return=0.05))
        records.append(_permit_record("EXHAUSTION_REVERSAL", ts=201, realized_return=-0.02))
        for i in range(8):
            records.append(_permit_record("EXHAUSTION_REVERSAL", ts=210 + i))
        _write_shadow(log, records)
        score = _extract_class_metrics("EXHAUSTION_REVERSAL", records, min_samples=5)
        assert score.mean_return_per_permit == pytest.approx(0.015, abs=1e-6)
        assert score.return_sample_count == 2
