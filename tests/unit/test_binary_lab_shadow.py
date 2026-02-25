"""
Tests for Binary Lab S1 — Shadow Sleeve (signals, adapter, runner).

Covers:
    - Signal extraction from state surfaces
    - Eligibility gate (all 7 conditions)
    - Direction mapping (regime → UP/DOWN, trend_slope fallback)
    - Conviction band mapping
    - Simulated fill model (conservative bias)
    - Round boundary detection & ID generation
    - Shadow runner lifecycle (entry → resolution → state machine feed)
    - Trade log emission (append-only, execution_mode: SHADOW)
    - No-trade logging with deny reasons
    - Kill line enforcement through state machine
    - Concurrent cap enforcement
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from execution.binary_lab_signals import (
    BinaryLabSignal,
    EligibilityResult,
    _BAND_RANK,
    _regime_to_direction,
    _score_to_band,
    check_eligibility,
    extract_signal,
)
from execution.binary_lab_shadow import (
    ENTRY_OFFSET_S,
    ENTRY_WINDOW_S,
    FEE_BPS,
    ROUND_DURATION_S,
    SLIPPAGE_BUFFER_BPS,
    BinaryLabShadowRunner,
    BinaryLabTradeWriter,
    OpenRound,
    RoundOutcome,
    SimulatedExecutionAdapter,
    SimulatedFill,
    _make_round_id,
    _round_start_unix,
)
from execution.binary_lab_executor import BinaryLabMode


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _limits() -> Dict[str, Any]:
    return {
        "_meta": {"sleeve_id": "binary_lab_s1", "time_horizon_minutes": 15},
        "capital": {"sleeve_total_usd": 2000, "per_round_usd": 20},
        "position_rules": {"max_concurrent": 3},
        "kill_conditions": {"kill_nav_usd": 1700, "sleeve_drawdown_usd": 300, "sleeve_drawdown_pct": 0.15},
        "entry_gate": {
            "blocked_regimes": ["CHOPPY"],
            "regime_confidence_fallback_min": 0.60,
            "min_conviction_band": "medium",
            "signal_source": "futures_pipeline",
        },
        "time_horizon": {"round_minutes": 15, "locked": True},
    }


def _signal(**overrides: Any) -> BinaryLabSignal:
    defaults = dict(
        symbol="BTCUSDT",
        regime="TREND_UP",
        regime_confidence=0.88,
        direction="UP",
        conviction_score=0.88,
        conviction_band="high",
        trend_slope=0.001,
        hybrid_score=0.50,
        ts="2026-02-25T12:00:00+00:00",
    )
    defaults.update(overrides)
    return BinaryLabSignal(**defaults)


def _sentinel_json(
    regime: str = "TREND_UP",
    confidence: float = 0.88,
    trend_slope: float = 0.001,
) -> Dict[str, Any]:
    return {
        "updated_ts": "2026-02-25T12:00:00+00:00",
        "primary_regime": regime,
        "smoothed_probs": {regime: confidence},
        "regime_probs": {regime: confidence},
        "features": {"trend_slope": trend_slope},
    }


def _scores_json(symbol: str = "BTCUSDT", hybrid: float = 0.50) -> Dict[str, Any]:
    return {
        "updated_ts": 1772050785.0,
        "symbols": [{"symbol": symbol, "hybrid_score": hybrid, "score": hybrid}],
    }


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


# ===========================================================================
# Tests: Direction mapping
# ===========================================================================

class TestDirectionMapping:

    def test_trend_up(self):
        assert _regime_to_direction("TREND_UP", None) == "UP"

    def test_trend_down(self):
        assert _regime_to_direction("TREND_DOWN", None) == "DOWN"

    def test_choppy_no_direction(self):
        assert _regime_to_direction("CHOPPY", 0.1) is None

    def test_crisis_no_direction(self):
        assert _regime_to_direction("CRISIS", -0.1) is None

    def test_mean_revert_positive_slope(self):
        assert _regime_to_direction("MEAN_REVERT", 0.05) == "UP"

    def test_mean_revert_negative_slope(self):
        assert _regime_to_direction("MEAN_REVERT", -0.05) == "DOWN"

    def test_mean_revert_no_slope(self):
        assert _regime_to_direction("MEAN_REVERT", None) is None

    def test_breakout_positive_slope(self):
        assert _regime_to_direction("BREAKOUT", 0.01) == "UP"

    def test_breakout_negative_slope(self):
        assert _regime_to_direction("BREAKOUT", -0.01) == "DOWN"

    def test_unknown_regime(self):
        assert _regime_to_direction("UNKNOWN", 0.1) is None


# ===========================================================================
# Tests: Conviction band mapping
# ===========================================================================

class TestConvictionBandMapping:

    def test_very_high(self):
        assert _score_to_band(0.95) == "very_high"

    def test_high(self):
        assert _score_to_band(0.85) == "high"

    def test_medium(self):
        assert _score_to_band(0.65) == "medium"

    def test_low(self):
        assert _score_to_band(0.45) == "low"

    def test_very_low(self):
        assert _score_to_band(0.25) == "very_low"

    def test_below_very_low(self):
        assert _score_to_band(0.10) == "very_low"

    def test_boundary_very_high(self):
        assert _score_to_band(0.92) == "very_high"

    def test_boundary_high(self):
        assert _score_to_band(0.80) == "high"

    def test_boundary_medium(self):
        assert _score_to_band(0.60) == "medium"


# ===========================================================================
# Tests: Signal extraction
# ===========================================================================

class TestExtractSignal:

    def test_basic_extraction(self, tmp_path):
        sp = tmp_path / "sentinel_x.json"
        sc = tmp_path / "scores.json"
        _write_json(sp, _sentinel_json())
        _write_json(sc, _scores_json())

        sig = extract_signal("BTCUSDT", sentinel_path=sp, scores_path=sc)
        assert sig is not None
        assert sig.direction == "UP"
        assert sig.regime == "TREND_UP"
        assert sig.regime_confidence == 0.88
        assert sig.conviction_band == "high"

    def test_trend_down(self, tmp_path):
        sp = tmp_path / "sentinel_x.json"
        sc = tmp_path / "scores.json"
        _write_json(sp, _sentinel_json(regime="TREND_DOWN", confidence=0.75))
        _write_json(sc, _scores_json(hybrid=0.30))

        sig = extract_signal("BTCUSDT", sentinel_path=sp, scores_path=sc)
        assert sig is not None
        assert sig.direction == "DOWN"
        assert sig.regime == "TREND_DOWN"

    def test_missing_sentinel(self, tmp_path):
        sig = extract_signal("BTCUSDT", sentinel_path=tmp_path / "nope.json")
        assert sig is None

    def test_choppy_yields_no_direction(self, tmp_path):
        sp = tmp_path / "sentinel_x.json"
        _write_json(sp, _sentinel_json(regime="CHOPPY", confidence=0.70))

        sig = extract_signal("BTCUSDT", sentinel_path=sp, scores_path=tmp_path / "nope.json")
        assert sig is not None
        assert sig.direction is None


# ===========================================================================
# Tests: Eligibility gate
# ===========================================================================

class TestEligibilityGate:

    def test_all_conditions_met(self):
        result = check_eligibility(
            _signal(), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is True
        assert result.deny_reason is None

    def test_regime_blocked(self):
        result = check_eligibility(
            _signal(regime="CHOPPY"), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is False
        assert "regime_blocked" in result.deny_reason

    def test_confidence_below_min(self):
        result = check_eligibility(
            _signal(regime_confidence=0.50), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is False
        assert "confidence_below_min" in result.deny_reason

    def test_conviction_band_below_min(self):
        result = check_eligibility(
            _signal(conviction_band="low"), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is False
        assert "conviction_band_below_min" in result.deny_reason

    def test_concurrent_cap_hit(self):
        result = check_eligibility(
            _signal(), _limits(),
            current_nav_usd=2000, open_positions=3, freeze_intact=True,
        )
        assert result.eligible is False
        assert "concurrent_cap" in result.deny_reason

    def test_no_direction_signal(self):
        result = check_eligibility(
            _signal(direction=None), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is False
        assert result.deny_reason == "no_direction_signal"

    def test_freeze_broken(self):
        result = check_eligibility(
            _signal(), _limits(),
            current_nav_usd=2000, open_positions=0, freeze_intact=False,
        )
        assert result.eligible is False
        assert result.deny_reason == "freeze_broken"

    def test_kill_line_breached(self):
        result = check_eligibility(
            _signal(), _limits(),
            current_nav_usd=1700, open_positions=0, freeze_intact=True,
        )
        assert result.eligible is False
        assert "kill_line" in result.deny_reason

    def test_band_rank_ordering(self):
        assert _BAND_RANK["very_high"] > _BAND_RANK["high"] > _BAND_RANK["medium"]
        assert _BAND_RANK["medium"] > _BAND_RANK["low"] > _BAND_RANK["very_low"]


# ===========================================================================
# Tests: Round boundary helpers
# ===========================================================================

class TestRoundBoundary:

    def test_round_start_alignment(self):
        # 12:03:45 should snap to 12:00:00
        ts = datetime(2026, 2, 25, 12, 3, 45, tzinfo=timezone.utc).timestamp()
        start = _round_start_unix(ts)
        expected = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        assert start == expected

    def test_round_start_exact_boundary(self):
        ts = datetime(2026, 2, 25, 12, 15, 0, tzinfo=timezone.utc).timestamp()
        start = _round_start_unix(ts)
        assert start == ts

    def test_round_id_format(self):
        ts = datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc).timestamp()
        assert _make_round_id(ts) == "R_20260225_1430"

    def test_round_id_deterministic(self):
        ts = datetime(2026, 2, 25, 9, 45, 0, tzinfo=timezone.utc).timestamp()
        assert _make_round_id(ts) == _make_round_id(ts)


# ===========================================================================
# Tests: Simulated fill model
# ===========================================================================

class TestSimulatedFill:

    def test_up_entry_fills_at_ask_plus_slippage(self):
        adapter = SimulatedExecutionAdapter(slippage_buffer_bps=5.0)
        snap = {"best_bid": 90000.0, "best_ask": 90010.0, "mid": 90005.0}
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=snap):
            fill = adapter.simulate_fill("BTCUSDT", "UP", is_entry=True)
        assert fill is not None
        expected = 90010.0 * (1 + 5.0 / 10_000)
        assert abs(fill.fill_price - expected) < 0.01

    def test_up_exit_fills_at_bid_minus_slippage(self):
        adapter = SimulatedExecutionAdapter(slippage_buffer_bps=5.0)
        snap = {"best_bid": 91000.0, "best_ask": 91010.0, "mid": 91005.0}
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=snap):
            fill = adapter.simulate_fill("BTCUSDT", "UP", is_entry=False)
        assert fill is not None
        expected = 91000.0 * (1 - 5.0 / 10_000)
        assert abs(fill.fill_price - expected) < 0.01

    def test_down_entry_fills_at_bid_minus_slippage(self):
        adapter = SimulatedExecutionAdapter(slippage_buffer_bps=5.0)
        snap = {"best_bid": 90000.0, "best_ask": 90010.0, "mid": 90005.0}
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=snap):
            fill = adapter.simulate_fill("BTCUSDT", "DOWN", is_entry=True)
        assert fill is not None
        expected = 90000.0 * (1 - 5.0 / 10_000)
        assert abs(fill.fill_price - expected) < 0.01

    def test_down_exit_fills_at_ask_plus_slippage(self):
        adapter = SimulatedExecutionAdapter(slippage_buffer_bps=5.0)
        snap = {"best_bid": 89000.0, "best_ask": 89010.0, "mid": 89005.0}
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=snap):
            fill = adapter.simulate_fill("BTCUSDT", "DOWN", is_entry=False)
        assert fill is not None
        expected = 89010.0 * (1 + 5.0 / 10_000)
        assert abs(fill.fill_price - expected) < 0.01

    def test_missing_book_returns_none(self):
        adapter = SimulatedExecutionAdapter()
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=None):
            fill = adapter.simulate_fill("BTCUSDT", "UP", is_entry=True)
        assert fill is None

    def test_fill_records_snapshot(self):
        adapter = SimulatedExecutionAdapter(slippage_buffer_bps=5.0)
        snap = {"best_bid": 90000.0, "best_ask": 90010.0, "mid": 90005.0}
        with mock.patch.object(adapter, "_fetch_book_snapshot", return_value=snap):
            fill = adapter.simulate_fill("BTCUSDT", "UP", is_entry=True)
        assert fill.best_bid == 90000.0
        assert fill.best_ask == 90010.0
        assert fill.mid_price == 90005.0
        assert fill.slippage_bps == 5.0


# ===========================================================================
# Tests: Trade writer (append-only)
# ===========================================================================

class TestTradeWriter:

    def test_append_only(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        w = BinaryLabTradeWriter(log)
        w.write({"round_id": "R1", "event_type": "ENTRY"})
        w.write({"round_id": "R2", "event_type": "ROUND_CLOSED"})

        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["round_id"] == "R1"
        assert json.loads(lines[1])["round_id"] == "R2"

    def test_write_failure_does_not_raise(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        w = BinaryLabTradeWriter(log)
        # Force a write failure by making the file unwritable
        with mock.patch.object(w, "_path") as mocked_path:
            mocked_path.open.side_effect = PermissionError("denied")
            w.write({"test": True})
        assert w.write_failure_count == 1


# ===========================================================================
# Tests: Shadow runner lifecycle
# ===========================================================================

class TestShadowRunner:

    def _make_runner(self, tmp_path, limits=None, **kwargs):
        """Create a runner with mocked adapter (no exchange calls)."""
        lim = limits or _limits()
        trade_log = tmp_path / "trades.jsonl"
        sentinel_path = tmp_path / "sentinel_x.json"
        scores_path = tmp_path / "scores.json"

        _write_json(sentinel_path, _sentinel_json())
        _write_json(scores_path, _scores_json())

        runner = BinaryLabShadowRunner(
            limits=lim,
            trade_log_path=trade_log,
            sentinel_path=sentinel_path,
            scores_path=scores_path,
            **kwargs,
        )
        return runner, trade_log, sentinel_path, scores_path

    def _mock_adapter_fill(self, runner, entry_price=90000.0, exit_price=90100.0):
        """Patch the adapter to return deterministic fills."""
        entry_fill = SimulatedFill(
            fill_price=entry_price, mid_price=entry_price,
            best_bid=entry_price - 5, best_ask=entry_price + 5,
            slippage_bps=5.0, ts="2026-02-25T12:00:30+00:00",
        )
        exit_fill = SimulatedFill(
            fill_price=exit_price, mid_price=exit_price,
            best_bid=exit_price - 5, best_ask=exit_price + 5,
            slippage_bps=5.0, ts="2026-02-25T12:15:00+00:00",
        )

        def _sim(symbol, direction, *, is_entry=True):
            return entry_fill if is_entry else exit_fill

        runner._adapter.simulate_fill = _sim

    def test_entry_at_round_boundary(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        # Hit entry window: round_start + offset_s
        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        entry_ts = round_start + timedelta(seconds=ENTRY_OFFSET_S + 5)
        ts_iso = entry_ts.isoformat()

        changed = runner.tick(ts_iso)
        assert changed is True
        assert runner.open_round_count == 1

        # Check trade log
        lines = trade_log.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "ENTRY"
        assert entry["execution_mode"] == "SHADOW"
        assert entry["round_id"] == "R_20260225_1200"
        assert entry["side"] == "UP"

    def test_no_duplicate_entry(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        entry_ts = round_start + timedelta(seconds=ENTRY_OFFSET_S + 5)
        ts_iso = entry_ts.isoformat()

        runner.tick(ts_iso)
        assert runner.open_round_count == 1

        # Second tick in same window should not duplicate
        runner.tick(ts_iso)
        assert runner.open_round_count == 1

    def test_outside_entry_window_no_entry(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        # Outside entry window (too late into round)
        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        late_ts = round_start + timedelta(seconds=ENTRY_OFFSET_S + ENTRY_WINDOW_S + 60)
        ts_iso = late_ts.isoformat()

        changed = runner.tick(ts_iso)
        assert changed is False
        assert runner.open_round_count == 0

    def test_resolution_after_15min(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner, entry_price=90000.0, exit_price=90100.0)

        # Enter at round start + offset
        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        entry_ts = round_start + timedelta(seconds=ENTRY_OFFSET_S + 5)
        runner.tick(entry_ts.isoformat())
        assert runner.open_round_count == 1

        # Resolve at round end
        resolve_ts = round_start + timedelta(seconds=ROUND_DURATION_S + 1)
        changed = runner.tick(resolve_ts.isoformat())
        assert changed is True
        assert runner.open_round_count == 0

        # Check trade log has ENTRY + ROUND_CLOSED
        lines = trade_log.read_text().strip().split("\n")
        assert len(lines) == 2
        resolved = json.loads(lines[1])
        assert resolved["event_type"] == "ROUND_CLOSED"
        assert resolved["execution_mode"] == "SHADOW"
        assert resolved["resolved_outcome"] in ("WIN", "LOSS")
        assert "pnl_usd" in resolved

    def test_pnl_computation_up_win(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner, entry_price=90000.0, exit_price=90200.0)

        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        runner.tick((round_start + timedelta(seconds=35)).isoformat())
        runner.tick((round_start + timedelta(seconds=901)).isoformat())

        lines = trade_log.read_text().strip().split("\n")
        resolved = json.loads(lines[1])
        # Gross PnL: (90200 - 90000) * (20 / 90000) = 200 * 0.000222 ≈ 0.04444
        # Fee: 20 * 4/10000 * 2 = 0.016
        # Net ≈ 0.02844
        assert resolved["resolved_outcome"] == "WIN"
        assert resolved["pnl_usd"] > 0

    def test_pnl_computation_down_loss(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        # DOWN direction entry at 90000, exit at 90500 → price went UP → LOSS
        _write_json(tmp_path / "sentinel_x.json", _sentinel_json(regime="TREND_DOWN"))
        self._mock_adapter_fill(runner, entry_price=90000.0, exit_price=90500.0)

        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        runner.tick((round_start + timedelta(seconds=35)).isoformat())
        runner.tick((round_start + timedelta(seconds=901)).isoformat())

        lines = trade_log.read_text().strip().split("\n")
        resolved = json.loads(lines[1])
        # DOWN: PnL = (entry - exit) * qty = (90000 - 90500) * (20/90000) < 0
        assert resolved["resolved_outcome"] == "LOSS"
        assert resolved["pnl_usd"] < 0

    def test_ineligible_logs_no_trade(self, tmp_path):
        runner, trade_log, sentinel, _ = self._make_runner(tmp_path)
        # Make regime CHOPPY (blocked)
        _write_json(sentinel, _sentinel_json(regime="CHOPPY"))
        self._mock_adapter_fill(runner)

        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        runner.tick((round_start + timedelta(seconds=35)).isoformat())

        lines = trade_log.read_text().strip().split("\n")
        assert len(lines) == 1
        no_trade = json.loads(lines[0])
        assert no_trade["event_type"] == "NO_TRADE"
        assert no_trade["execution_mode"] == "SHADOW"
        assert "regime_blocked" in no_trade["deny_reason"]
        assert runner.open_round_count == 0

    def test_concurrent_cap_enforced(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        # Pre-populate open rounds to hit the cap (max_concurrent=3)
        for i in range(3):
            fake_fill = SimulatedFill(
                fill_price=90000.0, mid_price=90000.0,
                best_bid=89995.0, best_ask=90005.0,
                slippage_bps=5.0, ts="2026-02-25T11:00:30+00:00",
            )
            runner._open_rounds.append(OpenRound(
                round_id=f"R_PREFILL_{i}",
                symbol="BTCUSDT",
                direction="UP",
                entry_fill=fake_fill,
                notional_usd=20.0,
                entry_ts="2026-02-25T11:00:30+00:00",
                entry_ts_unix=datetime(2026, 2, 25, 11, 0, 30, tzinfo=timezone.utc).timestamp(),
                # Set resolution far in the future so they stay open
                resolution_ts_unix=datetime(2026, 2, 25, 23, 0, 0, tzinfo=timezone.utc).timestamp(),
                conviction_band="high",
                conviction_score=0.88,
                regime="TREND_UP",
                regime_confidence=0.88,
            ))

        assert runner.open_round_count == 3

        # New entry attempt should be denied
        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        runner.tick((round_start + timedelta(seconds=35)).isoformat())
        assert runner.open_round_count == 3  # still 3

        lines = trade_log.read_text().strip().split("\n")
        last = json.loads(lines[-1])
        assert last["event_type"] == "NO_TRADE"
        assert "concurrent_cap" in last["deny_reason"]

    def test_execution_mode_tag_always_shadow(self, tmp_path):
        runner, trade_log, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        round_start = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
        runner.tick((round_start + timedelta(seconds=35)).isoformat())
        runner.tick((round_start + timedelta(seconds=901)).isoformat())

        lines = trade_log.read_text().strip().split("\n")
        for line in lines:
            record = json.loads(line)
            assert record["execution_mode"] == "SHADOW"

    def test_round_id_deterministic(self, tmp_path):
        runner, _, _, _ = self._make_runner(tmp_path)
        self._mock_adapter_fill(runner)

        round_start = datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc)
        ts = (round_start + timedelta(seconds=35)).isoformat()
        runner.tick(ts)

        assert "R_20260225_1430" in runner._processed_round_ids


# ===========================================================================
# Tests: SHADOW mode enum
# ===========================================================================

class TestBinaryLabModeEnum:

    def test_shadow_mode_exists(self):
        assert BinaryLabMode.SHADOW.value == "SHADOW"

    def test_paper_mode_exists(self):
        assert BinaryLabMode.PAPER.value == "PAPER"

    def test_live_mode_exists(self):
        assert BinaryLabMode.LIVE.value == "LIVE"

    def test_shadow_from_string(self):
        assert BinaryLabMode("SHADOW") == BinaryLabMode.SHADOW


# ===========================================================================
# Tests: Frozen constants
# ===========================================================================

class TestFrozenConstants:

    def test_round_duration(self):
        assert ROUND_DURATION_S == 900

    def test_entry_offset(self):
        assert ENTRY_OFFSET_S == 30

    def test_slippage_buffer(self):
        assert SLIPPAGE_BUFFER_BPS == 5.0

    def test_fee_bps(self):
        assert FEE_BPS == 4.0
