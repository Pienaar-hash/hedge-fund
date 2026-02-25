"""Tests for prediction.round_observer — Layer 3 Round Observer.

Covers:
    - TerminalSnapshot dataclass and serialization
    - RoundState dataclass: direction_from_oracle, compute_bundle_cost,
      compute_net_ev, to_record, spread stats
    - read_oracle_tick_near (boundary search, direction, stale cutoff)
    - read_latest_oracle_tick
    - read_clob_state_for_asset (latest bid/ask extraction)
    - RoundObserver init, _capture_book_state, _capture_terminal_snapshot,
      _capture_start_state, _capture_end_state, _collect_spreads,
      _check_market_resolution, _finalize_round
    - Edge cases: missing data, FLAT direction, partial snapshots,
      empty logs, stale oracle
    - _append_jsonl invariant
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from prediction.round_observer import (
    INTRAROUND_INTERVAL_S,
    ORACLE_STALE_MS,
    POLYMARKET_FEE_RATE,
    SNAPSHOT_OFFSETS_S,
    RoundObserver,
    RoundState,
    TerminalSnapshot,
    _append_jsonl,
    _ensure_parent,
    _now_iso,
    _now_ms,
    _tail_lines,
    polymarket_taker_fee,
    read_clob_ask_depth,
    read_clob_state_for_asset,
    read_latest_oracle_tick,
    read_oracle_tick_near,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_round_state(
    oracle_start: Optional[float] = 98000.0,
    oracle_end: Optional[float] = 98100.0,
    **kwargs: Any,
) -> RoundState:
    """Build a minimal RoundState for testing."""
    defaults = dict(
        slug="btc-updown-15m-1771833600",
        question="BTC Up or Down?",
        timeframe="15m",
        up_token="UP_TOKEN",
        down_token="DOWN_TOKEN",
        round_start_iso="2026-02-23T10:00:00+00:00",
        round_end_iso="2026-02-23T10:15:00Z",
        round_start_ts=1771833600.0,
        round_end_ts=1771834500.0,
        condition_id="0xabc123",
    )
    defaults.update(kwargs)
    rnd = RoundState(**defaults)
    rnd.oracle_start = oracle_start
    rnd.oracle_end = oracle_end
    return rnd


def _inject_snapshot(
    rnd: RoundState,
    label: str = "t_minus_3",
    ask_up: float = 0.55,
    ask_down: float = 0.48,
    bid_up: float = 0.53,
    bid_down: float = 0.46,
) -> None:
    """Inject a terminal snapshot into a round."""
    snap = TerminalSnapshot(
        offset_label=label,
        captured_at="2026-02-23T10:14:57+00:00",
        captured_at_ms=1771834497000,
        best_bid_up=bid_up,
        best_ask_up=ask_up,
        best_bid_down=bid_down,
        best_ask_down=ask_down,
        spread_up=round(ask_up - bid_up, 6),
        spread_down=round(ask_down - bid_down, 6),
        mid_up=round((bid_up + ask_up) / 2, 6),
        mid_down=round((bid_down + ask_down) / 2, 6),
        implied_prob_up=round((bid_up + ask_up) / 2, 6),
        oracle_price=98050.0,
        oracle_ts_ms=1771834497000,
    )
    rnd.snapshots[label] = snap


# ---------------------------------------------------------------------------
# Tests — TerminalSnapshot
# ---------------------------------------------------------------------------
class TestTerminalSnapshot:
    def test_to_dict_full(self) -> None:
        snap = TerminalSnapshot(
            offset_label="t_minus_10",
            captured_at="2026-02-23T10:14:50+00:00",
            captured_at_ms=1771834490000,
            best_bid_up=0.53,
            best_ask_up=0.55,
            best_bid_down=0.46,
            best_ask_down=0.48,
            spread_up=0.02,
            spread_down=0.02,
            mid_up=0.54,
            mid_down=0.47,
            implied_prob_up=0.54,
            oracle_price=98050.0,
            oracle_ts_ms=1771834490000,
        )
        d = snap.to_dict()
        assert d["best_bid_up"] == 0.53
        assert d["best_ask_up"] == 0.55
        assert d["spread_up"] == 0.02
        assert d["mid_up"] == 0.54
        assert d["implied_prob_up"] == 0.54
        assert d["oracle_price"] == 98050.0
        assert d["captured_at_ms"] == 1771834490000

    def test_to_dict_empty(self) -> None:
        snap = TerminalSnapshot(offset_label="t_minus_30")
        d = snap.to_dict()
        assert d["best_bid_up"] is None
        assert d["best_ask_up"] is None
        assert d["oracle_price"] is None
        assert d["captured_at"] == ""


# ---------------------------------------------------------------------------
# Tests — RoundState.direction_from_oracle
# ---------------------------------------------------------------------------
class TestDirectionFromOracle:
    def test_up(self) -> None:
        rnd = _make_round_state(oracle_start=98000.0, oracle_end=98100.0)
        assert rnd.direction_from_oracle() == "UP"

    def test_down(self) -> None:
        rnd = _make_round_state(oracle_start=98100.0, oracle_end=98000.0)
        assert rnd.direction_from_oracle() == "DOWN"

    def test_flat(self) -> None:
        rnd = _make_round_state(oracle_start=98000.0, oracle_end=98000.0)
        assert rnd.direction_from_oracle() == "FLAT"

    def test_missing_start(self) -> None:
        rnd = _make_round_state(oracle_start=None, oracle_end=98000.0)
        assert rnd.direction_from_oracle() is None

    def test_missing_end(self) -> None:
        rnd = _make_round_state(oracle_start=98000.0, oracle_end=None)
        assert rnd.direction_from_oracle() is None

    def test_both_missing(self) -> None:
        rnd = _make_round_state(oracle_start=None, oracle_end=None)
        assert rnd.direction_from_oracle() is None


# ---------------------------------------------------------------------------
# Tests — RoundState.compute_bundle_cost
# ---------------------------------------------------------------------------
class TestComputeBundleCost:
    def test_with_t3_snapshot(self) -> None:
        rnd = _make_round_state()
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)
        result = rnd.compute_bundle_cost()
        assert result == pytest.approx(1.03, abs=1e-6)

    def test_falls_back_to_t10(self) -> None:
        rnd = _make_round_state()
        _inject_snapshot(rnd, "t_minus_10", ask_up=0.54, ask_down=0.49)
        result = rnd.compute_bundle_cost()
        assert result == pytest.approx(1.03, abs=1e-6)

    def test_prefers_t3_over_t10(self) -> None:
        rnd = _make_round_state()
        _inject_snapshot(rnd, "t_minus_10", ask_up=0.60, ask_down=0.60)
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)
        result = rnd.compute_bundle_cost()
        assert result == pytest.approx(1.03, abs=1e-6)  # Uses T-3

    def test_none_when_flat(self) -> None:
        rnd = _make_round_state(oracle_start=100.0, oracle_end=100.0)
        _inject_snapshot(rnd, "t_minus_3")
        assert rnd.compute_bundle_cost() is None

    def test_none_when_no_oracle(self) -> None:
        rnd = _make_round_state(oracle_start=None, oracle_end=None)
        _inject_snapshot(rnd, "t_minus_3")
        assert rnd.compute_bundle_cost() is None

    def test_none_when_no_snapshot(self) -> None:
        rnd = _make_round_state()
        assert rnd.compute_bundle_cost() is None

    def test_none_when_partial_ask(self) -> None:
        """Snapshot exists but one ask is missing."""
        rnd = _make_round_state()
        snap = TerminalSnapshot(offset_label="t_minus_3")
        snap.best_ask_up = 0.55
        snap.best_ask_down = None
        rnd.snapshots["t_minus_3"] = snap
        assert rnd.compute_bundle_cost() is None


# ---------------------------------------------------------------------------
# Tests — RoundState.compute_net_ev
# ---------------------------------------------------------------------------
class TestComputeNetEV:
    def test_up_direction(self) -> None:
        rnd = _make_round_state(oracle_start=98000.0, oracle_end=98100.0)
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)
        net = rnd.compute_net_ev(fee_bps=0)
        # 1.0 - 0.55 - 0 = 0.45
        assert net == pytest.approx(0.45, abs=1e-6)

    def test_down_direction(self) -> None:
        rnd = _make_round_state(oracle_start=98100.0, oracle_end=98000.0)
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)
        net = rnd.compute_net_ev(fee_bps=0)
        # 1.0 - 0.48 - 0 = 0.52
        assert net == pytest.approx(0.52, abs=1e-6)

    def test_with_fee(self) -> None:
        rnd = _make_round_state(oracle_start=98000.0, oracle_end=98100.0)
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)
        net = rnd.compute_net_ev(fee_bps=200)  # 2%
        # 1.0 - 0.55 - (0.55 * 200/10000) = 1.0 - 0.55 - 0.011 = 0.439
        assert net == pytest.approx(0.439, abs=1e-6)

    def test_none_when_flat(self) -> None:
        rnd = _make_round_state(oracle_start=100.0, oracle_end=100.0)
        _inject_snapshot(rnd, "t_minus_3")
        assert rnd.compute_net_ev() is None

    def test_none_when_no_snapshot(self) -> None:
        rnd = _make_round_state()
        assert rnd.compute_net_ev() is None

    def test_none_when_cost_missing(self) -> None:
        rnd = _make_round_state()
        snap = TerminalSnapshot(offset_label="t_minus_3")
        snap.best_ask_up = None  # Missing
        rnd.snapshots["t_minus_3"] = snap
        assert rnd.compute_net_ev() is None


# ---------------------------------------------------------------------------
# Tests — RoundState.to_record
# ---------------------------------------------------------------------------
class TestToRecord:
    def test_full_record(self) -> None:
        rnd = _make_round_state()
        rnd.oracle_start_ts_ms = 1771833600000
        rnd.oracle_end_ts_ms = 1771834500500
        rnd.start_best_bid_up = 0.50
        rnd.start_best_ask_up = 0.52
        rnd.start_spread_up = 0.02
        rnd.spreads_up = [0.02, 0.03, 0.01]
        rnd.spreads_down = [0.02, 0.04]
        rnd.event_count = 42
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)

        rec = rnd.to_record()

        assert rec["slug"] == "btc-updown-15m-1771833600"
        assert rec["direction"] == "UP"
        assert rec["oracle_start"] == 98000.0
        assert rec["oracle_end"] == 98100.0
        assert rec["bundle_cost"] == pytest.approx(1.03, abs=1e-6)
        assert rec["net_ev_after_fee"] is not None
        assert rec["event_count"] == 42
        assert rec["start_book"]["best_bid_up"] == 0.50
        assert rec["spread_stats_up"]["min"] == 0.01
        assert rec["spread_stats_up"]["max"] == 0.03
        assert rec["spread_stats_up"]["mean"] == pytest.approx(0.02, abs=1e-6)
        assert rec["spread_stats_up"]["samples"] == 3
        assert rec["spread_stats_down"]["samples"] == 2
        assert "t_minus_3" in rec["snapshots"]
        assert rec["logged_at"]  # non-empty
        assert rec["timeframe"] == "15m"

    def test_oracle_misalignment_computed(self) -> None:
        rnd = _make_round_state()
        rnd.oracle_start_ts_ms = 1771833600200  # 200ms late
        rnd.oracle_end_ts_ms = 1771834500500  # 500ms late

        rec = rnd.to_record()
        assert rec["oracle_start_misalign_ms"] == 200
        assert rec["oracle_end_misalign_ms"] == 500

    def test_no_oracle_misalignment_when_missing(self) -> None:
        rnd = _make_round_state(oracle_start=None, oracle_end=None)
        rec = rnd.to_record()
        assert "oracle_start_misalign_ms" not in rec
        assert "oracle_end_misalign_ms" not in rec

    def test_empty_spreads(self) -> None:
        rnd = _make_round_state()
        rec = rnd.to_record()
        assert rec["spread_stats_up"]["samples"] == 0
        assert rec["spread_stats_up"]["min"] is None
        assert rec["spread_stats_up"]["max"] is None
        assert rec["spread_stats_up"]["mean"] is None

    def test_direction_none_when_no_oracle(self) -> None:
        rnd = _make_round_state(oracle_start=None, oracle_end=None)
        rec = rnd.to_record()
        assert rec["direction"] is None
        assert rec["bundle_cost"] is None

    def test_tick_size_changes_logged(self) -> None:
        rnd = _make_round_state()
        rnd.tick_size_changes = [
            {"ts_arrival_ms": 1234, "old_tick_size": 0.01, "new_tick_size": 0.001}
        ]
        rec = rnd.to_record()
        assert len(rec["tick_size_changes"]) == 1


# ---------------------------------------------------------------------------
# Tests — read_oracle_tick_near
# ---------------------------------------------------------------------------
class TestReadOracleTickNear:
    def test_finds_tick_before_target(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97900.0, "oracle_ts_ms": 1771833590000},
            {"price": 98000.0, "oracle_ts_ms": 1771833598000},
            {"price": 98001.0, "oracle_ts_ms": 1771833602000},
        ]
        _write_jsonl(log, ticks)

        # Target is at 1771833600000 (second 0), direction "before"
        result = read_oracle_tick_near(1771833600.0, log, max_age_ms=10000)
        assert result is not None
        assert result["price"] == 98000.0
        assert result["oracle_ts_ms"] == 1771833598000

    def test_skips_future_ticks_in_before_mode(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97900.0, "oracle_ts_ms": 1771833590000},
            {"price": 98050.0, "oracle_ts_ms": 1771833610000},  # After target
        ]
        _write_jsonl(log, ticks)

        result = read_oracle_tick_near(1771833600.0, log, max_age_ms=15000)
        assert result is not None
        assert result["price"] == 97900.0

    def test_nearest_mode_picks_closest(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97900.0, "oracle_ts_ms": 1771833550000},  # 50s before
            {"price": 98010.0, "oracle_ts_ms": 1771833602000},  # 2s after
        ]
        _write_jsonl(log, ticks)

        result = read_oracle_tick_near(
            1771833600.0, log, max_age_ms=60000, direction="nearest",
        )
        assert result is not None
        assert result["price"] == 98010.0  # Closer

    def test_returns_none_when_stale(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97800.0, "oracle_ts_ms": 1771833500000},  # 100s before
        ]
        _write_jsonl(log, ticks)

        result = read_oracle_tick_near(
            1771833600.0, log, max_age_ms=10000,
        )
        assert result is None  # 100s > 10s stale threshold

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        log = tmp_path / "nonexistent.jsonl"
        assert read_oracle_tick_near(1771833600.0, log) is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        log.write_text("")
        assert read_oracle_tick_near(1771833600.0, log) is None

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        with open(log, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"price": 98000.0, "oracle_ts_ms": 1771833599000}) + "\n")
            f.write("{broken\n")
        result = read_oracle_tick_near(1771833600.0, log, max_age_ms=10000)
        assert result is not None
        assert result["price"] == 98000.0

    def test_skips_ticks_without_oracle_ts_ms(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97900.0},  # Missing oracle_ts_ms
            {"price": 98000.0, "oracle_ts_ms": 1771833599000},
        ]
        _write_jsonl(log, ticks)

        result = read_oracle_tick_near(1771833600.0, log, max_age_ms=10000)
        assert result is not None
        assert result["price"] == 98000.0


# ---------------------------------------------------------------------------
# Tests — read_latest_oracle_tick
# ---------------------------------------------------------------------------
class TestReadLatestOracleTick:
    def test_returns_last_tick(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        ticks = [
            {"price": 97900.0, "oracle_ts_ms": 1000},
            {"price": 98100.0, "oracle_ts_ms": 2000},
        ]
        _write_jsonl(log, ticks)

        result = read_latest_oracle_tick(log)
        assert result is not None
        assert result["price"] == 98100.0

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert read_latest_oracle_tick(tmp_path / "nope.jsonl") is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        log.write_text("")
        assert read_latest_oracle_tick(log) is None

    def test_skips_trailing_empty_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "oracle.jsonl"
        with open(log, "w") as f:
            f.write(json.dumps({"price": 98000.0}) + "\n")
            f.write("\n\n")
        result = read_latest_oracle_tick(log)
        assert result is not None
        assert result["price"] == 98000.0


# ---------------------------------------------------------------------------
# Tests — read_clob_state_for_asset
# ---------------------------------------------------------------------------
class TestReadClobStateForAsset:
    def test_finds_latest_for_asset(self, tmp_path: Path) -> None:
        log = tmp_path / "clob.jsonl"
        records = [
            {"asset_id": "A", "best_bid": 0.50, "best_ask": 0.52, "ts_arrival_ms": 100},
            {"asset_id": "B", "best_bid": 0.45, "best_ask": 0.48, "ts_arrival_ms": 200},
            {"asset_id": "A", "best_bid": 0.51, "best_ask": 0.53, "ts_arrival_ms": 300},
        ]
        _write_jsonl(log, records)

        result = read_clob_state_for_asset("A", log)
        assert result is not None
        assert result["best_bid"] == 0.51
        assert result["best_ask"] == 0.53
        assert result["spread"] == pytest.approx(0.02, abs=1e-6)
        assert result["mid"] == pytest.approx(0.52, abs=1e-6)

    def test_returns_none_for_unknown_asset(self, tmp_path: Path) -> None:
        log = tmp_path / "clob.jsonl"
        _write_jsonl(log, [{"asset_id": "X", "best_bid": 0.5, "best_ask": 0.6}])
        assert read_clob_state_for_asset("Y", log) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert read_clob_state_for_asset("A", tmp_path / "nope.jsonl") is None

    def test_skips_records_without_bid_ask(self, tmp_path: Path) -> None:
        log = tmp_path / "clob.jsonl"
        records = [
            {"asset_id": "A", "event_type": "tick_size_change"},
            {"asset_id": "A", "best_bid": 0.50, "best_ask": 0.52},
        ]
        _write_jsonl(log, records)

        result = read_clob_state_for_asset("A", log)
        assert result is not None
        assert result["best_bid"] == 0.50


# ---------------------------------------------------------------------------
# Tests — _append_jsonl
# ---------------------------------------------------------------------------
class TestAppendJsonl:
    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "deep" / "rounds.jsonl"
        _append_jsonl(path, {"test": 1})
        assert path.exists()
        records = _read_jsonl(path)
        assert len(records) == 1
        assert records[0]["test"] == 1

    def test_appends_not_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "rounds.jsonl"
        _append_jsonl(path, {"a": 1})
        _append_jsonl(path, {"b": 2})
        records = _read_jsonl(path)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Tests — RoundObserver init
# ---------------------------------------------------------------------------
class TestRoundObserverInit:
    def test_default_init(self) -> None:
        obs = RoundObserver()
        assert obs.timeframe == "15m"
        assert obs.max_rounds == 0  # unlimited
        assert obs._completed_rounds == 0
        assert obs._current_round is None

    def test_custom_paths(self, tmp_path: Path) -> None:
        obs = RoundObserver(
            rounds_log=tmp_path / "rounds.jsonl",
            oracle_log=tmp_path / "oracle.jsonl",
            market_log=tmp_path / "market.jsonl",
        )
        assert obs.rounds_log == tmp_path / "rounds.jsonl"

    def test_custom_timeframe(self) -> None:
        obs = RoundObserver(timeframe="5m")
        assert obs.timeframe == "5m"


# ---------------------------------------------------------------------------
# Tests — RoundObserver._capture_book_state
# ---------------------------------------------------------------------------
class TestCaptureBookState:
    def test_captures_both_tokens(self, tmp_path: Path) -> None:
        log = tmp_path / "clob.jsonl"
        records = [
            {"asset_id": "UP", "best_bid": 0.53, "best_ask": 0.55, "ts_arrival_ms": 100},
            {"asset_id": "DOWN", "best_bid": 0.46, "best_ask": 0.48, "ts_arrival_ms": 100},
        ]
        _write_jsonl(log, records)

        obs = RoundObserver(market_log=log)
        result = obs._capture_book_state("UP", "DOWN")
        assert result["up"]["best_bid"] == 0.53
        assert result["down"]["best_ask"] == 0.48

    def test_handles_missing_token(self, tmp_path: Path) -> None:
        log = tmp_path / "clob.jsonl"
        _write_jsonl(log, [{"asset_id": "UP", "best_bid": 0.5, "best_ask": 0.6}])

        obs = RoundObserver(market_log=log)
        result = obs._capture_book_state("UP", "DOWN")
        assert result["up"] is not None
        assert result["down"] is None


# ---------------------------------------------------------------------------
# Tests — RoundObserver._capture_start_state
# ---------------------------------------------------------------------------
class TestCaptureStartState:
    def test_sets_oracle_and_book(self, tmp_path: Path) -> None:
        oracle_log = tmp_path / "oracle.jsonl"
        market_log = tmp_path / "clob.jsonl"

        _write_jsonl(oracle_log, [
            {"price": 98000.0, "oracle_ts_ms": 1771833599000},
        ])
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52, "ts_arrival_ms": 100},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.47, "best_ask": 0.49, "ts_arrival_ms": 100},
        ])

        obs = RoundObserver(oracle_log=oracle_log, market_log=market_log)
        rnd = _make_round_state(oracle_start=None, oracle_end=None)

        obs._capture_start_state(rnd)

        assert rnd.oracle_start == 98000.0
        assert rnd.oracle_start_ts_ms == 1771833599000
        assert rnd.start_best_bid_up == 0.50
        assert rnd.start_best_ask_down == 0.49

    def test_fallback_to_latest_tick(self, tmp_path: Path) -> None:
        """When no boundary tick found, should use latest available."""
        oracle_log = tmp_path / "oracle.jsonl"
        market_log = tmp_path / "clob.jsonl"

        # Oracle tick far from boundary (> ORACLE_STALE_MS)
        _write_jsonl(oracle_log, [
            {"price": 97500.0, "oracle_ts_ms": 1771833000000},
        ])
        _write_jsonl(market_log, [])

        obs = RoundObserver(oracle_log=oracle_log, market_log=market_log)
        rnd = _make_round_state(oracle_start=None, oracle_end=None)

        obs._capture_start_state(rnd)

        # Falls back to latest tick even if stale
        assert rnd.oracle_start == 97500.0


# ---------------------------------------------------------------------------
# Tests — RoundObserver._capture_end_state
# ---------------------------------------------------------------------------
class TestCaptureEndState:
    def test_sets_oracle_end(self, tmp_path: Path) -> None:
        oracle_log = tmp_path / "oracle.jsonl"
        _write_jsonl(oracle_log, [
            {"price": 97900.0, "oracle_ts_ms": 1771834490000},
            {"price": 98100.0, "oracle_ts_ms": 1771834499000},
        ])

        obs = RoundObserver(oracle_log=oracle_log)
        rnd = _make_round_state(oracle_end=None)

        obs._capture_end_state(rnd)
        assert rnd.oracle_end == 98100.0
        assert rnd.oracle_end_ts_ms == 1771834499000


# ---------------------------------------------------------------------------
# Tests — RoundObserver._collect_spreads
# ---------------------------------------------------------------------------
class TestCollectSpreads:
    def test_collects_spreads_within_window(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        start_ms = 1771833600000
        end_ms = 1771834500000

        records = [
            # Outside window (before start) — must be first so reverse scan
            # encounters it last, not first
            {"asset_id": "UP_TOKEN", "best_bid": 0.48, "best_ask": 0.60,
             "ts_arrival_ms": start_ms - 1000},
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52,
             "ts_arrival_ms": start_ms + 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.46, "best_ask": 0.49,
             "ts_arrival_ms": start_ms + 2000},
            {"asset_id": "UP_TOKEN", "best_bid": 0.51, "best_ask": 0.53,
             "ts_arrival_ms": start_ms + 3000},
        ]
        _write_jsonl(market_log, records)

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()

        obs._collect_spreads(rnd)

        assert len(rnd.spreads_up) == 2
        assert len(rnd.spreads_down) == 1
        assert rnd.event_count == 3  # 3 records in window

    def test_handles_missing_log(self, tmp_path: Path) -> None:
        obs = RoundObserver(market_log=tmp_path / "nope.jsonl")
        rnd = _make_round_state()
        obs._collect_spreads(rnd)  # Should not raise
        assert len(rnd.spreads_up) == 0

    def test_tracks_tick_size_changes(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        start_ms = 1771833600000

        records = [
            {
                "asset_id": "UP_TOKEN",
                "event_type": "tick_size_change",
                "old_tick_size": 0.01,
                "new_tick_size": 0.001,
                "ts_arrival_ms": start_ms + 5000,
            },
        ]
        _write_jsonl(market_log, records)

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        obs._collect_spreads(rnd)

        assert len(rnd.tick_size_changes) == 1
        assert rnd.tick_size_changes[0]["new_tick_size"] == 0.001


# ---------------------------------------------------------------------------
# Tests — RoundObserver._check_market_resolution
# ---------------------------------------------------------------------------
class TestCheckMarketResolution:
    def test_detects_up_resolution(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        end_ms = 1771834500000

        records = [
            {
                "event_type": "market_resolved",
                "winning_asset_id": "UP_TOKEN",
                "ts_arrival_ms": end_ms + 1000,
            },
        ]
        _write_jsonl(market_log, records)

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        obs._check_market_resolution(rnd)

        assert rnd.market_resolved_received is True
        assert rnd.resolved_outcome == "Up"
        assert rnd.resolution_source == "market_resolved"

    def test_detects_down_resolution(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        end_ms = 1771834500000

        records = [
            {
                "event_type": "market_resolved",
                "winning_asset_id": "DOWN_TOKEN",
                "ts_arrival_ms": end_ms + 1000,
            },
        ]
        _write_jsonl(market_log, records)

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        obs._check_market_resolution(rnd)

        assert rnd.resolved_outcome == "Down"

    def test_ignores_old_resolution(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        end_ms = 1771834500000

        records = [
            {
                "event_type": "market_resolved",
                "winning_asset_id": "UP_TOKEN",
                "ts_arrival_ms": end_ms - 120_000,  # Too old (>60s before end)
            },
        ]
        _write_jsonl(market_log, records)

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        obs._check_market_resolution(rnd)

        assert rnd.market_resolved_received is False

    def test_handles_missing_log(self, tmp_path: Path) -> None:
        obs = RoundObserver(market_log=tmp_path / "nope.jsonl")
        rnd = _make_round_state()
        obs._check_market_resolution(rnd)
        assert rnd.market_resolved_received is False


# ---------------------------------------------------------------------------
# Tests — RoundObserver._finalize_round
# ---------------------------------------------------------------------------
class TestFinalizeRound:
    def test_writes_record_to_log(self, tmp_path: Path) -> None:
        rounds_log = tmp_path / "rounds.jsonl"
        oracle_log = tmp_path / "oracle.jsonl"
        market_log = tmp_path / "clob.jsonl"

        # Oracle ticks covering the round
        _write_jsonl(oracle_log, [
            {"price": 98000.0, "oracle_ts_ms": 1771833599000},
            {"price": 98100.0, "oracle_ts_ms": 1771834499000},
        ])
        # Some CLOB events
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52,
             "ts_arrival_ms": 1771833601000},
        ])

        obs = RoundObserver(
            rounds_log=rounds_log,
            oracle_log=oracle_log,
            market_log=market_log,
        )
        rnd = _make_round_state(oracle_start=None, oracle_end=None)
        _inject_snapshot(rnd, "t_minus_3", ask_up=0.55, ask_down=0.48)

        record = obs._finalize_round(rnd)

        assert record["slug"] == "btc-updown-15m-1771833600"
        assert record["oracle_end"] == 98100.0
        assert rnd.resolved is True

        # Check JSONL was written
        records = _read_jsonl(rounds_log)
        assert len(records) == 1
        assert records[0]["slug"] == "btc-updown-15m-1771833600"

    def test_increments_completed_count(self, tmp_path: Path) -> None:
        obs = RoundObserver(
            rounds_log=tmp_path / "rounds.jsonl",
            oracle_log=tmp_path / "oracle.jsonl",
            market_log=tmp_path / "clob.jsonl",
        )
        _write_jsonl(tmp_path / "oracle.jsonl", [])
        _write_jsonl(tmp_path / "clob.jsonl", [])

        assert obs._completed_rounds == 0

        rnd = _make_round_state()
        obs._finalize_round(rnd)
        assert obs._completed_rounds == 1

        rnd2 = _make_round_state(slug="btc-updown-15m-other")
        obs._finalize_round(rnd2)
        assert obs._completed_rounds == 2

    def test_oracle_direction_fallback(self, tmp_path: Path) -> None:
        """When no market_resolved, use oracle direction."""
        obs = RoundObserver(
            rounds_log=tmp_path / "rounds.jsonl",
            oracle_log=tmp_path / "oracle.jsonl",
            market_log=tmp_path / "clob.jsonl",
        )
        _write_jsonl(tmp_path / "oracle.jsonl", [
            {"price": 98100.0, "oracle_ts_ms": 1771834499000},
        ])
        _write_jsonl(tmp_path / "clob.jsonl", [])

        rnd = _make_round_state(oracle_start=98000.0, oracle_end=None)
        obs._finalize_round(rnd)

        assert rnd.resolved_outcome == "Up"
        assert rnd.resolution_source == "oracle_direction"


# ---------------------------------------------------------------------------
# Tests — _ensure_parent
# ---------------------------------------------------------------------------
class TestEnsureParent:
    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "file.txt"
        _ensure_parent(path)
        assert path.parent.exists()


# ---------------------------------------------------------------------------
# Tests — helper functions
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_now_ms_is_int(self) -> None:
        ms = _now_ms()
        assert isinstance(ms, int)
        assert ms > 0

    def test_now_iso_is_string(self) -> None:
        iso = _now_iso()
        assert isinstance(iso, str)
        assert "T" in iso


# ---------------------------------------------------------------------------
# Tests — _capture_terminal_snapshot integration
# ---------------------------------------------------------------------------
class TestCaptureTerminalSnapshot:
    def test_populates_from_logs(self, tmp_path: Path) -> None:
        oracle_log = tmp_path / "oracle.jsonl"
        market_log = tmp_path / "clob.jsonl"

        _write_jsonl(oracle_log, [
            {"price": 98050.0, "oracle_ts_ms": int(time.time() * 1000)},
        ])
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.53, "best_ask": 0.55, "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.46, "best_ask": 0.48, "ts_arrival_ms": 1000},
        ])

        obs = RoundObserver(oracle_log=oracle_log, market_log=market_log)
        rnd = _make_round_state()

        snap = obs._capture_terminal_snapshot("t_minus_10", rnd)

        assert snap.offset_label == "t_minus_10"
        assert snap.best_bid_up == 0.53
        assert snap.best_ask_up == 0.55
        assert snap.best_bid_down == 0.46
        assert snap.best_ask_down == 0.48
        assert snap.spread_up == pytest.approx(0.02, abs=1e-6)
        assert snap.mid_up == pytest.approx(0.54, abs=1e-6)
        assert snap.implied_prob_up == pytest.approx(0.54, abs=1e-6)
        assert snap.oracle_price == 98050.0
        assert snap.captured_at != ""
        assert snap.captured_at_ms > 0

    def test_handles_empty_logs(self, tmp_path: Path) -> None:
        obs = RoundObserver(
            oracle_log=tmp_path / "oracle.jsonl",
            market_log=tmp_path / "clob.jsonl",
        )
        rnd = _make_round_state()

        snap = obs._capture_terminal_snapshot("t_minus_3", rnd)

        assert snap.best_bid_up is None
        assert snap.oracle_price is None


# ---------------------------------------------------------------------------
# Tests — Intraround Sampling
# ---------------------------------------------------------------------------
class TestComputeIntraaroundStats:
    def test_empty_samples(self) -> None:
        rnd = _make_round_state()
        stats = rnd._compute_intraround_stats()
        assert stats["sample_count"] == 0

    def test_no_dislocations(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "ts": "t1"},
            {"bundle_cost": 1.005, "ts": "t2"},
            {"bundle_cost": 1.001, "ts": "t3"},
        ]
        stats = rnd._compute_intraround_stats()
        assert stats["sample_count"] == 3
        assert stats["dislocation_count"] == 0
        assert stats["min_bundle_cost"] == pytest.approx(1.001, abs=1e-6)
        assert stats["max_bundle_cost"] == pytest.approx(1.01, abs=1e-6)
        assert stats["mean_bundle_cost"] == pytest.approx(1.005333, abs=1e-3)
        assert stats["min_dislocation"] is None

    def test_with_dislocations(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "ts": "t1"},
            {"bundle_cost": 0.995, "ts": "t2"},
            {"bundle_cost": 0.992, "ts": "t3"},
            {"bundle_cost": 1.005, "ts": "t4"},
        ]
        stats = rnd._compute_intraround_stats()
        assert stats["sample_count"] == 4
        assert stats["dislocation_count"] == 2
        assert stats["min_dislocation"] == pytest.approx(0.992, abs=1e-6)
        assert stats["min_bundle_cost"] == pytest.approx(0.992, abs=1e-6)

    def test_none_bundle_costs_skipped(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": None, "ts": "t1"},
            {"bundle_cost": 1.01, "ts": "t2"},
        ]
        stats = rnd._compute_intraround_stats()
        assert stats["sample_count"] == 1
        assert stats["dislocation_count"] == 0

    def test_all_none_bundle_costs(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": None, "ts": "t1"},
            {"bundle_cost": None, "ts": "t2"},
        ]
        stats = rnd._compute_intraround_stats()
        assert stats["sample_count"] == 2
        assert "dislocation_count" not in stats


class TestToRecordIntraaroundFields:
    def test_intraround_fields_present(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "ts": "t1", "ask_up": 0.51, "ask_down": 0.50},
        ]
        _inject_snapshot(rnd, "t_minus_3")
        rec = rnd.to_record()

        assert "intraround_samples" in rec
        assert len(rec["intraround_samples"]) == 1
        assert "intraround_stats" in rec
        assert rec["intraround_stats"]["sample_count"] == 1
        assert rec["intraround_stats"]["dislocation_count"] == 0

    def test_intraround_empty_when_no_samples(self) -> None:
        rnd = _make_round_state()
        _inject_snapshot(rnd, "t_minus_3")
        rec = rnd.to_record()

        assert rec["intraround_samples"] == []
        assert rec["intraround_stats"]["sample_count"] == 0


class TestCaptureIntraaroundSample:
    def test_captures_from_logs(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52, "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.47, "best_ask": 0.49, "ts_arrival_ms": 1000},
        ])

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 120  # started 2 min ago

        sample = obs._capture_intraround_sample(rnd)

        assert sample["ask_up"] == 0.52
        assert sample["ask_down"] == 0.49
        assert sample["bundle_cost"] == pytest.approx(1.01, abs=1e-6)
        assert sample["spread_up"] == pytest.approx(0.02, abs=1e-6)
        assert sample["spread_down"] == pytest.approx(0.02, abs=1e-6)
        assert sample["mid_up"] == pytest.approx(0.51, abs=1e-6)
        assert sample["mid_down"] == pytest.approx(0.48, abs=1e-6)
        assert sample["elapsed_s"] > 0
        assert sample["ts"] != ""

    def test_handles_missing_data(self, tmp_path: Path) -> None:
        obs = RoundObserver(market_log=tmp_path / "clob.jsonl")
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 60

        sample = obs._capture_intraround_sample(rnd)

        assert sample["ask_up"] is None
        assert sample["ask_down"] is None
        assert sample["bundle_cost"] is None

    def test_bundle_cost_sub_one(self, tmp_path: Path) -> None:
        """Verify bundle_cost correctly reflects ask sums below 1.0."""
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.48, "best_ask": 0.49, "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.49, "best_ask": 0.50, "ts_arrival_ms": 1000},
        ])

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 60

        sample = obs._capture_intraround_sample(rnd)

        assert sample["bundle_cost"] == pytest.approx(0.99, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests — IDCP: polymarket_taker_fee
# ---------------------------------------------------------------------------
class TestPolymarketTakerFee:
    def test_midpoint_price(self) -> None:
        # At 0.50: fee = 0.02 * min(0.50, 0.50) = 0.01
        assert polymarket_taker_fee(0.50) == pytest.approx(0.01, abs=1e-6)

    def test_high_price(self) -> None:
        # At 0.90: fee = 0.02 * min(0.90, 0.10) = 0.002
        assert polymarket_taker_fee(0.90) == pytest.approx(0.002, abs=1e-6)

    def test_low_price(self) -> None:
        # At 0.10: fee = 0.02 * min(0.10, 0.90) = 0.002
        assert polymarket_taker_fee(0.10) == pytest.approx(0.002, abs=1e-6)

    def test_zero_price(self) -> None:
        assert polymarket_taker_fee(0.0) == 0.0

    def test_one_price(self) -> None:
        assert polymarket_taker_fee(1.0) == 0.0

    def test_negative_price(self) -> None:
        assert polymarket_taker_fee(-0.5) == 0.0

    def test_symmetric(self) -> None:
        # fee(p) == fee(1-p)
        assert polymarket_taker_fee(0.3) == pytest.approx(
            polymarket_taker_fee(0.7), abs=1e-10
        )


# ---------------------------------------------------------------------------
# Tests — IDCP: read_clob_ask_depth
# ---------------------------------------------------------------------------
class TestReadClobAskDepth:
    def test_returns_depth_from_price_change(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {
                "asset_id": "UP_TOKEN",
                "event_type": "price_change",
                "side": "SELL",
                "price": 0.52,
                "best_ask": 0.52,
                "best_bid": 0.50,
                "size": "150.0",
                "ts_arrival_ms": 1000,
            },
        ])
        depth = read_clob_ask_depth("UP_TOKEN", market_log)
        assert depth == pytest.approx(150.0, abs=1e-6)

    def test_ignores_buy_side(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {
                "asset_id": "UP_TOKEN",
                "event_type": "price_change",
                "side": "BUY",
                "price": 0.50,
                "best_ask": 0.52,
                "best_bid": 0.50,
                "size": "200.0",
                "ts_arrival_ms": 1000,
            },
        ])
        depth = read_clob_ask_depth("UP_TOKEN", market_log)
        assert depth is None

    def test_ignores_non_top_of_book(self, tmp_path: Path) -> None:
        """Depth should only come from levels matching best_ask."""
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {
                "asset_id": "UP_TOKEN",
                "event_type": "price_change",
                "side": "SELL",
                "price": 0.55,  # Above best_ask — not top of book
                "best_ask": 0.52,
                "best_bid": 0.50,
                "size": "100.0",
                "ts_arrival_ms": 1000,
            },
        ])
        depth = read_clob_ask_depth("UP_TOKEN", market_log)
        assert depth is None

    def test_missing_file(self, tmp_path: Path) -> None:
        depth = read_clob_ask_depth("UP_TOKEN", tmp_path / "nope.jsonl")
        assert depth is None

    def test_prefers_most_recent(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {
                "asset_id": "UP_TOKEN",
                "event_type": "price_change",
                "side": "SELL",
                "price": 0.52,
                "best_ask": 0.52,
                "best_bid": 0.50,
                "size": "50.0",
                "ts_arrival_ms": 500,
            },
            {
                "asset_id": "UP_TOKEN",
                "event_type": "price_change",
                "side": "SELL",
                "price": 0.52,
                "best_ask": 0.52,
                "best_bid": 0.50,
                "size": "300.0",
                "ts_arrival_ms": 1000,
            },
        ])
        depth = read_clob_ask_depth("UP_TOKEN", market_log)
        assert depth == pytest.approx(300.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests — IDCP: _capture_intraround_sample fee + depth fields
# ---------------------------------------------------------------------------
class TestCaptureIntraaroundSampleIDCP:
    def test_fee_adjusted_bundle(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52,
             "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.47, "best_ask": 0.49,
             "ts_arrival_ms": 1000},
        ])

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 120

        sample = obs._capture_intraround_sample(rnd)

        # bundle = 0.52 + 0.49 = 1.01
        assert sample["bundle_cost"] == pytest.approx(1.01, abs=1e-6)

        # fee_up = 0.02 * min(0.52, 0.48) = 0.02 * 0.48 = 0.0096
        assert sample["fee_up"] == pytest.approx(0.0096, abs=1e-4)
        # fee_down = 0.02 * min(0.49, 0.51) = 0.02 * 0.49 = 0.0098
        assert sample["fee_down"] == pytest.approx(0.0098, abs=1e-4)

        # fee_adjusted = 1.01 + 0.0096 + 0.0098 = 1.0294
        assert sample["fee_adjusted_bundle"] == pytest.approx(1.0294, abs=1e-3)
        assert sample["bundle_sub_1"] is False
        assert sample["fee_adjusted_sub_1"] is False

    def test_fee_adjusted_sub_one(self, tmp_path: Path) -> None:
        """When raw bundle < 1.0, verify both flags."""
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.40, "best_ask": 0.42,
             "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.48, "best_ask": 0.50,
             "ts_arrival_ms": 1000},
        ])

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 60

        sample = obs._capture_intraround_sample(rnd)

        # bundle = 0.42 + 0.50 = 0.92
        assert sample["bundle_cost"] == pytest.approx(0.92, abs=1e-6)
        assert sample["bundle_sub_1"] is True

        # fee_up = 0.02 * min(0.42, 0.58) = 0.0084
        # fee_down = 0.02 * min(0.50, 0.50) = 0.01
        # fee_adj = 0.92 + 0.0084 + 0.01 = 0.9384 (still < 1.0)
        assert sample["fee_adjusted_bundle"] == pytest.approx(0.9384, abs=1e-3)
        assert sample["fee_adjusted_sub_1"] is True

    def test_depth_fields_present(self, tmp_path: Path) -> None:
        market_log = tmp_path / "clob.jsonl"
        _write_jsonl(market_log, [
            {"asset_id": "UP_TOKEN", "best_bid": 0.50, "best_ask": 0.52,
             "ts_arrival_ms": 1000},
            {"asset_id": "DOWN_TOKEN", "best_bid": 0.47, "best_ask": 0.49,
             "ts_arrival_ms": 1000},
            # price_change with depth at best_ask for UP
            {"asset_id": "UP_TOKEN", "event_type": "price_change",
             "side": "SELL", "price": 0.52, "best_ask": 0.52,
             "best_bid": 0.50, "size": "200.0", "ts_arrival_ms": 1100},
        ])

        obs = RoundObserver(market_log=market_log)
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 60

        sample = obs._capture_intraround_sample(rnd)

        assert sample["depth_up"] == pytest.approx(200.0, abs=1e-6)
        assert sample["depth_down"] is None  # No SELL price_change for DOWN

    def test_missing_data_has_none_idcp_fields(self, tmp_path: Path) -> None:
        obs = RoundObserver(market_log=tmp_path / "clob.jsonl")
        rnd = _make_round_state()
        rnd.round_start_ts = time.time() - 60

        sample = obs._capture_intraround_sample(rnd)

        assert sample["fee_up"] is None
        assert sample["fee_down"] is None
        assert sample["fee_adjusted_bundle"] is None
        assert sample["bundle_sub_1"] is None
        assert sample["fee_adjusted_sub_1"] is None
        assert sample["depth_up"] is None
        assert sample["depth_down"] is None


# ---------------------------------------------------------------------------
# Tests — IDCP: _compute_intraround_stats duration + fee-adjusted
# ---------------------------------------------------------------------------
class TestComputeIntraaroundStatsIDCP:
    def test_fee_adjusted_stats(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "fee_adjusted_bundle": 1.03,
             "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 1.01,
             "elapsed_s": 120, "ts": "t2"},
            {"bundle_cost": 0.98, "fee_adjusted_bundle": 0.995,
             "elapsed_s": 180, "ts": "t3"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["min_fee_adjusted_bundle"] == pytest.approx(0.995, abs=1e-3)
        assert stats["max_fee_adjusted_bundle"] == pytest.approx(1.03, abs=1e-3)
        assert stats["dislocation_fee_adjusted_count"] == 1
        assert stats["min_fee_adjusted_dislocation"] == pytest.approx(
            0.995, abs=1e-3
        )

    def test_no_fee_adjusted_sub_one(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 1.005,
             "elapsed_s": 60, "ts": "t1"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_fee_adjusted_count"] == 0
        assert stats["min_fee_adjusted_dislocation"] is None

    def test_dislocation_duration_single_window(self) -> None:
        """Window is based on fee_adjusted_bundle < 1.0 (not raw bundle)."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "fee_adjusted_bundle": 1.02, "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 0.99, "elapsed_s": 120, "ts": "t2"},
            {"bundle_cost": 0.98, "fee_adjusted_bundle": 0.98, "elapsed_s": 180, "ts": "t3"},
            {"bundle_cost": 1.02, "fee_adjusted_bundle": 1.03, "elapsed_s": 240, "ts": "t4"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_window_count"] == 1
        assert stats["max_dislocation_window_s"] == pytest.approx(120.0, abs=0.1)
        assert stats["total_dislocation_s"] == pytest.approx(120.0, abs=0.1)

    def test_dislocation_duration_two_windows(self) -> None:
        """Windows use fee_adjusted_bundle — raw sub-1 with fee >= 1 are not counted."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 0.99, "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 0.98, "fee_adjusted_bundle": 0.98, "elapsed_s": 120, "ts": "t2"},
            {"bundle_cost": 1.01, "fee_adjusted_bundle": 1.02, "elapsed_s": 180, "ts": "t3"},
            {"bundle_cost": 0.97, "fee_adjusted_bundle": 0.97, "elapsed_s": 240, "ts": "t4"},
            {"bundle_cost": 1.02, "fee_adjusted_bundle": 1.03, "elapsed_s": 300, "ts": "t5"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_window_count"] == 2
        # Window 1: 60→180 = 120s  Window 2: 240→300 = 60s
        assert stats["max_dislocation_window_s"] == pytest.approx(120.0, abs=0.1)
        assert stats["total_dislocation_s"] == pytest.approx(180.0, abs=0.1)
        assert stats["mean_dislocation_window_s"] == pytest.approx(90.0, abs=0.1)

    def test_dislocation_window_open_at_end(self) -> None:
        """Dislocation running through end of round (fee-adjusted) is counted."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "fee_adjusted_bundle": 1.02, "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 0.99, "elapsed_s": 120, "ts": "t2"},
            {"bundle_cost": 0.98, "fee_adjusted_bundle": 0.98, "elapsed_s": 180, "ts": "t3"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_window_count"] == 1
        # Open window from 120→180 = 60s
        assert stats["max_dislocation_window_s"] == pytest.approx(60.0, abs=0.1)

    def test_no_dislocation_no_windows(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 1.005, "elapsed_s": 120, "ts": "t2"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_window_count"] == 0
        assert "max_dislocation_window_s" not in stats

    def test_raw_sub_1_but_fee_adjusted_above_1_not_counted(self) -> None:
        """Raw bundle < 1.0 but fee-adjusted >= 1.0 must NOT count as window."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 1.005, "elapsed_s": 60, "ts": "t1"},
            {"bundle_cost": 0.98, "fee_adjusted_bundle": 1.002, "elapsed_s": 120, "ts": "t2"},
            {"bundle_cost": 0.97, "fee_adjusted_bundle": 1.001, "elapsed_s": 180, "ts": "t3"},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["dislocation_window_count"] == 0
        assert "max_dislocation_window_s" not in stats

    def test_skew_stats_in_intraround(self) -> None:
        """Sync audit skew stats are computed from skew_ms field."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "elapsed_s": 60, "ts": "t1", "skew_ms": 500},
            {"bundle_cost": 0.99, "fee_adjusted_bundle": 0.99, "elapsed_s": 120, "ts": "t2", "skew_ms": 45000},
            {"bundle_cost": 1.01, "elapsed_s": 180, "ts": "t3", "skew_ms": 200},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["skew_ms_min"] == 200
        assert stats["skew_ms_max"] == 45000
        assert stats["skew_samples_gt_1s"] == 1
        assert stats["skew_samples_gt_30s"] == 1

    def test_depth_stats(self) -> None:
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "elapsed_s": 60, "ts": "t1",
             "depth_up": 100.0, "depth_down": 200.0},
            {"bundle_cost": 1.005, "elapsed_s": 120, "ts": "t2",
             "depth_up": 150.0, "depth_down": 300.0},
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["mean_depth_up"] == pytest.approx(125.0, abs=0.1)
        assert stats["min_depth_up"] == pytest.approx(100.0, abs=0.1)
        assert stats["mean_depth_down"] == pytest.approx(250.0, abs=0.1)
        assert stats["min_depth_down"] == pytest.approx(200.0, abs=0.1)

    def test_missing_fee_adjusted_skipped(self) -> None:
        """Samples without fee_adjusted_bundle are gracefully excluded."""
        rnd = _make_round_state()
        rnd.intraround_samples = [
            {"bundle_cost": 1.01, "elapsed_s": 60, "ts": "t1"},
        ]
        stats = rnd._compute_intraround_stats()

        assert "min_fee_adjusted_bundle" not in stats
        assert "dislocation_fee_adjusted_count" not in stats

    def test_eligible_strict_all_pass(self) -> None:
        """Sample with fee_adj<1, skew<=1000, fresh legs → eligible strict."""
        rnd = _make_round_state()
        now_ms = 1700000000000
        rnd.intraround_samples = [
            {
                "bundle_cost": 0.96, "fee_adjusted_bundle": 0.98,
                "elapsed_s": 60, "ts": "t1",
                "skew_ms": 200,
                "sample_ts_ms": now_ms,
                "ts_up_ms": now_ms - 500,
                "ts_down_ms": now_ms - 700,
                "staleness_up_ms": 500,
                "staleness_down_ms": 700,
                "depth_up": 100.0, "depth_down": 200.0,
            },
            {
                "bundle_cost": 0.97, "fee_adjusted_bundle": 0.99,
                "elapsed_s": 120, "ts": "t2",
                "skew_ms": 800,
                "sample_ts_ms": now_ms + 60000,
                "ts_up_ms": now_ms + 60000 - 400,
                "ts_down_ms": now_ms + 60000 - 1200,
                "staleness_up_ms": 400,
                "staleness_down_ms": 1200,
                "depth_up": 150.0, "depth_down": 250.0,
            },
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["eligible_strict_count"] == 2
        assert stats["eligible_lenient_count"] == 2
        assert stats["eligible_strict_min_fab"] == pytest.approx(0.98, abs=1e-4)
        assert stats["eligible_strict_skew_max"] == 800
        assert stats["eligible_strict_median_depth_up"] == pytest.approx(150.0, abs=0.1)
        assert stats["eligible_strict_window_count"] >= 1

    def test_eligible_skew_too_high_rejects(self) -> None:
        """Sample with skew > 2000 is not eligible even lenient."""
        rnd = _make_round_state()
        now_ms = 1700000000000
        rnd.intraround_samples = [
            {
                "bundle_cost": 0.96, "fee_adjusted_bundle": 0.98,
                "elapsed_s": 60, "ts": "t1",
                "skew_ms": 5000,
                "sample_ts_ms": now_ms,
                "ts_up_ms": now_ms - 500,
                "ts_down_ms": now_ms - 5500,
                "staleness_up_ms": 500,
                "staleness_down_ms": 5500,
            },
        ]
        stats = rnd._compute_intraround_stats()

        # Skew exceeds both thresholds AND staleness_down > 2000
        assert stats["eligible_strict_count"] == 0
        assert stats["eligible_lenient_count"] == 0

    def test_eligible_stale_legs_rejected(self) -> None:
        """Both legs aligned (skew=0) but both stale → not eligible."""
        rnd = _make_round_state()
        now_ms = 1700000000000
        rnd.intraround_samples = [
            {
                "bundle_cost": 0.96, "fee_adjusted_bundle": 0.98,
                "elapsed_s": 60, "ts": "t1",
                "skew_ms": 0,
                "sample_ts_ms": now_ms,
                "ts_up_ms": now_ms - 10000,
                "ts_down_ms": now_ms - 10000,
                "staleness_up_ms": 10000,
                "staleness_down_ms": 10000,
            },
        ]
        stats = rnd._compute_intraround_stats()

        # Skew=0 but both legs 10s stale → freshness fails
        assert stats["eligible_strict_count"] == 0
        assert stats["eligible_lenient_count"] == 0

    def test_eligible_lenient_not_strict(self) -> None:
        """Sample with skew=1500 passes lenient but not strict."""
        rnd = _make_round_state()
        now_ms = 1700000000000
        rnd.intraround_samples = [
            {
                "bundle_cost": 0.96, "fee_adjusted_bundle": 0.98,
                "elapsed_s": 60, "ts": "t1",
                "skew_ms": 1500,
                "sample_ts_ms": now_ms,
                "ts_up_ms": now_ms - 300,
                "ts_down_ms": now_ms - 1800,
                "staleness_up_ms": 300,
                "staleness_down_ms": 1800,
                "depth_up": 100.0, "depth_down": 200.0,
            },
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["eligible_strict_count"] == 0
        assert stats["eligible_lenient_count"] == 1
        assert stats["eligible_lenient_min_fab"] == pytest.approx(0.98, abs=1e-4)

    def test_eligible_fee_adjusted_above_1_excluded(self) -> None:
        """Sample with fee_adjusted >= 1.0 is never eligible."""
        rnd = _make_round_state()
        now_ms = 1700000000000
        rnd.intraround_samples = [
            {
                "bundle_cost": 0.99, "fee_adjusted_bundle": 1.005,
                "elapsed_s": 60, "ts": "t1",
                "skew_ms": 100,
                "sample_ts_ms": now_ms,
                "ts_up_ms": now_ms - 100,
                "ts_down_ms": now_ms - 200,
                "staleness_up_ms": 100,
                "staleness_down_ms": 200,
            },
        ]
        stats = rnd._compute_intraround_stats()

        assert stats["eligible_strict_count"] == 0
        assert stats["eligible_lenient_count"] == 0


# ---------------------------------------------------------------------------
# _tail_lines helper
# ---------------------------------------------------------------------------

class TestTailLines:
    """Tests for _tail_lines — memory-safe file tail reader."""

    def test_reads_small_file_fully(self, tmp_path: Path) -> None:
        p = tmp_path / "small.jsonl"
        lines = [f'{{"n":{i}}}' for i in range(10)]
        p.write_text("\n".join(lines) + "\n")
        result = _tail_lines(p, max_bytes=10_000)
        assert len(result) == 10

    def test_reads_last_bytes_only(self, tmp_path: Path) -> None:
        """Large file: only tail portion returned."""
        p = tmp_path / "big.jsonl"
        # Each line ~20 chars → 100 lines ≈ 2000 bytes
        all_lines = [f'{{"n":{i},"pad":"x"}}' for i in range(100)]
        p.write_text("\n".join(all_lines) + "\n")
        # Read only last 200 bytes — should get ~10 lines
        result = _tail_lines(p, max_bytes=200)
        assert len(result) < 100
        assert len(result) >= 5  # at least a handful

    def test_drops_partial_first_line(self, tmp_path: Path) -> None:
        """When seeking mid-file, the first partial line is dropped."""
        p = tmp_path / "partial.jsonl"
        all_lines = [f'{{"n":{i}}}' for i in range(50)]
        content = "\n".join(all_lines) + "\n"
        p.write_text(content)
        # max_bytes smaller than file → seek lands mid-line
        result = _tail_lines(p, max_bytes=100)
        for line in result:
            # Every returned line should be valid JSON (no partial)
            json.loads(line)

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _tail_lines(p) == []

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist.jsonl"
        assert _tail_lines(p) == []

    def test_single_line_file(self, tmp_path: Path) -> None:
        p = tmp_path / "single.jsonl"
        p.write_text('{"key": "value"}\n')
        result = _tail_lines(p, max_bytes=10_000)
        assert len(result) == 1
        assert json.loads(result[0])["key"] == "value"

    def test_max_bytes_clamped_to_ceiling(self, tmp_path: Path) -> None:
        """Passing max_bytes > _TAIL_MAX_BYTES is silently clamped."""
        from prediction.round_observer import _TAIL_MAX_BYTES

        p = tmp_path / "small.jsonl"
        p.write_text('{"a":1}\n')
        # Requesting 100 MB should not crash or read 100 MB
        result = _tail_lines(p, max_bytes=100_000_000)
        assert len(result) == 1
        # Confirm the constant is 8 MB
        assert _TAIL_MAX_BYTES == 8_000_000

