"""
Futures S2 Proxy — unit tests.

Covers:
  - Signal→direction translation (edge→LONG/SHORT/NO_TRADE)
  - DD kill switch logic
  - PnL computation (gross, fee, net, log_return)
  - Round boundary / dedup logic
  - State persistence and recovery
  - Price region classifier
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from execution.futures_s2_proxy import (
    ENTRY_OFFSET_S,
    ENTRY_WINDOW_S,
    FuturesS2ProxyRunner,
    OpenFuturesTrade,
    _make_round_id,
    _price_region,
    _round_start_unix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_LIMITS: Dict[str, Any] = {
    "symbols": ["BTCUSDT"],
    "notional_usd": 200,
    "hold_duration_s": 900,
    "max_concurrent_per_symbol": 1,
    "min_edge": 0.03,
    "taker_fee_rate": 0.0004,
    "max_quote_age_s": 90,
    "dd_kill_usd": -500,
}


@dataclass
class _FakeSignal:
    """Minimal stand-in for BinaryLabS2Signal."""
    p_yes_mid: float = 0.50
    p_model_yes: float = 0.55
    edge_yes: float = 0.05
    quote_reconstruction_mode: str = "clob_live"
    spread: float = 0.02
    calibration_confident: bool = True
    quote_age_s: float = 10.0


def _make_runner(
    *,
    limits: Optional[Dict[str, Any]] = None,
    tmp_dir: Optional[str] = None,
) -> FuturesS2ProxyRunner:
    """Create a runner with isolated log/state paths."""
    td = tmp_dir or tempfile.mkdtemp()
    td_path = Path(td)
    lim = dict(_BASE_LIMITS, **(limits or {}))
    runner = FuturesS2ProxyRunner(
        limits=lim,
        model=MagicMock(),
        trade_log_path=td_path / "trades.jsonl",
        state_path=td_path / "futures_s2_proxy_state.json",
    )
    return runner


# ---------------------------------------------------------------------------
# Price region classifier
# ---------------------------------------------------------------------------

class TestPriceRegion:
    def test_extreme_low(self):
        assert _price_region(0.10) == "extreme_low"

    def test_low(self):
        assert _price_region(0.20) == "low"

    def test_mid_low(self):
        assert _price_region(0.40) == "mid_low"

    def test_center(self):
        assert _price_region(0.50) == "center"

    def test_mid_high(self):
        assert _price_region(0.60) == "mid_high"

    def test_high(self):
        assert _price_region(0.80) == "high"

    def test_extreme_high(self):
        assert _price_region(0.90) == "extreme_high"


# ---------------------------------------------------------------------------
# Round helpers
# ---------------------------------------------------------------------------

class TestRoundHelpers:
    def test_round_start_aligns_to_900s(self):
        # 1711199700.0 is a round boundary (1711199700 % 900 == 0)
        ts = 1711199700.0 + 450  # mid-round
        assert _round_start_unix(ts) == 1711199700.0

    def test_round_id_prefix(self):
        rid = _make_round_id(1711200000.0)
        assert rid.startswith("FS2_")

    def test_round_id_format(self):
        rid = _make_round_id(1711200000.0)
        # Should be FS2_YYYYMMDD_HHMM
        parts = rid.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # date
        assert len(parts[2]) == 4  # time


# ---------------------------------------------------------------------------
# Signal → direction translation
# ---------------------------------------------------------------------------

class TestEdgeToDirection:
    """Verify that _maybe_enter translates edge correctly."""

    def _run_entry(
        self,
        edge: float,
        *,
        min_edge: float = 0.03,
    ) -> Optional[str]:
        """Return the direction if entry made, else None."""
        limits = dict(_BASE_LIMITS, min_edge=min_edge)
        td = tempfile.mkdtemp()
        runner = _make_runner(limits=limits, tmp_dir=td)

        signal = _FakeSignal(
            p_yes_mid=0.50,
            p_model_yes=0.50 + edge,
            edge_yes=edge,
        )

        captured_side = {}
        def fake_send_order(**kwargs):
            captured_side["positionSide"] = kwargs.get("positionSide")
            return {"orderId": "test123", "avgPrice": "50000", "executedQty": "0.004", "status": "FILLED"}

        # Place time inside entry window of a round
        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10  # within window
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.exchange_utils.get_price", return_value=50000.0), \
             patch("execution.exchange_precision.normalize_qty", return_value=0.004), \
             patch("execution.exchange_utils.send_order", side_effect=fake_send_order):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        if entered:
            return captured_side.get("positionSide")
        return None

    def test_strong_positive_edge_goes_long(self):
        assert self._run_entry(0.05) == "LONG"

    def test_strong_negative_edge_goes_short(self):
        assert self._run_entry(-0.05) == "SHORT"

    def test_edge_below_threshold_no_trade(self):
        assert self._run_entry(0.02) is None

    def test_edge_at_boundary_goes_long(self):
        assert self._run_entry(0.03) == "LONG"

    def test_negative_edge_at_boundary_goes_short(self):
        assert self._run_entry(-0.03) == "SHORT"

    def test_zero_edge_no_trade(self):
        assert self._run_entry(0.0) is None


# ---------------------------------------------------------------------------
# DD kill switch
# ---------------------------------------------------------------------------

class TestDDKill:
    def test_dd_kill_blocks_entry(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        runner._cumulative_pnl = -501.0  # below -500 kill

        signal = _FakeSignal(edge_yes=0.05)

        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert not entered
        assert runner._dd_kill_active is True

    def test_dd_kill_at_exact_threshold(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        runner._cumulative_pnl = -500.0  # exactly at kill limit

        signal = _FakeSignal(edge_yes=0.05)

        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert not entered
        assert runner._dd_kill_active is True

    def test_dd_kill_inactive_allows_entry(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        runner._cumulative_pnl = -499.99  # just above kill

        signal = _FakeSignal(edge_yes=0.05)

        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.exchange_utils.get_price", return_value=50000.0), \
             patch("execution.exchange_precision.normalize_qty", return_value=0.004), \
             patch("execution.exchange_utils.send_order", return_value={"orderId": "t1", "avgPrice": "50000", "executedQty": "0.004", "status": "FILLED"}):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert entered


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

class TestPnLComputation:
    def test_long_profit(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
        )

        with patch("execution.exchange_utils.send_order", return_value={"orderId": "exit1", "avgPrice": "51000", "executedQty": "0.004", "status": "FILLED"}):
            outcome = runner._close_trade(trade, "2025-03-23T12:15:30+00:00", 1711196130.0)

        assert outcome is not None
        # gross = (51000-50000) * 0.004 * 1 = 4.0
        assert abs(outcome.gross_pnl - 4.0) < 0.01
        # fee = (50000*0.004 + 51000*0.004) * 0.0004 = (200 + 204) * 0.0004 = 0.1616
        assert abs(outcome.fee_usd - 0.1616) < 0.001
        # net = 4.0 - 0.1616 = 3.8384
        assert abs(outcome.net_pnl - 3.8384) < 0.01
        # log_return = ln(51000/50000) * 1 - 2*0.0004
        expected_log = math.log(51000 / 50000) - 0.0008
        assert abs(outcome.log_return - expected_log) < 1e-6

    def test_short_profit(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
        )

        with patch("execution.exchange_utils.send_order", return_value={"orderId": "exit1", "avgPrice": "49000", "executedQty": "0.004", "status": "FILLED"}):
            outcome = runner._close_trade(trade, "2025-03-23T12:15:30+00:00", 1711196130.0)

        assert outcome is not None
        # gross = (49000-50000) * 0.004 * (-1) = 4.0
        assert abs(outcome.gross_pnl - 4.0) < 0.01
        # log_return = ln(49000/50000) * (-1) - 2*0.0004
        expected_log = math.log(49000 / 50000) * (-1) - 0.0008
        assert abs(outcome.log_return - expected_log) < 1e-6

    def test_long_loss(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
        )

        with patch("execution.exchange_utils.send_order", return_value={"orderId": "exit1", "avgPrice": "49000", "executedQty": "0.004", "status": "FILLED"}):
            outcome = runner._close_trade(trade, "2025-03-23T12:15:30+00:00", 1711196130.0)

        assert outcome is not None
        # gross = (49000-50000) * 0.004 * 1 = -4.0
        assert abs(outcome.gross_pnl - (-4.0)) < 0.01
        assert outcome.net_pnl < 0


# ---------------------------------------------------------------------------
# Round boundary / dedup
# ---------------------------------------------------------------------------

class TestRoundBoundary:
    def test_before_entry_window_rejected(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05)
        round_start = _round_start_unix(1711200000.0)
        # Before the entry window
        now_unix = round_start + 5  # only 5s in, before ENTRY_OFFSET_S=30
        now_ts = "2025-03-23T12:00:05+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert not entered

    def test_after_entry_window_rejected(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05)
        round_start = _round_start_unix(1711200000.0)
        # After the window
        now_unix = round_start + ENTRY_OFFSET_S + ENTRY_WINDOW_S + 10
        now_ts = "2025-03-23T12:02:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert not entered

    def test_dedup_prevents_double_entry(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05)
        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.exchange_utils.get_price", return_value=50000.0), \
             patch("execution.exchange_precision.normalize_qty", return_value=0.004), \
             patch("execution.exchange_utils.send_order", return_value={"orderId": "t1", "avgPrice": "50000", "executedQty": "0.004", "status": "FILLED"}):
            first = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        # Second call: same round, should be dedup'd
        now_unix2 = round_start + ENTRY_OFFSET_S + 30
        now_ts2 = "2025-03-23T12:01:00+00:00"
        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            second = runner._maybe_enter("BTCUSDT", now_unix2, now_ts2)

        assert first is True
        assert second is False


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_state_round_trip(self):
        """Write state, create new runner — state restored."""
        td = tempfile.mkdtemp()
        td_path = Path(td)
        state_file = td_path / "futures_s2_proxy_state.json"

        runner = _make_runner(tmp_dir=td)
        runner._cumulative_pnl = -123.45
        runner._total_entries = 10
        runner._total_exits = 8
        runner._win_count = 5
        runner._sum_log_return = 0.004
        runner._dd_kill_active = False
        runner._write_state()

        assert state_file.exists()

        # New runner should load persisted state
        runner2 = _make_runner(tmp_dir=td)
        assert abs(runner2._cumulative_pnl - (-123.45)) < 0.01
        assert runner2._total_entries == 10
        assert runner2._total_exits == 8
        assert runner2._dd_kill_active is False

    def test_state_restores_open_trades(self):
        td = tempfile.mkdtemp()

        runner = _make_runner(tmp_dir=td)
        runner._open_trades.append(OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={"edge": 0.05},
        ))
        runner._write_state()

        runner2 = _make_runner(tmp_dir=td)
        assert len(runner2._open_trades) == 1
        assert runner2._open_trades[0].round_id == "FS2_20250323_1200"
        assert runner2._open_trades[0].side == "LONG"
        assert runner2._open_trades[0].entry_price == 50000.0


# ---------------------------------------------------------------------------
# Fixed-horizon exit timing
# ---------------------------------------------------------------------------

class TestFixedHorizonExit:
    def test_exit_triggered_at_deadline(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
        )
        runner._open_trades.append(trade)

        with patch("execution.exchange_utils.send_order", return_value={"orderId": "exit1", "avgPrice": "50500", "executedQty": "0.004", "status": "FILLED"}):
            exited = runner._maybe_exit_all(1711196130.0, "2025-03-23T12:15:30+00:00")

        assert exited is True
        assert len(runner._open_trades) == 0
        assert runner._total_exits == 1

    def test_no_exit_before_deadline(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
        )
        runner._open_trades.append(trade)

        # 5 minutes before deadline
        exited = runner._maybe_exit_all(1711195830.0, "2025-03-23T12:10:30+00:00")

        assert exited is False
        assert len(runner._open_trades) == 1


# ---------------------------------------------------------------------------
# Stale quote rejection
# ---------------------------------------------------------------------------

class TestStaleQuoteRejection:
    def test_stale_quote_blocks_entry(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05, quote_age_s=100.0)  # >90s

        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert not entered

    def test_fresh_quote_allows_entry(self):
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05, quote_age_s=10.0)

        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.exchange_utils.get_price", return_value=50000.0), \
             patch("execution.exchange_precision.normalize_qty", return_value=0.004), \
             patch("execution.exchange_utils.send_order", return_value={"orderId": "t1", "avgPrice": "50000", "executedQty": "0.004", "status": "FILLED"}):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert entered


# ---------------------------------------------------------------------------
# Entry / exit timing decomposition
# ---------------------------------------------------------------------------

class TestTimingDecomposition:
    def test_entry_timestamps_populated(self):
        """After _maybe_enter, OpenFuturesTrade carries 4 entry timestamps."""
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        signal = _FakeSignal(edge_yes=0.05)
        round_start = _round_start_unix(1711200000.0)
        now_unix = round_start + ENTRY_OFFSET_S + 10
        now_ts = "2025-03-23T12:00:40+00:00"

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.exchange_utils.get_price", return_value=50000.0), \
             patch("execution.exchange_precision.normalize_qty", return_value=0.004), \
             patch("execution.exchange_utils.send_order", return_value={"orderId": "t1", "avgPrice": "50000", "executedQty": "0.004", "status": "FILLED"}):
            entered = runner._maybe_enter("BTCUSDT", now_unix, now_ts)

        assert entered
        trade = runner._open_trades[0]
        assert trade.round_open_ts == round_start
        assert trade.entry_signal_ready_ts > 0
        assert trade.entry_order_submitted_ts >= trade.entry_signal_ready_ts
        assert trade.entry_fill_ts >= trade.entry_order_submitted_ts

    def test_exit_timestamps_populated(self):
        """After _close_trade, FuturesTradeOutcome carries exit + carried entry timestamps."""
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        trade = OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
            round_open_ts=1711195200.0,
            entry_signal_ready_ts=1711195225.0,
            entry_order_submitted_ts=1711195228.0,
            entry_fill_ts=1711195229.5,
        )

        with patch("execution.exchange_utils.send_order", return_value={"orderId": "exit1", "avgPrice": "51000", "executedQty": "0.004", "status": "FILLED"}):
            outcome = runner._close_trade(trade, "2025-03-23T12:15:30+00:00", 1711196130.0)

        assert outcome is not None
        assert outcome.hold_expiry_ts == 1711196130.0
        assert outcome.exit_decision_ts > 0
        assert outcome.exit_order_submitted_ts >= outcome.exit_decision_ts
        assert outcome.exit_fill_confirmed_ts >= outcome.exit_order_submitted_ts
        # Entry timestamps carried forward
        assert outcome.entry_signal_ready_ts == 1711195225.0
        assert outcome.entry_order_submitted_ts == 1711195228.0
        assert outcome.entry_fill_ts == 1711195229.5

    def test_timing_fields_survive_state_round_trip(self):
        """Entry timing fields persist through _write_state / _load_state."""
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)
        runner._open_trades.append(OpenFuturesTrade(
            round_id="FS2_20250323_1200",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_ts="2025-03-23T12:00:30+00:00",
            entry_ts_unix=1711195230.0,
            exit_ts_unix=1711196130.0,
            qty=0.004,
            notional_usd=200.0,
            order_id="entry1",
            signal_snapshot={},
            round_open_ts=1711195200.0,
            entry_signal_ready_ts=1711195225.123,
            entry_order_submitted_ts=1711195228.456,
            entry_fill_ts=1711195229.789,
        ))
        runner._write_state()

        runner2 = _make_runner(tmp_dir=td)
        assert len(runner2._open_trades) == 1
        restored = runner2._open_trades[0]
        assert restored.round_open_ts == 1711195200.0
        assert restored.entry_signal_ready_ts == 1711195225.123
        assert restored.entry_order_submitted_ts == 1711195228.456
        assert restored.entry_fill_ts == 1711195229.789


# ---------------------------------------------------------------------------
# Daemon thread
# ---------------------------------------------------------------------------

class TestDaemonThread:
    def test_daemon_starts_and_ticks(self):
        """Daemon thread starts, is_alive, and ticks at least once."""
        import time as _time
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        assert not runner.daemon_alive()
        runner.start_daemon(interval=0.1)

        assert runner.daemon_alive()
        _time.sleep(0.5)  # allow a few ticks
        assert runner._tick_count >= 1

        # Clean shutdown
        runner._daemon.stop(timeout=2)
        _time.sleep(0.2)
        assert not runner.daemon_alive()

    def test_daemon_double_start_is_noop(self):
        """Calling start_daemon twice does not spawn a second thread."""
        import time as _time
        td = tempfile.mkdtemp()
        runner = _make_runner(tmp_dir=td)

        runner.start_daemon(interval=0.1)
        first_daemon = runner._daemon
        runner.start_daemon(interval=0.1)  # should be no-op
        assert runner._daemon is first_daemon

        runner._daemon.stop(timeout=2)
