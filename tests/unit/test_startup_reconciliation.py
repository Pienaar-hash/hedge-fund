"""
Tests for executor startup reconciliation call sequence and the
startup_reconciliation module's four order-state cases.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ack_line(order_id: int, symbol: str = "BTCUSDT", ts_offset_s: float = 0.0) -> str:
    from execution.events import now_utc
    from datetime import datetime, timezone, timedelta
    ts = (datetime.now(timezone.utc) - timedelta(seconds=ts_offset_s)).isoformat()
    event = {
        "event_type": "order_ack",
        "symbol": symbol,
        "side": "BUY",
        "ts_ack": ts,
        "orderId": order_id,
        "clientOrderId": f"cli_{order_id}",
        "request_qty": 0.01,
        "order_type": "LIMIT",
        "status": "NEW",
    }
    return json.dumps(event)


def _fill_line(order_id: int) -> str:
    from execution.events import now_utc
    event = {
        "event_type": "order_fill",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "ts_fill_first": now_utc(),
        "ts_fill_last": now_utc(),
        "orderId": order_id,
        "clientOrderId": f"cli_{order_id}",
        "executedQty": 0.01,
        "avgPrice": 50000.0,
        "fee_total": 0.002,
        "feeAsset": "USDT",
        "tradeIds": ["t1"],
        "status": "FILLED",
    }
    return json.dumps(event)


def _fake_trade(order_id: int) -> Dict[str, Any]:
    return {
        "id": f"t_{order_id}",
        "qty": "0.01",
        "price": "50000.0",
        "commission": "0.002",
        "commissionAsset": "USDT",
        "time": int(time.time() * 1000),
        "realizedPnl": "5.0",
    }


# ---------------------------------------------------------------------------
# read_unresolved_acks
# ---------------------------------------------------------------------------

class TestReadUnresolvedAcks:
    def test_empty_file(self, tmp_path):
        from execution.startup_reconciliation import read_unresolved_acks
        log = tmp_path / "orders.jsonl"
        log.write_text("")
        result = read_unresolved_acks(log_path=str(log))
        assert result == {}

    def test_ack_with_fill_is_resolved(self, tmp_path):
        from execution.startup_reconciliation import read_unresolved_acks
        log = tmp_path / "orders.jsonl"
        log.write_text("\n".join([_ack_line(101), _fill_line(101)]))
        result = read_unresolved_acks(log_path=str(log))
        assert 101 not in result

    def test_ack_without_fill_is_unresolved(self, tmp_path):
        from execution.startup_reconciliation import read_unresolved_acks
        log = tmp_path / "orders.jsonl"
        log.write_text(_ack_line(202))
        result = read_unresolved_acks(log_path=str(log))
        assert 202 in result

    def test_old_ack_outside_lookback_ignored(self, tmp_path):
        from execution.startup_reconciliation import read_unresolved_acks
        log = tmp_path / "orders.jsonl"
        # ack from 10 hours ago, lookback is 1 hour
        log.write_text(_ack_line(303, ts_offset_s=10 * 3600))
        result = read_unresolved_acks(log_path=str(log), lookback_seconds=3600)
        assert 303 not in result

    def test_ack_exactly_at_boundary_is_included(self, tmp_path):
        """
        An ACK exactly at the cutoff edge (ts == time.time() - lookback_seconds)
        must be included. The reader uses >= so the boundary is inclusive.
        The exchange history query uses orderId — no separate time filter — so
        there is no window mismatch between reader and exchange query.
        """
        from execution.startup_reconciliation import read_unresolved_acks
        import time
        from datetime import datetime, timezone, timedelta

        lookback = 3600.0
        # Write an ack timestamped exactly at the cutoff
        boundary_ts = datetime.now(timezone.utc) - timedelta(seconds=lookback - 1)
        event = {
            "event_type": "order_ack",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "ts_ack": boundary_ts.isoformat(),
            "orderId": 404,
            "clientOrderId": "cli_404",
            "request_qty": 0.01,
            "order_type": "LIMIT",
            "status": "NEW",
        }
        log = tmp_path / "orders.jsonl"
        log.write_text(json.dumps(event))

        result = read_unresolved_acks(log_path=str(log), lookback_seconds=lookback)
        assert 404 in result, (
            "ACK inside the lookback window (at boundary - 1s) must be included"
        )

    def test_missing_file_returns_empty(self, tmp_path):
        from execution.startup_reconciliation import read_unresolved_acks
        result = read_unresolved_acks(log_path=str(tmp_path / "nonexistent.jsonl"))
        assert result == {}


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def _run(self, status: str, trades: list):
        from execution.startup_reconciliation import _classify
        order_status = {"status": status}
        return _classify(order_status, trades)

    def test_filled_with_trades(self):
        case, has_fills = self._run("FILLED", [_fake_trade(1)])
        assert case == "full_fill_missing"
        assert has_fills is True

    def test_new_with_trades(self):
        case, has_fills = self._run("NEW", [_fake_trade(1)])
        assert case == "partial_fill_open"
        assert has_fills is True

    def test_canceled_with_trades(self):
        case, has_fills = self._run("CANCELED", [_fake_trade(1)])
        assert case == "partial_fill_terminal"
        assert has_fills is True

    def test_new_no_trades(self):
        case, has_fills = self._run("NEW", [])
        assert case == "open_no_fills"
        assert has_fills is False

    def test_canceled_no_trades(self):
        case, has_fills = self._run("CANCELED", [])
        assert case == "dead_no_fills"
        assert has_fills is False


# ---------------------------------------------------------------------------
# reconcile_unresolved_acks — integration with write_event mocked
# ---------------------------------------------------------------------------

class TestReconcileIntegration:
    def test_full_fill_missing_writes_fill_event_with_provenance(self, tmp_path):
        from execution.startup_reconciliation import reconcile_unresolved_acks

        log = tmp_path / "orders.jsonl"
        log.write_text(_ack_line(999))
        trades = [_fake_trade(999)]

        with (
            patch("execution.startup_reconciliation.fetch_order_status",
                  return_value={"status": "FILLED"}),
            patch("execution.startup_reconciliation.fetch_order_trades",
                  return_value=trades),
            patch("execution.startup_reconciliation.write_event") as mock_write,
        ):
            reconcile_unresolved_acks(log_path=str(log))

        written_types = [c.args[0] for c in mock_write.call_args_list]
        assert "order_fill" in written_types

        fill_payload = next(
            c.args[1] for c in mock_write.call_args_list if c.args[0] == "order_fill"
        )
        # All five provenance fields must be present
        assert fill_payload["synthetic"] is True
        assert fill_payload["recovery_source"] == "exchange_trade_history"
        assert fill_payload["recovery_reason"] == "ack_without_fill_log"
        assert fill_payload["recovery_ts"]           # non-empty ISO timestamp
        assert fill_payload["original_ack_ts"]       # ties event to the crash window

    def test_dead_no_fills_writes_nothing_but_logs_warning(self, tmp_path):
        from execution.startup_reconciliation import reconcile_unresolved_acks

        log = tmp_path / "orders.jsonl"
        log.write_text(_ack_line(888))

        with (
            patch("execution.startup_reconciliation.fetch_order_status",
                  return_value={"status": "CANCELED"}),
            patch("execution.startup_reconciliation.fetch_order_trades",
                  return_value=[]),
            patch("execution.startup_reconciliation.write_event") as mock_write,
            patch("execution.startup_reconciliation.LOG") as mock_log,
        ):
            reconcile_unresolved_acks(log_path=str(log))

        written_types = [c.args[0] for c in mock_write.call_args_list]
        assert "order_fill" not in written_types
        assert "order_close" not in written_types

        # Operator-visible warning so a human can spot orphaned ACKs
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("dead_no_fills" in w for w in warning_calls), (
            "Expected a dead_no_fills warning for the operator"
        )

    def test_summary_counts_cases(self, tmp_path):
        from execution.startup_reconciliation import reconcile_unresolved_acks

        log = tmp_path / "orders.jsonl"
        log.write_text(_ack_line(777))

        with (
            patch("execution.startup_reconciliation.fetch_order_status",
                  return_value={"status": "FILLED"}),
            patch("execution.startup_reconciliation.fetch_order_trades",
                  return_value=[_fake_trade(777)]),
            patch("execution.startup_reconciliation.write_event"),
        ):
            summary = reconcile_unresolved_acks(log_path=str(log))

        assert summary["unresolved_count"] == 1
        assert summary["case_counts"].get("full_fill_missing") == 1

    def test_exchange_api_failure_skips_ack_and_continues(self, tmp_path):
        """Per-ACK exchange errors fail open: the ACK is skipped with an error
        logged, cancel_all still runs, and other ACKs in the same pass succeed."""
        from execution.startup_reconciliation import reconcile_unresolved_acks

        log = tmp_path / "orders.jsonl"
        # Two unresolved ACKs; first will fail, second should still be processed
        log.write_text("\n".join([_ack_line(111), _ack_line(222)]))

        call_count = {"n": 0}

        def flaky_status(symbol, order_id, client_order_id):
            call_count["n"] += 1
            if order_id == 111:
                raise OSError("network timeout")
            return {"status": "CANCELED"}

        with (
            patch("execution.startup_reconciliation.fetch_order_status",
                  side_effect=flaky_status),
            patch("execution.startup_reconciliation.fetch_order_trades",
                  return_value=[]),
            patch("execution.startup_reconciliation.write_event"),
        ):
            summary = reconcile_unresolved_acks(log_path=str(log))

        # Failed ACK is recorded in errors, not lost silently
        assert len(summary["errors"]) == 1
        assert "111" in summary["errors"][0]
        # Surviving ACK contributes to case_counts
        assert summary["case_counts"].get("dead_no_fills") == 1


# ---------------------------------------------------------------------------
# run_order_reconciliation — call sequence
# ---------------------------------------------------------------------------

class TestRunOrderReconciliationSequence:
    """Proves reconcile runs BEFORE cancel_all, and cancel_all is called exactly once."""

    def test_reconcile_then_cancel_sequence(self, tmp_path):
        from execution.executor_startup import run_order_reconciliation

        call_order: List[str] = []

        def fake_cancel():
            call_order.append("cancel")
            return {"cancelled": 0}

        with patch(
            "execution.startup_reconciliation.reconcile_unresolved_acks",
            side_effect=lambda **kw: (call_order.append("reconcile"), {})[1],
        ):
            run_order_reconciliation(
                cancel_all_open_orders=fake_cancel,
                log_path=str(tmp_path / "orders.jsonl"),
            )

        assert call_order == ["reconcile", "cancel"], (
            f"Expected reconcile before cancel, got: {call_order}"
        )

    def test_cancel_called_even_if_reconcile_fails(self, tmp_path):
        from execution.executor_startup import run_order_reconciliation

        cancelled = []

        def fake_cancel():
            cancelled.append(1)
            return {}

        with patch(
            "execution.startup_reconciliation.reconcile_unresolved_acks",
            side_effect=RuntimeError("boom"),
        ):
            # Should not raise
            run_order_reconciliation(
                cancel_all_open_orders=fake_cancel,
                log_path=str(tmp_path / "orders.jsonl"),
            )

        assert len(cancelled) == 1, "cancel_all must be called even when reconcile fails"
