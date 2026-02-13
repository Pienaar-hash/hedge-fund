"""
Tests for episode_ledger v7.9-E2 fills-faithful changes.

Tests:
  - Ghost fill filtering (event_type != order_fill, qty == 0)
  - Multi-file fill loading (rotated + current)
  - Multi-fill exit tracking (exit_fills > 1)
  - Orphan fill detection (exits exceeding entry qty)
  - Reconciliation summary (fills_total, fills_consumed, fills_orphaned)
  - Deduplication across rotated + current log overlap
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from execution.episode_ledger import (
    _load_execution_log,
    build_episode_ledger,
    Episode,
    EpisodeLedger,
    EXECUTION_LOG_DIR,
    EXECUTION_LOG_PATH,
)


@pytest.fixture
def log_dir(tmp_path):
    """Create a temporary log directory with execution logs."""
    exec_dir = tmp_path / "logs" / "execution"
    exec_dir.mkdir(parents=True)
    state_dir = tmp_path / "logs" / "state"
    state_dir.mkdir(parents=True)
    return exec_dir


def _write_log(path: Path, records: list[dict]):
    """Write records as JSONL."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fill(symbol="BTCUSDT", side="BUY", pos_side="LONG", qty=0.1,
          price=50000.0, fee=0.02, reduce=False, ts="2026-02-12T10:00:00",
          order_id="1", strategy="vol_target", exit_reason=None):
    """Build a minimal order_fill record."""
    meta = {"strategy": strategy}
    if exit_reason:
        meta["exit_reason"] = exit_reason
        meta["exit"] = {"reason": exit_reason}
    return {
        "event_type": "order_fill",
        "symbol": symbol,
        "side": side,
        "positionSide": pos_side,
        "executedQty": qty,
        "avgPrice": price,
        "fee_total": fee,
        "reduceOnly": reduce,
        "status": "FILLED",
        "ts": ts,
        "ts_fill_first": ts,
        "orderId": order_id,
        "metadata": meta,
    }


def _ghost(symbol="BTCUSDT", side="SELL", pos_side="LONG"):
    """Build a ghost fill (order_ack, qty=0)."""
    return {
        "event_type": "order_ack",
        "symbol": symbol,
        "side": side,
        "positionSide": pos_side,
        "executedQty": 0,
        "status": "NEW",
        "reduceOnly": True,
    }


class TestFillLoading:
    """Test _load_execution_log reads all files and filters correctly."""

    def test_filters_ghost_fills(self, log_dir):
        """Ghost fills (order_ack, qty=0) are excluded."""
        records = [
            _fill(order_id="1"),
            _ghost(),
            _ghost(),
            {"event_type": "order_close", "executedQty": 0, "symbol": "BTC"},
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            fills = _load_execution_log()

        assert len(fills) == 1
        assert fills[0]["orderId"] == "1"

    def test_reads_rotated_files(self, log_dir):
        """Fills from rotated log files are included."""
        old_fill = _fill(ts="2026-02-10T10:00:00", order_id="old1")
        new_fill = _fill(ts="2026-02-12T10:00:00", order_id="new1")

        _write_log(log_dir / "orders_executed.1.jsonl", [old_fill])
        _write_log(log_dir / "orders_executed.jsonl", [new_fill])

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            fills = _load_execution_log()

        assert len(fills) == 2
        order_ids = {f["orderId"] for f in fills}
        assert "old1" in order_ids
        assert "new1" in order_ids

    def test_deduplicates_overlapping_fills(self, log_dir):
        """Same fill in rotated + current log is only loaded once."""
        fill = _fill(order_id="dup1", ts="2026-02-12T10:00:00")

        _write_log(log_dir / "orders_executed.1.jsonl", [fill])
        _write_log(log_dir / "orders_executed.jsonl", [fill])

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            fills = _load_execution_log()

        assert len(fills) == 1

    def test_zero_qty_fills_excluded(self, log_dir):
        """Fills with executedQty=0 are excluded even if event_type=order_fill."""
        records = [
            {**_fill(order_id="1"), "executedQty": 0},
            _fill(order_id="2", qty=0.1),
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            fills = _load_execution_log()

        assert len(fills) == 1
        assert fills[0]["orderId"] == "2"


class TestMultiFillEpisodes:
    """Test that multi-fill exits are tracked properly."""

    def test_multi_fill_exit(self, log_dir):
        """3 exit fills to close one entry → 1 episode with exit_fills=3."""
        records = [
            _fill(side="BUY", qty=0.3, ts="2026-02-12T10:00:00", order_id="e1"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T11:00:00", order_id="x1"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T11:30:00", order_id="x2"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T12:00:00", order_id="x3",
                  exit_reason="REGIME_FLIP"),
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        assert len(ledger.episodes) == 1
        ep = ledger.episodes[0]
        assert ep.entry_fills == 1
        assert ep.exit_fills == 3
        assert ep.total_qty == pytest.approx(0.3, abs=1e-6)

    def test_multi_entry_single_exit(self, log_dir):
        """2 entry fills + 1 exit fill → 1 episode with entry_fills=2."""
        records = [
            _fill(side="BUY", qty=0.1, ts="2026-02-12T10:00:00", order_id="e1"),
            _fill(side="BUY", qty=0.1, ts="2026-02-12T10:05:00", order_id="e2"),
            _fill(side="SELL", qty=0.2, reduce=True, ts="2026-02-12T12:00:00", order_id="x1",
                  exit_reason="tp"),
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        assert len(ledger.episodes) == 1
        ep = ledger.episodes[0]
        assert ep.entry_fills == 2
        assert ep.exit_fills == 1


class TestOrphanFills:
    """Test that orphan fills are detected and counted."""

    def test_exits_before_entries_are_orphans(self, log_dir):
        """Exit fills before any entry → counted as orphans."""
        records = [
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T09:00:00", order_id="orphan1"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T09:30:00", order_id="orphan2"),
            _fill(side="BUY", qty=0.1, ts="2026-02-12T10:00:00", order_id="e1"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T12:00:00", order_id="x1"),
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        recon = ledger.stats.get("reconciliation", {})
        assert recon.get("fills_orphaned", 0) == 2
        assert recon.get("fills_consumed", 0) == 2  # 1 entry + 1 exit
        assert len(ledger.episodes) == 1


class TestReconciliationSummary:
    """Test reconciliation summary in stats."""

    def test_clean_reconciliation(self, log_dir):
        """All fills consumed, no orphans."""
        records = [
            _fill(side="BUY", qty=0.1, ts="2026-02-12T10:00:00", order_id="e1"),
            _fill(side="SELL", qty=0.1, reduce=True, ts="2026-02-12T12:00:00", order_id="x1"),
        ]
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        recon = ledger.stats["reconciliation"]
        assert recon["fills_total"] == 2
        assert recon["fills_consumed"] == 2
        assert recon["fills_orphaned"] == 0

    def test_churn_pattern_reconciliation(self, log_dir):
        """Simulate churn: 1 entry, 20 exits → 19 orphans."""
        records = [
            _fill(side="BUY", qty=0.1, ts="2026-02-12T10:00:00", order_id="e1"),
        ]
        for i in range(20):
            ts = f"2026-02-12T10:{i+1:02d}:00"
            records.append(
                _fill(side="SELL", qty=0.1, reduce=True, ts=ts,
                      order_id=f"x{i}", exit_reason="REGIME_FLIP")
            )
        _write_log(log_dir / "orders_executed.jsonl", records)

        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        assert len(ledger.episodes) == 1  # One proper episode
        recon = ledger.stats["reconciliation"]
        assert recon["fills_total"] == 21  # 1 entry + 20 exits
        assert recon["fills_consumed"] == 2  # 1 entry + 1 exit (closes position)
        assert recon["fills_orphaned"] == 19  # 19 excess exits

    def test_empty_log(self, log_dir):
        """No log files → empty ledger with zero reconciliation."""
        # Don't create any log files
        with patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_DIR"]),
            "EXECUTION_LOG_DIR", log_dir,
        ), patch.object(
            __import__("execution.episode_ledger", fromlist=["EXECUTION_LOG_PATH"]),
            "EXECUTION_LOG_PATH", log_dir / "orders_executed.jsonl",
        ):
            ledger = build_episode_ledger()

        assert len(ledger.episodes) == 0
        assert ledger.stats.get("total_fills", 0) == 0
