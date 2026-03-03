"""Tests for execution.fill_tracker — sync/async parity, poll behaviour, timeouts."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from execution.fill_tracker import (
    OrderAckInfo,
    a_confirm_order_fill,
    a_fetch_order_status,
    a_fetch_order_trades,
    confirm_order_fill,
    start_fill_task,
    wait_fill_task,
)
from execution.pnl_tracker import PositionTracker


# ── helpers ───────────────────────────────────────────────────────────

def _make_ack(**overrides: Any) -> OrderAckInfo:
    defaults: Dict[str, Any] = dict(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        request_qty=1.0,
        position_side="LONG",
        reduce_only=False,
        order_id=111,
        client_order_id="cli-111",
        status="NEW",
        latency_ms=10.0,
        attempt_id="att-1",
        intent_id="int-1",
        ts_ack="2025-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return OrderAckInfo(**defaults)


_TWO_TRADES: List[Dict[str, Any]] = [
    {"qty": "0.5", "price": "100.0", "commission": "0.01",
     "commissionAsset": "USDT", "time": 1700000000000, "id": 1},
    {"qty": "0.5", "price": "101.0", "commission": "0.01",
     "commissionAsset": "USDT", "time": 1700000005000, "id": 2},
]


# ── Async wrappers ────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_fetch_order_status_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import execution.fill_tracker as ft
    monkeypatch.setattr(ft, "fetch_order_status", lambda s, oid, cid: {"status": "FILLED"})
    result = await a_fetch_order_status("BTCUSDT", 111, None)
    assert result == {"status": "FILLED"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_fetch_order_trades_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import execution.fill_tracker as ft
    monkeypatch.setattr(ft, "fetch_order_trades", lambda s, oid: _TWO_TRADES)
    result = await a_fetch_order_trades("BTCUSDT", 111)
    assert len(result) == 2


# ── Sync / async parity ──────────────────────────────────────────────

class TestSyncAsyncParity:
    """Verify sync wrapper produces identical results to running the async core."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {"status": "FILLED"})
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: list(_TWO_TRADES))
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)

    @pytest.mark.unit
    def test_sync_returns_correct_fill(self) -> None:
        ack = _make_ack()
        tracker = PositionTracker()
        result = confirm_order_fill(ack, position_tracker=tracker)
        assert result is not None
        assert pytest.approx(result.executed_qty) == 1.0
        assert pytest.approx(result.avg_price) == 100.5
        assert result.status == "FILLED"
        assert pytest.approx(result.fee_total) == 0.02

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_returns_correct_fill(self) -> None:
        ack = _make_ack()
        tracker = PositionTracker()
        result = await a_confirm_order_fill(ack, position_tracker=tracker)
        assert result is not None
        assert pytest.approx(result.executed_qty) == 1.0
        assert pytest.approx(result.avg_price) == 100.5
        assert result.status == "FILLED"
        assert pytest.approx(result.fee_total) == 0.02

    @pytest.mark.unit
    def test_sync_async_field_identity(self) -> None:
        """Every field of the sync and async result must match exactly."""
        ack = _make_ack()
        sync_result = confirm_order_fill(ack, position_tracker=PositionTracker())
        async_result = asyncio.run(
            a_confirm_order_fill(ack, position_tracker=PositionTracker())
        )
        assert sync_result is not None
        assert async_result is not None
        for field in (
            "executed_qty", "avg_price", "status", "fee_total",
            "fee_asset", "trade_ids",
        ):
            assert getattr(sync_result, field) == getattr(async_result, field), (
                f"Mismatch on {field}: {getattr(sync_result, field)} != {getattr(async_result, field)}"
            )


# ── Poll iteration behaviour ─────────────────────────────────────────

class TestPollIterations:
    @pytest.mark.unit
    def test_polls_until_filled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Status changes from NEW → NEW → FILLED over 3 polls."""
        import execution.fill_tracker as ft
        statuses = iter(["NEW", "NEW", "FILLED"])
        call_count = {"status": 0, "trades": 0}

        def fake_status(*a: Any, **kw: Any) -> Dict[str, Any]:
            call_count["status"] += 1
            return {"status": next(statuses, "FILLED")}

        def fake_trades(*a: Any, **kw: Any) -> List[Dict[str, Any]]:
            call_count["trades"] += 1
            # Return trades only on the third poll
            if call_count["trades"] >= 3:
                return list(_TWO_TRADES)
            return []

        monkeypatch.setattr(ft, "fetch_order_status", fake_status)
        monkeypatch.setattr(ft, "fetch_order_trades", fake_trades)
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        # Speed up: zero sleep
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        assert result is not None
        assert result.status == "FILLED"
        assert call_count["status"] == 3
        assert call_count["trades"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_polls_until_filled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same test through async path."""
        import execution.fill_tracker as ft
        statuses = iter(["NEW", "NEW", "FILLED"])
        call_count = {"status": 0, "trades": 0}

        def fake_status(*a: Any, **kw: Any) -> Dict[str, Any]:
            call_count["status"] += 1
            return {"status": next(statuses, "FILLED")}

        def fake_trades(*a: Any, **kw: Any) -> List[Dict[str, Any]]:
            call_count["trades"] += 1
            if call_count["trades"] >= 3:
                return list(_TWO_TRADES)
            return []

        monkeypatch.setattr(ft, "fetch_order_status", fake_status)
        monkeypatch.setattr(ft, "fetch_order_trades", fake_trades)
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = await a_confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        assert result is not None
        assert result.status == "FILLED"
        assert call_count["status"] == 3


# ── Timeout ───────────────────────────────────────────────────────────

class TestTimeout:
    @pytest.mark.unit
    def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {"status": "NEW"})
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: [])
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(ft, "FILL_POLL_TIMEOUT", 0.0)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {"status": "NEW"})
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: [])
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(ft, "FILL_POLL_TIMEOUT", 0.0)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = await a_confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        assert result is None


# ── Partial fills ─────────────────────────────────────────────────────

class TestPartialFills:
    @pytest.mark.unit
    def test_cumulative_qty_across_polls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First poll: 1 trade. Second poll: 1 more + FILLED. Assert cumulative."""
        import execution.fill_tracker as ft
        poll = {"n": 0}

        def fake_status(*a: Any, **kw: Any) -> Dict[str, Any]:
            return {"status": "NEW" if poll["n"] < 2 else "FILLED"}

        def fake_trades(*a: Any, **kw: Any) -> List[Dict[str, Any]]:
            poll["n"] += 1
            if poll["n"] == 1:
                return [_TWO_TRADES[0]]
            elif poll["n"] == 2:
                return list(_TWO_TRADES)  # both trades; id=1 is already seen
            return list(_TWO_TRADES)

        monkeypatch.setattr(ft, "fetch_order_status", fake_status)
        monkeypatch.setattr(ft, "fetch_order_trades", fake_trades)
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        assert result is not None
        assert pytest.approx(result.executed_qty) == 1.0
        assert len(result.trade_ids) == 2


# ── Error resilience ─────────────────────────────────────────────────

class TestErrorResilience:
    @pytest.mark.unit
    def test_no_order_id_returns_none(self) -> None:
        ack = _make_ack(order_id=None, client_order_id=None)
        result = confirm_order_fill(ack, position_tracker=PositionTracker())
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_no_order_id_returns_none(self) -> None:
        ack = _make_ack(order_id=None, client_order_id=None)
        result = await a_confirm_order_fill(ack, position_tracker=PositionTracker())
        assert result is None

    @pytest.mark.unit
    def test_status_fetch_fails_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_order_status returns {} on error; trades still processed."""
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {})
        # Return trades with FILLED-like data but status won't change from ack
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: list(_TWO_TRADES))
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(ft, "FILL_POLL_TIMEOUT", 0.05)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        result = confirm_order_fill(_make_ack(), position_tracker=PositionTracker())
        # Trades are processed even though status is stuck on NEW
        assert result is not None
        assert pytest.approx(result.executed_qty) == 1.0
        assert result.status == "NEW"  # status never updated to FILLED


# ── FillTaskHandle lifecycle ──────────────────────────────────────────

class TestFillTaskHandle:
    @pytest.fixture(autouse=True)
    def _patch_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {"status": "FILLED"})
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: list(_TWO_TRADES))
        monkeypatch.setattr(ft, "write_event", lambda *a, **kw: None)

    @pytest.mark.unit
    def test_handle_lifecycle(self) -> None:
        """start → wait → result matches direct confirm_order_fill."""
        ack = _make_ack()
        tracker = PositionTracker()
        handle = start_fill_task(ack, position_tracker=tracker)
        assert not handle._done
        assert handle._result is None

        result = wait_fill_task(handle)
        assert result is not None
        assert handle._done
        assert pytest.approx(result.executed_qty) == 1.0
        assert pytest.approx(result.avg_price) == 100.5

    @pytest.mark.unit
    def test_wait_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling wait_fill_task twice returns cached result, no re-poll."""
        import execution.fill_tracker as ft
        call_count = {"n": 0}
        orig_confirm = ft.confirm_order_fill

        def counting_confirm(*a: Any, **kw: Any) -> Any:
            call_count["n"] += 1
            return orig_confirm(*a, **kw)

        monkeypatch.setattr(ft, "confirm_order_fill", counting_confirm)

        handle = start_fill_task(_make_ack(), position_tracker=PositionTracker())
        r1 = wait_fill_task(handle)
        r2 = wait_fill_task(handle)
        assert r1 is r2
        assert call_count["n"] == 1

    @pytest.mark.unit
    def test_handle_stores_arguments(self) -> None:
        ack = _make_ack()
        handle = start_fill_task(ack, metadata={"foo": 1}, strategy="trend")
        assert handle.ack is ack
        assert handle.metadata == {"foo": 1}
        assert handle.strategy == "trend"

    @pytest.mark.unit
    def test_handle_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import execution.fill_tracker as ft
        monkeypatch.setattr(ft, "fetch_order_status", lambda *a, **kw: {"status": "NEW"})
        monkeypatch.setattr(ft, "fetch_order_trades", lambda *a, **kw: [])
        monkeypatch.setattr(ft, "FILL_POLL_TIMEOUT", 0.0)
        monkeypatch.setattr(ft, "FILL_POLL_INTERVAL", 0.0)

        handle = start_fill_task(_make_ack(), position_tracker=PositionTracker())
        result = wait_fill_task(handle)
        assert result is None
        assert handle._done
