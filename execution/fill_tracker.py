"""Fill tracking: order acknowledgement, fill polling, and PnL close detection.

Extracted from executor_live.py (Commit 5/6, architecture repair sprint).
Zero behavioural change — move-only refactor.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from execution.events import now_utc, write_event
from execution.exchange_utils import _req
from execution.helpers import iso_to_ts, ms_to_iso, normalize_status, to_float
from execution.pnl_tracker import (
    CloseResult as PnlCloseResult,
    Fill as PnlFill,
    PositionTracker,
)

LOG = logging.getLogger("exutil")

# ──────────────────────────────────────────────────────────────────
# Constants  (previously module-level globals in executor_live.py)
# ──────────────────────────────────────────────────────────────────
FILL_POLL_INTERVAL = float(os.getenv("ORDER_FILL_POLL_INTERVAL", "0.5") or 0.5)
FILL_POLL_TIMEOUT = float(os.getenv("ORDER_FILL_POLL_TIMEOUT", "8.0") or 8.0)
FILL_FINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}

# Singleton tracker — same instance shared with executor_live via import.
POSITION_TRACKER = PositionTracker()


# ──────────────────────────────────────────────────────────────────
# FillTaskRunner — private asyncio loop in a daemon thread (Phase 4)
# ──────────────────────────────────────────────────────────────────
class FillTaskRunner:
    """Owns a private asyncio event-loop running in a daemon thread.

    Coroutines submitted via :meth:`submit` execute on that loop and
    return a :class:`concurrent.futures.Future` that the caller can
    block on (or poll) from any synchronous thread.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Spin up the background loop.  Idempotent if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fill-task-runner",
        )
        self._thread.start()
        self._started.wait()

    def _run(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the loop and join the thread."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._loop = None
        self._thread = None

    # ── public API ────────────────────────────────────────────────

    def submit(self, coro) -> Future:  # type: ignore[type-arg]
        """Schedule *coro* on the background loop, return a Future."""
        assert self._loop is not None and self._loop.is_running()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


_RUNNER: Optional[FillTaskRunner] = None


def _get_runner() -> FillTaskRunner:
    """Lazily start and return the module-level :class:`FillTaskRunner`."""
    global _RUNNER
    if _RUNNER is None or not _RUNNER.running:
        _RUNNER = FillTaskRunner()
        _RUNNER.start()
    return _RUNNER


def _shutdown_runner() -> None:
    """Best-effort cleanup on interpreter exit."""
    if _RUNNER is not None:
        try:
            _RUNNER.stop(timeout=2.0)
        except Exception:  # pragma: no cover
            pass


atexit.register(_shutdown_runner)


# ──────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────
@dataclass
class OrderAckInfo:
    symbol: str
    side: str
    order_type: str
    request_qty: Optional[float]
    position_side: Optional[str]
    reduce_only: bool
    order_id: Optional[int]
    client_order_id: Optional[str]
    status: str
    latency_ms: Optional[float]
    attempt_id: Optional[str] = None
    intent_id: Optional[str] = None
    ts_ack: str = ""


@dataclass
class FillSummary:
    executed_qty: float
    avg_price: Optional[float]
    status: str
    fee_total: float
    fee_asset: Optional[str]
    trade_ids: List[Any]
    ts_fill_first: Optional[str]
    ts_fill_last: Optional[str]
    latency_ms: Optional[float] = None
    is_maker: bool = False


# Type alias — keeps executor call-sites unchanged when FillResult
# evolves in a later phase.
FillResult = Optional[FillSummary]


@dataclass
class FillTaskHandle:
    """Deferred-execution handle for a fill-polling task.

    Phase 4: work begins immediately on :func:`start_fill_task` via the
    background :class:`FillTaskRunner`.  :func:`wait_fill_task` blocks on
    the ``Future``; :func:`poll_fill_task` checks without blocking.
    """

    ack: OrderAckInfo
    metadata: Optional[Mapping[str, Any]] = None
    strategy: Optional[str] = None
    position_tracker: Optional[PositionTracker] = None
    _result: FillResult = field(default=None, repr=False)
    _done: bool = field(default=False, repr=False)
    _future: Optional[Future] = field(default=None, repr=False)  # type: ignore[type-arg]


# ──────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────
def fetch_order_status(symbol: str, order_id: Optional[int], client_order_id: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"symbol": symbol}
    if order_id:
        params["orderId"] = int(order_id)
    elif client_order_id:
        params["origClientOrderId"] = client_order_id
    else:
        return {}
    try:
        resp = _req("GET", "/fapi/v1/order", signed=True, params=params, timeout=6.0)
        return resp.json() or {}
    except Exception as exc:
        LOG.debug("[fills] order_status_fetch_failed symbol=%s order_id=%s err=%s", symbol, order_id, exc)
        return {}


def fetch_order_trades(symbol: str, order_id: Optional[int]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"symbol": symbol}
    if order_id:
        params["orderId"] = int(order_id)
    try:
        resp = _req("GET", "/fapi/v1/userTrades", signed=True, params=params, timeout=6.0)
        data = resp.json() or []
        if not isinstance(data, list):
            return []
        return data
    except Exception as exc:
        LOG.debug("[fills] order_trades_fetch_failed symbol=%s order_id=%s err=%s", symbol, order_id, exc)
        return []


# ── Async wrappers (Phase 3) ─────────────────────────────────────

async def a_fetch_order_status(
    symbol: str, order_id: Optional[int], client_order_id: Optional[str],
) -> Dict[str, Any]:
    """Async wrapper — delegates to :func:`fetch_order_status` via thread."""
    return await asyncio.to_thread(fetch_order_status, symbol, order_id, client_order_id)


async def a_fetch_order_trades(
    symbol: str, order_id: Optional[int],
) -> List[Dict[str, Any]]:
    """Async wrapper — delegates to :func:`fetch_order_trades` via thread."""
    return await asyncio.to_thread(fetch_order_trades, symbol, order_id)


# ──────────────────────────────────────────────────────────────────
# Order ack / close gate / fill confirm
# ──────────────────────────────────────────────────────────────────
def emit_order_ack(
    symbol: str,
    side: str,
    order_type: str,
    request_qty: Optional[float],
    position_side: Optional[str],
    reduce_only: bool,
    resp: Mapping[str, Any],
    *,
    latency_ms: Optional[float],
    attempt_id: Optional[str],
    intent_id: Optional[str],
) -> Optional[OrderAckInfo]:
    status = normalize_status(resp.get("status"))
    order_id_raw = resp.get("orderId")
    try:
        order_id = int(order_id_raw) if order_id_raw is not None else None
    except (TypeError, ValueError):
        order_id = None
    client_order_id = resp.get("clientOrderId") or resp.get("orderId")
    if not order_id and not client_order_id:
        return None
    ts_ack = now_utc()

    ack = OrderAckInfo(
        symbol=symbol,
        side=str(side).upper(),
        order_type=str(order_type).upper(),
        request_qty=to_float(request_qty),
        position_side=position_side,
        reduce_only=bool(reduce_only),
        order_id=order_id,
        client_order_id=str(client_order_id) if client_order_id is not None else None,
        status=status,
        latency_ms=to_float(latency_ms),
        attempt_id=attempt_id,
        intent_id=intent_id,
        ts_ack=ts_ack,
    )
    payload: Dict[str, Any] = {
        "symbol": ack.symbol,
        "side": ack.side,
        "ts_ack": ts_ack,
        "orderId": ack.order_id,
        "clientOrderId": ack.client_order_id,
        "request_qty": ack.request_qty,
        "order_type": ack.order_type,
        "status": ack.status,
    }
    if ack.position_side:
        payload["positionSide"] = ack.position_side
    if ack.reduce_only:
        payload["reduceOnly"] = True
    if ack.latency_ms is not None:
        payload["latency_ms"] = ack.latency_ms
    if attempt_id:
        payload["attempt_id"] = attempt_id
    if intent_id:
        payload["intent_id"] = intent_id
    try:
        write_event("order_ack", payload)
    except Exception as exc:
        LOG.debug("[events] ack_write_failed %s %s", payload.get("orderId"), exc)
    return ack


def should_emit_close(ack: OrderAckInfo, close_results: List[PnlCloseResult]) -> bool:
    if not close_results:
        return False
    if ack.reduce_only:
        return True
    pos_before = close_results[0].position_before
    pos_after = close_results[-1].position_after
    if abs(pos_after) < 1e-8:
        return True
    if pos_before == 0.0:
        return False
    return pos_before * pos_after <= 0.0


async def a_confirm_order_fill(
    ack: OrderAckInfo,
    metadata: Optional[Mapping[str, Any]] = None,
    strategy: Optional[str] = None,
    *,
    position_tracker: Optional[PositionTracker] = None,
) -> Optional[FillSummary]:
    """Async core — poll Binance for fills and emit order_fill / order_close events.

    HTTP calls are offloaded to threads via :func:`asyncio.to_thread`;
    inter-poll waits use :func:`asyncio.sleep`.  All event writes and
    PnL tracking remain synchronous (local I/O, not worth threading).

    Args:
        position_tracker: Injected tracker. Falls back to module-level POSITION_TRACKER.
    """
    tracker = position_tracker or POSITION_TRACKER
    if not ack.order_id and not ack.client_order_id:
        return None
    start = time.time()
    seen_trade_ids: set[str] = set()
    executed_qty = 0.0
    cum_quote = 0.0
    fee_total = 0.0
    fee_asset: Optional[str] = None
    ts_first: Optional[str] = None
    ts_last: Optional[str] = None
    status = ack.status
    last_summary: Optional[FillSummary] = None
    fill_latency_ms: Optional[float] = None

    metadata_payload: Optional[Dict[str, Any]] = None
    if isinstance(metadata, Mapping):
        try:
            metadata_payload = dict(metadata)
        except Exception:
            metadata_payload = None

    while (time.time() - start) <= FILL_POLL_TIMEOUT:
        status_resp = await a_fetch_order_status(ack.symbol, ack.order_id, ack.client_order_id)
        if status_resp:
            status = normalize_status(status_resp.get("status"))

        trades = await a_fetch_order_trades(ack.symbol, ack.order_id)
        new_trades: List[Dict[str, Any]] = []
        for trade in trades:
            trade_id = trade.get("id")
            if trade_id is None:
                continue
            trade_id_str = str(trade_id)
            if trade_id_str in seen_trade_ids:
                continue
            seen_trade_ids.add(trade_id_str)
            new_trades.append(trade)
            qty = to_float(trade.get("qty")) or 0.0
            price = to_float(trade.get("price")) or 0.0
            executed_qty += qty
            cum_quote += qty * price
            commission = to_float(trade.get("commission")) or 0.0
            fee_total += commission
            fee_asset = fee_asset or trade.get("commissionAsset") or trade.get("marginAsset") or "USDT"
            trade_ts = ms_to_iso(trade.get("time"))
            now_iso = now_utc()
            if trade_ts:
                if ts_first is None or trade_ts < ts_first:
                    ts_first = trade_ts
                if ts_last is None or trade_ts > ts_last:
                    ts_last = trade_ts
            else:
                if ts_first is None:
                    ts_first = now_iso
                ts_last = now_iso

        if new_trades:
            avg_price = (cum_quote / executed_qty) if executed_qty else None
            fill_payload: Dict[str, Any] = {
                "symbol": ack.symbol,
                "side": ack.side,
                "ts_fill_first": ts_first or now_utc(),
                "ts_fill_last": ts_last or now_utc(),
                "orderId": ack.order_id,
                "clientOrderId": ack.client_order_id,
                "executedQty": executed_qty,
                "avgPrice": avg_price,
                "fee_total": fee_total,
                "feeAsset": fee_asset or "USDT",
                "tradeIds": sorted(seen_trade_ids),
                "status": status,
            }
            if strategy:
                fill_payload["strategy"] = strategy
            if metadata_payload:
                fill_payload["metadata"] = metadata_payload
            if ack.position_side:
                fill_payload["positionSide"] = ack.position_side
            if ack.reduce_only:
                fill_payload["reduceOnly"] = True
            if ack.attempt_id:
                fill_payload["attempt_id"] = ack.attempt_id
            if ack.intent_id:
                fill_payload["intent_id"] = ack.intent_id
            try:
                write_event("order_fill", fill_payload)
            except Exception as exc:
                LOG.debug("[events] fill_write_failed %s %s", ack.order_id, exc)

            close_results: List[PnlCloseResult] = []
            for trade in new_trades:
                qty = to_float(trade.get("qty")) or 0.0
                if qty <= 0:
                    continue
                price = to_float(trade.get("price")) or 0.0
                commission = to_float(trade.get("commission")) or 0.0
                fill_obj = PnlFill(
                    symbol=ack.symbol,
                    side=ack.side,
                    qty=qty,
                    price=price,
                    fee=commission,
                    position_side=ack.position_side,
                    reduce_only=ack.reduce_only,
                )
                close_res = tracker.apply_fill(fill_obj)
                if close_res:
                    close_results.append(close_res)

            if should_emit_close(ack, close_results):
                total_realized = sum(r.realized_pnl for r in close_results)
                total_fees = sum(r.fees for r in close_results)
                pos_before = close_results[0].position_before if close_results else 0.0
                pos_after = close_results[-1].position_after if close_results else 0.0
                closed_qty = sum(r.closed_qty for r in close_results)
                close_payload: Dict[str, Any] = {
                    "symbol": ack.symbol,
                    "ts_close": ts_last or now_utc(),
                    "orderId": ack.order_id,
                    "clientOrderId": ack.client_order_id,
                    "realizedPnlUsd": total_realized,
                    "fees_total": total_fees,
                    "position_size_before": pos_before,
                    "position_size_after": pos_after,
                }
                if strategy:
                    close_payload["strategy"] = strategy
                if metadata_payload:
                    close_payload["metadata"] = metadata_payload
                if ack.position_side:
                    close_payload["positionSide"] = ack.position_side
                if closed_qty > 0:
                    close_payload["closed_qty"] = closed_qty
                if ack.attempt_id:
                    close_payload["attempt_id"] = ack.attempt_id
                if ack.intent_id:
                    close_payload["intent_id"] = ack.intent_id
                try:
                    write_event("order_close", close_payload)
                except Exception as exc:
                    LOG.debug("[events] close_write_failed %s %s", ack.order_id, exc)

            if ts_last:
                ack_ts = iso_to_ts(ack.ts_ack)
                fill_ts = iso_to_ts(ts_last)
                if ack_ts is not None and fill_ts is not None:
                    fill_latency_ms = max(0.0, (fill_ts - ack_ts) * 1000.0)

            last_summary = FillSummary(
                executed_qty=executed_qty,
                avg_price=avg_price,
                status=status,
                fee_total=fee_total,
                fee_asset=fee_asset,
                trade_ids=sorted(seen_trade_ids),
                ts_fill_first=ts_first,
                ts_fill_last=ts_last,
                latency_ms=fill_latency_ms,
            )

        if status in FILL_FINAL_STATUSES:
            break
        if not new_trades:
            await asyncio.sleep(FILL_POLL_INTERVAL)

    return last_summary


def confirm_order_fill(
    ack: OrderAckInfo,
    metadata: Optional[Mapping[str, Any]] = None,
    strategy: Optional[str] = None,
    *,
    position_tracker: Optional[PositionTracker] = None,
) -> Optional[FillSummary]:
    """Sync wrapper — spins up an event loop to run the async core.

    Behavior-identical to the pre-Phase-3 implementation.  The async
    core (:func:`a_confirm_order_fill`) is the single source of truth.
    """
    return asyncio.run(
        a_confirm_order_fill(
            ack, metadata, strategy, position_tracker=position_tracker,
        )
    )


# ──────────────────────────────────────────────────────────────────
# Handle-based API (Phase 3 seam — Phase 4 makes wait non-blocking)
# ──────────────────────────────────────────────────────────────────

def start_fill_task(
    ack: OrderAckInfo,
    metadata: Optional[Mapping[str, Any]] = None,
    strategy: Optional[str] = None,
    *,
    position_tracker: Optional[PositionTracker] = None,
) -> FillTaskHandle:
    """Create a fill-polling handle and begin background polling immediately.

    The async core runs on the :class:`FillTaskRunner`'s event loop.
    :func:`wait_fill_task` or :func:`poll_fill_task` retrieve the result.
    """
    handle = FillTaskHandle(
        ack=ack,
        metadata=metadata,
        strategy=strategy,
        position_tracker=position_tracker,
    )
    try:
        runner = _get_runner()
        handle._future = runner.submit(
            a_confirm_order_fill(
                ack, metadata, strategy,
                position_tracker=position_tracker,
            )
        )
    except Exception:
        LOG.warning("[fills] runner_submit_failed; will fall back on wait", exc_info=True)
    return handle


def wait_fill_task(handle: FillTaskHandle) -> FillResult:
    """Block until fill polling completes and return the result.

    Idempotent: if the handle was already awaited, the cached result
    is returned without re-polling.
    """
    if handle._done:
        return handle._result
    if handle._future is not None:
        try:
            handle._result = handle._future.result(timeout=FILL_POLL_TIMEOUT + 5.0)
        except Exception:
            LOG.warning("[fills] future_result_failed; falling back to sync", exc_info=True)
            handle._result = confirm_order_fill(
                handle.ack,
                handle.metadata,
                handle.strategy,
                position_tracker=handle.position_tracker,
            )
    else:
        LOG.warning("[fills] no future available; falling back to sync")
        handle._result = confirm_order_fill(
            handle.ack,
            handle.metadata,
            handle.strategy,
            position_tracker=handle.position_tracker,
        )
    handle._done = True
    return handle._result


def poll_fill_task(handle: FillTaskHandle) -> FillResult:
    """Non-blocking check: return the result if ready, else ``None``."""
    if handle._done:
        return handle._result
    if handle._future is not None and handle._future.done():
        try:
            handle._result = handle._future.result(timeout=0)
        except Exception:
            LOG.warning("[fills] poll_result_failed", exc_info=True)
            handle._result = None
        handle._done = True
        return handle._result
    return None
