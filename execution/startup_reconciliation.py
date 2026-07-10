"""
Startup reconciliation for crash-window order gaps.

Scans recent order_ack events in orders_executed.jsonl that have no matching
order_fill, queries exchange state for each, and backfills synthetic fill/close
events with explicit recovery provenance so downstream consumers stay honest.

Entry point: reconcile_unresolved_acks()

Safe startup sequence:
    1. Call reconcile_unresolved_acks() — backfills any fills that happened
       during the crash window (cases: full fill missing, partial fill missing).
    2. Call cancel_all_open_orders() — cleans up orders that were open but
       unfilled (we never backfill those; cancellation is the correct action).
    3. Log the reconciliation summary event returned by reconcile_unresolved_acks().
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from execution.events import now_utc, write_event
from execution.fill_tracker import fetch_order_status, fetch_order_trades
from execution.helpers import ms_to_iso, normalize_status, to_float

LOG = logging.getLogger("startup_reconciliation")

_DEFAULT_LOG_PATH = "logs/execution/orders_executed.jsonl"
_DEFAULT_LOOKBACK_S = float(4 * 3600)  # 4 hours

# Terminal states — no further exchange action possible
_TERMINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}

# Provenance constants stamped on every recovery event
_RECOVERY_SOURCE = "exchange_trade_history"


# ---------------------------------------------------------------------------
# Step 1: collect unresolved acks from JSONL
# ---------------------------------------------------------------------------

def read_unresolved_acks(
    log_path: str = _DEFAULT_LOG_PATH,
    lookback_seconds: float = _DEFAULT_LOOKBACK_S,
) -> Dict[int, Dict[str, Any]]:
    """
    Scan the orders_executed.jsonl log for order_ack events within the lookback
    window that have no corresponding order_fill event.

    Returns a dict keyed by orderId. Value is the raw ack event dict.
    """
    path = Path(log_path)
    if not path.exists():
        LOG.info("[reconcile] log not found, nothing to reconcile path=%s", log_path)
        return {}

    cutoff_ts = time.time() - lookback_seconds
    acks: Dict[int, Dict[str, Any]] = {}
    filled_ids: set[int] = set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        LOG.warning("[reconcile] cannot read log path=%s err=%s", log_path, exc)
        return {}

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event_type")
        order_id_raw = event.get("orderId")
        try:
            order_id = int(order_id_raw) if order_id_raw is not None else None
        except (TypeError, ValueError):
            order_id = None

        if order_id is None:
            continue

        if event_type == "order_ack":
            # Only track recent acks
            ts_str = event.get("ts_ack") or event.get("ts") or ""
            try:
                from datetime import datetime
                event_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                event_ts = 0.0
            if event_ts >= cutoff_ts:
                acks[order_id] = event

        elif event_type == "order_fill":
            filled_ids.add(order_id)

    unresolved = {oid: ev for oid, ev in acks.items() if oid not in filled_ids}
    LOG.info(
        "[reconcile] scan complete: %d acks in window, %d already filled, %d unresolved",
        len(acks),
        len(filled_ids),
        len(unresolved),
    )
    return unresolved


# ---------------------------------------------------------------------------
# Step 2: classify each unresolved ack against exchange state
# ---------------------------------------------------------------------------

def _classify(
    order_status: Dict[str, Any],
    trades: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """
    Return (case_label, has_fills).

    Cases:
        "full_fill_missing"     — order FILLED, trades present, no local order_fill
        "partial_fill_open"     — order still open (NEW/PARTIALLY_FILLED), trades present
        "partial_fill_terminal" — order terminal (CANCELED/EXPIRED) with partial trades
        "open_no_fills"         — order still open, no trades
        "dead_no_fills"         — order terminal, no trades
    """
    status = normalize_status(order_status.get("status")) if order_status else "UNKNOWN"
    has_fills = bool(trades)
    is_terminal = status in _TERMINAL_STATUSES

    if status == "FILLED":
        return "full_fill_missing", has_fills
    if is_terminal and has_fills:
        return "partial_fill_terminal", True
    if is_terminal:
        return "dead_no_fills", False
    if has_fills:
        return "partial_fill_open", True
    return "open_no_fills", False


# ---------------------------------------------------------------------------
# Step 3: build and write recovery events
# ---------------------------------------------------------------------------

def _recovery_provenance(reason: str, ack_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Four required provenance fields stamped on every synthetic fill/close event.

    When reading orders_executed.jsonl, these fields identify that the event
    was not written by the live fill-polling path but was reconstructed at
    startup from exchange trade history:

        synthetic          — always True; distinguishes recovery events from live fills
        recovery_source    — where the fill data came from (always "exchange_trade_history")
        recovery_ts        — wall-clock time the reconciliation ran
        recovery_reason    — why the live event was missing (e.g. "ack_without_fill_log")
        original_ack_ts    — ts_ack from the original order_ack; anchors the crash window
    """
    return {
        "synthetic": True,
        "recovery_source": _RECOVERY_SOURCE,
        "recovery_ts": now_utc(),
        "recovery_reason": reason,
        "original_ack_ts": ack_ts,
    }


def _backfill_fill_event(
    ack: Dict[str, Any],
    trades: List[Dict[str, Any]],
    exchange_status: str,
    recovery_reason: str,
) -> Optional[Dict[str, Any]]:
    """Construct and write a synthetic order_fill event. Returns payload or None."""
    executed_qty = 0.0
    cum_quote = 0.0
    fee_total = 0.0
    fee_asset: Optional[str] = None
    trade_ids: List[str] = []
    ts_first: Optional[str] = None
    ts_last: Optional[str] = None

    for trade in trades:
        trade_id = trade.get("id")
        if trade_id is None:
            continue
        trade_ids.append(str(trade_id))
        qty = to_float(trade.get("qty")) or 0.0
        price = to_float(trade.get("price")) or 0.0
        executed_qty += qty
        cum_quote += qty * price
        fee_total += to_float(trade.get("commission")) or 0.0
        fee_asset = fee_asset or (
            trade.get("commissionAsset") or trade.get("marginAsset") or "USDT"
        )
        trade_ts = ms_to_iso(trade.get("time"))
        if trade_ts:
            if ts_first is None or trade_ts < ts_first:
                ts_first = trade_ts
            if ts_last is None or trade_ts > ts_last:
                ts_last = trade_ts

    if not trade_ids:
        return None

    avg_price = (cum_quote / executed_qty) if executed_qty else None
    now = now_utc()

    payload: Dict[str, Any] = {
        "symbol": ack.get("symbol", ""),
        "side": ack.get("side", ""),
        "ts_fill_first": ts_first or now,
        "ts_fill_last": ts_last or now,
        "orderId": ack.get("orderId"),
        "clientOrderId": ack.get("clientOrderId"),
        "executedQty": executed_qty,
        "avgPrice": avg_price,
        "fee_total": fee_total,
        "feeAsset": fee_asset or "USDT",
        "tradeIds": sorted(trade_ids),
        "status": exchange_status,
        **_recovery_provenance(recovery_reason, ack_ts=ack.get("ts_ack")),
    }
    # Preserve linkage fields from the original ack
    for key in ("attempt_id", "intent_id", "strategy", "positionSide", "reduceOnly"):
        if key in ack:
            payload[key] = ack[key]

    try:
        write_event("order_fill", payload)
        LOG.info(
            "[reconcile] wrote recovery order_fill orderId=%s symbol=%s qty=%.6f reason=%s",
            ack.get("orderId"),
            ack.get("symbol"),
            executed_qty,
            recovery_reason,
        )
    except Exception as exc:
        LOG.warning("[reconcile] fill_write_failed orderId=%s err=%s", ack.get("orderId"), exc)
        return None
    return payload


def _backfill_close_event(
    ack: Dict[str, Any],
    trades: List[Dict[str, Any]],
    fill_payload: Dict[str, Any],
    recovery_reason: str,
) -> None:
    """Write a synthetic order_close if this order reduced or closed a position."""
    reduce_only = ack.get("reduceOnly") or ack.get("reduce_only", False)
    if not reduce_only:
        # Without position tracker state we cannot determine whether this was a
        # closing trade. Only emit close events for explicit reduceOnly orders
        # to avoid fabricating incorrect PnL attribution.
        return

    realized_pnl = sum(to_float(t.get("realizedPnl")) or 0.0 for t in trades)
    fees_total = sum(to_float(t.get("commission")) or 0.0 for t in trades)
    closed_qty = sum(to_float(t.get("qty")) or 0.0 for t in trades)

    payload: Dict[str, Any] = {
        "symbol": ack.get("symbol", ""),
        "ts_close": fill_payload.get("ts_fill_last") or now_utc(),
        "orderId": ack.get("orderId"),
        "clientOrderId": ack.get("clientOrderId"),
        "realizedPnlUsd": realized_pnl,
        "fees_total": fees_total,
        # Position sizes unavailable without tracker state; mark as unknown
        "position_size_before": None,
        "position_size_after": None,
        "closed_qty": closed_qty,
        "recovery_pnl_approximate": True,
        **_recovery_provenance(recovery_reason, ack_ts=ack.get("ts_ack")),
    }
    for key in ("attempt_id", "intent_id", "strategy", "positionSide"):
        if key in ack:
            payload[key] = ack[key]

    try:
        write_event("order_close", payload)
        LOG.info(
            "[reconcile] wrote recovery order_close orderId=%s symbol=%s pnl=%.4f",
            ack.get("orderId"),
            ack.get("symbol"),
            realized_pnl,
        )
    except Exception as exc:
        LOG.warning("[reconcile] close_write_failed orderId=%s err=%s", ack.get("orderId"), exc)


# ---------------------------------------------------------------------------
# Step 4: process one unresolved ack
# ---------------------------------------------------------------------------

def _process_one(order_id: int, ack: Dict[str, Any]) -> str:
    """
    Query exchange, classify, backfill as needed.
    Returns the case label for summary logging.
    """
    symbol = ack.get("symbol", "")
    client_order_id = ack.get("clientOrderId")

    order_status = fetch_order_status(symbol, order_id, client_order_id)
    trades = fetch_order_trades(symbol, order_id)
    exchange_status = normalize_status(order_status.get("status")) if order_status else "UNKNOWN"

    case, has_fills = _classify(order_status, trades)
    LOG.info(
        "[reconcile] orderId=%s symbol=%s exchange_status=%s case=%s trades=%d",
        order_id,
        symbol,
        exchange_status,
        case,
        len(trades),
    )

    if case == "full_fill_missing":
        fill_payload = _backfill_fill_event(ack, trades, exchange_status, "ack_without_fill_log")
        if fill_payload:
            _backfill_close_event(ack, trades, fill_payload, "ack_without_close_log")

    elif case in ("partial_fill_open", "partial_fill_terminal"):
        _backfill_fill_event(ack, trades, exchange_status, f"ack_with_{case}")

    elif case == "dead_no_fills":
        # Terminal with no fills — order died cleanly (CANCELED/EXPIRED/REJECTED
        # before any trade). No synthetic event needed, but log so operators can
        # spot orphaned ACKs in the startup log and rule out a history-query gap.
        LOG.warning(
            "[reconcile] dead_no_fills orderId=%s symbol=%s status=%s "
            "— verify exchange history is not truncated for this window",
            order_id,
            symbol,
            exchange_status,
        )

    # "open_no_fills": cancel_all_open_orders() (called by caller) will cancel it.

    return case


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconcile_unresolved_acks(
    log_path: str = _DEFAULT_LOG_PATH,
    lookback_seconds: float = _DEFAULT_LOOKBACK_S,
) -> Dict[str, Any]:
    """
    Full reconciliation pass. Call this at executor startup BEFORE cancel_all_open_orders.

    Returns a summary dict suitable for writing as a reconciliation_summary event.
    """
    start_ts = now_utc()
    unresolved = read_unresolved_acks(log_path=log_path, lookback_seconds=lookback_seconds)

    case_counts: Dict[str, int] = {}
    errors: List[str] = []

    for order_id, ack in unresolved.items():
        try:
            case = _process_one(order_id, ack)
            case_counts[case] = case_counts.get(case, 0) + 1
        except Exception as exc:
            LOG.warning("[reconcile] process_failed orderId=%s err=%s", order_id, exc, exc_info=True)
            errors.append(f"orderId={order_id} err={exc}")

    summary: Dict[str, Any] = {
        "event_type": "reconciliation_summary",
        "ts": start_ts,
        "lookback_seconds": lookback_seconds,
        "unresolved_count": len(unresolved),
        "case_counts": case_counts,
        "errors": errors,
        "recovery_source": _RECOVERY_SOURCE,
    }
    LOG.info(
        "[reconcile] done: %d unresolved, cases=%s, errors=%d",
        len(unresolved),
        case_counts,
        len(errors),
    )
    return summary
