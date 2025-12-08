from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

from execution.log_utils import get_logger, log_event, safe_dump

__all__ = ["now_utc", "write_event", "validate_event", "event_logger", "write_twap_event"]


_LOG = logging.getLogger("execution.events")
_DEFAULT_EVENT_PATH = os.getenv("ORDER_EVENTS_PATH") or "logs/execution/orders_executed.jsonl"
_EVENT_LOGGER = get_logger(_DEFAULT_EVENT_PATH)

# TWAP events logger (v7.4 C1)
_TWAP_EVENT_PATH = os.getenv("TWAP_EVENTS_PATH") or "logs/execution/twap_events.jsonl"
_TWAP_LOGGER = get_logger(_TWAP_EVENT_PATH)

_REQUIRED_FIELDS = {
    "order_ack": {"symbol", "side", "ts_ack", "orderId", "clientOrderId", "request_qty", "order_type", "status"},
    "order_fill": {
        "symbol",
        "side",
        "ts_fill_first",
        "ts_fill_last",
        "orderId",
        "clientOrderId",
        "executedQty",
        "avgPrice",
        "fee_total",
        "feeAsset",
        "tradeIds",
        "status",
    },
    "order_close": {
        "symbol",
        "ts_close",
        "orderId",
        "clientOrderId",
        "realizedPnlUsd",
        "fees_total",
        "position_size_before",
        "position_size_after",
    },
}


def now_utc() -> str:
    """Return current UTC timestamp as ISO8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def event_logger() -> Any:
    """Expose the default logger so callers can override if desired."""
    return _EVENT_LOGGER


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


def validate_event(event_type: str, payload: Mapping[str, Any]) -> None:
    """Validate that required keys for an event are present."""
    required = _REQUIRED_FIELDS.get(event_type)
    if not required:
        return
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{event_type} missing fields: {', '.join(sorted(missing))}")
    if event_type == "order_fill":
        trade_ids = payload.get("tradeIds")
        if trade_ids is not None and not _is_sequence(trade_ids):
            raise ValueError("order_fill tradeIds must be a sequence")


def write_event(event_type: str, payload: Mapping[str, Any], *, logger: Any | None = None) -> None:
    """Write a validated event to the shared execution log."""
    target_logger = logger or _EVENT_LOGGER
    body: MutableMapping[str, Any] = safe_dump(payload or {})
    body["event_type"] = event_type
    try:
        validate_event(event_type, body)
    except Exception as exc:  # pragma: no cover - validation failures should be rare
        _LOG.warning("skip_event invalid=%s error=%s payload=%s", event_type, exc, payload)
        return
    try:
        log_event(target_logger, event_type, body)
    except Exception as exc:
        _LOG.warning("event_write_failed type=%s err=%s", event_type, exc)


# ---------------------------------------------------------------------------
# TWAP Events (v7.4 C1)
# ---------------------------------------------------------------------------

def write_twap_event(
    event_type: str,
    symbol: str,
    side: str,
    slice_index: int,
    slice_count: int,
    slice_qty: float,
    parent_gross_usd: float,
    twap_cfg: Mapping[str, Any],
    order_result: Mapping[str, Any] | None = None,
) -> None:
    """
    Write a TWAP execution event with standardized structure.
    
    Args:
        event_type: Event type (e.g., 'twap_slice_complete', 'twap_start')
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        slice_index: Index of this slice (0-based)
        slice_count: Total number of slices
        slice_qty: Quantity for this slice
        parent_gross_usd: Total gross USD of parent order
        twap_cfg: TWAP configuration dict
        order_result: Optional order result for completed slices
    """
    body: MutableMapping[str, Any] = {
        "event_type": event_type,
        "execution_style": "twap",
        "symbol": symbol,
        "side": side,
        "twap": {
            "slice_index": slice_index,
            "slice_count": slice_count,
            "slice_qty": slice_qty,
            "parent_gross_usd": parent_gross_usd,
            "twap_cfg": {
                "min_notional_usd": twap_cfg.get("min_notional_usd", 0),
                "slices": twap_cfg.get("slices", 1),
                "interval_seconds": twap_cfg.get("interval_seconds", 0),
            },
        },
        "ts": now_utc(),
    }
    
    if order_result:
        body["order_id"] = order_result.get("order_id", "")
        body["status"] = order_result.get("status", "UNKNOWN")
        body["filled_qty"] = order_result.get("filled_qty", 0)
        body["avg_price"] = order_result.get("avg_price")
        body["is_maker"] = order_result.get("is_maker", False)
        body["slippage_bps"] = order_result.get("slippage_bps", 0)
    
    try:
        log_event(_TWAP_LOGGER, event_type, body)
    except Exception as exc:
        _LOG.warning("twap_event_write_failed type=%s err=%s", event_type, exc)
