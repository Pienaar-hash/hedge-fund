from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

from execution.log_utils import get_logger, log_event, safe_dump

__all__ = ["now_utc", "write_event", "validate_event", "event_logger"]


_LOG = logging.getLogger("execution.events")
_DEFAULT_EVENT_PATH = os.getenv("ORDER_EVENTS_PATH") or "logs/execution/orders_executed.jsonl"
_EVENT_LOGGER = get_logger(_DEFAULT_EVENT_PATH)

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
