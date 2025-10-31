from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Tuple

from execution import exchange_utils as ex
from execution.log_utils import get_logger, log_event, safe_dump

__all__ = ["route_order", "route_intent"]


_LOG = logging.getLogger("order_router")
LOG_ORDERS = get_logger("logs/execution/orders_executed.jsonl")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_side(intent: Mapping[str, Any]) -> str:
    raw = intent.get("side") or intent.get("signal")
    side = str(raw or "").upper()
    if side in ("BUY", "LONG"):
        return "BUY"
    if side in ("SELL", "SHORT"):
        return "SELL"
    raise ValueError(f"invalid side: {raw}")


def _as_str_quantity(value: Any) -> str:
    if value is None:
        raise ValueError("quantity missing")
    if isinstance(value, (int, float)):
        return f"{value}"
    if hasattr(value, "quantize"):
        return f"{value:f}"
    return str(value)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _slippage_bps(side: str, mark_px: float | None, fill_px: float | None) -> float | None:
    if mark_px is None or fill_px is None:
        return None
    if mark_px == 0:
        return None
    diff = (fill_px - mark_px) / mark_px * 10_000.0
    if side.upper() == "SELL":
        diff *= -1.0
    return diff


def route_order(intent: Mapping[str, Any], risk_ctx: Mapping[str, Any], dry_run: bool) -> Dict[str, Any]:
    """
    Normalize the intent payload and dispatch via exchange utils.

    Returns a structured result with keys:
        accepted: bool
        reason: str | None
        order_id: Any
        price: float | None
        qty: float | None
        raw: Dict[str, Any] (optional raw exchange response)
        request_id: str | None
        latency_ms: float | None
        exchange_filters_used: Dict[str, Any]
        rounded_qty: float | None
        rounded_price: float | None

    Raises:
        Exception: Propagated exchange error after logging.
    """

    try:
        side = _normalize_side(intent)
    except ValueError as exc:
        return {
            "accepted": False,
            "reason": str(exc),
            "order_id": None,
            "price": None,
            "qty": None,
            "raw": None,
        }

    symbol = str(intent.get("symbol") or intent.get("pair") or "").upper()
    if not symbol:
        return {
            "accepted": False,
            "reason": "missing_symbol",
            "order_id": None,
            "price": None,
            "qty": None,
            "raw": None,
        }

    qty = intent.get("quantity", intent.get("qty"))
    if qty is None:
        payload = risk_ctx.get("payload") or {}
        qty = payload.get("quantity")
    try:
        qty_str = _as_str_quantity(qty)
    except ValueError as exc:
        return {
            "accepted": False,
            "reason": str(exc),
            "order_id": None,
            "price": None,
            "qty": None,
            "raw": None,
        }

    price = risk_ctx.get("price") or intent.get("price")
    order_type = str(intent.get("type") or risk_ctx.get("type") or "MARKET").upper()
    position_side = intent.get("positionSide") or risk_ctx.get("positionSide")
    reduce_only = intent.get("reduceOnly", risk_ctx.get("reduceOnly"))

    payload = dict(risk_ctx.get("payload") or {})
    if not payload:
        payload = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": qty_str,
        }
        if price not in (None, "", 0, 0.0):
            payload["price"] = str(price)

    payload["symbol"] = symbol
    payload["side"] = side
    payload["type"] = order_type
    payload["quantity"] = qty_str
    if position_side:
        payload["positionSide"] = str(position_side).upper()
    if reduce_only is not None and _truthy(reduce_only):
        payload["reduceOnly"] = "true"
    elif "reduceOnly" in payload:
        payload.pop("reduceOnly", None)

    ex.set_dry_run(bool(dry_run))

    try:
        filters_snapshot = ex.get_symbol_filters(symbol)
    except Exception:
        filters_snapshot = {}

    latency_ms: float | None = None
    try:
        t0 = time.perf_counter()
        resp = ex.send_order(**payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        error_payload: Dict[str, Any] = {
            "exc": repr(exc),
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "price": _to_float(price),
            "qty": _to_float(qty),
            "payload": payload,
            "dry_run": bool(dry_run),
            "position_side": payload.get("positionSide"),
            "reduce_only": payload.get("reduceOnly"),
        }
        if filters_snapshot:
            error_payload["exchange_filters_used"] = filters_snapshot
        log_event(LOG_ORDERS, "order_error", safe_dump(error_payload))
        _LOG.error("route_order failed: %s", exc)
        raise

    order_id = resp.get("orderId")
    avg_price = resp.get("avgPrice") or price
    executed_qty = resp.get("executedQty") or resp.get("origQty") or qty
    reason = "dry_run" if resp.get("dryRun") else None
    request_id = (
        resp.get("clientOrderId")
        or payload.get("newClientOrderId")
        or payload.get("clientOrderId")
    )
    rounded_qty = _to_float(payload.get("quantity"))
    if rounded_qty is None:
        rounded_qty = _to_float(executed_qty)
    rounded_price = _to_float(payload.get("price"))
    if rounded_price is None:
        rounded_price = _to_float(avg_price)

    result: Dict[str, Any] = {
        "accepted": True,
        "reason": reason,
        "order_id": order_id,
        "price": _to_float(avg_price),
        "qty": _to_float(executed_qty),
        "raw": resp,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "exchange_filters_used": filters_snapshot,
        "rounded_qty": rounded_qty,
        "rounded_price": rounded_price,
    }
    log_event(LOG_ORDERS, "order_ack", safe_dump(result))
    return result


def route_intent(intent: Dict[str, Any], attempt_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Route an order intent and emit routing metrics."""
    router_ctx = dict(intent.get("router_ctx") or {})
    dry_run = bool(intent.get("dry_run", False))
    timing = dict(intent.get("timing") or {})

    base_intent: Dict[str, Any] = {
        key: value
        for key, value in intent.items()
        if key not in {"router_ctx", "dry_run", "timing", "attempt_id", "intent_id"}
    }

    retry_count = int(intent.get("retry_count", 0) or 0)
    mark_px = (
        _to_float(intent.get("mark_price"))
        or _to_float(router_ctx.get("mark_price"))
        or _to_float(router_ctx.get("price"))
        or _to_float(base_intent.get("price"))
    )
    submit_px = _to_float(base_intent.get("price")) or _to_float(router_ctx.get("price"))

    try:
        exchange_response = route_order(base_intent, router_ctx, dry_run)
    except Exception:
        router_metrics: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "venue": "binance_futures",
            "route": base_intent.get("route", "market"),
            "prices": {"mark": mark_px, "submitted": submit_px, "avg_fill": None},
            "qty": {"contracts": _to_float(base_intent.get("quantity")), "notional_usd": None},
            "timing_ms": {
                "decision": _to_float(timing.get("decision")),
                "submit": _to_float(timing.get("submit")),
                "ack": None,
                "fill": None,
            },
            "result": {"status": "rejected", "retries": retry_count, "cancelled": False},
            "fees_usd": None,
            "slippage_bps": None,
        }
        raise

    side = str(base_intent.get("side") or base_intent.get("signal") or "").upper()
    if not side:
        try:
            side = _normalize_side(base_intent)
        except ValueError:
            side = "BUY"

    avg_fill = exchange_response.get("price")
    qty_val = exchange_response.get("qty")
    if qty_val is None:
        qty_val = _to_float(base_intent.get("quantity"))
    qty_float = _to_float(qty_val)

    notional = None
    ref_price = mark_px or submit_px or _to_float(avg_fill)
    if ref_price is not None and qty_float is not None:
        notional = ref_price * qty_float

    raw = exchange_response.get("raw") or {}
    raw_status = str(raw.get("status") or "").lower()
    if raw_status in ("", "new"):
        status = "filled" if exchange_response.get("accepted") else "rejected"
    elif "partial" in raw_status:
        status = "partial"
    elif "cancel" in raw_status:
        status = "rejected"
    else:
        status = raw_status

    cancelled = "cancel" in raw_status
    fees = _to_float(raw.get("commission")) or _to_float(raw.get("cumQuote"))

    slippage = _slippage_bps(side, mark_px, _to_float(avg_fill))
    router_metrics = {
        "attempt_id": attempt_id,
        "venue": "binance_futures",
        "route": base_intent.get("route", "market"),
        "prices": {"mark": mark_px, "submitted": submit_px, "avg_fill": _to_float(avg_fill)},
        "qty": {"contracts": qty_float, "notional_usd": notional},
        "timing_ms": {
            "decision": _to_float(timing.get("decision")),
            "submit": _to_float(timing.get("submit")),
            "ack": exchange_response.get("latency_ms"),
            "fill": _to_float(timing.get("fill")),
        },
        "result": {
            "status": status,
            "retries": retry_count,
            "cancelled": bool(cancelled),
        },
        "fees_usd": fees,
        "slippage_bps": slippage,
    }
    return exchange_response, router_metrics
