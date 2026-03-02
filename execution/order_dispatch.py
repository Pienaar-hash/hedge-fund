"""
Order dispatch functions extracted from ``executor_live._send_order``.

All business logic here is **identical** to the original closures.  No
imports from ``executor_live`` — every dependency is injected via explicit
parameters or the :class:`DispatchRetryContext` dataclass.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

__all__ = [
    "DispatchRetryContext",
    "dispatch_to_exchange",
    "dispatch_with_retry",
    "attempt_maker_first",
    "build_maker_metrics",
    "meta_float",
]

LOG = logging.getLogger("executor")


# ── Utilities ─────────────────────────────────────────────────────────


def meta_float(val: Any, fallback: float) -> float:
    """Safely coerce *val* to float, returning *fallback* on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


# ── Dispatch to exchange ──────────────────────────────────────────────


def dispatch_to_exchange(
    payload: Dict[str, Any],
    positions: List[Dict[str, Any]],
    send_fn: Callable[..., Dict[str, Any]],
    close_position_fn: Callable[..., tuple],
) -> Dict[str, Any]:
    """Send an order to the exchange, converting to closePosition when needed.

    Extracted from the ``_dispatch`` closure inside ``_send_order``.
    """
    request_payload = dict(payload)
    order_type = str(request_payload.get("type") or "MARKET").upper()
    convert_close, _close_qty = close_position_fn(
        request_payload.get("symbol"),
        request_payload.get("side"),
        request_payload.get("positionSide"),
        request_payload.get("reduceOnly"),
        order_type=order_type,
        positions=positions,
    )
    if convert_close and order_type != "MARKET":
        request_payload.pop("reduceOnly", None)
        request_payload.pop("quantity", None)
        request_payload["closePosition"] = True
    ro_val = request_payload.get("reduceOnly")
    if isinstance(ro_val, str):
        ro_val = ro_val.lower() in ("1", "true", "yes", "on")
    return send_fn(
        symbol=request_payload["symbol"],
        side=request_payload["side"],
        type=request_payload.get("type", "MARKET"),
        quantity=request_payload.get("quantity"),
        positionSide=request_payload.get("positionSide"),
        reduceOnly=ro_val,
        price=request_payload.get("price"),
        closePosition=request_payload.get("closePosition"),
        timeInForce=request_payload.get("timeInForce"),
        newClientOrderId=request_payload.get("newClientOrderId"),
        positions=positions,
    )


# ── Maker-first attempt ──────────────────────────────────────────────


def attempt_maker_first(
    px: float,
    qty: float,
    symbol: str,
    side: str,
    *,
    submit_limit_fn: Optional[Callable] = None,
    effective_px_fn: Optional[Callable] = None,
) -> Optional[Any]:
    """Try a POST_ONLY maker-first limit order.

    Returns a ``PlaceOrderResult`` on success, ``None`` on failure or
    when router functions are unavailable.
    """
    if submit_limit_fn is None or effective_px_fn is None:
        return None
    if px <= 0 or qty <= 0:
        return None
    try:
        post_px = effective_px_fn(px, side, is_maker=True) or px
        return submit_limit_fn(symbol, post_px, qty, side)
    except Exception as exc:
        LOG.warning("[executor] maker_first_failed symbol=%s err=%s", symbol, exc)
        return None


# ── Maker metrics builder ────────────────────────────────────────────


def build_maker_metrics(
    result: Any,
    attempt_id: str,
    price_hint: float,
    decision_latency_ms: float,
) -> Dict[str, Any]:
    """Build router-style metrics dict from a ``PlaceOrderResult``."""
    avg_fill = result.price if result.price is not None else None
    return {
        "attempt_id": attempt_id,
        "venue": "binance_futures",
        "route": "maker_first",
        "prices": {
            "mark": price_hint,
            "submitted": result.price,
            "avg_fill": avg_fill,
        },
        "qty": {
            "contracts": result.filled_qty or result.qty,
            "notional_usd": (
                (avg_fill or price_hint)
                * (result.filled_qty or result.qty or 0.0)
            )
            if (avg_fill or price_hint)
            else None,
        },
        "timing_ms": {
            "decision": decision_latency_ms,
            "submit": None,
            "ack": None,
            "fill": None,
        },
        "result": {
            "status": "FILLED" if (result.filled_qty or 0.0) > 0 else "NEW",
            "retries": result.rejections,
            "cancelled": False,
        },
        "fees_usd": None,
        "slippage_bps": result.slippage_bps,
    }


# ── Dispatch retry context ───────────────────────────────────────────


@dataclass
class DispatchRetryContext:
    """All parameters needed by :func:`dispatch_with_retry`.

    Module-level globals from ``executor_live`` are injected here so that
    ``order_dispatch`` never imports from ``executor_live``.
    """

    # Order identity
    symbol: str
    side: str
    pos_side: str
    gross_target: float

    # Observability context
    meta: Dict[str, Any]
    payload_view: Dict[str, Any]
    normalized_ctx: Dict[str, Any]

    # Retry config
    max_retries: int
    retry_backoff_s: float

    # Injected callbacks — no executor_live imports
    note_error_fn: Callable          # _RISK_STATE.note_error
    log_order_error_fn: Callable     # _log_order_error
    publish_audit_fn: Callable       # publish_order_audit
    classify_error_fn: Callable      # classify_binance_error


def dispatch_with_retry(
    dispatch_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
    ctx: DispatchRetryContext,
) -> Optional[Dict[str, Any]]:
    """Execute *dispatch_fn* with transient-error retry.

    Returns
    -------
    dict
        Exchange response on success.
    None
        Signals caller to abort (e.g. precision error ``-1111``).

    Raises
    ------
    requests.HTTPError
        Non-retriable HTTP error.
    Exception
        Any other dispatch error.
    """
    assert ctx.symbol, "DispatchRetryContext.symbol must be non-empty"
    assert ctx.side in ("BUY", "SELL"), (
        f"DispatchRetryContext.side must be BUY or SELL, got {ctx.side!r}"
    )

    dispatch_attempt = 0
    while True:
        try:
            return dispatch_fn(payload)
        except requests.HTTPError as exc:
            dispatch_attempt += 1
            try:
                ctx.note_error_fn(time.time())
            except Exception:
                LOG.warning(
                    "[RISK_NOTE_ERROR] note_error failed for %s (http)",
                    ctx.symbol, exc_info=True,
                )
            err_code = None
            try:
                if exc.response is not None:
                    err_code = exc.response.json().get("code")
            except Exception:
                err_code = None
            classification = ctx.classify_error_fn(
                exc, getattr(exc, "response", None),
            )
            LOG.error(
                "[executor] ORDER_ERR code=%s symbol=%s "
                "side=%s meta=%s payload=%s err=%s",
                err_code,
                ctx.symbol,
                ctx.side,
                ctx.meta,
                ctx.payload_view,
                exc,
            )
            retriable = (
                bool(classification.get("retriable"))
                and dispatch_attempt <= ctx.max_retries
            )
            ctx.log_order_error_fn(
                symbol=ctx.symbol,
                side=ctx.side,
                notional=ctx.gross_target,
                reason="http_error",
                classification=classification,
                retried=retriable,
                exc=exc,
                component="exchange",
                context={
                    "code": err_code,
                    "payload": ctx.payload_view,
                    "attempt": dispatch_attempt,
                },
            )
            try:
                ctx.publish_audit_fn(
                    ctx.symbol,
                    {
                        "phase": "error",
                        "side": ctx.side,
                        "positionSide": ctx.pos_side,
                        "error": str(exc),
                        "code": err_code,
                        "normalized": ctx.normalized_ctx,
                        "payload": ctx.payload_view,
                    },
                )
            except Exception:
                pass
            if err_code == -1111:
                LOG.error(
                    "[executor] ORDER_PRECISION ctx=%s payload=%s",
                    ctx.normalized_ctx,
                    ctx.payload_view,
                )
                return None
            if retriable:
                time.sleep(ctx.retry_backoff_s)
                continue
            raise
        except Exception as exc:
            try:
                ctx.note_error_fn(time.time())
            except Exception:
                LOG.warning(
                    "[RISK_NOTE_ERROR] note_error failed for %s (generic)",
                    ctx.symbol, exc_info=True,
                )
            ctx.log_order_error_fn(
                symbol=ctx.symbol,
                side=ctx.side,
                notional=ctx.gross_target,
                reason="dispatch_error",
                classification=None,
                retried=False,
                exc=exc,
                component="executor",
            )
            try:
                ctx.publish_audit_fn(
                    ctx.symbol,
                    {
                        "phase": "error",
                        "side": ctx.side,
                        "positionSide": ctx.pos_side,
                        "error": str(exc),
                    },
                )
            except Exception:
                LOG.warning(
                    "[DISPATCH_ERROR_AUDIT] publish_order_audit failed for %s",
                    ctx.symbol, exc_info=True,
                )
            raise
