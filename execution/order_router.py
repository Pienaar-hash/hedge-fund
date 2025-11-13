"""
v5.9 Execution Hardening — Router upgrades
- Maker-first POST_ONLY with smart fallback
- Fee-aware effective price calculator
- Child-order aggregation by min-notional
- Auto-cancel/refresh on low fill ratio
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import requests

from execution import exchange_utils as ex
from execution.log_utils import get_logger, log_event, safe_dump

try:  # optional dependency
    import yaml
except Exception:  # pragma: no cover - best-effort fallback when PyYAML absent
    yaml = None  # type: ignore[assignment]

__all__ = [
    "route_order",
    "route_intent",
    "is_ack_ok",
    "PlaceOrderResult",
    "effective_px",
    "chunk_qty",
    "submit_limit",
    "monitor_and_refresh",
]

def _load_runtime_cfg() -> Dict[str, Any]:
    path = Path(os.getenv("RUNTIME_CONFIG") or "config/runtime.yaml")
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_RUNTIME_CFG = _load_runtime_cfg()
_TRADING_WINDOW = _RUNTIME_CFG.get("trading_window") or {}
_OFFPEAK_CFG = _RUNTIME_CFG.get("offpeak") or {}
_PRIORITY_CFG = _RUNTIME_CFG.get("priority") or {}
_FEES_CFG = (
    _RUNTIME_CFG.get("fees")
    or (_RUNTIME_CFG.get("execution") or {}).get("fees")
    or {}
)


def _runtime_flag(key: str, default: Any) -> Any:
    value = _RUNTIME_CFG.get(key)
    return value if value is not None else default


def _min_child_from_runtime(default: float) -> float:
    candidates: list[float] = []
    for section in (_OFFPEAK_CFG, _PRIORITY_CFG):
        try:
            val = float(section.get("min_child_notional", 0.0))
        except (TypeError, ValueError):
            continue
        if val > 0:
            candidates.append(val)
    return min(candidates) if candidates else default


def _fee_from_runtime(key: str, env_key: str, default: float) -> float:
    env_val = os.getenv(env_key)
    if env_val is not None:
        try:
            return float(env_val)
        except (TypeError, ValueError):
            pass
    cfg_val = _FEES_CFG.get(key)
    if cfg_val is not None:
        try:
            return float(cfg_val)
        except (TypeError, ValueError):
            pass
    return default


# --- Tunables sourced from runtime.yaml / env overrides ---
POST_ONLY_DEFAULT = bool(_runtime_flag("post_only_default", True))
SLIP_MAX_BPS = int(_runtime_flag("router_slip_max_bps", 3))  # switch to taker if mid drifts beyond this
REJECTS_MAX = int(_runtime_flag("router_rejects_max", 2))  # post-only rejects before fallback
MIN_CHILD_NOTIONAL = _min_child_from_runtime(30.0)  # USDT per child
LOW_FILL_WINDOW_S = int(_runtime_flag("low_fill_window_s", 60))
MIN_FILL_RATIO = float(_runtime_flag("min_fill_ratio", 0.40))

# Exchange fee tier (bps). Negative for maker rebates.
TAKER_BPS = _fee_from_runtime("taker_bps", "TAKER_FEE_BPS", 5.0)
MAKER_BPS = _fee_from_runtime("maker_bps", "MAKER_FEE_BPS", -1.0)

# expose trading window metadata for downstream schedulers
TRADING_WINDOW = _TRADING_WINDOW
OFFPEAK_CFG = _OFFPEAK_CFG
PRIORITY_CFG = _PRIORITY_CFG


@dataclass
class PlaceOrderResult:
    order_id: str
    side: str
    price: float | None
    qty: float
    is_maker: bool
    rejected_post_only: bool = False
    rejections: int = 0
    slippage_bps: float = 0.0
    placed_ts: float = field(default_factory=time.time)
    filled_qty: float = 0.0
    raw: Dict[str, Any] | None = None


def _bps(a: float | None, b: float | None) -> float:
    if a is None or b in (None, 0):
        return 0.0
    return (a - b) / b * 1e4


def effective_px(px: float | None, side: str, is_maker: bool = True) -> float | None:
    """Price adjusted for fees/rebates from the strategy's perspective."""
    if px is None:
        return None
    bps = MAKER_BPS if is_maker else TAKER_BPS
    adj = (bps / 1e4) * px
    return px + adj if side.upper() == "BUY" else px - adj


def MAX(a: float, b: float) -> float:
    """Tiny local helper to avoid importing math for a single call."""
    return a if a > b else b


def chunk_qty(total_qty: float, px: float) -> list[float]:
    """Split into children ensuring min notional."""
    if total_qty <= 0:
        return []
    if px <= 0:
        return [total_qty]
    min_child = MAX(1.0, MIN_CHILD_NOTIONAL)
    chunks = max(1, int((total_qty * px) // min_child))
    return [total_qty / chunks] * chunks


def place_order(
    symbol: str,
    side: str,
    order_type: str,
    price: float | None,
    qty: float,
    flags: Mapping[str, Any] | None = None,
) -> PlaceOrderResult:
    """
    Thin wrapper around exchange client (existing implementation below).
    This function is expected to:
      - honor postOnly if flags={"postOnly": True}
      - set rejected_post_only=True if post-only would cross
      - populate order_id and partial fills via polling hook elsewhere
    """
    flags = dict(flags or {})
    payload = {
        "symbol": symbol.upper(),
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": _as_str_quantity(qty),
    }
    if price not in (None, "", 0, 0.0) and payload["type"] != "MARKET":
        payload["price"] = str(price)
    payload.update(flags)
    if flags.get("postOnly") and "timeInForce" not in payload:
        payload["timeInForce"] = "GTX"

    rejection_result = PlaceOrderResult(
        order_id="",
        side=payload["side"],
        price=_to_float(price),
        qty=float(qty),
        is_maker=bool(flags.get("postOnly")),
        rejected_post_only=True,
        rejections=1,
    )

    try:
        resp = ex.send_order(**payload)
    except requests.HTTPError as exc:
        if flags.get("postOnly") and exc.response is not None:
            try:
                body = exc.response.json()
                msg = str(body.get("msg", "")).lower()
                if body.get("code") == -2010 or "immediately match" in msg:
                    return rejection_result
            except Exception:
                pass
        raise

    avg_px = _to_float(resp.get("avgPrice")) or _to_float(price)
    executed_qty = _to_float(resp.get("executedQty")) or 0.0
    orig_qty = _to_float(resp.get("origQty")) or float(qty)
    status = _normalize_status(resp.get("status"))
    was_post_only = bool(flags.get("postOnly"))
    rejected_post_only = was_post_only and status in {"REJECTED", "EXPIRED"}

    result = PlaceOrderResult(
        order_id=str(resp.get("orderId") or ""),
        side=payload["side"],
        price=avg_px,
        qty=orig_qty,
        is_maker=bool(resp.get("maker")) if "maker" in resp else was_post_only,
        rejected_post_only=rejected_post_only,
        filled_qty=executed_qty,
        slippage_bps=_bps(avg_px, _to_float(price)) if price is not None else 0.0,
        raw=resp,
    )
    return result


def submit_limit(
    symbol: str,
    px: float,
    qty: float,
    side: str,
    post_only: bool = POST_ONLY_DEFAULT,
    prev: PlaceOrderResult | None = None,
    place_func: Callable[..., PlaceOrderResult] | None = None,
) -> PlaceOrderResult:
    """Maker-first with bounded smart fallback."""
    place_cb = place_func or place_order
    flags = {"postOnly": post_only} if post_only else {}
    result = place_cb(symbol, side, "LIMIT", px, qty, flags)
    prior_rejections = prev.rejections if prev else 0
    prior_rejections = max(0, prior_rejections or 0)
    current_rejections = max(0, result.rejections or 0)
    if result.rejected_post_only:
        result.rejections = max(prior_rejections, current_rejections) + 1
    else:
        result.rejections = max(prior_rejections, current_rejections)

    slip = result.slippage_bps or 0.0
    if (result.rejections >= REJECTS_MAX) or (slip > SLIP_MAX_BPS):
        return place_cb(symbol, side, "MARKET", None, qty, flags={"postOnly": False})
    return result


def monitor_and_refresh(
    order: PlaceOrderResult,
    get_state: Callable[[str], Any],
    cancel: Callable[[str], Any],
    reprice_wider: Callable[[PlaceOrderResult], Any],
    now: float | None = None,
) -> None:
    """
    Generic watcher: if order is live longer than LOW_FILL_WINDOW_S and
    fill ratio < MIN_FILL_RATIO -> cancel and reprice.
    """
    if not order or not order.order_id:
        return
    state = get_state(order.order_id)
    if not state:
        return
    placed_ts = order.placed_ts or 0.0
    age = (now or time.time()) - placed_ts

    filled_qty = getattr(state, "filled_qty", None)
    if filled_qty is None and isinstance(state, Mapping):
        filled_qty = state.get("filled_qty") or state.get("executedQty")
    try:
        filled_qty = float(filled_qty or 0.0)
    except (TypeError, ValueError):
        filled_qty = 0.0

    denom = order.qty if order.qty not in (None, 0) else 1e-12
    fill_ratio = filled_qty / denom

    if age > LOW_FILL_WINDOW_S and fill_ratio < MIN_FILL_RATIO:
        cancel(order.order_id)
        reprice_wider(order)
        return

_LOG = logging.getLogger("order_router")
LOG_ORDERS = get_logger("logs/execution/orders_executed.jsonl")

_ACK_OK_STATUSES = {"NEW", "PARTIALLY_FILLED"}


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


def _normalize_status(status: Any) -> str:
    if not status:
        return "UNKNOWN"
    try:
        normalized = str(status).upper()
        if normalized == "CANCELLED":  # handle alternative spelling
            return "CANCELED"
        return normalized
    except Exception:
        return "UNKNOWN"


def is_ack_ok(status: Any) -> bool:
    """Return True when an ACK status represents an accepted order."""
    normalized = _normalize_status(status)
    return normalized in _ACK_OK_STATUSES


def route_order(intent: Mapping[str, Any], risk_ctx: Mapping[str, Any], dry_run: bool) -> Dict[str, Any]:
    """
    Normalize the intent payload and dispatch via exchange utils.

    Returns a structured result with keys:
        accepted: bool
        reason: str | None
        order_id: Any
        status: str
        price: float | None (only present when exchange reported executedQty > 0)
        qty: float | None (only present when exchange reported executedQty > 0)
        raw: Dict[str, Any] (optional raw exchange response)
        request_id: str | None
        client_order_id: str | None
        transact_time: Any | None
        latency_ms: float | None
        exchange_filters_used: Dict[str, Any]

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

    risk_ctx = dict(risk_ctx or {})
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

    maker_qty = _to_float(risk_ctx.get("maker_qty"))
    maker_price = _to_float(risk_ctx.get("maker_price") or price)
    maker_enabled = (
        bool(risk_ctx.get("maker_first"))
        and not bool(payload.get("reduceOnly"))
        and maker_qty is not None
        and maker_qty > 0
        and maker_price is not None
        and maker_price > 0
    )

    latency_ms: float | None = None
    resp: Dict[str, Any] | None = None
    if maker_enabled:
        try:
            t0 = time.perf_counter()
            maker_px = effective_px(maker_price, side, is_maker=True) or maker_price
            maker_result = submit_limit(symbol, maker_px, maker_qty, side)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            maker_result = None
            _LOG.warning("maker_first_failed symbol=%s err=%s", symbol, exc)
        else:
            resp = maker_result.raw or {
                "orderId": maker_result.order_id,
                "status": "FILLED" if maker_result.filled_qty else "NEW",
                "avgPrice": maker_result.price,
                "executedQty": maker_result.filled_qty,
                "origQty": maker_result.qty,
            }
            resp["maker_result"] = {
                "is_maker": maker_result.is_maker,
                "rejections": maker_result.rejections,
                "slippage_bps": maker_result.slippage_bps,
            }

    if resp is None:
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
    status = _normalize_status(resp.get("status"))
    executed_qty = resp.get("executedQty")
    avg_price = resp.get("avgPrice")
    reason = "dry_run" if resp.get("dryRun") else None
    request_id = (
        resp.get("clientOrderId")
        or payload.get("newClientOrderId")
        or payload.get("clientOrderId")
    )
    executed_qty_float = _to_float(executed_qty)
    if executed_qty_float is not None and executed_qty_float <= 0.0:
        executed_qty_float = None
    avg_price_float = _to_float(avg_price)
    if avg_price_float is not None and avg_price_float <= 0.0:
        avg_price_float = None

    accepted = is_ack_ok(status) or bool(resp.get("dryRun"))

    result: Dict[str, Any] = {
        "accepted": accepted,
        "reason": reason,
        "order_id": order_id,
        "status": status,
        "price": avg_price_float,
        "qty": executed_qty_float,
        "raw": resp,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "exchange_filters_used": filters_snapshot,
        "client_order_id": request_id,
        "transact_time": resp.get("transactTime"),
    }
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
    raw_status = _normalize_status(raw.get("status"))
    status = raw_status if raw_status != "UNKNOWN" else ("ACCEPTED" if exchange_response.get("accepted") else "REJECTED")

    cancelled = raw_status in {"CANCELED", "CANCELLED"}
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
