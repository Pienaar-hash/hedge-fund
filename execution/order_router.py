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
from typing import Any, Callable, Dict, Mapping, Tuple

import requests

from execution import exchange_utils as ex
from execution.log_utils import get_logger, log_event, safe_dump
from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.intel.router_policy import router_policy, RouterPolicy
from execution.intel.router_autotune_apply_v6 import (
    APPLY_ENABLED as AUTOTUNE_APPLY_ENABLED,
    get_symbol_suggestion,
    get_current_risk_mode,
    apply_router_suggestion,
)
from execution.runtime_config import load_runtime_config
from execution.utils.execution_health import record_execution_error

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

_RUNTIME_CFG = load_runtime_config()
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
SLIP_MAX_BPS = int(_runtime_flag("router_slip_max_bps", 5))  # v6.4: increased from 3 - switch to taker if mid drifts beyond this
REJECTS_MAX = int(_runtime_flag("router_rejects_max", 4))  # v6.4: increased from 2 - post-only rejects before fallback
MIN_CHILD_NOTIONAL = _min_child_from_runtime(30.0)  # USDT per child
LOW_FILL_WINDOW_S = int(_runtime_flag("low_fill_window_s", 60))
MIN_FILL_RATIO = float(_runtime_flag("min_fill_ratio", 0.40))
MAX_SPREAD_FOR_MAKER_BPS = float(_runtime_flag("router_max_spread_bps", 12.0))
WIDE_SPREAD_OFFSET_CLAMP_BPS = float(_runtime_flag("router_offset_spread_clamp_bps", 6.0))

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


def _apply_offset(mid: float, bps: float, side: str) -> float:
    """
    Apply a signed bps offset relative to mid.
    BUY -> quote below mid, SELL -> quote above mid.
    """
    if mid <= 0:
        return mid
    try:
        side_norm = side.upper()
    except Exception:
        side_norm = "BUY"
    delta = (bps / 10_000.0) * mid
    if side_norm == "SELL":
        return mid + delta
    return mid - delta


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
    status_raw = getattr(state, "status", None)
    if status_raw is None and isinstance(state, Mapping):
        status_raw = state.get("status")
    if status_raw:
        status_norm = _normalize_status(status_raw)
        if status_norm in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
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
LOG_ROUTER_DECISIONS = get_logger("logs/execution/router_decisions.jsonl")

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

    ctx_original = risk_ctx if isinstance(risk_ctx, dict) else None
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

    is_market_close = order_type == "MARKET" and bool(payload.get("reduceOnly"))
    if is_market_close:
        pos_qty = _to_float(risk_ctx.get("pos_qty"))
        try:
            order_qty = float(qty)
        except Exception:
            order_qty = pos_qty
        testnet_flag = str(os.getenv("BINANCE_TESTNET", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if testnet_flag and pos_qty is not None and order_qty is not None and order_qty == pos_qty:
            reduce_only = False
            payload.pop("reduceOnly", None)
            _LOG.info("[router][testnet-guard] Disabling reduceOnly full-close")

    ex.set_dry_run(bool(dry_run))

    try:
        filters_snapshot = ex.get_symbol_filters(symbol)
    except Exception:
        filters_snapshot = {}

    spread_bps = _to_float(
        risk_ctx.get("spread_bps")
        or intent.get("spread_bps")
        or risk_ctx.get("book_spread_bps")
    )
    maker_qty = _to_float(risk_ctx.get("maker_qty"))
    maker_price = _to_float(risk_ctx.get("maker_price") or price)
    policy = router_policy(symbol)
    try:
        base_offset_bps = float(policy.offset_bps) if policy.offset_bps is not None else float(suggest_maker_offset_bps(symbol))
    except Exception:
        base_offset_bps = 2.0
    policy_snapshot = {
        "maker_first": policy.maker_first,
        "taker_bias": policy.taker_bias,
        "quality": policy.quality,
        "reason": policy.reason,
        "offset_bps": base_offset_bps,
    }
    router_decision: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "route": None,
        "maker_requested": bool(risk_ctx.get("maker_first")),
        "policy_quality": policy.quality,
        "policy_maker_first": policy.maker_first,
        "policy_taker_bias": policy.taker_bias,
        "offset_bps": base_offset_bps,
        "spread_bps": spread_bps,
        "reasons": [],
    }
    policy_before_snapshot = dict(policy_snapshot)
    policy_after_snapshot = dict(policy_snapshot)
    adjusted_offset_bps = base_offset_bps
    risk_mode = "normal"
    autotune_applied = False
    if AUTOTUNE_APPLY_ENABLED:
        suggestion = get_symbol_suggestion(symbol)
        risk_mode = get_current_risk_mode()
        new_policy_dict, applied, new_offset = apply_router_suggestion(
            policy_snapshot,
            suggestion=suggestion,
            symbol=symbol,
            risk_mode=risk_mode,
            current_offset_bps=adjusted_offset_bps,
        )
        if applied:
            autotune_applied = True
            adjusted_offset_bps = new_offset
            policy_after_snapshot.update(
                {
                    "maker_first": new_policy_dict.get("maker_first", policy.maker_first),
                    "taker_bias": new_policy_dict.get("taker_bias", policy.taker_bias),
                    "offset_bps": adjusted_offset_bps,
                }
            )
            policy = RouterPolicy(
                maker_first=policy_after_snapshot["maker_first"],
                taker_bias=policy_after_snapshot["taker_bias"],
                quality=policy.quality,
                reason=policy.reason,
                offset_bps=adjusted_offset_bps,
            )
    router_decision["offset_bps"] = adjusted_offset_bps
    target_ctx = ctx_original if isinstance(ctx_original, dict) else risk_ctx
    target_ctx["router_policy"] = policy_after_snapshot
    target_ctx["autotune"] = {
        "applied": autotune_applied,
        "before": policy_before_snapshot,
        "after": policy_after_snapshot,
        "risk_mode": risk_mode,
    }
    taker_bias = str(getattr(policy, "taker_bias", "") or "").lower()
    prefer_taker_bias = taker_bias == "prefer_taker"
    reduce_only_flag = bool(payload.get("reduceOnly"))
    maker_enabled = True
    if not router_decision["maker_requested"]:
        maker_enabled = False
        router_decision["reasons"].append("maker_not_requested")
    if not policy.maker_first:
        maker_enabled = False
        router_decision["reasons"].append("policy_maker_disabled")
    # v6.5: Allow maker orders for "ok", "good", and "degraded" (for recovery)
    # Only "broken" quality should completely disable maker attempts
    # This enables maker attempts during bootstrap and recovery phases
    if policy.quality == "broken":
        maker_enabled = False
        router_decision["reasons"].append("policy_quality_broken")
    # v6.5: Only disable maker for explicit "prefer_taker" bias
    # "balanced" bias should still allow maker attempts for recovery
    if prefer_taker_bias:
        maker_enabled = False
        router_decision["reasons"].append("policy_bias_prefers_taker")
    # v6.4: Allow maker for reduce-only exits unless it's an urgent stop-loss
    # Take-profit exits can afford to wait for maker fills
    is_urgent_exit = bool(intent.get("is_stop_loss") or intent.get("urgent") or risk_ctx.get("is_stop_loss"))
    if reduce_only_flag and is_urgent_exit:
        maker_enabled = False
        router_decision["reasons"].append("reduce_only_urgent")
    elif reduce_only_flag:
        # Allow maker for non-urgent exits (take-profit)
        router_decision["reasons"].append("reduce_only_maker_allowed")
    if maker_qty is None or maker_qty <= 0:
        maker_enabled = False
        router_decision["reasons"].append("missing_maker_qty")
    if maker_price is None or maker_price <= 0:
        maker_enabled = False
        router_decision["reasons"].append("missing_maker_price")
    if spread_bps is not None and spread_bps > MAX_SPREAD_FOR_MAKER_BPS:
        maker_enabled = False
        router_decision["reasons"].append("spread_too_wide")
    router_decision["maker_allowed"] = maker_enabled
    router_decision["maker_qty"] = maker_qty
    router_decision["maker_price"] = maker_price

    spread_clamped = False
    if maker_enabled and spread_bps is not None and spread_bps > WIDE_SPREAD_OFFSET_CLAMP_BPS:
        adjusted_offset_bps = min(adjusted_offset_bps, WIDE_SPREAD_OFFSET_CLAMP_BPS)
        router_decision["reasons"].append("wide_spread_clamped")
        spread_clamped = True
        router_decision["offset_bps"] = adjusted_offset_bps

    latency_ms: float | None = None
    resp: Dict[str, Any] | None = None
    maker_used = False
    router_decision["maker_started"] = maker_enabled
    if maker_enabled:
        try:
            t0 = time.perf_counter()
            adaptive_bps = adjusted_offset_bps
            adjusted_price = _apply_offset(maker_price, adaptive_bps, side)
            maker_px = effective_px(adjusted_price, side, is_maker=True) or adjusted_price
            maker_result = submit_limit(symbol, maker_px, maker_qty, side)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            router_decision["maker_offset_bps"] = adaptive_bps
        except Exception as exc:
            maker_result = None
            router_decision["reasons"].append("maker_submit_failed")
            router_decision["maker_error"] = str(exc)
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
            maker_used = bool(maker_result.is_maker)
            router_decision["maker_used"] = maker_used
            router_decision["route"] = "maker" if maker_used else "taker"
            if not maker_result.is_maker:
                router_decision["reasons"].append("maker_fallback_to_taker")

    if resp is None:
        try:
            t0 = time.perf_counter()
            resp = ex.send_order(**payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            router_decision["route"] = "taker"
        except Exception as exc:
            router_decision["reasons"].append("taker_submit_failed")
            router_decision["taker_error"] = str(exc)
            classification = ex.classify_binance_error(exc, getattr(exc, "response", None))
            router_decision["error_classification"] = classification
            error_payload: Dict[str, Any] = {
                "type": "order_error",
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
                "classification": classification,
                "context": "router",
            }
            if filters_snapshot:
                error_payload["exchange_filters_used"] = filters_snapshot
            try:
                record_execution_error(
                    "router",
                    symbol=symbol,
                    message="taker_submit_failed",
                    classification=classification,
                    context={"side": side, "order_type": order_type},
                )
            except Exception:
                pass
            log_event(LOG_ORDERS, "order_error", safe_dump(error_payload))
            _LOG.error("route_order failed: %s", exc)
            raise

    route_choice = "maker" if maker_used else "taker"
    router_decision["route"] = router_decision.get("route") or route_choice
    router_decision["used_fallback"] = bool(maker_enabled and not maker_used and router_decision["maker_started"])
    router_decision["latency_ms"] = latency_ms
    try:
        router_decision.setdefault("type", "route_decision")
        router_decision.setdefault("context", "router")
        log_event(LOG_ROUTER_DECISIONS, "route_decision", safe_dump(router_decision))
    except Exception:
        pass
    risk_ctx["routed_as"] = route_choice
    if ctx_original is not None:
        try:
            ctx_original["routed_as"] = route_choice
        except Exception:
            pass

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

    router_meta = {
        "maker_start": bool(maker_enabled),
        "is_maker_final": bool(maker_used),
        "used_fallback": bool(maker_enabled and not maker_used),
        "router_policy": policy_after_snapshot,
        "autotune": {
            "applied": autotune_applied,
            "before": policy_before_snapshot,
            "after": policy_after_snapshot,
            "risk_mode": risk_mode,
        },
        "decision": {**router_decision, "spread_clamped": spread_clamped},
    }
    try:
        target_ctx["route_decision"] = router_meta["decision"]
    except Exception:
        pass
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
        "router_meta": router_meta,
    }
    return result


def route_intent(intent: Dict[str, Any], attempt_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Route an order intent and emit routing metrics."""
    router_ctx_raw = intent.get("router_ctx")
    router_ctx = router_ctx_raw if isinstance(router_ctx_raw, dict) else {}
    dry_run = bool(intent.get("dry_run", False))
    timing = dict(intent.get("timing") or {})

    base_intent: Dict[str, Any] = {
        key: value
        for key, value in intent.items()
        if key not in {"router_ctx", "dry_run", "timing", "attempt_id", "intent_id"}
    }

    symbol = str(base_intent.get("symbol") or base_intent.get("pair") or "").upper()
    if symbol and "router_policy" not in router_ctx:
        try:
            policy_probe = router_policy(symbol)
            router_ctx["router_policy"] = {
                "maker_first": policy_probe.maker_first,
                "taker_bias": policy_probe.taker_bias,
                "quality": policy_probe.quality,
                "reason": policy_probe.reason,
            }
        except Exception as exc:
            router_ctx["router_policy"] = {"error": str(exc)}

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
        policy_meta = router_ctx.get("router_policy") or {}
        router_metrics["policy"] = {
            "maker_first": bool(policy_meta.get("maker_first")),
            "taker_bias": policy_meta.get("taker_bias"),
            "quality": policy_meta.get("quality"),
            "reason": policy_meta.get("reason"),
            "offset_bps": policy_meta.get("offset_bps"),
        }
        autotune_meta = router_ctx.get("autotune") or {}
        router_metrics["autotune_applied"] = bool(autotune_meta.get("applied"))
        router_metrics["policy_before"] = autotune_meta.get("before")
        router_metrics["policy_after"] = autotune_meta.get("after")
        started_maker = bool(router_ctx.get("maker_first")) and bool(router_ctx.get("maker_qty"))
        router_metrics["maker_start"] = bool(started_maker)
        router_metrics["is_maker_final"] = False
        router_metrics["used_fallback"] = False
        router_metrics["decision"] = router_ctx.get("route_decision")
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
        "ack_latency_ms": exchange_response.get("latency_ms"),
    }
    router_meta = exchange_response.get("router_meta") or {}
    started_maker = bool(router_meta.get("maker_start"))
    is_maker_final = bool(router_meta.get("is_maker_final"))
    used_fallback = bool(router_meta.get("used_fallback"))
    if not router_meta:
        started_maker = bool(router_ctx.get("maker_first"))
        is_maker_final = False
        used_fallback = False
    router_metrics["maker_start"] = started_maker
    router_metrics["is_maker_final"] = is_maker_final
    router_metrics["used_fallback"] = used_fallback
    policy_meta = (
        router_meta.get("router_policy")
        if isinstance(router_meta, Mapping)
        else None
    ) or router_ctx.get("router_policy") or {}
    router_metrics["policy"] = {
        "maker_first": bool(policy_meta.get("maker_first")),
        "taker_bias": policy_meta.get("taker_bias"),
        "quality": policy_meta.get("quality"),
        "reason": policy_meta.get("reason"),
        "offset_bps": policy_meta.get("offset_bps"),
    }
    autotune_meta = (
        router_meta.get("autotune")
        if isinstance(router_meta, Mapping)
        else None
    ) or router_ctx.get("autotune") or {}
    router_metrics["autotune_applied"] = bool(autotune_meta.get("applied"))
    router_metrics["policy_before"] = autotune_meta.get("before")
    router_metrics["policy_after"] = autotune_meta.get("after")
    router_metrics["decision"] = router_meta.get("decision") if isinstance(router_meta, Mapping) else router_ctx.get("route_decision")
    return exchange_response, router_metrics
