"""
v5.9 Execution Hardening — Router upgrades
- Maker-first POST_ONLY with smart fallback
- Fee-aware effective price calculator
- Child-order aggregation by min-notional
- Auto-cancel/refresh on low fill ratio
v7.5_B1: Added slippage tracking, spread-aware TWAP, liquidity buckets
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import requests

from execution import exchange_utils as ex
from execution.exchange_precision import normalize_price, normalize_qty
from execution.log_utils import get_logger, log_event, safe_dump
from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.intel.router_autotune_shared import suggest_autotune_for_symbol
from execution.intel.router_policy import router_policy, RouterPolicy
from execution.intel.router_autotune_apply_v6 import (
    APPLY_ENABLED as AUTOTUNE_APPLY_ENABLED,
    get_symbol_suggestion,
    get_current_risk_mode,
    apply_router_suggestion,
)
from execution.diagnostics_metrics import record_order_placed, record_router_event
from execution.runtime_config import load_runtime_config, get_twap_config, TWAPConfig
from execution.utils.execution_health import record_execution_error

# v7.5_B1: Slippage and liquidity imports
try:
    from execution.liquidity_model import (
        get_liquidity_model,
        get_bucket_for_symbol,
        LiquidityBucketConfig,
    )
    from execution.slippage_model import (
        SlippageObservation,
        estimate_expected_slippage_bps,
        compute_realized_slippage_bps,
        compute_spread_bps,
        record_slippage_observation,
        load_slippage_config,
        SlippageConfig,
    )
    _SLIPPAGE_AVAILABLE = True
except ImportError:
    _SLIPPAGE_AVAILABLE = False

__all__ = [
    "route_order",
    "route_intent",
    "is_ack_ok",
    "PlaceOrderResult",
    "effective_px",
    "chunk_qty",
    "submit_limit",
    "monitor_and_refresh",
    "should_use_twap",
    "split_twap_slices",
    "ChildOrderResult",
    "TWAPResult",
    "RouterStats",
    "get_router_stats_snapshot",
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

# v7.5_B1: Load slippage config
_SLIPPAGE_CFG: Optional["SlippageConfig"] = None


def _get_slippage_config() -> Optional["SlippageConfig"]:
    """Get cached slippage configuration."""
    global _SLIPPAGE_CFG
    if _SLIPPAGE_CFG is None and _SLIPPAGE_AVAILABLE:
        try:
            _SLIPPAGE_CFG = load_slippage_config()
        except Exception:
            _SLIPPAGE_CFG = None
    return _SLIPPAGE_CFG


def get_market_microstructure(
    symbol: str,
    depth_levels: int = 5,
) -> Tuple[float, float, float, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Fetch market microstructure data from exchange.
    
    Returns:
        (best_bid, best_ask, spread_bps, bids, asks)
        where bids/asks are lists of (price, qty) up to depth_levels.
    """
    try:
        book = ex.get_orderbook(symbol, limit=depth_levels)
        bids = [(float(p), float(q)) for p, q in book.get("bids", [])[:depth_levels]]
        asks = [(float(p), float(q)) for p, q in book.get("asks", [])[:depth_levels]]
        
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        
        if best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * 10000.0
        else:
            spread_bps = 0.0
        
        return best_bid, best_ask, spread_bps, bids, asks
    except Exception as exc:
        _LOG.debug("failed to get orderbook for %s: %s", symbol, exc)
        return 0.0, 0.0, 0.0, [], []


def _record_slippage(
    symbol: str,
    side: str,
    notional_usd: float,
    fill_price: float,
    mid_price: float,
    spread_bps: float,
    is_maker: bool,
    depth: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """
    Record a slippage observation after order fill.
    
    Args:
        symbol: Trading symbol
        side: "BUY" or "SELL"
        notional_usd: Filled notional in USD
        fill_price: Average fill price
        mid_price: Mid price at time of order
        spread_bps: Spread in bps at time of order
        is_maker: Whether this was a maker fill
        depth: Order book depth (asks for BUY, bids for SELL)
    """
    if not _SLIPPAGE_AVAILABLE:
        return
    
    try:
        # Compute realized slippage
        realized_bps = compute_realized_slippage_bps(side, fill_price, mid_price)
        
        # Compute expected slippage from depth
        expected_bps = 0.0
        if depth and mid_price > 0:
            qty = notional_usd / mid_price if mid_price > 0 else 0.0
            expected_bps = estimate_expected_slippage_bps(side, qty, depth, mid_price)
        
        obs = SlippageObservation(
            symbol=symbol,
            side=side,
            notional_usd=notional_usd,
            expected_bps=expected_bps,
            realized_bps=realized_bps,
            spread_bps=spread_bps,
            maker=is_maker,
        )
        
        record_slippage_observation(obs)
        
        _LOG.debug(
            "[slippage] %s %s: expected=%.2f realized=%.2f spread=%.2f maker=%s",
            symbol, side, expected_bps, realized_bps, spread_bps, is_maker
        )
    except Exception as exc:
        _LOG.debug("failed to record slippage: %s", exc)


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


class RouterStats:
    """
    Lightweight accumulator for router microstructure stats.

    Tracks slippage, latency, and TWAP usage in a rolling window.
    """

    def __init__(
        self,
        window_seconds: float = 900.0,
        min_events: int = 5,
        ema_alpha: float = 0.3,
    ) -> None:
        self.window_seconds = max(float(window_seconds), 0.0)
        self.min_events = max(int(min_events or 0), 0)
        self.ema_alpha = float(ema_alpha)
        self._events: Dict[str, deque[Dict[str, float]]] = {}
        self._slip_ema: Dict[str, float] = {}

    def _trim(self, symbol: str, now: float) -> None:
        events = self._events.get(symbol)
        if not events:
            return
        if self.window_seconds <= 0:
            return
        cutoff = now - self.window_seconds
        while len(events) > max(self.min_events, 0) and events and events[0].get("ts_fill", 0.0) < cutoff:
            events.popleft()

    def _update_ema(self, symbol: str, value: float) -> float:
        prev = self._slip_ema.get(symbol)
        if prev is None:
            self._slip_ema[symbol] = value
        else:
            self._slip_ema[symbol] = (self.ema_alpha * value) + ((1.0 - self.ema_alpha) * prev)
        return self._slip_ema[symbol]

    def update_on_fill(
        self,
        symbol: str,
        intended_price: float | None,
        fill_price: float | None,
        ts_sent: float | None,
        ts_fill: float | None,
        is_twap_child: bool,
        notional: float | None,
        *,
        intended_notional: float | None = None,
    ) -> None:
        """
        Record a fill event for microstructure stats.

        Args:
            symbol: Trading symbol
            intended_price: Target/quoted price for the order
            fill_price: Executed fill price
            ts_sent: Timestamp when the order was sent (epoch seconds)
            ts_fill: Timestamp when the fill/ack was received (epoch seconds)
            is_twap_child: Whether this fill came from a TWAP child slice
            notional: Filled notional (quote currency)
            intended_notional: Intended notional for the order (optional)
        """
        sym = str(symbol or "").upper()
        if not sym:
            return

        now = time.time()
        ts_sent_val = float(ts_sent) if ts_sent is not None else now
        ts_fill_val = float(ts_fill) if ts_fill is not None else now
        latency_ms = max(0.0, (ts_fill_val - ts_sent_val) * 1000.0)
        filled_notional = abs(float(notional or 0.0))
        target_notional = abs(float(intended_notional if intended_notional is not None else filled_notional))

        slip_bps = 0.0
        if intended_price and fill_price:
            slip_bps = abs(_bps(float(fill_price), float(intended_price)))

        events = self._events.setdefault(sym, deque())
        events.append(
            {
                "ts_sent": ts_sent_val,
                "ts_fill": ts_fill_val,
                "slippage_bps": slip_bps,
                "latency_ms": latency_ms,
                "is_twap_child": 1.0 if is_twap_child else 0.0,
                "notional": filled_notional,
                "filled_notional": filled_notional,
                "intended_notional": target_notional,
            }
        )
        self._trim(sym, ts_fill_val)
        self._update_ema(sym, slip_bps)

    def snapshot(self, now: float | None = None) -> Dict[str, Any]:
        """Return a per-symbol snapshot for state publishing."""
        ts_now = float(now if now is not None else time.time())
        per_symbol: Dict[str, Any] = {}
        for sym, events in self._events.items():
            if not events:
                continue
            filtered = list(events)
            total_notional = sum(e.get("notional", 0.0) for e in filtered)
            weight = total_notional if total_notional > 0 else float(len(filtered))
            if weight <= 0:
                continue

            weighted = lambda key: sum((e.get(key) or 0.0) * (e.get("notional", 0.0) if total_notional > 0 else 1.0) for e in filtered) / weight  # noqa: E731
            avg_slip = weighted("slippage_bps")
            avg_latency = weighted("latency_ms")
            twap_notional = sum(e.get("notional", 0.0) for e in filtered if e.get("is_twap_child"))
            twap_usage = twap_notional / total_notional if total_notional > 0 else 0.0
            child_events = [e for e in filtered if e.get("is_twap_child")]
            child_count = len(child_events)
            child_intended = sum(e.get("intended_notional", 0.0) for e in child_events)
            child_filled = sum(e.get("filled_notional", 0.0) for e in child_events)
            child_fill_ratio = (child_filled / child_intended) if child_intended > 0 else (1.0 if child_count else 0.0)

            per_symbol[sym] = {
                "symbol": sym,
                "avg_slippage_bps": avg_slip,
                "slippage_drift_bps": self._slip_ema.get(sym, avg_slip),
                "avg_latency_ms": avg_latency,
                "twap_usage_ratio": twap_usage,
                "last_order_ts": max(e.get("ts_sent", 0.0) for e in filtered),
                "last_fill_ts": max(e.get("ts_fill", 0.0) for e in filtered),
                "total_notional": total_notional,
                "twap_notional": twap_notional,
                "event_count": len(filtered),
                "child_orders": {
                    "count": child_count,
                    "fill_ratio": child_fill_ratio,
                },
            }

        return {
            "updated_ts": datetime.fromtimestamp(ts_now, tz=timezone.utc).isoformat(),
            "per_symbol": per_symbol,
            "window_seconds": self.window_seconds,
            "min_events": self.min_events,
        }


def _init_router_stats() -> RouterStats:
    try:
        from execution.router_metrics import load_router_quality_config

        cfg = load_router_quality_config()
        return RouterStats(
            window_seconds=cfg.stats_window_seconds,
            min_events=cfg.stats_window_min_events,
        )
    except Exception:
        return RouterStats()


_ROUTER_STATS = _init_router_stats()


def get_router_stats_snapshot(now: float | None = None) -> Dict[str, Any]:
    """Expose router stats snapshot for state publishing."""
    return _ROUTER_STATS.snapshot(now)


# ---------------------------------------------------------------------------
# TWAP Execution Support (v7.4 C1)
# ---------------------------------------------------------------------------

@dataclass
class ChildOrderResult:
    """Result of a single TWAP slice execution."""
    slice_index: int
    slice_count: int
    slice_qty: float
    order_id: str
    status: str
    filled_qty: float
    avg_price: float | None
    is_maker: bool
    slippage_bps: float
    latency_ms: float | None
    raw: Dict[str, Any] | None = None


@dataclass
class TWAPResult:
    """Aggregate result of TWAP execution."""
    parent_symbol: str
    parent_side: str
    parent_qty: float
    parent_gross_usd: float
    execution_style: str  # "twap" or "single"
    slices: int
    interval_seconds: float
    children: list[ChildOrderResult] = field(default_factory=list)
    total_filled_qty: float = 0.0
    weighted_avg_price: float | None = None
    avg_slippage_bps: float = 0.0
    total_latency_ms: float = 0.0
    maker_count: int = 0
    taker_count: int = 0
    success: bool = True
    error: str | None = None


def should_use_twap(
    gross_usd: float,
    twap_cfg: TWAPConfig,
) -> bool:
    """
    Determine if TWAP should be used for this order.
    
    Args:
        gross_usd: Gross notional value of the order in USD
        twap_cfg: TWAP configuration
    
    Returns:
        True if TWAP should be used
    """
    return (
        twap_cfg.enabled
        and gross_usd >= twap_cfg.min_notional_usd
        and twap_cfg.slices > 1
    )


def split_twap_slices(
    total_qty: float,
    slices: int,
    min_slice_notional: float = 0.0,
    price: float = 1.0,
) -> list[float]:
    """
    Split total quantity into TWAP slices.
    
    Args:
        total_qty: Total quantity to split
        slices: Desired number of slices
        min_slice_notional: Minimum notional per slice (optional)
        price: Reference price for notional calculation
    
    Returns:
        List of slice quantities (sum equals total_qty)
    """
    if total_qty <= 0 or slices <= 0:
        return []
    
    if slices == 1:
        return [total_qty]
    
    # Check if each slice meets minimum notional
    if min_slice_notional > 0 and price > 0:
        slice_qty = total_qty / slices
        slice_notional = slice_qty * price
        
        # If slice is too small, reduce number of slices
        while slice_notional < min_slice_notional and slices > 1:
            slices -= 1
            slice_qty = total_qty / slices
            slice_notional = slice_qty * price
    
    if slices == 1:
        return [total_qty]
    
    # Compute equal slices
    base_qty = total_qty / slices
    quantities = [base_qty] * slices
    
    # Adjust last slice for rounding error
    remainder = total_qty - sum(quantities)
    if remainder != 0:
        quantities[-1] += remainder
    
    return quantities


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
        try:
            record_router_event()
        except Exception:
            pass
        reprice_wider(order)
        return


# ---------------------------------------------------------------------------
# TWAP Execution Implementation (v7.4 C1)
# ---------------------------------------------------------------------------

LOG_TWAP_EVENTS = None  # Lazy init to avoid circular imports


def _get_twap_logger():
    """Get TWAP events logger (lazy init)."""
    global LOG_TWAP_EVENTS
    if LOG_TWAP_EVENTS is None:
        LOG_TWAP_EVENTS = get_logger("logs/execution/twap_events.jsonl")
    return LOG_TWAP_EVENTS


def _route_twap(
    intent: Mapping[str, Any],
    risk_ctx: Mapping[str, Any],
    twap_cfg: TWAPConfig,
    dry_run: bool = False,
    sleep_func: Callable[[float], None] | None = None,
) -> TWAPResult:
    """
    Execute a TWAP for the given intent.
    
    Splits quantity into N slices, calls existing maker-first routing for each,
    and sleeps between slices as configured.
    
    Args:
        intent: Order intent with symbol, side, qty, etc.
        risk_ctx: Risk context from check_order
        twap_cfg: TWAP configuration
        dry_run: If True, skip actual order placement
        sleep_func: Optional sleep function (for testing)
    
    Returns:
        TWAPResult with aggregated execution data
    """
    sleep_fn = sleep_func or time.sleep
    symbol = str(intent.get("symbol") or intent.get("pair") or "").upper()
    side = str(intent.get("side") or intent.get("signal") or "").upper()
    if side in ("LONG",):
        side = "BUY"
    elif side in ("SHORT",):
        side = "SELL"
    
    total_qty = float(intent.get("quantity") or intent.get("qty") or 0)
    price = float(intent.get("price") or risk_ctx.get("price") or 0)
    gross_usd = total_qty * price if price > 0 else 0
    
    result = TWAPResult(
        parent_symbol=symbol,
        parent_side=side,
        parent_qty=total_qty,
        parent_gross_usd=gross_usd,
        execution_style="twap",
        slices=twap_cfg.slices,
        interval_seconds=twap_cfg.interval_seconds,
    )
    
    # Compute slice quantities
    slice_quantities = split_twap_slices(
        total_qty=total_qty,
        slices=twap_cfg.slices,
        min_slice_notional=MIN_CHILD_NOTIONAL,
        price=price,
    )
    
    actual_slices = len(slice_quantities)
    result.slices = actual_slices
    
    if actual_slices == 0:
        result.success = False
        result.error = "no_valid_slices"
        return result
    
    # Log TWAP start
    twap_logger = _get_twap_logger()
    
    # v7.5_B1: Get liquidity bucket and slippage config for spread-aware TWAP
    bucket_config = None
    slippage_cfg = _get_slippage_config()
    spread_pause_factor = 1.5
    if _SLIPPAGE_AVAILABLE:
        try:
            bucket_config = get_bucket_for_symbol(symbol)
            if slippage_cfg:
                spread_pause_factor = slippage_cfg.spread_pause_factor
        except Exception:
            pass
    
    twap_start_event = {
        "event": "twap_start",
        "symbol": symbol,
        "side": side,
        "total_qty": total_qty,
        "gross_usd": gross_usd,
        "slices": actual_slices,
        "interval_seconds": twap_cfg.interval_seconds,
        "slice_quantities": slice_quantities,
        "liquidity_bucket": bucket_config.name if bucket_config else None,
        "ts": time.time(),
    }
    try:
        log_event(twap_logger, "twap_start", safe_dump(twap_start_event))
    except Exception:
        pass
    
    skipped_slices = 0
    
    # Execute each slice
    for i, slice_qty in enumerate(slice_quantities):
        # v7.5_B1: Check spread before each slice
        if bucket_config and slippage_cfg and slippage_cfg.enabled:
            try:
                _, _, current_spread_bps, _, _ = get_market_microstructure(
                    symbol, slippage_cfg.depth_levels
                )
                max_spread_bps = bucket_config.max_spread_bps
                pause_thresh_bps = max_spread_bps * spread_pause_factor
                
                if current_spread_bps > pause_thresh_bps:
                    _LOG.info(
                        "[twap-spread] skipping slice %d/%d: symbol=%s spread_bps=%.2f thresh=%.2f",
                        i + 1, actual_slices, symbol, current_spread_bps, pause_thresh_bps
                    )
                    skipped_slices += 1
                    
                    # Log skipped slice
                    skip_event = {
                        "event": "twap_slice_skipped",
                        "symbol": symbol,
                        "side": side,
                        "slice_index": i,
                        "slice_count": actual_slices,
                        "reason": "spread_too_wide",
                        "spread_bps": current_spread_bps,
                        "threshold_bps": pause_thresh_bps,
                        "ts": time.time(),
                    }
                    try:
                        log_event(twap_logger, "twap_slice_skipped", safe_dump(skip_event))
                    except Exception:
                        pass
                    
                    # Sleep before trying next slice
                    sleep_fn(twap_cfg.interval_seconds)
                    continue
            except Exception as exc:
                _LOG.debug("spread check failed for %s: %s", symbol, exc)
        
        slice_intent = dict(intent)
        slice_intent["quantity"] = slice_qty
        slice_intent["qty"] = slice_qty
        slice_intent["_twap_slice"] = True
        slice_intent["_twap_slice_index"] = i
        slice_intent["_twap_slice_count"] = actual_slices
        
        slice_risk_ctx = dict(risk_ctx)
        
        try:
            t0 = time.perf_counter()
            exchange_response = route_order(slice_intent, slice_risk_ctx, dry_run)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            child = ChildOrderResult(
                slice_index=i,
                slice_count=actual_slices,
                slice_qty=slice_qty,
                order_id="",
                status="FAILED",
                filled_qty=0.0,
                avg_price=None,
                is_maker=False,
                slippage_bps=0.0,
                latency_ms=None,
                raw={"error": str(exc)},
            )
            result.children.append(child)
            result.success = False
            result.error = str(exc)
            
            # Log slice error
            _log_twap_slice(
                twap_logger, symbol, side, i, actual_slices, slice_qty,
                gross_usd, twap_cfg, child, "error"
            )
            continue
        
        # Extract result data
        raw = exchange_response.get("raw") or {}
        router_meta = exchange_response.get("router_meta") or {}
        is_maker = bool(router_meta.get("is_maker_final"))
        avg_price = exchange_response.get("price")
        filled_qty = exchange_response.get("qty") or 0.0
        status = exchange_response.get("status") or "UNKNOWN"
        
        # Compute slippage for this slice
        mark_px = float(intent.get("mark_price") or price or 0)
        slip_bps = 0.0
        if mark_px > 0 and avg_price:
            diff = (avg_price - mark_px) / mark_px * 10_000.0
            if side == "SELL":
                diff *= -1.0
            slip_bps = diff
        
        child = ChildOrderResult(
            slice_index=i,
            slice_count=actual_slices,
            slice_qty=slice_qty,
            order_id=str(exchange_response.get("order_id") or ""),
            status=status,
            filled_qty=float(filled_qty) if filled_qty else 0.0,
            avg_price=float(avg_price) if avg_price else None,
            is_maker=is_maker,
            slippage_bps=slip_bps,
            latency_ms=latency_ms,
            raw=raw,
        )
        result.children.append(child)
        
        # Update aggregates
        result.total_filled_qty += child.filled_qty
        result.total_latency_ms += latency_ms or 0.0
        if is_maker:
            result.maker_count += 1
        else:
            result.taker_count += 1
        
        # Log slice completion
        _log_twap_slice(
            twap_logger, symbol, side, i, actual_slices, slice_qty,
            gross_usd, twap_cfg, child, "complete"
        )
        
        # Sleep between slices (except after last)
        if i < actual_slices - 1 and twap_cfg.interval_seconds > 0:
            sleep_fn(twap_cfg.interval_seconds)
    
    # Compute weighted average price
    total_value = 0.0
    total_qty_filled = 0.0
    for child in result.children:
        if child.avg_price and child.filled_qty > 0:
            total_value += child.avg_price * child.filled_qty
            total_qty_filled += child.filled_qty
    
    if total_qty_filled > 0:
        result.weighted_avg_price = total_value / total_qty_filled
    
    # Compute average slippage
    slip_sum = sum(c.slippage_bps for c in result.children)
    if result.children:
        result.avg_slippage_bps = slip_sum / len(result.children)
    
    # Log TWAP complete
    twap_complete_event = {
        "event": "twap_complete",
        "symbol": symbol,
        "side": side,
        "total_qty": total_qty,
        "total_filled_qty": result.total_filled_qty,
        "weighted_avg_price": result.weighted_avg_price,
        "avg_slippage_bps": result.avg_slippage_bps,
        "slices_executed": len(result.children),
        "slices_skipped": skipped_slices,  # v7.5_B1: track skipped slices
        "maker_count": result.maker_count,
        "taker_count": result.taker_count,
        "total_latency_ms": result.total_latency_ms,
        "success": result.success,
        "ts": time.time(),
    }
    try:
        log_event(twap_logger, "twap_complete", safe_dump(twap_complete_event))
    except Exception:
        pass
    
    return result


def _log_twap_slice(
    logger,
    symbol: str,
    side: str,
    slice_index: int,
    slice_count: int,
    slice_qty: float,
    parent_gross_usd: float,
    twap_cfg: TWAPConfig,
    child: ChildOrderResult,
    status: str,
) -> None:
    """Log a TWAP slice event."""
    event = {
        "event": f"twap_slice_{status}",
        "execution_style": "twap",
        "symbol": symbol,
        "side": side,
        "twap": {
            "slice_index": slice_index,
            "slice_count": slice_count,
            "slice_qty": slice_qty,
            "parent_gross_usd": parent_gross_usd,
            "twap_cfg": {
                "min_notional_usd": twap_cfg.min_notional_usd,
                "slices": twap_cfg.slices,
                "interval_seconds": twap_cfg.interval_seconds,
            },
        },
        "order_id": child.order_id,
        "status": child.status,
        "filled_qty": child.filled_qty,
        "avg_price": child.avg_price,
        "is_maker": child.is_maker,
        "slippage_bps": child.slippage_bps,
        "latency_ms": child.latency_ms,
        "ts": time.time(),
    }
    try:
        log_event(logger, event["event"], safe_dump(event))
    except Exception:
        pass


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

    is_twap_child = bool(intent.get("_twap_slice"))
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
    qty_float_intent = _to_float(qty)
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
    policy_probe = router_policy(symbol)
    try:
        base_offset_bps = float(policy_probe.offset_bps) if policy_probe.offset_bps is not None else float(suggest_maker_offset_bps(symbol))
    except Exception:
        base_offset_bps = 2.0
    adjusted_offset_bps = base_offset_bps
    dynamic_autotune: Dict[str, Any] = {}
    maker_first_override = policy_probe.maker_first
    try:
        dynamic_autotune = suggest_autotune_for_symbol(
            symbol,
            base_offset_bps,
            min_offset_bps=getattr(maker_offset, "MIN_OFFSET_BPS", 0.5),
        )
        adjusted_offset_bps = float(dynamic_autotune.get("adaptive_offset_bps") or base_offset_bps)
        maker_first_override = bool(dynamic_autotune.get("maker_first"))
    except Exception:
        dynamic_autotune = {}
    policy = RouterPolicy(
        maker_first=maker_first_override,
        taker_bias=policy_probe.taker_bias,
        quality=policy_probe.quality,
        reason=policy_probe.reason,
        offset_bps=adjusted_offset_bps,
    )
    policy_snapshot = {
        "maker_first": policy.maker_first,
        "taker_bias": policy.taker_bias,
        "quality": policy.quality,
        "reason": policy.reason,
        "offset_bps": adjusted_offset_bps,
    }
    router_decision: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "route": None,
        "maker_requested": bool(risk_ctx.get("maker_first")),
        "policy_quality": policy.quality,
        "policy_maker_first": policy.maker_first,
        "policy_taker_bias": policy.taker_bias,
        "offset_bps": adjusted_offset_bps,
        "spread_bps": spread_bps,
        "maker_reliability": float(dynamic_autotune.get("maker_reliability") or 0.0),
        "effective_reliability": float(dynamic_autotune.get("effective_reliability") or dynamic_autotune.get("maker_reliability") or 0.0),
        "risk_mode_dynamic": dynamic_autotune.get("risk_mode"),
        "reasons": [],
    }
    policy_before_snapshot = dict(policy_snapshot)
    policy_after_snapshot = dict(policy_snapshot)
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

    try:
        record_router_event()
    except Exception:
        pass

    latency_ms: float | None = None
    resp: Dict[str, Any] | None = None
    maker_used = False
    ts_sent_wall: float | None = None
    ts_fill_wall: float | None = None
    router_decision["maker_started"] = maker_enabled
    if maker_enabled:
        try:
            ts_sent_wall = time.time()
            t0 = time.perf_counter()
            adaptive_bps = adjusted_offset_bps
            adjusted_price = _apply_offset(maker_price, adaptive_bps, side)
            maker_px = effective_px(adjusted_price, side, is_maker=True) or adjusted_price
            # Normalize price to exchange tickSize precision
            maker_px = normalize_price(symbol, maker_px)
            maker_result = submit_limit(symbol, maker_px, maker_qty, side)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ts_fill_wall = time.time()
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
            ts_sent_wall = time.time()
            t0 = time.perf_counter()
            resp = ex.send_order(**payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ts_fill_wall = time.time()
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
            "dynamic": dynamic_autotune,
        },
        "decision": {**router_decision, "spread_clamped": spread_clamped},
    }
    
    # v7.5_B1: Record slippage observation for filled orders
    if accepted and executed_qty_float and executed_qty_float > 0 and avg_price_float:
        try:
            mid_price = _to_float(risk_ctx.get("mid_price") or maker_price)
            notional_usd = executed_qty_float * avg_price_float
            obs_spread_bps = spread_bps if spread_bps else 0.0
            
            # Get depth for expected slippage calculation
            depth = None
            if _SLIPPAGE_AVAILABLE:
                slippage_cfg = _get_slippage_config()
                if slippage_cfg and slippage_cfg.enabled:
                    try:
                        _, _, _, bids, asks = get_market_microstructure(
                            symbol, slippage_cfg.depth_levels
                        )
                        # Use asks for BUY, bids for SELL
                        depth = asks if side == "BUY" else bids
                    except Exception:
                        pass
            
            _record_slippage(
                symbol=symbol,
                side=side,
                notional_usd=notional_usd,
                fill_price=avg_price_float,
                mid_price=mid_price or avg_price_float,
                spread_bps=obs_spread_bps,
                is_maker=maker_used,
                depth=depth,
            )
        except Exception as exc:
            _LOG.debug("slippage recording failed: %s", exc)
    
    try:
        target_ctx["route_decision"] = router_meta["decision"]
    except Exception:
        pass
    if accepted:
        try:
            record_order_placed()
        except Exception:
            pass

    if accepted and executed_qty_float and executed_qty_float > 0 and avg_price_float:
        intended_price_for_stats = maker_price if maker_used and maker_price else _to_float(payload.get("price"))
        if intended_price_for_stats is None:
            intended_price_for_stats = _to_float(risk_ctx.get("mid_price") or risk_ctx.get("price"))
        intended_notional = None
        try:
            if qty_float_intent and intended_price_for_stats:
                intended_notional = abs(qty_float_intent * intended_price_for_stats)
        except Exception:
            intended_notional = None
        filled_notional = executed_qty_float * avg_price_float
        try:
            _ROUTER_STATS.update_on_fill(
                symbol=symbol,
                intended_price=intended_price_for_stats or avg_price_float,
                fill_price=avg_price_float,
                ts_sent=ts_sent_wall,
                ts_fill=ts_fill_wall or time.time(),
                is_twap_child=is_twap_child,
                notional=filled_notional,
                intended_notional=intended_notional,
            )
        except Exception:
            pass
        try:
            record_router_event()
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
    
    # --- TWAP Decision (v7.4 C1) ---
    # Check if this intent should use TWAP execution
    twap_cfg = get_twap_config()
    qty_float_check = _to_float(base_intent.get("quantity") or base_intent.get("qty"))
    price_check = mark_px or submit_px or 0.0
    gross_usd = (qty_float_check or 0) * price_check
    
    # Skip TWAP for slices (already in TWAP) or explicitly disabled
    is_twap_slice = bool(base_intent.get("_twap_slice"))
    use_twap = (
        not is_twap_slice
        and should_use_twap(gross_usd, twap_cfg)
    )
    
    if use_twap:
        # Execute via TWAP path
        twap_result = _route_twap(
            intent=base_intent,
            risk_ctx=router_ctx,
            twap_cfg=twap_cfg,
            dry_run=dry_run,
        )
        
        # Convert TWAPResult to exchange_response format
        exchange_response = _twap_result_to_exchange_response(twap_result)
        
        # Build router_metrics for TWAP
        router_metrics = _build_twap_metrics(
            twap_result=twap_result,
            attempt_id=attempt_id,
            timing=timing,
            mark_px=mark_px,
            submit_px=submit_px,
            retry_count=retry_count,
            router_ctx=router_ctx,
        )
        
        return exchange_response, router_metrics

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
    router_metrics["execution_style"] = "single"  # v7.4 C1: tag non-TWAP orders
    return exchange_response, router_metrics


# ---------------------------------------------------------------------------
# TWAP Result Converters (v7.4 C1)
# ---------------------------------------------------------------------------

def _twap_result_to_exchange_response(twap_result: TWAPResult) -> Dict[str, Any]:
    """
    Convert TWAPResult to exchange_response format for compatibility.
    
    Args:
        twap_result: TWAP execution result
    
    Returns:
        Dict matching route_order return format
    """
    # Aggregate order IDs from children
    child_order_ids = [c.order_id for c in twap_result.children if c.order_id]
    first_order_id = child_order_ids[0] if child_order_ids else ""
    
    # Determine overall status
    statuses = [c.status for c in twap_result.children]
    if all(s in ("FILLED", "NEW", "PARTIALLY_FILLED") for s in statuses):
        status = "FILLED" if twap_result.total_filled_qty >= twap_result.parent_qty * 0.99 else "PARTIALLY_FILLED"
    elif any(s == "FAILED" for s in statuses):
        status = "PARTIALLY_FILLED" if twap_result.total_filled_qty > 0 else "REJECTED"
    else:
        status = "FILLED"
    
    accepted = status not in ("REJECTED", "FAILED")
    
    return {
        "accepted": accepted,
        "reason": twap_result.error,
        "order_id": first_order_id,
        "status": status,
        "price": twap_result.weighted_avg_price,
        "qty": twap_result.total_filled_qty,
        "raw": {
            "twap": True,
            "slices": twap_result.slices,
            "children": [
                {
                    "slice_index": c.slice_index,
                    "order_id": c.order_id,
                    "status": c.status,
                    "filled_qty": c.filled_qty,
                    "avg_price": c.avg_price,
                    "is_maker": c.is_maker,
                    "slippage_bps": c.slippage_bps,
                }
                for c in twap_result.children
            ],
            "total_filled_qty": twap_result.total_filled_qty,
            "weighted_avg_price": twap_result.weighted_avg_price,
        },
        "request_id": None,
        "latency_ms": twap_result.total_latency_ms,
        "exchange_filters_used": {},
        "client_order_id": None,
        "transact_time": None,
        "router_meta": {
            "maker_start": True,
            "is_maker_final": twap_result.maker_count > twap_result.taker_count,
            "used_fallback": twap_result.taker_count > 0,
            "execution_style": "twap",
            "twap": {
                "slices": twap_result.slices,
                "interval_seconds": twap_result.interval_seconds,
                "maker_count": twap_result.maker_count,
                "taker_count": twap_result.taker_count,
                "avg_slippage_bps": twap_result.avg_slippage_bps,
            },
        },
    }


def _build_twap_metrics(
    twap_result: TWAPResult,
    attempt_id: str,
    timing: Dict[str, Any],
    mark_px: float | None,
    submit_px: float | None,
    retry_count: int,
    router_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build router_metrics dict for TWAP execution.
    
    Args:
        twap_result: TWAP execution result
        attempt_id: Unique attempt identifier
        timing: Timing information
        mark_px: Mark price
        submit_px: Submitted price
        retry_count: Number of retries
        router_ctx: Router context
    
    Returns:
        Router metrics dict
    """
    policy_meta = router_ctx.get("router_policy") or {}
    
    return {
        "attempt_id": attempt_id,
        "venue": "binance_futures",
        "route": "twap",
        "execution_style": "twap",
        "prices": {
            "mark": mark_px,
            "submitted": submit_px,
            "avg_fill": twap_result.weighted_avg_price,
        },
        "qty": {
            "contracts": twap_result.parent_qty,
            "notional_usd": twap_result.parent_gross_usd,
        },
        "timing_ms": {
            "decision": _to_float(timing.get("decision")),
            "submit": _to_float(timing.get("submit")),
            "ack": twap_result.total_latency_ms,
            "fill": _to_float(timing.get("fill")),
        },
        "result": {
            "status": "FILLED" if twap_result.success else "REJECTED",
            "retries": retry_count,
            "cancelled": False,
        },
        "fees_usd": None,
        "slippage_bps": twap_result.avg_slippage_bps,
        "ack_latency_ms": twap_result.total_latency_ms,
        "maker_start": True,
        "is_maker_final": twap_result.maker_count > twap_result.taker_count,
        "used_fallback": twap_result.taker_count > 0,
        "policy": {
            "maker_first": bool(policy_meta.get("maker_first")),
            "taker_bias": policy_meta.get("taker_bias"),
            "quality": policy_meta.get("quality"),
            "reason": policy_meta.get("reason"),
            "offset_bps": policy_meta.get("offset_bps"),
        },
        "autotune_applied": False,
        "policy_before": None,
        "policy_after": None,
        "decision": None,
        "twap": {
            "slices": twap_result.slices,
            "slices_executed": len(twap_result.children),
            "interval_seconds": twap_result.interval_seconds,
            "maker_count": twap_result.maker_count,
            "taker_count": twap_result.taker_count,
            "total_filled_qty": twap_result.total_filled_qty,
            "weighted_avg_price": twap_result.weighted_avg_price,
            "avg_slippage_bps": twap_result.avg_slippage_bps,
            "children": [
                {
                    "slice_index": c.slice_index,
                    "order_id": c.order_id,
                    "status": c.status,
                    "filled_qty": c.filled_qty,
                    "is_maker": c.is_maker,
                    "slippage_bps": c.slippage_bps,
                }
                for c in twap_result.children
            ],
        },
    }
