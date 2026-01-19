"""
Minotaur Integration — v7.9_P3

Integration layer connecting Minotaur Execution Engine to the existing
executor and Hydra pipelines.

This module provides:
1. Intent-to-plan conversion for merged intents
2. Execution plan scheduling and child order dispatch
3. Fill processing for quality tracking
4. Runtime state management

Usage:
    from execution.minotaur_integration import (
        run_minotaur_for_intents,
        process_fill_for_quality,
        get_minotaur_runtime_state,
    )
    
    if is_minotaur_enabled(strategy_config):
        plans, state = run_minotaur_for_intents(merged_intents, nav, cfg, ...)
        for plan in plans:
            child_orders = plan_child_orders(plan)
            # dispatch child orders through router
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from execution.minotaur_engine import (
    # Config
    MinotaurConfig,
    load_minotaur_config,
    is_minotaur_enabled,
    # Dataclasses
    MicrostructureSnapshot,
    ExecutionRegime,
    ExecutionPlan,
    ChildOrder,
    ExecutionQualityStats,
    MinotaurState,
    # Constants
    REGIME_NORMAL,
    REGIME_THIN,
    REGIME_WIDE_SPREAD,
    REGIME_SPIKE,
    REGIME_CRUNCH,
    MODE_INSTANT,
    MODE_TWAP,
    # Functions
    build_microstructure_snapshot,
    classify_execution_regime,
    build_execution_plan,
    plan_child_orders,
    check_throttling,
    apply_throttling,
    calculate_slippage_bps,
    update_quality_stats,
    save_execution_quality_state,
    log_plan_event,
    log_slippage_event,
    log_throttle_event,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime State
# ---------------------------------------------------------------------------

# Global runtime state for Minotaur
_MINOTAUR_STATE: Optional[MinotaurState] = None
_QUALITY_STATS: Dict[str, ExecutionQualityStats] = {}


def get_minotaur_runtime_state() -> MinotaurState:
    """Get current Minotaur runtime state."""
    global _MINOTAUR_STATE
    if _MINOTAUR_STATE is None:
        _MINOTAUR_STATE = MinotaurState()
    return _MINOTAUR_STATE


def reset_cycle_state() -> None:
    """Reset per-cycle state counters."""
    global _MINOTAUR_STATE
    if _MINOTAUR_STATE is None:
        _MINOTAUR_STATE = MinotaurState()
    _MINOTAUR_STATE.orders_this_cycle = 0


def get_quality_stats() -> Dict[str, ExecutionQualityStats]:
    """Get current quality stats dict."""
    return _QUALITY_STATS


# ---------------------------------------------------------------------------
# Microstructure Data Providers
# ---------------------------------------------------------------------------


def get_microstructure_for_symbol(
    symbol: str,
    market_data_cache: Optional[Dict[str, Any]] = None,
    router_health: Optional[Dict[str, Any]] = None,
) -> MicrostructureSnapshot:
    """
    Get microstructure snapshot for a symbol.
    
    Uses available market data sources to build the snapshot.
    
    Args:
        symbol: Trading symbol
        market_data_cache: Optional dict with orderbook/ticker data
        router_health: Optional router_health state for slippage history
        
    Returns:
        MicrostructureSnapshot
    """
    # Default values
    best_bid = 0.0
    best_ask = 0.0
    bid_depth_usd = 0.0
    ask_depth_usd = 0.0
    recent_slippage_bps = 0.0
    
    # Try to get data from market_data_cache
    if market_data_cache:
        symbol_data = market_data_cache.get(symbol, {})
        
        # Ticker data
        best_bid = float(symbol_data.get("bidPrice", 0) or 0)
        best_ask = float(symbol_data.get("askPrice", 0) or 0)
        
        # Orderbook depth if available
        orderbook = symbol_data.get("orderbook", {})
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 1.0
            bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:5]) if bids else 0.0
            ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks[:5]) if asks else 0.0
    
    # Try to get slippage from router_health
    if router_health:
        symbols_health = router_health.get("symbols", {})
        symbol_health = symbols_health.get(symbol, {})
        recent_slippage_bps = float(symbol_health.get("avg_slippage_bps", 0) or 0)
    
    return build_microstructure_snapshot(
        symbol=symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_usd=bid_depth_usd,
        ask_depth_usd=ask_depth_usd,
        recent_slippage_bps=recent_slippage_bps,
    )


# ---------------------------------------------------------------------------
# Intent Processing
# ---------------------------------------------------------------------------


def run_minotaur_for_intents(
    merged_intents: List[Dict[str, Any]],
    nav_usd: float,
    minotaur_cfg: MinotaurConfig,
    market_data_cache: Optional[Dict[str, Any]] = None,
    router_health: Optional[Dict[str, Any]] = None,
    trend_biases: Optional[Dict[str, float]] = None,
    head_importances: Optional[Dict[str, float]] = None,
) -> Tuple[List[ExecutionPlan], MinotaurState]:
    """
    Run Minotaur execution planning for merged intents.
    
    Args:
        merged_intents: List of merged intents from Hydra/Cerberus
        nav_usd: Current NAV in USD
        minotaur_cfg: Minotaur configuration
        market_data_cache: Optional market data for microstructure
        router_health: Optional router health state for slippage history
        trend_biases: Optional dict of symbol -> trend bias [-1, 1]
        head_importances: Optional dict of symbol -> head importance weight
        
    Returns:
        Tuple of (list of ExecutionPlan, MinotaurState)
    """
    global _MINOTAUR_STATE
    
    if _MINOTAUR_STATE is None:
        _MINOTAUR_STATE = MinotaurState(enabled=minotaur_cfg.enabled)
    
    _MINOTAUR_STATE.enabled = minotaur_cfg.enabled
    
    if not minotaur_cfg.enabled:
        # Minotaur disabled: return simple instant plans
        plans = []
        for intent in merged_intents:
            symbol = intent.get("symbol", "")
            side = intent.get("side", intent.get("direction", "LONG")).upper()
            qty = float(intent.get("qty", 0) or 0)
            price = float(intent.get("price", intent.get("entry_price", 0)) or 0)
            
            if qty > 0 and price > 0:
                plans.append(ExecutionPlan(
                    symbol=symbol,
                    side=side,
                    total_qty=qty,
                    total_notional=qty * price,
                    slicing_mode=MODE_INSTANT,
                    slice_count=1,
                    schedule_seconds=0,
                    aggressiveness=0.6,
                    regime=REGIME_NORMAL,
                    notes="minotaur disabled",
                ))
        
        return plans, _MINOTAUR_STATE
    
    # Build regimes for all symbols
    regimes: Dict[str, ExecutionRegime] = {}
    snapshots: Dict[str, MicrostructureSnapshot] = {}
    
    for intent in merged_intents:
        symbol = intent.get("symbol", "")
        if symbol and symbol not in snapshots:
            snapshot = get_microstructure_for_symbol(
                symbol, market_data_cache, router_health
            )
            snapshots[symbol] = snapshot
            regimes[symbol] = classify_execution_regime(snapshot, minotaur_cfg)
    
    # Update state with regime summary
    _MINOTAUR_STATE.symbols_in_thin = [
        s for s, r in regimes.items() if r.regime == REGIME_THIN
    ]
    _MINOTAUR_STATE.symbols_in_crunch = [
        s for s, r in regimes.items() if r.regime == REGIME_CRUNCH
    ]
    
    # Check throttling
    throttle_active, halt_new, throttle_reason = check_throttling(
        list(regimes.values()),
        _MINOTAUR_STATE.orders_this_cycle,
        minotaur_cfg,
    )
    _MINOTAUR_STATE.throttling_active = throttle_active
    _MINOTAUR_STATE.halt_new_positions = halt_new
    
    if throttle_active:
        log_throttle_event(throttle_reason, _MINOTAUR_STATE.symbols_in_thin)
    
    # Build execution plans
    raw_plans = []
    for intent in merged_intents:
        symbol = intent.get("symbol", "")
        side = intent.get("side", intent.get("direction", "LONG")).upper()
        qty = float(intent.get("qty", 0) or 0)
        price = float(intent.get("price", intent.get("entry_price", 0)) or 0)
        
        if qty <= 0 or price <= 0:
            continue
        
        regime = regimes.get(symbol, ExecutionRegime(symbol=symbol))
        trend_bias = trend_biases.get(symbol) if trend_biases else None
        head_importance = head_importances.get(symbol) if head_importances else None
        
        plan = build_execution_plan(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            cfg=minotaur_cfg,
            regime=regime,
            trend_bias=trend_bias,
            head_importance=head_importance,
        )
        raw_plans.append(plan)
        log_plan_event(plan)
    
    # Apply throttling if active
    if throttle_active:
        final_plans = apply_throttling(raw_plans, regimes, minotaur_cfg)
    else:
        final_plans = raw_plans
    
    # Update order count
    _MINOTAUR_STATE.orders_this_cycle += sum(p.slice_count for p in final_plans)
    _MINOTAUR_STATE.updated_ts = time.time()
    
    _LOG.info(
        "[minotaur] planned %d intents -> %d plans, %d child orders, throttle=%s",
        len(merged_intents),
        len(final_plans),
        sum(p.slice_count for p in final_plans),
        throttle_active,
    )
    
    return final_plans, _MINOTAUR_STATE


# ---------------------------------------------------------------------------
# Fill Processing
# ---------------------------------------------------------------------------


def process_fill_for_quality(
    symbol: str,
    side: str,
    fill_price: float,
    model_price: float,
    fill_qty: float,
    target_qty: float,
    used_twap: bool,
    regime: str = REGIME_NORMAL,
    cfg: Optional[MinotaurConfig] = None,
) -> ExecutionQualityStats:
    """
    Process a fill and update execution quality stats.
    
    Args:
        symbol: Trading symbol
        side: "LONG" or "SHORT"
        fill_price: Actual fill price
        model_price: Model/expected price
        fill_qty: Filled quantity
        target_qty: Target quantity
        used_twap: Whether TWAP execution was used
        regime: Current execution regime
        cfg: Optional config for quality targets
        
    Returns:
        Updated ExecutionQualityStats
    """
    global _QUALITY_STATS
    
    # Get or create stats for this symbol
    if symbol not in _QUALITY_STATS:
        _QUALITY_STATS[symbol] = ExecutionQualityStats(symbol=symbol)
    
    stats = _QUALITY_STATS[symbol]
    
    # Calculate slippage
    slippage_bps = calculate_slippage_bps(fill_price, model_price, side)
    
    # Calculate fill ratio
    fill_ratio = fill_qty / target_qty if target_qty > 0 else 1.0
    
    # Calculate notional
    notional = fill_qty * fill_price
    
    # Get lookback from config
    lookback = cfg.quality_lookback_trades if cfg else 200
    
    # Update stats
    updated_stats = update_quality_stats(
        stats=stats,
        slippage_bps=slippage_bps,
        fill_ratio=fill_ratio,
        notional=notional,
        used_twap=used_twap,
        regime=regime,
        lookback=lookback,
    )
    
    _QUALITY_STATS[symbol] = updated_stats
    
    # Log slippage spike events
    max_slippage = cfg.max_slippage_bps if cfg else 10.0
    if abs(slippage_bps) > max_slippage:
        log_slippage_event(
            symbol=symbol,
            slippage_bps=slippage_bps,
            threshold_bps=max_slippage,
            is_spike=True,
        )
    
    return updated_stats


def save_quality_state(cfg: MinotaurConfig) -> None:
    """Save quality stats to state file."""
    global _QUALITY_STATS, _MINOTAUR_STATE
    save_execution_quality_state(_QUALITY_STATS, cfg, _MINOTAUR_STATE)


# ---------------------------------------------------------------------------
# Child Order Dispatch
# ---------------------------------------------------------------------------


@dataclass
class ChildOrderScheduler:
    """
    Scheduler for dispatching child orders over time.
    
    Maintains a queue of pending child orders and dispatches them
    when their scheduled time arrives.
    """
    pending_orders: List[ChildOrder] = field(default_factory=list)
    dispatched_orders: List[ChildOrder] = field(default_factory=list)
    
    def add_plan(self, plan: ExecutionPlan) -> List[ChildOrder]:
        """
        Add an execution plan and generate child orders.
        
        Returns the generated child orders.
        """
        children = plan_child_orders(plan)
        self.pending_orders.extend(children)
        return children
    
    def get_ready_orders(self, current_ts: Optional[float] = None) -> List[ChildOrder]:
        """
        Get child orders ready for dispatch.
        
        Args:
            current_ts: Current timestamp (default: now)
            
        Returns:
            List of child orders ready to be sent
        """
        if current_ts is None:
            current_ts = time.time()
        
        ready = []
        still_pending = []
        
        for order in self.pending_orders:
            if order.earliest_ts <= current_ts:
                ready.append(order)
                self.dispatched_orders.append(order)
            else:
                still_pending.append(order)
        
        self.pending_orders = still_pending
        return ready
    
    def clear(self) -> None:
        """Clear all pending orders."""
        self.pending_orders.clear()
    
    @property
    def pending_count(self) -> int:
        """Number of pending orders."""
        return len(self.pending_orders)
    
    @property
    def dispatched_count(self) -> int:
        """Number of dispatched orders."""
        return len(self.dispatched_orders)


# Global scheduler instance
_CHILD_ORDER_SCHEDULER = ChildOrderScheduler()


def get_child_order_scheduler() -> ChildOrderScheduler:
    """Get the global child order scheduler."""
    return _CHILD_ORDER_SCHEDULER


def dispatch_ready_child_orders() -> List[ChildOrder]:
    """
    Get and return child orders ready for immediate dispatch.
    
    Returns:
        List of ChildOrder objects ready to be submitted to router
    """
    return _CHILD_ORDER_SCHEDULER.get_ready_orders()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def convert_intent_to_router_params(
    child_order: ChildOrder,
    base_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a child order to router submission parameters.
    
    Args:
        child_order: ChildOrder from Minotaur planner
        base_params: Optional base parameters to merge
        
    Returns:
        Dict suitable for router.submit()
    """
    params = base_params.copy() if base_params else {}
    
    params.update({
        "symbol": child_order.symbol,
        "side": child_order.side,
        "qty": child_order.target_qty,
        "execution_aggressiveness": child_order.aggressiveness_hint,
        "minotaur_sequence": child_order.sequence,
        "minotaur_plan_id": child_order.parent_plan_id,
    })
    
    return params


def get_symbol_execution_summary(symbol: str) -> Dict[str, Any]:
    """
    Get execution quality summary for a symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dict with quality metrics or empty dict if no data
    """
    if symbol not in _QUALITY_STATS:
        return {}
    
    stats = _QUALITY_STATS[symbol]
    return {
        "avg_slippage_bps": stats.avg_slippage_bps,
        "p95_slippage_bps": stats.p95_slippage_bps,
        "max_slippage_bps": stats.max_slippage_bps,
        "fill_ratio": stats.fill_ratio,
        "twap_usage_pct": stats.twap_usage_pct,
        "last_regime": stats.last_regime,
        "trade_count": stats.trade_count,
    }


def get_execution_quality_summary() -> Dict[str, Any]:
    """
    Get overall execution quality summary.
    
    Returns:
        Dict with aggregate metrics across all symbols
    """
    if not _QUALITY_STATS:
        return {
            "total_symbols": 0,
            "avg_slippage_bps": 0.0,
            "avg_fill_ratio": 1.0,
            "twap_usage_pct": 0.0,
        }
    
    total = len(_QUALITY_STATS)
    avg_slip = sum(s.avg_slippage_bps for s in _QUALITY_STATS.values()) / total
    avg_fill = sum(s.fill_ratio for s in _QUALITY_STATS.values()) / total
    avg_twap = sum(s.twap_usage_pct for s in _QUALITY_STATS.values()) / total
    
    return {
        "total_symbols": total,
        "avg_slippage_bps": round(avg_slip, 3),
        "avg_fill_ratio": round(avg_fill, 4),
        "twap_usage_pct": round(avg_twap, 4),
        "symbols_tracked": list(_QUALITY_STATS.keys()),
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # State
    "get_minotaur_runtime_state",
    "reset_cycle_state",
    "get_quality_stats",
    # Microstructure
    "get_microstructure_for_symbol",
    # Intent processing
    "run_minotaur_for_intents",
    # Fill processing
    "process_fill_for_quality",
    "save_quality_state",
    # Child orders
    "ChildOrderScheduler",
    "get_child_order_scheduler",
    "dispatch_ready_child_orders",
    "convert_intent_to_router_params",
    # Utilities
    "get_symbol_execution_summary",
    "get_execution_quality_summary",
    # Execution Alpha integration
    "process_fill_for_alpha",
    "get_alpha_aggregator",
    "save_alpha_state",
]


# ---------------------------------------------------------------------------
# Execution Alpha Integration (v7.9_P4)
# ---------------------------------------------------------------------------

# Global alpha aggregator instance
_ALPHA_AGGREGATOR = None
_ALPHA_CONFIG = None
_ALPHA_ALERTS: List[Dict[str, Any]] = []


def _get_alpha_aggregator():
    """Get or initialize the global alpha aggregator."""
    global _ALPHA_AGGREGATOR, _ALPHA_CONFIG
    
    if _ALPHA_AGGREGATOR is None:
        try:
            from execution.execution_alpha import (
                AlphaAggregator,
                load_execution_alpha_config,
            )
            _ALPHA_CONFIG = load_execution_alpha_config()
            _ALPHA_AGGREGATOR = AlphaAggregator(
                lookback_trades=_ALPHA_CONFIG.lookback_trades
            )
        except ImportError:
            _LOG.debug("execution_alpha module not available")
            return None
    
    return _ALPHA_AGGREGATOR


def get_alpha_aggregator():
    """Public accessor for alpha aggregator."""
    return _get_alpha_aggregator()


def process_fill_for_alpha(
    fill: Dict[str, Any],
    quotes: Optional[Dict[str, Any]] = None,
    regime: Optional[str] = None,
    head_contributions: Optional[Dict[str, float]] = None,
    cfg = None,
) -> Optional[Any]:
    """
    Process a fill for execution alpha tracking.
    
    This should be called after each fill is received, alongside
    process_fill_for_quality.
    
    Args:
        fill: Fill record with symbol, side, qty, price
        quotes: Quote cache for model price resolution
        regime: Execution regime (NORMAL, THIN, etc.)
        head_contributions: Head contribution weights
        cfg: Optional ExecutionAlphaConfig override
    
    Returns:
        AlphaSample if alpha tracking is enabled, None otherwise
    """
    global _ALPHA_ALERTS
    
    try:
        from execution.execution_alpha import (
            load_execution_alpha_config,
            resolve_model_price,
            create_alpha_sample,
            check_and_generate_alerts,
            log_alert_event,
        )
    except ImportError:
        return None
    
    if cfg is None:
        cfg = load_execution_alpha_config()
    
    if not cfg.enabled:
        return None
    
    aggregator = _get_alpha_aggregator()
    if aggregator is None:
        return None
    
    # Resolve model price
    model_price = resolve_model_price(fill, quotes, cfg)
    if model_price is None:
        # Fallback to fill price (zero alpha)
        model_price = float(fill.get("price", fill.get("fill_price", 0)))
    
    # Get head contributions from fill if not provided
    if head_contributions is None:
        head_contributions = fill.get("head_contributions", {})
    
    # Create sample
    sample = create_alpha_sample(
        fill=fill,
        model_price=model_price,
        regime=regime,
        head_contributions=head_contributions,
    )
    
    # Add to aggregator
    aggregator.add_sample(sample)
    
    # Check for alerts
    symbol_stats = aggregator.get_symbol_stats(sample.symbol)
    alerts = check_and_generate_alerts(sample, symbol_stats, cfg)
    
    for alert in alerts:
        log_alert_event(alert)
        _ALPHA_ALERTS.append(alert)
    
    # Keep only last 100 alerts in memory
    _ALPHA_ALERTS = _ALPHA_ALERTS[-100:]
    
    return sample


def save_alpha_state(cfg = None) -> None:
    """
    Save execution alpha state to disk.
    
    Called at the end of each executor cycle.
    """
    global _ALPHA_ALERTS
    
    try:
        from execution.execution_alpha import (
            load_execution_alpha_config,
            save_execution_alpha_state,
        )
    except ImportError:
        return
    
    if cfg is None:
        cfg = load_execution_alpha_config()
    
    if not cfg.enabled:
        return
    
    aggregator = _get_alpha_aggregator()
    if aggregator is None:
        return
    
    save_execution_alpha_state(aggregator, cfg, _ALPHA_ALERTS[-10:])

