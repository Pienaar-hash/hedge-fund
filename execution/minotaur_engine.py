"""
Minotaur Execution Engine — v7.9_P3

Microstructure-aware execution layer that:
1. Senses liquidity, spread, volatility, and microstructure regimes per symbol
2. Dynamically chooses aggressiveness (instant vs sliced, passive vs aggressive)
3. Runs liquidity-aware TWAP slicing for larger trades / thin markets
4. Implements execution throttling when microstructure conditions are hostile
5. Tracks execution quality (slippage vs model, impact, fill quality)

This module sits between intent generation (Hydra/Cerberus) and order submission (Router).

Usage:
    from execution.minotaur_engine import (
        load_minotaur_config,
        is_minotaur_enabled,
        build_microstructure_snapshot,
        classify_execution_regime,
        build_execution_plan,
        plan_child_orders,
    )

    if is_minotaur_enabled(strategy_config):
        snapshot = build_microstructure_snapshot(symbol, ...)
        regime = classify_execution_regime(snapshot, cfg)
        plan = build_execution_plan(symbol, side, qty, price, cfg, regime)
        child_orders = plan_child_orders(plan)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_AGGRESSIVENESS = 0.6
DEFAULT_TREND_BONUS = 0.1
DEFAULT_CRISIS_CAP = 0.3
DEFAULT_WIDE_SPREAD_BPS = 10.0
DEFAULT_THIN_DEPTH_USD = 2000.0
DEFAULT_VOL_SPIKE_MULT = 2.5
DEFAULT_MIN_NOTIONAL_FOR_TWAP = 500.0
DEFAULT_MAX_CHILD_NOTIONAL = 150.0
DEFAULT_MIN_SLICE_COUNT = 2
DEFAULT_MAX_SLICE_COUNT = 12
DEFAULT_TWAP_MIN_SECONDS = 60
DEFAULT_TWAP_MAX_SECONDS = 900

# Execution regimes
REGIME_NORMAL = "NORMAL"
REGIME_THIN = "THIN"
REGIME_WIDE_SPREAD = "WIDE_SPREAD"
REGIME_SPIKE = "SPIKE"
REGIME_CRUNCH = "CRUNCH"

# Slicing modes
MODE_INSTANT = "INSTANT"
MODE_TWAP = "TWAP"
MODE_STEPPED = "STEPPED"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MicrostructureSnapshot:
    """
    Point-in-time microstructure features for a symbol.
    
    Captures liquidity, spread, volatility, and recent execution quality.
    """
    symbol: str
    spread_bps: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    top_depth_usd: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    book_imbalance: float = 0.0          # (bid - ask) / (bid + ask), [-1, 1]
    trade_imbalance: float = 0.0         # net buy pressure, [-1, 1]
    realized_vol_1m: float = 0.0         # 1-minute realized vol
    realized_vol_5m: float = 0.0         # 5-minute realized vol
    recent_slippage_bps: float = 0.0     # rolling avg slippage
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionRegime:
    """
    Classified execution regime for a symbol.
    
    Regimes:
    - NORMAL: Good liquidity, tight spreads, stable conditions
    - THIN: Low depth, may cause slippage on larger orders
    - WIDE_SPREAD: High bid/ask spread, passive orders preferred
    - SPIKE: Volatility spike, reduce aggressiveness
    - CRUNCH: Both thin and wide — hostile conditions
    """
    symbol: str
    regime: str = REGIME_NORMAL
    liquidity_score: float = 1.0     # 0 = illiquid, 1 = highly liquid
    risk_score: float = 0.0          # 0 = low risk, 1 = high risk
    spread_bps: float = 0.0
    depth_usd: float = 0.0
    vol_ratio: float = 1.0           # vol_1m / vol_5m
    notes: str = ""


@dataclass
class ExecutionPlan:
    """
    Execution plan for a single intent.
    
    Describes how to execute a trade: instant vs sliced,
    number of slices, timing, and aggressiveness.
    """
    symbol: str
    side: str                        # "LONG" or "SHORT"
    total_qty: float
    total_notional: float
    slicing_mode: str = MODE_INSTANT  # INSTANT, TWAP, STEPPED
    slice_count: int = 1
    schedule_seconds: int = 0
    aggressiveness: float = 0.6       # 0 = fully passive, 1 = fully aggressive
    regime: str = REGIME_NORMAL
    notes: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ChildOrder:
    """
    A single child order in a TWAP execution plan.
    """
    symbol: str
    side: str
    target_qty: float
    target_notional: float
    sequence: int                     # 0, 1, 2, ...
    earliest_ts: float               # Unix timestamp when order can be sent
    aggressiveness_hint: float       # Hint for router
    parent_plan_id: str = ""
    notes: str = ""


@dataclass
class MinotaurConfig:
    """
    Configuration for Minotaur execution engine.
    """
    enabled: bool = False
    min_notional_for_twap_usd: float = DEFAULT_MIN_NOTIONAL_FOR_TWAP
    max_child_order_notional_usd: float = DEFAULT_MAX_CHILD_NOTIONAL
    min_slice_count: int = DEFAULT_MIN_SLICE_COUNT
    max_slice_count: int = DEFAULT_MAX_SLICE_COUNT
    twap_min_seconds: int = DEFAULT_TWAP_MIN_SECONDS
    twap_max_seconds: int = DEFAULT_TWAP_MAX_SECONDS
    aggressiveness_base: float = DEFAULT_BASE_AGGRESSIVENESS
    aggressiveness_trend_bonus: float = DEFAULT_TREND_BONUS
    aggressiveness_crisis_cap: float = DEFAULT_CRISIS_CAP
    wide_spread_bps: float = DEFAULT_WIDE_SPREAD_BPS
    thin_depth_usd: float = DEFAULT_THIN_DEPTH_USD
    vol_spike_mult: float = DEFAULT_VOL_SPIKE_MULT
    max_symbols_in_thin_liquidity: int = 3
    max_new_orders_per_cycle: int = 25
    halt_on_liquidity_crunch: bool = True
    max_slippage_bps: float = 10.0
    max_tail_slippage_bps: float = 30.0
    quality_lookback_trades: int = 200


@dataclass
class ExecutionQualityStats:
    """
    Per-symbol execution quality statistics.
    """
    symbol: str
    avg_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    fill_ratio: float = 1.0           # Filled qty / target qty
    mean_notional: float = 0.0
    twap_usage_pct: float = 0.0       # % of orders using TWAP
    last_regime: str = REGIME_NORMAL
    trade_count: int = 0
    last_updated_ts: float = field(default_factory=time.time)


@dataclass
class MinotaurState:
    """
    Runtime state for Minotaur engine.
    """
    enabled: bool = False
    symbols_in_thin: List[str] = field(default_factory=list)
    symbols_in_crunch: List[str] = field(default_factory=list)
    orders_this_cycle: int = 0
    throttling_active: bool = False
    halt_new_positions: bool = False
    quality_stats: Dict[str, ExecutionQualityStats] = field(default_factory=dict)
    updated_ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------


def load_minotaur_config(strategy_cfg: Mapping[str, Any]) -> MinotaurConfig:
    """
    Load Minotaur configuration from strategy config.
    
    Args:
        strategy_cfg: Strategy configuration dict
        
    Returns:
        MinotaurConfig dataclass
    """
    minotaur_block = strategy_cfg.get("execution_minotaur", {})
    
    agg = minotaur_block.get("aggressiveness", {})
    thresholds = minotaur_block.get("microstructure_thresholds", {})
    throttling = minotaur_block.get("throttling", {})
    quality = minotaur_block.get("quality_targets", {})
    
    return MinotaurConfig(
        enabled=minotaur_block.get("enabled", False),
        min_notional_for_twap_usd=minotaur_block.get("min_notional_for_twap_usd", DEFAULT_MIN_NOTIONAL_FOR_TWAP),
        max_child_order_notional_usd=minotaur_block.get("max_child_order_notional_usd", DEFAULT_MAX_CHILD_NOTIONAL),
        min_slice_count=minotaur_block.get("min_slice_count", DEFAULT_MIN_SLICE_COUNT),
        max_slice_count=minotaur_block.get("max_slice_count", DEFAULT_MAX_SLICE_COUNT),
        twap_min_seconds=minotaur_block.get("twap_min_seconds", DEFAULT_TWAP_MIN_SECONDS),
        twap_max_seconds=minotaur_block.get("twap_max_seconds", DEFAULT_TWAP_MAX_SECONDS),
        aggressiveness_base=agg.get("base", DEFAULT_BASE_AGGRESSIVENESS),
        aggressiveness_trend_bonus=agg.get("trend_bonus", DEFAULT_TREND_BONUS),
        aggressiveness_crisis_cap=agg.get("crisis_cap", DEFAULT_CRISIS_CAP),
        wide_spread_bps=thresholds.get("wide_spread_bps", DEFAULT_WIDE_SPREAD_BPS),
        thin_depth_usd=thresholds.get("thin_depth_usd", DEFAULT_THIN_DEPTH_USD),
        vol_spike_mult=thresholds.get("vol_spike_mult", DEFAULT_VOL_SPIKE_MULT),
        max_symbols_in_thin_liquidity=throttling.get("max_symbols_in_thin_liquidity", 3),
        max_new_orders_per_cycle=throttling.get("max_new_orders_per_cycle", 25),
        halt_on_liquidity_crunch=throttling.get("halt_on_liquidity_crunch", True),
        max_slippage_bps=quality.get("max_slippage_bps", 10.0),
        max_tail_slippage_bps=quality.get("max_tail_slippage_bps", 30.0),
        quality_lookback_trades=quality.get("lookback_trades", 200),
    )


def is_minotaur_enabled(strategy_cfg: Mapping[str, Any]) -> bool:
    """Check if Minotaur execution engine is enabled."""
    return strategy_cfg.get("execution_minotaur", {}).get("enabled", False)


# ---------------------------------------------------------------------------
# Microstructure Feature Extraction
# ---------------------------------------------------------------------------


def build_microstructure_snapshot(
    symbol: str,
    best_bid: float,
    best_ask: float,
    bid_depth_usd: float = 0.0,
    ask_depth_usd: float = 0.0,
    trade_imbalance: float = 0.0,
    realized_vol_1m: float = 0.0,
    realized_vol_5m: float = 0.0,
    recent_slippage_bps: float = 0.0,
) -> MicrostructureSnapshot:
    """
    Build microstructure snapshot from market data.
    
    Args:
        symbol: Trading symbol
        best_bid: Best bid price
        best_ask: Best ask price
        bid_depth_usd: USD depth on bid side (top N levels)
        ask_depth_usd: USD depth on ask side (top N levels)
        trade_imbalance: Net buy/sell pressure [-1, 1]
        realized_vol_1m: 1-minute realized volatility
        realized_vol_5m: 5-minute realized volatility
        recent_slippage_bps: Rolling average slippage in bps
        
    Returns:
        MicrostructureSnapshot
    """
    mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = 0.0
    if mid_price > 0:
        spread_bps = (best_ask - best_bid) / mid_price * 10_000
    
    top_depth_usd = min(bid_depth_usd, ask_depth_usd) if bid_depth_usd > 0 and ask_depth_usd > 0 else 0.0
    
    # Book imbalance: positive = more bid depth, negative = more ask depth
    book_imbalance = 0.0
    total_depth = bid_depth_usd + ask_depth_usd
    if total_depth > 0:
        book_imbalance = (bid_depth_usd - ask_depth_usd) / total_depth
    
    return MicrostructureSnapshot(
        symbol=symbol.upper(),
        spread_bps=spread_bps,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        top_depth_usd=top_depth_usd,
        bid_depth_usd=bid_depth_usd,
        ask_depth_usd=ask_depth_usd,
        book_imbalance=book_imbalance,
        trade_imbalance=trade_imbalance,
        realized_vol_1m=realized_vol_1m,
        realized_vol_5m=realized_vol_5m,
        recent_slippage_bps=recent_slippage_bps,
    )


def build_snapshot_from_orderbook(
    symbol: str,
    orderbook: Dict[str, Any],
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    recent_slippage_bps: float = 0.0,
) -> MicrostructureSnapshot:
    """
    Build microstructure snapshot from orderbook data structure.
    
    Args:
        symbol: Trading symbol
        orderbook: Dict with 'bids' and 'asks' lists of [price, qty] pairs
        recent_trades: Optional list of recent trades for vol calculation
        recent_slippage_bps: Rolling average slippage
        
    Returns:
        MicrostructureSnapshot
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    
    # Calculate depth in USD for top 5 levels
    bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:5]) if bids else 0.0
    ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks[:5]) if asks else 0.0
    
    # Calculate realized vol from recent trades if available
    realized_vol_1m = 0.0
    realized_vol_5m = 0.0
    trade_imbalance = 0.0
    
    if recent_trades:
        # Simple volatility: std of returns
        prices = [float(t.get("price", 0)) for t in recent_trades[-60:] if t.get("price")]
        if len(prices) >= 2:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            if returns:
                realized_vol_1m = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        
        prices_5m = [float(t.get("price", 0)) for t in recent_trades[-300:] if t.get("price")]
        if len(prices_5m) >= 2:
            returns_5m = [(prices_5m[i] - prices_5m[i-1]) / prices_5m[i-1] for i in range(1, len(prices_5m))]
            if returns_5m:
                realized_vol_5m = (sum(r**2 for r in returns_5m) / len(returns_5m)) ** 0.5
        
        # Trade imbalance: net buy pressure
        buys = sum(1 for t in recent_trades[-100:] if t.get("isBuyerMaker") is False)
        sells = sum(1 for t in recent_trades[-100:] if t.get("isBuyerMaker") is True)
        total_trades = buys + sells
        if total_trades > 0:
            trade_imbalance = (buys - sells) / total_trades
    
    return build_microstructure_snapshot(
        symbol=symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_usd=bid_depth_usd,
        ask_depth_usd=ask_depth_usd,
        trade_imbalance=trade_imbalance,
        realized_vol_1m=realized_vol_1m,
        realized_vol_5m=realized_vol_5m,
        recent_slippage_bps=recent_slippage_bps,
    )


# ---------------------------------------------------------------------------
# Regime Classification
# ---------------------------------------------------------------------------


def classify_execution_regime(
    snapshot: MicrostructureSnapshot,
    cfg: MinotaurConfig,
) -> ExecutionRegime:
    """
    Classify execution regime based on microstructure features.
    
    Args:
        snapshot: Microstructure snapshot for a symbol
        cfg: Minotaur configuration
        
    Returns:
        ExecutionRegime with regime classification and scores
    """
    symbol = snapshot.symbol
    spread_bps = snapshot.spread_bps
    depth_usd = snapshot.top_depth_usd
    vol_1m = snapshot.realized_vol_1m
    vol_5m = snapshot.realized_vol_5m
    
    # Calculate volatility ratio
    vol_ratio = vol_1m / vol_5m if vol_5m > 0 else 1.0
    
    # Determine regime
    is_thin = depth_usd < cfg.thin_depth_usd
    is_wide = spread_bps > cfg.wide_spread_bps
    is_spike = vol_ratio > cfg.vol_spike_mult
    
    if is_thin and is_wide:
        regime = REGIME_CRUNCH
        notes = f"thin depth ({depth_usd:.0f}$) + wide spread ({spread_bps:.1f}bps)"
    elif is_spike:
        regime = REGIME_SPIKE
        notes = f"vol spike: 1m/5m ratio = {vol_ratio:.2f}"
    elif is_thin:
        regime = REGIME_THIN
        notes = f"thin depth: {depth_usd:.0f}$ < {cfg.thin_depth_usd:.0f}$"
    elif is_wide:
        regime = REGIME_WIDE_SPREAD
        notes = f"wide spread: {spread_bps:.1f}bps > {cfg.wide_spread_bps:.1f}bps"
    else:
        regime = REGIME_NORMAL
        notes = "normal conditions"
    
    # Calculate liquidity score [0, 1]
    # Higher depth and tighter spread = higher liquidity
    depth_score = min(depth_usd / (cfg.thin_depth_usd * 5), 1.0) if cfg.thin_depth_usd > 0 else 1.0
    spread_score = max(0.0, 1.0 - spread_bps / (cfg.wide_spread_bps * 2)) if cfg.wide_spread_bps > 0 else 1.0
    liquidity_score = (depth_score * 0.6 + spread_score * 0.4)
    
    # Calculate risk score [0, 1]
    # Higher vol spike + recent slippage = higher risk
    vol_risk = min(vol_ratio / (cfg.vol_spike_mult * 2), 1.0) if cfg.vol_spike_mult > 0 else 0.0
    slippage_risk = min(snapshot.recent_slippage_bps / 20.0, 1.0) if snapshot.recent_slippage_bps > 0 else 0.0
    risk_score = (vol_risk * 0.6 + slippage_risk * 0.4)
    
    return ExecutionRegime(
        symbol=symbol,
        regime=regime,
        liquidity_score=round(liquidity_score, 3),
        risk_score=round(risk_score, 3),
        spread_bps=round(spread_bps, 2),
        depth_usd=round(depth_usd, 2),
        vol_ratio=round(vol_ratio, 2),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Execution Plan Builder
# ---------------------------------------------------------------------------


def build_execution_plan(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    cfg: MinotaurConfig,
    regime: ExecutionRegime,
    trend_bias: Optional[float] = None,
    head_importance: Optional[float] = None,
) -> ExecutionPlan:
    """
    Build execution plan for a trade.
    
    Args:
        symbol: Trading symbol
        side: "LONG" or "SHORT"
        qty: Order quantity
        price: Current price (for notional calculation)
        cfg: Minotaur configuration
        regime: Classified execution regime
        trend_bias: Optional trend factor [-1, 1] in direction of trade
        head_importance: Optional head weight from Hydra [0, 1]
        
    Returns:
        ExecutionPlan describing how to execute the trade
    """
    notional = qty * price
    side_upper = side.upper()
    
    # Decide slicing mode
    if notional < cfg.min_notional_for_twap_usd and regime.regime == REGIME_NORMAL:
        # Small order in normal conditions: execute instantly
        return ExecutionPlan(
            symbol=symbol.upper(),
            side=side_upper,
            total_qty=qty,
            total_notional=notional,
            slicing_mode=MODE_INSTANT,
            slice_count=1,
            schedule_seconds=0,
            aggressiveness=_compute_aggressiveness(cfg, regime, trend_bias),
            regime=regime.regime,
            notes="instant execution: small notional + normal regime",
        )
    
    # Calculate slice count based on notional and max child order size
    if cfg.max_child_order_notional_usd > 0:
        raw_slices = math.ceil(notional / cfg.max_child_order_notional_usd)
    else:
        raw_slices = cfg.min_slice_count
    
    slice_count = max(cfg.min_slice_count, min(raw_slices, cfg.max_slice_count))
    
    # Calculate schedule duration based on regime
    schedule_seconds = _compute_schedule_seconds(cfg, regime, slice_count)
    
    # Determine slicing mode
    if regime.regime in (REGIME_THIN, REGIME_WIDE_SPREAD):
        slicing_mode = MODE_TWAP
    elif regime.regime in (REGIME_SPIKE, REGIME_CRUNCH):
        slicing_mode = MODE_STEPPED  # More careful in hostile conditions
    else:
        slicing_mode = MODE_TWAP
    
    # Compute aggressiveness
    aggressiveness = _compute_aggressiveness(cfg, regime, trend_bias)
    
    notes = f"{slicing_mode}: {slice_count} slices over {schedule_seconds}s, regime={regime.regime}"
    
    return ExecutionPlan(
        symbol=symbol.upper(),
        side=side_upper,
        total_qty=qty,
        total_notional=notional,
        slicing_mode=slicing_mode,
        slice_count=slice_count,
        schedule_seconds=schedule_seconds,
        aggressiveness=round(aggressiveness, 3),
        regime=regime.regime,
        notes=notes,
    )


def _compute_aggressiveness(
    cfg: MinotaurConfig,
    regime: ExecutionRegime,
    trend_bias: Optional[float],
) -> float:
    """
    Compute execution aggressiveness [0, 1].
    
    Higher = more aggressive (market orders, tighter limits).
    Lower = more passive (wider limits, patient execution).
    """
    agg = cfg.aggressiveness_base
    
    # Trend bonus: if trend is in our favour, be more aggressive
    if trend_bias is not None and trend_bias > 0:
        agg += cfg.aggressiveness_trend_bonus * trend_bias
    
    # Crisis cap: reduce aggressiveness in hostile regimes
    if regime.regime in (REGIME_SPIKE, REGIME_CRUNCH):
        agg = min(agg, cfg.aggressiveness_crisis_cap)
    elif regime.regime == REGIME_THIN:
        agg = min(agg, cfg.aggressiveness_base * 0.8)
    elif regime.regime == REGIME_WIDE_SPREAD:
        agg = min(agg, cfg.aggressiveness_base * 0.7)
    
    return max(0.0, min(1.0, agg))


def _compute_schedule_seconds(
    cfg: MinotaurConfig,
    regime: ExecutionRegime,
    slice_count: int,
) -> int:
    """
    Compute TWAP schedule duration based on regime.
    """
    base_duration = cfg.twap_min_seconds
    
    # Scale up duration for adverse regimes
    if regime.regime == REGIME_NORMAL:
        duration_mult = 1.0
    elif regime.regime == REGIME_THIN:
        duration_mult = 2.0
    elif regime.regime == REGIME_WIDE_SPREAD:
        duration_mult = 1.5
    elif regime.regime == REGIME_SPIKE:
        duration_mult = 2.5
    elif regime.regime == REGIME_CRUNCH:
        duration_mult = 3.0
    else:
        duration_mult = 1.0
    
    # Also scale by slice count (more slices = more time)
    slice_mult = max(1.0, slice_count / 4.0)
    
    duration = int(base_duration * duration_mult * slice_mult)
    
    return max(cfg.twap_min_seconds, min(duration, cfg.twap_max_seconds))


# ---------------------------------------------------------------------------
# TWAP / Slicing Planner
# ---------------------------------------------------------------------------


def plan_child_orders(
    plan: ExecutionPlan,
    start_ts: Optional[float] = None,
    plan_id: Optional[str] = None,
) -> List[ChildOrder]:
    """
    Plan child orders for a TWAP execution.
    
    Args:
        plan: ExecutionPlan from build_execution_plan
        start_ts: Optional start timestamp (default: now)
        plan_id: Optional parent plan ID for linking
        
    Returns:
        List of ChildOrder objects
    """
    if start_ts is None:
        start_ts = time.time()
    
    if plan_id is None:
        plan_id = f"{plan.symbol}_{plan.side}_{int(start_ts)}"
    
    if plan.slicing_mode == MODE_INSTANT or plan.slice_count <= 1:
        # Single order
        return [ChildOrder(
            symbol=plan.symbol,
            side=plan.side,
            target_qty=plan.total_qty,
            target_notional=plan.total_notional,
            sequence=0,
            earliest_ts=start_ts,
            aggressiveness_hint=plan.aggressiveness,
            parent_plan_id=plan_id,
            notes="instant execution",
        )]
    
    child_orders = []
    per_slice_qty = plan.total_qty / plan.slice_count
    per_slice_notional = plan.total_notional / plan.slice_count
    
    if plan.slicing_mode == MODE_TWAP:
        # Even distribution over time
        interval = plan.schedule_seconds / plan.slice_count
        for i in range(plan.slice_count):
            child_orders.append(ChildOrder(
                symbol=plan.symbol,
                side=plan.side,
                target_qty=per_slice_qty,
                target_notional=per_slice_notional,
                sequence=i,
                earliest_ts=start_ts + (i * interval),
                aggressiveness_hint=plan.aggressiveness,
                parent_plan_id=plan_id,
                notes=f"TWAP slice {i+1}/{plan.slice_count}",
            ))
    
    elif plan.slicing_mode == MODE_STEPPED:
        # Front-load with decreasing sizes (for hostile conditions)
        # First half gets 60%, second half gets 40%
        weights = _generate_stepped_weights(plan.slice_count)
        interval = plan.schedule_seconds / plan.slice_count
        
        for i, weight in enumerate(weights):
            child_orders.append(ChildOrder(
                symbol=plan.symbol,
                side=plan.side,
                target_qty=plan.total_qty * weight,
                target_notional=plan.total_notional * weight,
                sequence=i,
                earliest_ts=start_ts + (i * interval),
                aggressiveness_hint=plan.aggressiveness * (1 - i * 0.05),  # Decrease over time
                parent_plan_id=plan_id,
                notes=f"STEPPED slice {i+1}/{plan.slice_count} (weight={weight:.3f})",
            ))
    
    return child_orders


def _generate_stepped_weights(slice_count: int) -> List[float]:
    """
    Generate decreasing weights for STEPPED execution.
    
    First slices are larger, last slices are smaller.
    """
    if slice_count <= 1:
        return [1.0]
    
    # Linear decreasing weights, normalized to sum to 1
    raw_weights = [slice_count - i for i in range(slice_count)]
    total = sum(raw_weights)
    return [w / total for w in raw_weights]


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def check_throttling(
    regimes: List[ExecutionRegime],
    pending_orders: int,
    cfg: MinotaurConfig,
) -> Tuple[bool, bool, str]:
    """
    Check if throttling should be applied.
    
    Args:
        regimes: List of ExecutionRegime for all symbols with pending orders
        pending_orders: Number of orders pending for this cycle
        cfg: Minotaur configuration
        
    Returns:
        Tuple of (throttle_active, halt_new_positions, reason)
    """
    thin_symbols = [r.symbol for r in regimes if r.regime == REGIME_THIN]
    crunch_symbols = [r.symbol for r in regimes if r.regime == REGIME_CRUNCH]
    
    throttle_active = False
    halt_new = False
    reasons = []
    
    # Check thin liquidity limit
    if len(thin_symbols) > cfg.max_symbols_in_thin_liquidity:
        throttle_active = True
        reasons.append(f"{len(thin_symbols)} symbols in THIN liquidity")
    
    # Check crunch symbols
    if crunch_symbols and cfg.halt_on_liquidity_crunch:
        if len(crunch_symbols) >= 2:
            halt_new = True
            reasons.append(f"{len(crunch_symbols)} symbols in CRUNCH")
    
    # Check order rate limit
    if pending_orders > cfg.max_new_orders_per_cycle:
        throttle_active = True
        reasons.append(f"{pending_orders} orders > max {cfg.max_new_orders_per_cycle}")
    
    reason = "; ".join(reasons) if reasons else "no throttling"
    return throttle_active, halt_new, reason


def apply_throttling(
    plans: List[ExecutionPlan],
    regimes: Dict[str, ExecutionRegime],
    cfg: MinotaurConfig,
    max_orders: Optional[int] = None,
) -> List[ExecutionPlan]:
    """
    Apply throttling to execution plans.
    
    Reduces slice counts and drops non-critical plans when throttling active.
    
    Args:
        plans: List of ExecutionPlan objects
        regimes: Dict of symbol -> ExecutionRegime
        cfg: Minotaur configuration
        max_orders: Optional override for max orders this cycle
        
    Returns:
        Filtered and adjusted list of ExecutionPlan objects
    """
    if max_orders is None:
        max_orders = cfg.max_new_orders_per_cycle
    
    # Sort plans by total_notional (descending) - prioritize larger trades
    sorted_plans = sorted(plans, key=lambda p: p.total_notional, reverse=True)
    
    result = []
    total_child_orders = 0
    
    for plan in sorted_plans:
        regime = regimes.get(plan.symbol)
        
        # Skip new positions in CRUNCH if halt enabled
        if regime and regime.regime == REGIME_CRUNCH and cfg.halt_on_liquidity_crunch:
            _LOG.info("[minotaur] dropping plan for %s: CRUNCH regime + halt enabled", plan.symbol)
            continue
        
        # Adjust slice count if we're running low on order budget
        remaining_budget = max_orders - total_child_orders
        if remaining_budget <= 0:
            _LOG.info("[minotaur] throttling: order budget exhausted")
            break
        
        if plan.slice_count > remaining_budget:
            # Reduce slice count to fit budget
            adjusted_plan = ExecutionPlan(
                symbol=plan.symbol,
                side=plan.side,
                total_qty=plan.total_qty,
                total_notional=plan.total_notional,
                slicing_mode=MODE_INSTANT if remaining_budget == 1 else plan.slicing_mode,
                slice_count=remaining_budget,
                schedule_seconds=int(plan.schedule_seconds * remaining_budget / plan.slice_count),
                aggressiveness=plan.aggressiveness,
                regime=plan.regime,
                notes=f"{plan.notes} [THROTTLED: {plan.slice_count}->{remaining_budget} slices]",
            )
            result.append(adjusted_plan)
            total_child_orders += remaining_budget
        else:
            result.append(plan)
            total_child_orders += plan.slice_count
    
    return result


# ---------------------------------------------------------------------------
# Execution Quality Metrics
# ---------------------------------------------------------------------------


def calculate_slippage_bps(
    fill_price: float,
    model_price: float,
    side: str,
) -> float:
    """
    Calculate slippage in basis points.
    
    Positive = worse than model (paid more or received less).
    Negative = better than model (paid less or received more).
    
    Args:
        fill_price: Actual fill price
        model_price: Model/expected price (mid or limit)
        side: "LONG" or "SHORT"
        
    Returns:
        Slippage in basis points
    """
    if model_price <= 0:
        return 0.0
    
    diff_pct = (fill_price - model_price) / model_price
    
    # For LONG, positive diff is bad (paid more)
    # For SHORT, negative diff is bad (received less)
    if side.upper() == "LONG":
        return diff_pct * 10_000
    else:
        return -diff_pct * 10_000


def update_quality_stats(
    stats: ExecutionQualityStats,
    slippage_bps: float,
    fill_ratio: float,
    notional: float,
    used_twap: bool,
    regime: str,
    lookback: int = 200,
) -> ExecutionQualityStats:
    """
    Update execution quality statistics with a new fill.
    
    Uses exponential moving average for rolling stats.
    
    Args:
        stats: Existing stats object
        slippage_bps: Slippage for this fill
        fill_ratio: Filled qty / target qty for this order
        notional: Order notional
        used_twap: Whether this order used TWAP execution
        regime: Current execution regime
        lookback: Lookback window for weighting
        
    Returns:
        Updated ExecutionQualityStats
    """
    # EMA decay factor
    alpha = 2.0 / (lookback + 1) if lookback > 0 else 0.1
    
    # Update averages
    if stats.trade_count == 0:
        new_avg = slippage_bps
        new_notional = notional
        new_twap_pct = 1.0 if used_twap else 0.0
        new_fill_ratio = fill_ratio
    else:
        new_avg = alpha * slippage_bps + (1 - alpha) * stats.avg_slippage_bps
        new_notional = alpha * notional + (1 - alpha) * stats.mean_notional
        twap_val = 1.0 if used_twap else 0.0
        new_twap_pct = alpha * twap_val + (1 - alpha) * stats.twap_usage_pct
        new_fill_ratio = alpha * fill_ratio + (1 - alpha) * stats.fill_ratio
    
    # Update max (simple running max)
    new_max = max(stats.max_slippage_bps, slippage_bps)
    
    # Approximate p95 using max and avg
    # More accurate p95 would require keeping a buffer of recent values
    new_p95 = min(new_max, stats.avg_slippage_bps * 2.0 + 5.0) if stats.trade_count > 10 else new_max
    
    return ExecutionQualityStats(
        symbol=stats.symbol,
        avg_slippage_bps=round(new_avg, 3),
        p95_slippage_bps=round(new_p95, 3),
        max_slippage_bps=round(new_max, 3),
        fill_ratio=round(new_fill_ratio, 4),
        mean_notional=round(new_notional, 2),
        twap_usage_pct=round(new_twap_pct, 4),
        last_regime=regime,
        trade_count=stats.trade_count + 1,
        last_updated_ts=time.time(),
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

_STATE_DIR = Path("logs/state")
_QUALITY_STATE_FILE = _STATE_DIR / "execution_quality.json"


def load_execution_quality_state() -> Dict[str, Any]:
    """Load execution quality state from disk."""
    try:
        if _QUALITY_STATE_FILE.exists():
            return json.loads(_QUALITY_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        _LOG.warning("[minotaur] failed to load execution_quality state: %s", e)
    return {"updated_ts": None, "symbols": {}, "meta": {}}


def save_execution_quality_state(
    quality_stats: Dict[str, ExecutionQualityStats],
    cfg: MinotaurConfig,
    minotaur_state: Optional[MinotaurState] = None,
) -> None:
    """
    Save execution quality state to disk.
    
    Args:
        quality_stats: Dict of symbol -> ExecutionQualityStats
        cfg: Minotaur configuration (for meta)
        minotaur_state: Optional runtime state
    """
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        
        symbols = {}
        for symbol, stats in quality_stats.items():
            symbols[symbol] = {
                "avg_slippage_bps": stats.avg_slippage_bps,
                "p95_slippage_bps": stats.p95_slippage_bps,
                "max_slippage_bps": stats.max_slippage_bps,
                "fill_ratio": stats.fill_ratio,
                "mean_notional": stats.mean_notional,
                "twap_usage_pct": stats.twap_usage_pct,
                "last_regime": stats.last_regime,
                "trade_count": stats.trade_count,
                "last_updated_ts": stats.last_updated_ts,
            }
        
        meta = {
            "enabled": cfg.enabled,
            "max_slippage_target_bps": cfg.max_slippage_bps,
            "thin_liquidity_symbols": minotaur_state.symbols_in_thin if minotaur_state else [],
            "crunch_symbols": minotaur_state.symbols_in_crunch if minotaur_state else [],
            "throttling_active": minotaur_state.throttling_active if minotaur_state else False,
        }
        
        state = {
            "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbols": symbols,
            "meta": meta,
        }
        
        _QUALITY_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _LOG.debug("[minotaur] saved execution_quality state with %d symbols", len(symbols))
        
    except Exception as e:
        _LOG.warning("[minotaur] failed to save execution_quality state: %s", e)


# ---------------------------------------------------------------------------
# Event Logging
# ---------------------------------------------------------------------------

_EXEC_LOG_DIR = Path("logs/execution")
_EXEC_EVENTS_FILE = _EXEC_LOG_DIR / "execution_events.jsonl"


def log_execution_event(event: Dict[str, Any]) -> None:
    """
    Log an execution event to JSONL file.
    
    Args:
        event: Event dict with at minimum 'symbol' and 'event' keys
    """
    try:
        _EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        with open(_EXEC_EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            
    except Exception as e:
        _LOG.warning("[minotaur] failed to log execution event: %s", e)


def log_plan_event(plan: ExecutionPlan) -> None:
    """Log an execution plan creation event."""
    log_execution_event({
        "symbol": plan.symbol,
        "event": f"PLAN_{plan.slicing_mode}",
        "slice_count": plan.slice_count,
        "schedule_seconds": plan.schedule_seconds,
        "regime": plan.regime,
        "notional": round(plan.total_notional, 2),
        "aggressiveness": plan.aggressiveness,
    })


def log_slippage_event(
    symbol: str,
    slippage_bps: float,
    threshold_bps: float,
    is_spike: bool = False,
) -> None:
    """Log a slippage observation event."""
    log_execution_event({
        "symbol": symbol,
        "event": "SLIPPAGE_SPIKE" if is_spike else "SLIPPAGE",
        "slippage_bps": round(slippage_bps, 2),
        "threshold_bps": threshold_bps,
    })


def log_throttle_event(reason: str, symbols: List[str]) -> None:
    """Log a throttling event."""
    log_execution_event({
        "symbol": ",".join(symbols) if symbols else "GLOBAL",
        "event": "THROTTLE",
        "reason": reason,
    })


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "REGIME_NORMAL",
    "REGIME_THIN",
    "REGIME_WIDE_SPREAD",
    "REGIME_SPIKE",
    "REGIME_CRUNCH",
    "MODE_INSTANT",
    "MODE_TWAP",
    "MODE_STEPPED",
    # Dataclasses
    "MicrostructureSnapshot",
    "ExecutionRegime",
    "ExecutionPlan",
    "ChildOrder",
    "MinotaurConfig",
    "ExecutionQualityStats",
    "MinotaurState",
    # Config
    "load_minotaur_config",
    "is_minotaur_enabled",
    # Microstructure
    "build_microstructure_snapshot",
    "build_snapshot_from_orderbook",
    # Regime
    "classify_execution_regime",
    # Planning
    "build_execution_plan",
    "plan_child_orders",
    # Throttling
    "check_throttling",
    "apply_throttling",
    # Quality
    "calculate_slippage_bps",
    "update_quality_stats",
    # State
    "load_execution_quality_state",
    "save_execution_quality_state",
    # Logging
    "log_execution_event",
    "log_plan_event",
    "log_slippage_event",
    "log_throttle_event",
]
