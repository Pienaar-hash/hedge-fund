"""
Execution Alpha Engine — v7.9_P4

Tracks and attributes execution alpha (the difference between fill price
and model price) per fill, per symbol, and per head.

Key concepts:
- **Execution Alpha**: The gain or loss purely from fill price vs model price
- **Drag**: Negative alpha (execution cost)
- **Model Price**: Reference price at order time (router model, mid, bid/ask)

This module provides:
- Alpha sample computation per fill
- Rolling statistics per symbol (cum_alpha, avg_bps, p95/p99, regime breakdown)
- Per-head attribution using head_contributions weights
- Optional penalty multipliers for Universe Optimizer & Cerberus
- Alert generation for tail slippage events

All features are config-gated and disabled by default.

Author: Execution Alpha Engine v7.9_P4
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Deque
import statistics

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Paths
# ---------------------------------------------------------------------------

_STATE_DIR = Path("logs/state")
_ALPHA_STATE_FILE = _STATE_DIR / "execution_alpha.json"
_EVENTS_DIR = Path("logs/execution")
_EVENTS_FILE = _EVENTS_DIR / "execution_alpha_events.jsonl"


# ---------------------------------------------------------------------------
# Config Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionAlphaConfig:
    """Configuration for the Execution Alpha Engine."""
    
    enabled: bool = False
    
    # Model price source
    model_price_source: str = "router_model"  # router_model | mid | bid_ask_side
    use_expected_fill_price_field: bool = True
    
    # Aggregation settings
    lookback_trades: int = 500
    min_samples_for_penalty: int = 50
    
    # Penalty application flags (all disabled by default)
    apply_symbol_penalty_to_universe: bool = False
    apply_head_penalty_to_cerberus: bool = False
    apply_head_penalty_to_hydra_pnl: bool = False
    
    # Penalty thresholds
    symbol_drag_bps_soft: float = 8.0
    symbol_drag_bps_hard: float = 20.0
    symbol_min_multiplier: float = 0.60
    
    head_drag_bps_soft: float = 6.0
    head_drag_bps_hard: float = 15.0
    head_min_multiplier: float = 0.70
    
    # Alert thresholds
    alerts_enabled: bool = True
    tail_slippage_bps: float = 30.0
    drag_bps_over_rolling: float = 12.0


def load_execution_alpha_config(
    strategy_cfg: Optional[Dict[str, Any]] = None
) -> ExecutionAlphaConfig:
    """
    Load ExecutionAlphaConfig from strategy_config.json.
    
    Falls back to defaults if config not present.
    """
    if strategy_cfg is None:
        try:
            cfg_path = Path("config/strategy_config.json")
            if cfg_path.exists():
                strategy_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            _LOG.warning("Failed to load strategy_config.json: %s", e)
            return ExecutionAlphaConfig()
    
    if strategy_cfg is None:
        return ExecutionAlphaConfig()
    
    alpha_cfg = strategy_cfg.get("execution_alpha", {})
    if not alpha_cfg:
        return ExecutionAlphaConfig()
    
    penalty = alpha_cfg.get("penalty", {})
    alerts = alpha_cfg.get("alerts", {})
    
    return ExecutionAlphaConfig(
        enabled=alpha_cfg.get("enabled", False),
        model_price_source=alpha_cfg.get("model_price_source", "router_model"),
        use_expected_fill_price_field=alpha_cfg.get("use_expected_fill_price_field", True),
        lookback_trades=alpha_cfg.get("lookback_trades", 500),
        min_samples_for_penalty=alpha_cfg.get("min_samples_for_penalty", 50),
        apply_symbol_penalty_to_universe=alpha_cfg.get("apply_symbol_penalty_to_universe", False),
        apply_head_penalty_to_cerberus=alpha_cfg.get("apply_head_penalty_to_cerberus", False),
        apply_head_penalty_to_hydra_pnl=alpha_cfg.get("apply_head_penalty_to_hydra_pnl", False),
        symbol_drag_bps_soft=penalty.get("symbol_drag_bps_soft", 8.0),
        symbol_drag_bps_hard=penalty.get("symbol_drag_bps_hard", 20.0),
        symbol_min_multiplier=penalty.get("symbol_min_multiplier", 0.60),
        head_drag_bps_soft=penalty.get("head_drag_bps_soft", 6.0),
        head_drag_bps_hard=penalty.get("head_drag_bps_hard", 15.0),
        head_min_multiplier=penalty.get("head_min_multiplier", 0.70),
        alerts_enabled=alerts.get("enabled", True),
        tail_slippage_bps=alerts.get("tail_slippage_bps", 30.0),
        drag_bps_over_rolling=alerts.get("drag_bps_over_rolling", 12.0),
    )


def is_execution_alpha_enabled(
    strategy_cfg: Optional[Dict[str, Any]] = None
) -> bool:
    """Check if execution alpha is enabled."""
    cfg = load_execution_alpha_config(strategy_cfg)
    return cfg.enabled


# ---------------------------------------------------------------------------
# Alpha Sample Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AlphaSample:
    """
    Single execution alpha observation.
    
    Records the alpha (USD and bps) for a single fill, along with
    attribution to heads via head_contributions.
    """
    ts: float
    symbol: str
    side: str  # BUY or SELL
    qty: float
    fill_price: float
    model_price: float
    alpha_usd: float
    alpha_bps: float
    drag_bps: float  # max(0, -alpha_bps)
    regime: Optional[str] = None
    head_contributions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ts": self.ts,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "fill_price": self.fill_price,
            "model_price": self.model_price,
            "alpha_usd": self.alpha_usd,
            "alpha_bps": self.alpha_bps,
            "drag_bps": self.drag_bps,
            "regime": self.regime,
            "head_contributions": self.head_contributions,
        }


# ---------------------------------------------------------------------------
# Alpha Statistics Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegimeBreakdown:
    """Alpha statistics for a specific execution regime."""
    samples: int = 0
    avg_alpha_bps: float = 0.0
    cum_alpha_usd: float = 0.0


@dataclass
class SymbolAlphaStats:
    """
    Aggregated alpha statistics for a symbol.
    
    Tracks cumulative alpha, average bps, percentiles, and regime breakdown.
    """
    symbol: str
    samples: int = 0
    cum_alpha_usd: float = 0.0
    avg_alpha_bps: float = 0.0
    p95_alpha_bps: float = 0.0
    p99_alpha_bps: float = 0.0
    avg_drag_bps: float = 0.0
    regime_breakdown: Dict[str, RegimeBreakdown] = field(default_factory=dict)
    suggested_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "samples": self.samples,
            "cum_alpha_usd": round(self.cum_alpha_usd, 4),
            "avg_alpha_bps": round(self.avg_alpha_bps, 2),
            "p95_alpha_bps": round(self.p95_alpha_bps, 2),
            "p99_alpha_bps": round(self.p99_alpha_bps, 2),
            "avg_drag_bps": round(self.avg_drag_bps, 2),
            "regime_breakdown": {
                regime: {
                    "samples": rb.samples,
                    "avg_alpha_bps": round(rb.avg_alpha_bps, 2),
                    "cum_alpha_usd": round(rb.cum_alpha_usd, 4),
                }
                for regime, rb in self.regime_breakdown.items()
            },
            "suggested_multiplier": round(self.suggested_multiplier, 4),
        }


@dataclass
class HeadAlphaStats:
    """
    Aggregated alpha statistics for a head.
    
    Alpha is attributed to heads using the head_contributions weights
    from each fill.
    """
    head: str
    samples: int = 0
    cum_alpha_usd: float = 0.0
    avg_alpha_bps: float = 0.0
    p95_alpha_bps: float = 0.0
    p99_alpha_bps: float = 0.0
    avg_drag_bps: float = 0.0
    suggested_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "samples": self.samples,
            "cum_alpha_usd": round(self.cum_alpha_usd, 4),
            "avg_alpha_bps": round(self.avg_alpha_bps, 2),
            "p95_alpha_bps": round(self.p95_alpha_bps, 2),
            "p99_alpha_bps": round(self.p99_alpha_bps, 2),
            "avg_drag_bps": round(self.avg_drag_bps, 2),
            "suggested_multiplier": round(self.suggested_multiplier, 4),
        }


@dataclass
class ExecutionAlphaState:
    """
    Complete execution alpha state.
    
    Written to logs/state/execution_alpha.json
    """
    enabled: bool = False
    symbol_stats: Dict[str, SymbolAlphaStats] = field(default_factory=dict)
    head_stats: Dict[str, HeadAlphaStats] = field(default_factory=dict)
    total_cum_alpha_usd: float = 0.0
    avg_alpha_bps: float = 0.0
    last_alerts: List[Dict[str, Any]] = field(default_factory=list)
    updated_ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Model Price Resolution
# ---------------------------------------------------------------------------


def resolve_model_price(
    fill: Dict[str, Any],
    quotes: Optional[Dict[str, Any]] = None,
    cfg: Optional[ExecutionAlphaConfig] = None,
) -> Optional[float]:
    """
    Resolve the model price for a fill.
    
    Order of precedence:
    1. If use_expected_fill_price_field and fill has expected_fill_price or router_model_price
    2. If model_price_source == "router_model": use router's model price
    3. If model_price_source == "mid": use (best_bid + best_ask) / 2
    4. If model_price_source == "bid_ask_side": BUY uses ask, SELL uses bid
    
    Args:
        fill: Fill record containing price, side, symbol, and optional model prices
        quotes: Quote cache with bid/ask for symbols
        cfg: ExecutionAlphaConfig
    
    Returns:
        Model price as float, or None if cannot be resolved
    """
    if cfg is None:
        cfg = ExecutionAlphaConfig()
    
    # Priority 1: Use expected fill price field if enabled and available
    if cfg.use_expected_fill_price_field:
        for field_name in ("expected_fill_price", "router_model_price", "model_price"):
            if field_name in fill and fill[field_name] is not None:
                try:
                    return float(fill[field_name])
                except (ValueError, TypeError):
                    continue
    
    # Priority 2-4: Based on model_price_source
    source = cfg.model_price_source
    symbol = fill.get("symbol", "")
    side = fill.get("side", "").upper()
    
    if source == "router_model":
        # Try router_model_price or limit_price from fill
        for field_name in ("router_model_price", "limit_price", "expected_price"):
            if field_name in fill and fill[field_name] is not None:
                try:
                    return float(fill[field_name])
                except (ValueError, TypeError):
                    continue
        # Fall through to mid if no router model
        source = "mid"
    
    if quotes is None:
        return None
    
    quote = quotes.get(symbol, {})
    best_bid = quote.get("bid") or quote.get("best_bid")
    best_ask = quote.get("ask") or quote.get("best_ask")
    
    if source == "mid":
        if best_bid is not None and best_ask is not None:
            try:
                return (float(best_bid) + float(best_ask)) / 2.0
            except (ValueError, TypeError):
                pass
    
    elif source == "bid_ask_side":
        if side == "BUY" and best_ask is not None:
            try:
                return float(best_ask)
            except (ValueError, TypeError):
                pass
        elif side == "SELL" and best_bid is not None:
            try:
                return float(best_bid)
            except (ValueError, TypeError):
                pass
    
    # Final fallback: try fill price if no model price available
    # (this results in zero alpha, which is safe default)
    return None


# ---------------------------------------------------------------------------
# Execution Alpha Computation
# ---------------------------------------------------------------------------


def compute_alpha(
    fill_price: float,
    model_price: float,
    qty: float,
    side: str,
) -> tuple[float, float, float]:
    """
    Compute execution alpha for a fill.
    
    Direction sign:
    - BUY: positive alpha if we bought below model price (saved money)
    - SELL: positive alpha if we sold above model price (got more money)
    
    Args:
        fill_price: Actual fill price
        model_price: Expected/model price
        qty: Quantity filled
        side: BUY or SELL
    
    Returns:
        Tuple of (alpha_usd, alpha_bps, drag_bps)
    """
    if model_price <= 0:
        return 0.0, 0.0, 0.0
    
    side_upper = side.upper()
    direction = 1.0 if side_upper == "BUY" else -1.0
    
    # For BUY: positive alpha if fill_price < model_price (we paid less)
    # For SELL: positive alpha if fill_price > model_price (we received more)
    price_diff = model_price - fill_price
    
    alpha_usd = direction * price_diff * qty
    alpha_bps = direction * (price_diff / model_price) * 10_000
    
    # Drag is the negative part of alpha (execution cost)
    drag_bps = max(0.0, -alpha_bps)
    
    return alpha_usd, alpha_bps, drag_bps


def create_alpha_sample(
    fill: Dict[str, Any],
    model_price: float,
    regime: Optional[str] = None,
    head_contributions: Optional[Dict[str, float]] = None,
) -> AlphaSample:
    """
    Create an AlphaSample from a fill record.
    
    Args:
        fill: Fill record with symbol, side, qty, price
        model_price: Resolved model price
        regime: Current execution regime (NORMAL, THIN, etc.)
        head_contributions: Head contribution weights from the order
    
    Returns:
        AlphaSample instance
    """
    symbol = fill.get("symbol", "UNKNOWN")
    side = fill.get("side", "BUY").upper()
    qty = float(fill.get("qty", fill.get("quantity", 0.0)))
    fill_price = float(fill.get("price", fill.get("fill_price", 0.0)))
    
    alpha_usd, alpha_bps, drag_bps = compute_alpha(fill_price, model_price, qty, side)
    
    return AlphaSample(
        ts=fill.get("ts", time.time()),
        symbol=symbol,
        side=side,
        qty=qty,
        fill_price=fill_price,
        model_price=model_price,
        alpha_usd=alpha_usd,
        alpha_bps=alpha_bps,
        drag_bps=drag_bps,
        regime=regime,
        head_contributions=head_contributions or {},
    )


# ---------------------------------------------------------------------------
# Rolling Statistics Aggregator
# ---------------------------------------------------------------------------


class AlphaAggregator:
    """
    Maintains rolling alpha statistics per symbol and per head.
    
    Uses fixed-size deques to maintain the last N samples for rolling stats.
    """
    
    def __init__(self, lookback_trades: int = 500):
        self.lookback_trades = lookback_trades
        
        # Per-symbol sample buffers
        self._symbol_samples: Dict[str, Deque[AlphaSample]] = {}
        
        # Per-head sample buffers (stores (alpha_bps, alpha_usd, weight) tuples)
        self._head_samples: Dict[str, Deque[tuple[float, float, float]]] = {}
        
        # Computed stats
        self._symbol_stats: Dict[str, SymbolAlphaStats] = {}
        self._head_stats: Dict[str, HeadAlphaStats] = {}
    
    def add_sample(self, sample: AlphaSample) -> None:
        """
        Add an alpha sample and update rolling statistics.
        
        Args:
            sample: AlphaSample to add
        """
        symbol = sample.symbol
        
        # Add to symbol buffer
        if symbol not in self._symbol_samples:
            self._symbol_samples[symbol] = deque(maxlen=self.lookback_trades)
        self._symbol_samples[symbol].append(sample)
        
        # Attribute to heads
        for head, weight in sample.head_contributions.items():
            if weight <= 0:
                continue
            
            if head not in self._head_samples:
                self._head_samples[head] = deque(maxlen=self.lookback_trades)
            
            # Store (alpha_bps, alpha_usd * weight, weight) for attribution
            self._head_samples[head].append((
                sample.alpha_bps,
                sample.alpha_usd * weight,
                weight,
            ))
        
        # Update stats
        self._update_symbol_stats(symbol)
        for head in sample.head_contributions:
            self._update_head_stats(head)
    
    def _update_symbol_stats(self, symbol: str) -> None:
        """Update statistics for a symbol."""
        samples = list(self._symbol_samples.get(symbol, []))
        if not samples:
            return
        
        n = len(samples)
        cum_alpha_usd = sum(s.alpha_usd for s in samples)
        alpha_bps_list = [s.alpha_bps for s in samples]
        drag_bps_list = [s.drag_bps for s in samples]
        
        avg_alpha_bps = statistics.mean(alpha_bps_list) if alpha_bps_list else 0.0
        avg_drag_bps = statistics.mean(drag_bps_list) if drag_bps_list else 0.0
        
        # Compute percentiles (negative = worse)
        sorted_bps = sorted(alpha_bps_list)
        p95_idx = max(0, int(n * 0.05) - 1)  # 5th percentile (worst)
        p99_idx = max(0, int(n * 0.01) - 1)  # 1st percentile (worst)
        p95_alpha_bps = sorted_bps[p95_idx] if sorted_bps else 0.0
        p99_alpha_bps = sorted_bps[p99_idx] if sorted_bps else 0.0
        
        # Regime breakdown
        regime_samples: Dict[str, List[AlphaSample]] = {}
        for s in samples:
            regime = s.regime or "UNKNOWN"
            if regime not in regime_samples:
                regime_samples[regime] = []
            regime_samples[regime].append(s)
        
        regime_breakdown = {}
        for regime, regime_list in regime_samples.items():
            regime_breakdown[regime] = RegimeBreakdown(
                samples=len(regime_list),
                avg_alpha_bps=statistics.mean(s.alpha_bps for s in regime_list),
                cum_alpha_usd=sum(s.alpha_usd for s in regime_list),
            )
        
        self._symbol_stats[symbol] = SymbolAlphaStats(
            symbol=symbol,
            samples=n,
            cum_alpha_usd=cum_alpha_usd,
            avg_alpha_bps=avg_alpha_bps,
            p95_alpha_bps=p95_alpha_bps,
            p99_alpha_bps=p99_alpha_bps,
            avg_drag_bps=avg_drag_bps,
            regime_breakdown=regime_breakdown,
            suggested_multiplier=1.0,  # Computed separately
        )
    
    def _update_head_stats(self, head: str) -> None:
        """Update statistics for a head."""
        samples = list(self._head_samples.get(head, []))
        if not samples:
            return
        
        n = len(samples)
        # samples are (alpha_bps, weighted_alpha_usd, weight)
        cum_alpha_usd = sum(s[1] for s in samples)
        alpha_bps_list = [s[0] for s in samples]
        
        avg_alpha_bps = statistics.mean(alpha_bps_list) if alpha_bps_list else 0.0
        avg_drag_bps = statistics.mean(max(0, -bps) for bps in alpha_bps_list)
        
        sorted_bps = sorted(alpha_bps_list)
        p95_idx = max(0, int(n * 0.05) - 1)
        p99_idx = max(0, int(n * 0.01) - 1)
        p95_alpha_bps = sorted_bps[p95_idx] if sorted_bps else 0.0
        p99_alpha_bps = sorted_bps[p99_idx] if sorted_bps else 0.0
        
        self._head_stats[head] = HeadAlphaStats(
            head=head,
            samples=n,
            cum_alpha_usd=cum_alpha_usd,
            avg_alpha_bps=avg_alpha_bps,
            p95_alpha_bps=p95_alpha_bps,
            p99_alpha_bps=p99_alpha_bps,
            avg_drag_bps=avg_drag_bps,
            suggested_multiplier=1.0,  # Computed separately
        )
    
    def get_symbol_stats(self, symbol: str) -> Optional[SymbolAlphaStats]:
        """Get stats for a symbol."""
        return self._symbol_stats.get(symbol)
    
    def get_all_symbol_stats(self) -> Dict[str, SymbolAlphaStats]:
        """Get stats for all symbols."""
        return dict(self._symbol_stats)
    
    def get_head_stats(self, head: str) -> Optional[HeadAlphaStats]:
        """Get stats for a head."""
        return self._head_stats.get(head)
    
    def get_all_head_stats(self) -> Dict[str, HeadAlphaStats]:
        """Get stats for all heads."""
        return dict(self._head_stats)
    
    def get_total_cum_alpha_usd(self) -> float:
        """Get total cumulative alpha across all symbols."""
        return sum(s.cum_alpha_usd for s in self._symbol_stats.values())
    
    def get_avg_alpha_bps(self) -> float:
        """Get average alpha bps across all symbols."""
        stats = list(self._symbol_stats.values())
        if not stats:
            return 0.0
        return statistics.mean(s.avg_alpha_bps for s in stats)


# ---------------------------------------------------------------------------
# Penalty Multiplier Functions
# ---------------------------------------------------------------------------


def compute_penalty_multiplier(
    drag_bps: float,
    soft_threshold: float,
    hard_threshold: float,
    min_multiplier: float,
) -> float:
    """
    Compute a piecewise linear penalty multiplier based on drag bps.
    
    - Below soft threshold: 1.0 (no penalty)
    - Between soft and hard: linear ramp down to min_multiplier
    - Above hard threshold: min_multiplier
    
    Args:
        drag_bps: Average drag in basis points
        soft_threshold: Drag bps below which no penalty applies
        hard_threshold: Drag bps above which max penalty applies
        min_multiplier: Minimum multiplier (max penalty)
    
    Returns:
        Multiplier in [min_multiplier, 1.0]
    """
    if drag_bps <= soft_threshold:
        return 1.0
    
    if drag_bps >= hard_threshold:
        return min_multiplier
    
    # Linear interpolation
    k = (drag_bps - soft_threshold) / (hard_threshold - soft_threshold)
    return 1.0 - k * (1.0 - min_multiplier)


def compute_symbol_multipliers(
    stats: Dict[str, SymbolAlphaStats],
    cfg: ExecutionAlphaConfig,
) -> Dict[str, float]:
    """
    Compute penalty multipliers for all symbols.
    
    Only applies to symbols with enough samples.
    
    Args:
        stats: Per-symbol alpha stats
        cfg: ExecutionAlphaConfig
    
    Returns:
        Dict of symbol -> multiplier
    """
    multipliers = {}
    
    for symbol, stat in stats.items():
        if stat.samples < cfg.min_samples_for_penalty:
            multipliers[symbol] = 1.0
            stat.suggested_multiplier = 1.0
            continue
        
        mult = compute_penalty_multiplier(
            drag_bps=stat.avg_drag_bps,
            soft_threshold=cfg.symbol_drag_bps_soft,
            hard_threshold=cfg.symbol_drag_bps_hard,
            min_multiplier=cfg.symbol_min_multiplier,
        )
        multipliers[symbol] = mult
        stat.suggested_multiplier = mult
    
    return multipliers


def compute_head_multipliers(
    stats: Dict[str, HeadAlphaStats],
    cfg: ExecutionAlphaConfig,
) -> Dict[str, float]:
    """
    Compute penalty multipliers for all heads.
    
    Only applies to heads with enough samples.
    
    Args:
        stats: Per-head alpha stats
        cfg: ExecutionAlphaConfig
    
    Returns:
        Dict of head -> multiplier
    """
    multipliers = {}
    
    for head, stat in stats.items():
        if stat.samples < cfg.min_samples_for_penalty:
            multipliers[head] = 1.0
            stat.suggested_multiplier = 1.0
            continue
        
        mult = compute_penalty_multiplier(
            drag_bps=stat.avg_drag_bps,
            soft_threshold=cfg.head_drag_bps_soft,
            hard_threshold=cfg.head_drag_bps_hard,
            min_multiplier=cfg.head_min_multiplier,
        )
        multipliers[head] = mult
        stat.suggested_multiplier = mult
    
    return multipliers


# ---------------------------------------------------------------------------
# Alert Generation
# ---------------------------------------------------------------------------


def check_and_generate_alerts(
    sample: AlphaSample,
    symbol_stats: Optional[SymbolAlphaStats],
    cfg: ExecutionAlphaConfig,
) -> List[Dict[str, Any]]:
    """
    Check if a sample triggers any alerts.
    
    Alert conditions:
    - Tail slippage: alpha_bps worse than tail_slippage_bps threshold
    - Rolling drag: symbol's avg_drag_bps exceeds drag_bps_over_rolling
    
    Args:
        sample: The alpha sample to check
        symbol_stats: Current symbol stats (if available)
        cfg: ExecutionAlphaConfig
    
    Returns:
        List of alert events
    """
    if not cfg.alerts_enabled:
        return []
    
    alerts = []
    
    # Check tail slippage (very negative alpha)
    if sample.alpha_bps < -cfg.tail_slippage_bps:
        alerts.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "EXEC_ALPHA_TAIL",
            "symbol": sample.symbol,
            "alpha_bps": round(sample.alpha_bps, 2),
            "alpha_usd": round(sample.alpha_usd, 4),
            "regime": sample.regime,
            "head_contributions": sample.head_contributions,
            "threshold": cfg.tail_slippage_bps,
        })
    
    # Check rolling drag
    if symbol_stats and symbol_stats.avg_drag_bps > cfg.drag_bps_over_rolling:
        alerts.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "EXEC_DRAG_HIGH",
            "symbol": sample.symbol,
            "avg_drag_bps": round(symbol_stats.avg_drag_bps, 2),
            "samples": symbol_stats.samples,
            "threshold": cfg.drag_bps_over_rolling,
        })
    
    return alerts


def log_alert_event(alert: Dict[str, Any]) -> None:
    """Write an alert event to the events log."""
    try:
        _EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")
    except Exception as e:
        _LOG.warning("Failed to write alpha alert: %s", e)


# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------


def save_execution_alpha_state(
    aggregator: AlphaAggregator,
    cfg: ExecutionAlphaConfig,
    last_alerts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Save execution alpha state to disk.
    
    Writes to logs/state/execution_alpha.json
    """
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        
        symbol_stats = aggregator.get_all_symbol_stats()
        head_stats = aggregator.get_all_head_stats()
        
        # Compute multipliers
        compute_symbol_multipliers(symbol_stats, cfg)
        compute_head_multipliers(head_stats, cfg)
        
        state = {
            "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbols": {sym: stat.to_dict() for sym, stat in symbol_stats.items()},
            "heads": {head: stat.to_dict() for head, stat in head_stats.items()},
            "meta": {
                "enabled": cfg.enabled,
                "total_cum_alpha_usd": round(aggregator.get_total_cum_alpha_usd(), 4),
                "avg_alpha_bps": round(aggregator.get_avg_alpha_bps(), 2),
                "last_alerts": (last_alerts or [])[-10:],  # Keep last 10 alerts
                "lookback_trades": cfg.lookback_trades,
                "penalties_active": {
                    "universe": cfg.apply_symbol_penalty_to_universe,
                    "cerberus": cfg.apply_head_penalty_to_cerberus,
                    "hydra_pnl": cfg.apply_head_penalty_to_hydra_pnl,
                },
            },
        }
        
        _ALPHA_STATE_FILE.write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        
    except Exception as e:
        _LOG.error("Failed to save execution_alpha.json: %s", e)


def load_execution_alpha_state() -> Dict[str, Any]:
    """Load execution alpha state from disk."""
    try:
        if _ALPHA_STATE_FILE.exists():
            return json.loads(_ALPHA_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        _LOG.warning("Failed to load execution_alpha.json: %s", e)
    
    return {"updated_ts": None, "symbols": {}, "heads": {}, "meta": {}}


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def get_symbol_alpha_multiplier(
    symbol: str,
    state: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Get the execution alpha penalty multiplier for a symbol.
    
    For use by Universe Optimizer.
    
    Args:
        symbol: Symbol to look up
        state: Optional pre-loaded state (loads from disk if None)
    
    Returns:
        Multiplier in [min_multiplier, 1.0]
    """
    if state is None:
        state = load_execution_alpha_state()
    
    symbols = state.get("symbols", {})
    if symbol not in symbols:
        return 1.0
    
    return symbols[symbol].get("suggested_multiplier", 1.0)


def get_head_alpha_multiplier(
    head: str,
    state: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Get the execution alpha penalty multiplier for a head.
    
    For use by Cerberus.
    
    Args:
        head: Head name to look up
        state: Optional pre-loaded state (loads from disk if None)
    
    Returns:
        Multiplier in [min_multiplier, 1.0]
    """
    if state is None:
        state = load_execution_alpha_state()
    
    heads = state.get("heads", {})
    if head not in heads:
        return 1.0
    
    return heads[head].get("suggested_multiplier", 1.0)


def get_all_symbol_alpha_multipliers(
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Get all symbol alpha multipliers."""
    if state is None:
        state = load_execution_alpha_state()
    
    return {
        sym: data.get("suggested_multiplier", 1.0)
        for sym, data in state.get("symbols", {}).items()
    }


def get_all_head_alpha_multipliers(
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Get all head alpha multipliers."""
    if state is None:
        state = load_execution_alpha_state()
    
    return {
        head: data.get("suggested_multiplier", 1.0)
        for head, data in state.get("heads", {}).items()
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "ExecutionAlphaConfig",
    "load_execution_alpha_config",
    "is_execution_alpha_enabled",
    
    # Dataclasses
    "AlphaSample",
    "SymbolAlphaStats",
    "HeadAlphaStats",
    "RegimeBreakdown",
    "ExecutionAlphaState",
    
    # Model price
    "resolve_model_price",
    
    # Alpha computation
    "compute_alpha",
    "create_alpha_sample",
    
    # Aggregation
    "AlphaAggregator",
    
    # Penalties
    "compute_penalty_multiplier",
    "compute_symbol_multipliers",
    "compute_head_multipliers",
    
    # Alerts
    "check_and_generate_alerts",
    "log_alert_event",
    
    # State
    "save_execution_alpha_state",
    "load_execution_alpha_state",
    
    # Integration helpers
    "get_symbol_alpha_multiplier",
    "get_head_alpha_multiplier",
    "get_all_symbol_alpha_multipliers",
    "get_all_head_alpha_multipliers",
]
