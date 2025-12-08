"""
v7.5_B1 — Slippage Model

Models expected vs realized slippage for execution quality tracking.
Provides:
- Expected slippage estimation from order book depth
- Realized slippage calculation from fills
- EWMA tracking of slippage per symbol
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

LOG = logging.getLogger("slippage_model")

Side = Literal["BUY", "SELL"]

# Default state directory
DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
SLIPPAGE_STATE_FILE = "slippage_metrics.json"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SlippageObservation:
    """A single slippage observation from an executed order."""
    symbol: str
    side: Side
    notional_usd: float
    expected_bps: float
    realized_bps: float
    spread_bps: float
    maker: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SlippageStats:
    """EWMA slippage statistics for a symbol."""
    ewma_expected_bps: float = 0.0
    ewma_realized_bps: float = 0.0
    trade_count: int = 0
    last_obs_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ewma_expected_bps": round(self.ewma_expected_bps, 4),
            "ewma_realized_bps": round(self.ewma_realized_bps, 4),
            "trade_count": self.trade_count,
            "last_obs_ts": self.last_obs_ts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlippageStats":
        return cls(
            ewma_expected_bps=float(data.get("ewma_expected_bps", 0.0)),
            ewma_realized_bps=float(data.get("ewma_realized_bps", 0.0)),
            trade_count=int(data.get("trade_count", 0)),
            last_obs_ts=float(data.get("last_obs_ts", 0.0)),
        )


@dataclass
class SlippageConfig:
    """Slippage model configuration."""
    enabled: bool = True
    ewma_halflife_trades: int = 50
    max_expected_slippage_bps: float = 15.0
    depth_levels: int = 5
    spread_pause_factor: float = 1.5


# ---------------------------------------------------------------------------
# Expected Slippage Estimation
# ---------------------------------------------------------------------------

def estimate_expected_slippage_bps(
    side: Side,
    qty: float,
    depth: List[Tuple[float, float]],
    mid_price: float,
) -> float:
    """
    Estimate expected price impact in bps assuming we cross the book up to `qty`.
    
    Args:
        side: "BUY" or "SELL"
        qty: Quantity to fill
        depth: List of (price, qty) tuples for the relevant side of the book
               For BUY: asks (ascending price)
               For SELL: bids (descending price)
        mid_price: Current mid price
        
    Returns:
        Expected slippage in basis points (positive = cost, negative = improvement)
    """
    if not depth or qty <= 0 or mid_price <= 0:
        return 0.0
    
    remaining_qty = qty
    total_cost = 0.0
    total_filled = 0.0
    
    for level_price, level_qty in depth:
        if remaining_qty <= 0:
            break
        
        fill_qty = min(remaining_qty, level_qty)
        total_cost += fill_qty * level_price
        total_filled += fill_qty
        remaining_qty -= fill_qty
    
    if total_filled <= 0:
        # Could not fill any quantity - return large slippage estimate
        return 50.0  # 50 bps cap for insufficient depth
    
    # Calculate VWAP
    vwap = total_cost / total_filled
    
    # Calculate slippage vs mid price
    # For BUY: positive slippage if VWAP > mid (we pay more)
    # For SELL: positive slippage if VWAP < mid (we receive less)
    if side == "BUY":
        slippage_bps = (vwap - mid_price) / mid_price * 10000.0
    else:
        slippage_bps = (mid_price - vwap) / mid_price * 10000.0
    
    # Cap at reasonable maximum (200 bps allows for multi-level crossing)
    return min(max(slippage_bps, -10.0), 200.0)


def compute_realized_slippage_bps(
    side: Side,
    fill_price: float,
    mid_price: float,
) -> float:
    """
    Compute realized slippage from a fill.
    
    Args:
        side: "BUY" or "SELL"
        fill_price: Actual fill price
        mid_price: Mid price at time of order
        
    Returns:
        Realized slippage in basis points (positive = cost)
    """
    if mid_price <= 0:
        return 0.0
    
    if side == "BUY":
        # For BUY: positive slippage if fill_price > mid (we paid more)
        slippage_bps = (fill_price - mid_price) / mid_price * 10000.0
    else:
        # For SELL: positive slippage if fill_price < mid (we received less)
        slippage_bps = (mid_price - fill_price) / mid_price * 10000.0
    
    return slippage_bps


def compute_spread_bps(best_bid: float, best_ask: float) -> float:
    """
    Compute bid-ask spread in basis points.
    
    Args:
        best_bid: Best bid price
        best_ask: Best ask price
        
    Returns:
        Spread in basis points
    """
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    
    mid = (best_bid + best_ask) / 2.0
    spread_abs = best_ask - best_bid
    return spread_abs / mid * 10000.0


# ---------------------------------------------------------------------------
# EWMA Update
# ---------------------------------------------------------------------------

def compute_ewma_alpha(halflife_trades: int) -> float:
    """Compute EWMA alpha from halflife in number of trades."""
    if halflife_trades <= 0:
        return 1.0
    return 1.0 - math.pow(0.5, 1.0 / halflife_trades)


def update_slippage_ewma(
    prev: Optional[SlippageStats],
    obs: SlippageObservation,
    halflife_trades: int = 50,
) -> SlippageStats:
    """
    Update EWMA of expected & realized slippage with new observation.
    
    Args:
        prev: Previous slippage stats (or None for first observation)
        obs: New slippage observation
        halflife_trades: EWMA halflife in number of trades
        
    Returns:
        Updated SlippageStats
    """
    alpha = compute_ewma_alpha(halflife_trades)
    
    if prev is None or prev.trade_count == 0:
        # First observation - initialize with observation values
        return SlippageStats(
            ewma_expected_bps=obs.expected_bps,
            ewma_realized_bps=obs.realized_bps,
            trade_count=1,
            last_obs_ts=obs.timestamp,
        )
    
    # EWMA update
    new_expected = alpha * obs.expected_bps + (1 - alpha) * prev.ewma_expected_bps
    new_realized = alpha * obs.realized_bps + (1 - alpha) * prev.ewma_realized_bps
    
    return SlippageStats(
        ewma_expected_bps=new_expected,
        ewma_realized_bps=new_realized,
        trade_count=prev.trade_count + 1,
        last_obs_ts=obs.timestamp,
    )


# ---------------------------------------------------------------------------
# Slippage Metrics Store
# ---------------------------------------------------------------------------

class SlippageMetricsStore:
    """
    Persistent store for per-symbol slippage metrics.
    Stores EWMA statistics and provides query/update interface.
    """
    
    def __init__(self, state_dir: Path | None = None):
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_file = self.state_dir / SLIPPAGE_STATE_FILE
        self._cache: Dict[str, SlippageStats] = {}
        self._loaded = False
        self._halflife_trades = 50
    
    def _load(self) -> None:
        """Load metrics from disk."""
        if self._loaded:
            return
        
        if not self.state_file.exists():
            self._cache = {}
            self._loaded = True
            return
        
        try:
            with self.state_file.open() as f:
                data = json.load(f)
            
            self._cache = {
                symbol: SlippageStats.from_dict(stats)
                for symbol, stats in data.get("per_symbol", {}).items()
            }
            self._loaded = True
            LOG.debug("loaded slippage metrics for %d symbols", len(self._cache))
        except (json.JSONDecodeError, IOError) as exc:
            LOG.warning("failed to load slippage metrics: %s", exc)
            self._cache = {}
            self._loaded = True
    
    def _save(self) -> None:
        """Persist metrics to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "updated_ts": time.time(),
            "per_symbol": {
                symbol: stats.to_dict()
                for symbol, stats in self._cache.items()
            },
        }
        
        try:
            with self.state_file.open("w") as f:
                json.dump(data, f, indent=2)
        except IOError as exc:
            LOG.warning("failed to save slippage metrics: %s", exc)
    
    def get_stats(self, symbol: str) -> Optional[SlippageStats]:
        """Get slippage stats for a symbol."""
        self._load()
        return self._cache.get(symbol.upper())
    
    def update(self, obs: SlippageObservation) -> SlippageStats:
        """
        Update slippage metrics with a new observation.
        
        Args:
            obs: Slippage observation
            
        Returns:
            Updated SlippageStats for the symbol
        """
        self._load()
        symbol = obs.symbol.upper()
        
        prev = self._cache.get(symbol)
        new_stats = update_slippage_ewma(prev, obs, self._halflife_trades)
        self._cache[symbol] = new_stats
        
        # Persist periodically (every 10 updates)
        if new_stats.trade_count % 10 == 0:
            self._save()
        
        return new_stats
    
    def get_all_stats(self) -> Dict[str, SlippageStats]:
        """Get all symbol slippage stats."""
        self._load()
        return dict(self._cache)
    
    def set_halflife(self, halflife_trades: int) -> None:
        """Set the EWMA halflife."""
        self._halflife_trades = max(1, halflife_trades)
    
    def flush(self) -> None:
        """Force save to disk."""
        self._save()


# ---------------------------------------------------------------------------
# Singleton Store
# ---------------------------------------------------------------------------

_SLIPPAGE_STORE: Optional[SlippageMetricsStore] = None


def get_slippage_store() -> SlippageMetricsStore:
    """Get the global slippage metrics store."""
    global _SLIPPAGE_STORE
    if _SLIPPAGE_STORE is None:
        _SLIPPAGE_STORE = SlippageMetricsStore()
    return _SLIPPAGE_STORE


def record_slippage_observation(obs: SlippageObservation) -> SlippageStats:
    """
    Record a slippage observation and update metrics.
    
    Args:
        obs: Slippage observation from an executed order
        
    Returns:
        Updated SlippageStats for the symbol
    """
    return get_slippage_store().update(obs)


def get_symbol_slippage_stats(symbol: str) -> Optional[SlippageStats]:
    """Get slippage stats for a symbol."""
    return get_slippage_store().get_stats(symbol)


def get_all_slippage_stats() -> Dict[str, SlippageStats]:
    """Get all symbol slippage stats."""
    return get_slippage_store().get_all_stats()


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_slippage_config() -> SlippageConfig:
    """
    Load slippage configuration from runtime.yaml.
    
    Returns:
        SlippageConfig with settings
    """
    try:
        import yaml
        
        runtime_path = Path("config/runtime.yaml")
        if not runtime_path.exists():
            return SlippageConfig()
        
        with runtime_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        
        router_cfg = cfg.get("router", {})
        slippage_cfg = router_cfg.get("slippage", {})
        
        return SlippageConfig(
            enabled=slippage_cfg.get("enabled", True),
            ewma_halflife_trades=int(slippage_cfg.get("ewma_halflife_trades", 50)),
            max_expected_slippage_bps=float(slippage_cfg.get("max_expected_slippage_bps", 15.0)),
            depth_levels=int(slippage_cfg.get("depth_levels", 5)),
            spread_pause_factor=float(slippage_cfg.get("spread_pause_factor", 1.5)),
        )
    except Exception as exc:
        LOG.warning("failed to load slippage config: %s", exc)
        return SlippageConfig()


# ---------------------------------------------------------------------------
# Slippage Snapshot for State Publishing
# ---------------------------------------------------------------------------

def build_slippage_snapshot() -> Dict[str, Any]:
    """
    Build a slippage snapshot for state publishing.
    
    Returns:
        Dict with per_symbol slippage stats
    """
    all_stats = get_all_slippage_stats()
    
    return {
        "updated_ts": time.time(),
        "per_symbol": {
            symbol: stats.to_dict()
            for symbol, stats in all_stats.items()
        },
    }
