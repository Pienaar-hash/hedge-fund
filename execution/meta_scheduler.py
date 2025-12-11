"""
v7.8_P1 — Meta-Learning Weight Scheduler ("Slow Brain")

A slow-timescale governance layer that learns from realized factor/sector/system
performance over time and maintains long-horizon overlays on:

- Factor weights (on top of P2 adaptive IR/PnL)
- Conviction strength (on top of P1 + P6 regime curves)
- Category rotation intensity (on top of P3 category_momentum)

This module stores its state in `logs/state/meta_scheduler.json` and is:
- Fully optional (default: disabled)
- Non-breaking (identical to v7.7 when disabled)
- Research-grade governance, not a trading strategy

State Contract:
- Single writer: EdgeScanner (via this module's write_meta_scheduler_state)
- Never writes to other state surfaces
- All overlays are multiplicative and bounded
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_META_SCHEDULER_PATH = Path("logs/state/meta_scheduler.json")

# Neutral multiplier (no effect)
NEUTRAL_MULTIPLIER = 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MetaSchedulerConfig:
    """
    Configuration for the Meta-Learning Weight Scheduler.
    
    Attributes:
        enabled: Whether meta-scheduler is active (default: False for back-compat)
        learning_rate: Speed of overlay adjustments (e.g. 0.05)
        min_samples: Minimum snapshots before learning kicks in
        max_factor_shift: Maximum per-factor overlay deviation (±, e.g. 0.10)
        max_conviction_shift: Maximum global conviction overlay (±, e.g. 0.15)
        max_category_shift: Maximum per-category overlay (±, e.g. 0.15)
        decay: EMA decay for long-horizon stats (e.g. 0.90)
        ir_threshold: IR threshold for significant signal (e.g. 0.3)
        pnl_threshold: PnL threshold for significant signal (e.g. 0.0)
    """
    
    enabled: bool = False
    learning_rate: float = 0.05
    min_samples: int = 50
    max_factor_shift: float = 0.10
    max_conviction_shift: float = 0.15
    max_category_shift: float = 0.15
    decay: float = 0.90
    ir_threshold: float = 0.3
    pnl_threshold: float = 0.0


def load_meta_scheduler_config(
    strategy_cfg: Optional[Mapping[str, Any]] = None,
) -> MetaSchedulerConfig:
    """
    Load MetaSchedulerConfig from strategy_config.json.
    
    Expects a top-level "meta_scheduler" block:
    {
        "meta_scheduler": {
            "enabled": false,
            "learning_rate": 0.05,
            ...
        }
    }
    
    Args:
        strategy_cfg: Strategy config dict, or None for defaults.
        
    Returns:
        MetaSchedulerConfig instance
    """
    if strategy_cfg is None:
        return MetaSchedulerConfig()
    
    ms_cfg = strategy_cfg.get("meta_scheduler", {})
    if not isinstance(ms_cfg, Mapping):
        return MetaSchedulerConfig()
    
    return MetaSchedulerConfig(
        enabled=bool(ms_cfg.get("enabled", False)),
        learning_rate=float(ms_cfg.get("learning_rate", 0.05)),
        min_samples=int(ms_cfg.get("min_samples", 50)),
        max_factor_shift=float(ms_cfg.get("max_factor_shift", 0.10)),
        max_conviction_shift=float(ms_cfg.get("max_conviction_shift", 0.15)),
        max_category_shift=float(ms_cfg.get("max_category_shift", 0.15)),
        decay=float(ms_cfg.get("decay", 0.90)),
        ir_threshold=float(ms_cfg.get("ir_threshold", 0.3)),
        pnl_threshold=float(ms_cfg.get("pnl_threshold", 0.0)),
    )


# ---------------------------------------------------------------------------
# State Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FactorMetaState:
    """
    Meta-learning state for factor weight overlays.
    
    Attributes:
        meta_weights: Per-factor multiplier overlays ∈ [1-max_shift, 1+max_shift]
        ema_ir: EMA of per-factor information ratio
        ema_pnl: EMA of per-factor PnL contribution
    """
    
    meta_weights: Dict[str, float] = field(default_factory=dict)
    ema_ir: Dict[str, float] = field(default_factory=dict)
    ema_pnl: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "meta_weights": {k: round(v, 6) for k, v in self.meta_weights.items()},
            "ema_ir": {k: round(v, 6) for k, v in self.ema_ir.items()},
            "ema_pnl": {k: round(v, 6) for k, v in self.ema_pnl.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorMetaState":
        """Create from dictionary."""
        return cls(
            meta_weights=dict(data.get("meta_weights", {})),
            ema_ir=dict(data.get("ema_ir", {})),
            ema_pnl=dict(data.get("ema_pnl", {})),
        )


@dataclass
class ConvictionMetaState:
    """
    Meta-learning state for global conviction overlay.
    
    Attributes:
        global_strength: Overlay multiplier ∈ [1-max_shift, 1+max_shift]
        ema_health: EMA of strategy health score (long-horizon signal)
    """
    
    global_strength: float = NEUTRAL_MULTIPLIER
    ema_health: float = 0.5  # Neutral health signal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "global_strength": round(self.global_strength, 6),
            "ema_health": round(self.ema_health, 6),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvictionMetaState":
        """Create from dictionary."""
        return cls(
            global_strength=float(data.get("global_strength", NEUTRAL_MULTIPLIER)),
            ema_health=float(data.get("ema_health", 0.5)),
        )


@dataclass
class CategoryMetaState:
    """
    Meta-learning state for category rotation overlays.
    
    Attributes:
        category_overlays: Per-category multipliers ∈ [1-max_shift, 1+max_shift]
        ema_category_ir: EMA of per-category information ratio
        ema_category_pnl: EMA of per-category PnL
    """
    
    category_overlays: Dict[str, float] = field(default_factory=dict)
    ema_category_ir: Dict[str, float] = field(default_factory=dict)
    ema_category_pnl: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category_overlays": {k: round(v, 6) for k, v in self.category_overlays.items()},
            "ema_category_ir": {k: round(v, 6) for k, v in self.ema_category_ir.items()},
            "ema_category_pnl": {k: round(v, 6) for k, v in self.ema_category_pnl.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryMetaState":
        """Create from dictionary."""
        return cls(
            category_overlays=dict(data.get("category_overlays", {})),
            ema_category_ir=dict(data.get("ema_category_ir", {})),
            ema_category_pnl=dict(data.get("ema_category_pnl", {})),
        )


@dataclass
class MetaSchedulerState:
    """
    Complete meta-scheduler state.
    
    Attributes:
        updated_ts: ISO timestamp of last update
        factor_state: Factor weight overlay state
        conviction_state: Global conviction overlay state
        category_state: Category rotation overlay state
        stats: Extra tracking (sample_count, last_health_score, etc.)
    """
    
    updated_ts: str = ""
    factor_state: FactorMetaState = field(default_factory=FactorMetaState)
    conviction_state: ConvictionMetaState = field(default_factory=ConvictionMetaState)
    category_state: CategoryMetaState = field(default_factory=CategoryMetaState)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "factor_state": self.factor_state.to_dict(),
            "conviction_state": self.conviction_state.to_dict(),
            "category_state": self.category_state.to_dict(),
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaSchedulerState":
        """Create from dictionary."""
        return cls(
            updated_ts=str(data.get("updated_ts", "")),
            factor_state=FactorMetaState.from_dict(data.get("factor_state", {})),
            conviction_state=ConvictionMetaState.from_dict(data.get("conviction_state", {})),
            category_state=CategoryMetaState.from_dict(data.get("category_state", {})),
            stats=dict(data.get("stats", {})),
        )


def create_neutral_state() -> MetaSchedulerState:
    """
    Create a neutral meta-scheduler state with all multipliers at 1.0.
    
    Returns:
        MetaSchedulerState with neutral overlays
    """
    return MetaSchedulerState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        factor_state=FactorMetaState(),
        conviction_state=ConvictionMetaState(
            global_strength=NEUTRAL_MULTIPLIER,
            ema_health=0.5,
        ),
        category_state=CategoryMetaState(),
        stats={"sample_count": 0, "last_health_score": None},
    )


# ---------------------------------------------------------------------------
# State Loader & Writer
# ---------------------------------------------------------------------------


def load_meta_scheduler_state(
    path: Path | str = DEFAULT_META_SCHEDULER_PATH,
) -> Optional[MetaSchedulerState]:
    """
    Load meta-scheduler state from file.
    
    Args:
        path: Path to meta_scheduler.json
        
    Returns:
        MetaSchedulerState if file exists and is valid, else None
    """
    path = Path(path)
    if not path.exists():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return MetaSchedulerState.from_dict(data)
    except Exception:
        return None


def write_meta_scheduler_state(
    state: MetaSchedulerState,
    path: Path | str = DEFAULT_META_SCHEDULER_PATH,
) -> None:
    """
    Write meta-scheduler state to file (atomic write).
    
    This is the ONLY allowed writer for meta_scheduler.json.
    Must be called from EdgeScanner only.
    
    Args:
        state: MetaSchedulerState to write
        path: Output path (default: logs/state/meta_scheduler.json)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tmp = path.with_suffix(".tmp")
        payload = state.to_dict()
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(path)
    except Exception:
        # Fail silently - meta state is non-critical
        pass


# ---------------------------------------------------------------------------
# EMA Helpers
# ---------------------------------------------------------------------------


def _ema_update(prev: float, current: float, decay: float) -> float:
    """
    Update EMA: new_ema = decay * prev + (1 - decay) * current
    
    Args:
        prev: Previous EMA value
        current: Current observation
        decay: Decay factor (higher = slower adaptation)
        
    Returns:
        Updated EMA value
    """
    return decay * prev + (1.0 - decay) * current


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Learning Step
# ---------------------------------------------------------------------------


def _update_factor_emas(
    prev_state: FactorMetaState,
    factor_edges: Dict[str, Any],
    decay: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Update factor EMAs from current factor_edges.
    
    Args:
        prev_state: Previous factor meta state
        factor_edges: Current factor edges from EdgeScanner
        decay: EMA decay factor
        
    Returns:
        Tuple of (updated_ema_ir, updated_ema_pnl)
    """
    ema_ir = dict(prev_state.ema_ir)
    ema_pnl = dict(prev_state.ema_pnl)
    
    for factor_name, edge_data in factor_edges.items():
        if not isinstance(edge_data, dict):
            continue
            
        # Extract IR and PnL from factor edge
        current_ir = float(edge_data.get("ir", edge_data.get("edge_score", 0.0)))
        current_pnl = float(edge_data.get("pnl_contrib", edge_data.get("pnl", 0.0)))
        
        # Update EMAs
        prev_ir = ema_ir.get(factor_name, 0.0)
        prev_pnl = ema_pnl.get(factor_name, 0.0)
        
        ema_ir[factor_name] = _ema_update(prev_ir, current_ir, decay)
        ema_pnl[factor_name] = _ema_update(prev_pnl, current_pnl, decay)
    
    return ema_ir, ema_pnl


def _compute_factor_overlays(
    ema_ir: Dict[str, float],
    ema_pnl: Dict[str, float],
    prev_weights: Dict[str, float],
    cfg: MetaSchedulerConfig,
) -> Dict[str, float]:
    """
    Compute factor meta weight overlays based on EMAs.
    
    Logic:
    - If ema_ir >> 0 and ema_pnl > 0 → increase overlay
    - If ema_ir << 0 or ema_pnl < 0 → decrease overlay
    
    Args:
        ema_ir: EMA of factor IR values
        ema_pnl: EMA of factor PnL values
        prev_weights: Previous meta weights
        cfg: Meta-scheduler config
        
    Returns:
        Updated meta weights dict
    """
    meta_weights = dict(prev_weights)
    lr = cfg.learning_rate
    max_shift = cfg.max_factor_shift
    ir_thresh = cfg.ir_threshold
    pnl_thresh = cfg.pnl_threshold
    
    all_factors = set(ema_ir.keys()) | set(ema_pnl.keys()) | set(prev_weights.keys())
    
    for factor_name in all_factors:
        ir_val = ema_ir.get(factor_name, 0.0)
        pnl_val = ema_pnl.get(factor_name, 0.0)
        prev_w = prev_weights.get(factor_name, NEUTRAL_MULTIPLIER)
        
        # Compute adjustment direction
        adjustment = 0.0
        
        # Strong positive signal: IR above threshold AND positive PnL
        if ir_val > ir_thresh and pnl_val > pnl_thresh:
            # Boost proportional to IR strength
            adjustment = lr * min(ir_val / 2.0, 1.0)
        # Negative signal: IR below negative threshold OR significantly negative PnL
        elif ir_val < -ir_thresh or pnl_val < -abs(pnl_thresh) - 0.01:
            # Reduce proportional to signal strength
            signal_strength = max(abs(ir_val), abs(pnl_val))
            adjustment = -lr * min(signal_strength / 2.0, 1.0)
        # Mixed/weak signal: small mean reversion to neutral
        else:
            adjustment = -lr * 0.1 * (prev_w - NEUTRAL_MULTIPLIER)
        
        # Apply adjustment and clamp
        new_w = prev_w + adjustment
        meta_weights[factor_name] = _clamp(
            new_w,
            NEUTRAL_MULTIPLIER - max_shift,
            NEUTRAL_MULTIPLIER + max_shift,
        )
    
    return meta_weights


def _update_category_emas(
    prev_state: CategoryMetaState,
    category_edges: Dict[str, Any],
    decay: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Update category EMAs from current category_edges.
    
    Args:
        prev_state: Previous category meta state
        category_edges: Current category edges from EdgeScanner
        decay: EMA decay factor
        
    Returns:
        Tuple of (updated_ema_ir, updated_ema_pnl)
    """
    ema_ir = dict(prev_state.ema_category_ir)
    ema_pnl = dict(prev_state.ema_category_pnl)
    
    for cat_name, edge_data in category_edges.items():
        if not isinstance(edge_data, dict):
            continue
        
        current_ir = float(edge_data.get("ir", edge_data.get("edge_score", 0.0)))
        current_pnl = float(edge_data.get("total_pnl", edge_data.get("pnl", 0.0)))
        
        prev_ir = ema_ir.get(cat_name, 0.0)
        prev_pnl = ema_pnl.get(cat_name, 0.0)
        
        ema_ir[cat_name] = _ema_update(prev_ir, current_ir, decay)
        ema_pnl[cat_name] = _ema_update(prev_pnl, current_pnl, decay)
    
    return ema_ir, ema_pnl


def _compute_category_overlays(
    ema_ir: Dict[str, float],
    ema_pnl: Dict[str, float],
    prev_overlays: Dict[str, float],
    cfg: MetaSchedulerConfig,
) -> Dict[str, float]:
    """
    Compute category meta overlays based on EMAs.
    
    Similar logic to factor overlays but for categories.
    
    Args:
        ema_ir: EMA of category IR values
        ema_pnl: EMA of category PnL values
        prev_overlays: Previous category overlays
        cfg: Meta-scheduler config
        
    Returns:
        Updated category overlays dict
    """
    overlays = dict(prev_overlays)
    lr = cfg.learning_rate
    max_shift = cfg.max_category_shift
    ir_thresh = cfg.ir_threshold
    pnl_thresh = cfg.pnl_threshold
    
    all_categories = set(ema_ir.keys()) | set(ema_pnl.keys()) | set(prev_overlays.keys())
    
    for cat_name in all_categories:
        ir_val = ema_ir.get(cat_name, 0.0)
        pnl_val = ema_pnl.get(cat_name, 0.0)
        prev_o = prev_overlays.get(cat_name, NEUTRAL_MULTIPLIER)
        
        adjustment = 0.0
        
        if ir_val > ir_thresh and pnl_val > pnl_thresh:
            adjustment = lr * min(ir_val / 2.0, 1.0)
        elif ir_val < -ir_thresh or pnl_val < -abs(pnl_thresh) - 0.01:
            signal_strength = max(abs(ir_val), abs(pnl_val))
            adjustment = -lr * min(signal_strength / 2.0, 1.0)
        else:
            adjustment = -lr * 0.1 * (prev_o - NEUTRAL_MULTIPLIER)
        
        new_o = prev_o + adjustment
        overlays[cat_name] = _clamp(
            new_o,
            NEUTRAL_MULTIPLIER - max_shift,
            NEUTRAL_MULTIPLIER + max_shift,
        )
    
    return overlays


def _compute_conviction_strength(
    prev_state: ConvictionMetaState,
    factor_edges: Dict[str, Any],
    strategy_health: Optional[Dict[str, Any]],
    cfg: MetaSchedulerConfig,
) -> tuple[float, float]:
    """
    Compute global conviction meta strength based on system health.
    
    Logic:
    - If health_score high AND factor edges strong → increase global_strength
    - If health low OR factor edges weak → decrease global_strength
    
    Args:
        prev_state: Previous conviction meta state
        factor_edges: Current factor edges
        strategy_health: Current strategy health dict (from P7)
        cfg: Meta-scheduler config
        
    Returns:
        Tuple of (new_global_strength, new_ema_health)
    """
    lr = cfg.learning_rate
    max_shift = cfg.max_conviction_shift
    decay = cfg.decay
    
    # Get current health score
    current_health = 0.5  # Default neutral
    if strategy_health is not None:
        current_health = float(strategy_health.get("health_score", 0.5))
    
    # Update EMA health
    ema_health = _ema_update(prev_state.ema_health, current_health, decay)
    
    # Compute mean factor edge for additional signal
    factor_scores = []
    for edge_data in factor_edges.values():
        if isinstance(edge_data, dict):
            score = float(edge_data.get("edge_score", edge_data.get("ir", 0.0)))
            factor_scores.append(score)
    mean_factor_edge = sum(factor_scores) / len(factor_scores) if factor_scores else 0.0
    
    # Compute adjustment
    prev_strength = prev_state.global_strength
    adjustment = 0.0
    
    # Strong positive: health high AND factors strong
    if ema_health > 0.65 and mean_factor_edge > 0.2:
        adjustment = lr * (ema_health - 0.5) * 2.0  # Scale by excess health
    # Weak signal: health low OR factors weak
    elif ema_health < 0.45 or mean_factor_edge < -0.2:
        adjustment = -lr * max(0.5 - ema_health, abs(mean_factor_edge)) * 2.0
    # Neutral: mean revert
    else:
        adjustment = -lr * 0.1 * (prev_strength - NEUTRAL_MULTIPLIER)
    
    new_strength = _clamp(
        prev_strength + adjustment,
        NEUTRAL_MULTIPLIER - max_shift,
        NEUTRAL_MULTIPLIER + max_shift,
    )
    
    return new_strength, ema_health


def meta_learning_step(
    cfg: MetaSchedulerConfig,
    prev_state: Optional[MetaSchedulerState],
    factor_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    strategy_health: Optional[Dict[str, Any]],
) -> MetaSchedulerState:
    """
    Perform one meta-learning update step.
    
    This is the core learning function. It:
    1. Updates EMAs for factors and categories
    2. Computes new overlay multipliers based on EMAs
    3. Updates conviction strength based on system health
    
    Args:
        cfg: MetaSchedulerConfig
        prev_state: Previous MetaSchedulerState (or None for fresh start)
        factor_edges: Dict of factor edge data from EdgeScanner
        category_edges: Dict of category edge data from EdgeScanner
        strategy_health: Strategy health dict from P7 (or None)
        
    Returns:
        Updated MetaSchedulerState
    """
    # If disabled or no health data, return previous state or neutral
    if not cfg.enabled:
        if prev_state is not None:
            return prev_state
        return create_neutral_state()
    
    # Initialize from previous state or create neutral
    if prev_state is None:
        prev_state = create_neutral_state()
    
    # Get stats
    sample_count = int(prev_state.stats.get("sample_count", 0))
    
    # Update sample count
    sample_count += 1
    
    # If not enough samples yet, just update EMAs but don't adjust overlays
    if sample_count < cfg.min_samples:
        # Update factor EMAs
        ema_ir, ema_pnl = _update_factor_emas(
            prev_state.factor_state, factor_edges, cfg.decay
        )
        # Update category EMAs
        cat_ema_ir, cat_ema_pnl = _update_category_emas(
            prev_state.category_state, category_edges, cfg.decay
        )
        # Update conviction EMA
        current_health = 0.5
        if strategy_health is not None:
            current_health = float(strategy_health.get("health_score", 0.5))
        ema_health = _ema_update(prev_state.conviction_state.ema_health, current_health, cfg.decay)
        
        return MetaSchedulerState(
            updated_ts=datetime.now(timezone.utc).isoformat(),
            factor_state=FactorMetaState(
                meta_weights=prev_state.factor_state.meta_weights,  # Keep neutral
                ema_ir=ema_ir,
                ema_pnl=ema_pnl,
            ),
            conviction_state=ConvictionMetaState(
                global_strength=prev_state.conviction_state.global_strength,  # Keep neutral
                ema_health=ema_health,
            ),
            category_state=CategoryMetaState(
                category_overlays=prev_state.category_state.category_overlays,  # Keep neutral
                ema_category_ir=cat_ema_ir,
                ema_category_pnl=cat_ema_pnl,
            ),
            stats={
                "sample_count": sample_count,
                "last_health_score": strategy_health.get("health_score") if strategy_health else None,
                "learning_active": False,
            },
        )
    
    # Full learning step
    
    # 1. Update factor state
    ema_ir, ema_pnl = _update_factor_emas(
        prev_state.factor_state, factor_edges, cfg.decay
    )
    meta_weights = _compute_factor_overlays(
        ema_ir, ema_pnl, prev_state.factor_state.meta_weights, cfg
    )
    
    # 2. Update category state
    cat_ema_ir, cat_ema_pnl = _update_category_emas(
        prev_state.category_state, category_edges, cfg.decay
    )
    category_overlays = _compute_category_overlays(
        cat_ema_ir, cat_ema_pnl, prev_state.category_state.category_overlays, cfg
    )
    
    # 3. Update conviction state
    global_strength, ema_health = _compute_conviction_strength(
        prev_state.conviction_state, factor_edges, strategy_health, cfg
    )
    
    return MetaSchedulerState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        factor_state=FactorMetaState(
            meta_weights=meta_weights,
            ema_ir=ema_ir,
            ema_pnl=ema_pnl,
        ),
        conviction_state=ConvictionMetaState(
            global_strength=global_strength,
            ema_health=ema_health,
        ),
        category_state=CategoryMetaState(
            category_overlays=category_overlays,
            ema_category_ir=cat_ema_ir,
            ema_category_pnl=cat_ema_pnl,
        ),
        stats={
            "sample_count": sample_count,
            "last_health_score": strategy_health.get("health_score") if strategy_health else None,
            "learning_active": True,
        },
    )


# ---------------------------------------------------------------------------
# Public API (Read-Only Views)
# ---------------------------------------------------------------------------


def get_factor_meta_weights(
    state: Optional[MetaSchedulerState],
) -> Dict[str, float]:
    """
    Get factor meta weight overlays.
    
    Returns neutral (1.0) for all factors if state is None or empty.
    
    Args:
        state: MetaSchedulerState or None
        
    Returns:
        Dict of factor_name → multiplier
    """
    if state is None:
        return {}
    return dict(state.factor_state.meta_weights)


def get_conviction_meta_strength(
    state: Optional[MetaSchedulerState],
) -> float:
    """
    Get global conviction meta strength multiplier.
    
    Returns 1.0 (neutral) if state is None.
    
    Args:
        state: MetaSchedulerState or None
        
    Returns:
        Global conviction multiplier
    """
    if state is None:
        return NEUTRAL_MULTIPLIER
    return state.conviction_state.global_strength


def get_category_meta_overlays(
    state: Optional[MetaSchedulerState],
) -> Dict[str, float]:
    """
    Get category meta overlays.
    
    Returns empty dict if state is None.
    
    Args:
        state: MetaSchedulerState or None
        
    Returns:
        Dict of category_name → multiplier
    """
    if state is None:
        return {}
    return dict(state.category_state.category_overlays)


def is_meta_scheduler_active(
    cfg: MetaSchedulerConfig,
    state: Optional[MetaSchedulerState],
) -> bool:
    """
    Check if meta-scheduler is actively learning (enabled + min_samples reached).
    
    Args:
        cfg: MetaSchedulerConfig
        state: MetaSchedulerState or None
        
    Returns:
        True if actively learning, False otherwise
    """
    if not cfg.enabled:
        return False
    if state is None:
        return False
    sample_count = int(state.stats.get("sample_count", 0))
    return sample_count >= cfg.min_samples


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------


__all__ = [
    # Config
    "MetaSchedulerConfig",
    "load_meta_scheduler_config",
    # State classes
    "FactorMetaState",
    "ConvictionMetaState",
    "CategoryMetaState",
    "MetaSchedulerState",
    "create_neutral_state",
    # State I/O
    "load_meta_scheduler_state",
    "write_meta_scheduler_state",
    # Learning
    "meta_learning_step",
    # Public API
    "get_factor_meta_weights",
    "get_conviction_meta_strength",
    "get_category_meta_overlays",
    "is_meta_scheduler_active",
    # Constants
    "DEFAULT_META_SCHEDULER_PATH",
    "NEUTRAL_MULTIPLIER",
]
