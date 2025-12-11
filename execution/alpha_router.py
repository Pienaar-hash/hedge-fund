"""
v7.8_P2 — Alpha Router (Dynamic Capital Allocation Engine)

Codename: "Overmind"

A portfolio-level allocation layer that computes:

    target_allocation ∈ [0, 1]

Based on:
- StrategyHealth (P7)
- MetaScheduler overlays (P8)
- Factor edge consistency
- Router execution quality
- Vol regime and DD regime

This allocation then adjusts:
- Maximum NAV at risk (max_total_exposure_pct)
- Position sizing ceilings (max_per_trade_nav_pct)
- Symbol caps (symbol_cap_pct)

WITHOUT modifying:
- Risk engine vetoes
- Router logic
- Exit engine
- Trade semantics

All adjustments are pure math transforms applied BEFORE sizing.

State Contract:
- Single writer: executor (via this module's write_alpha_router_state)
- Surface: logs/state/alpha_router_state.json
- Behaviour when disabled: identical to v7.8_P1 baseline
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ALPHA_ROUTER_PATH = Path("logs/state/alpha_router_state.json")

# Neutral allocation (full exposure)
NEUTRAL_ALLOCATION = 1.0

# Minimum valid allocation (never zero to avoid flatline)
MIN_ALLOCATION = 0.05


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlphaRouterConfig:
    """
    Configuration for the Alpha Router (Dynamic Capital Allocation Engine).
    
    Attributes:
        enabled: Whether alpha router is active (default: False for back-compat)
        allocation_floor: Minimum allocation, e.g. 0.20 (never less than 20%)
        allocation_ceiling: Maximum allocation, e.g. 1.00 (full NAV at risk)
        health_thresholds: Thresholds for health-based allocation
            - "strong": health_score >= this → full allocation
            - "weak": health_score <= this → reduced allocation
        regime_penalties: Multipliers per volatility regime
            - e.g. {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.80, "CRISIS": 0.50}
        dd_penalties: Multipliers per drawdown state
            - e.g. {"NORMAL": 1.0, "RECOVERY": 0.85, "DRAWDOWN": 0.65}
        router_quality_thresholds: Thresholds for router execution quality
            - "good": >= this → no penalty
            - "moderate": >= this → small penalty
            - "poor": < moderate → larger penalty
        meta_influence: Maximum ± shift from meta-scheduler overlay
        smoothing_alpha: EMA alpha for allocation stability (0 = no smoothing)
    """
    
    enabled: bool = False
    allocation_floor: float = 0.20
    allocation_ceiling: float = 1.00
    health_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "strong": 0.75,
        "weak": 0.45,
    })
    regime_penalties: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 1.0,
        "NORMAL": 1.0,
        "HIGH": 0.80,
        "CRISIS": 0.50,
    })
    dd_penalties: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "RECOVERY": 0.85,
        "DRAWDOWN": 0.65,
    })
    router_quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "good": 0.65,
        "moderate": 0.45,
    })
    meta_influence: float = 0.10
    smoothing_alpha: float = 0.25


def load_alpha_router_config(
    strategy_cfg: Optional[Mapping[str, Any]] = None,
) -> AlphaRouterConfig:
    """
    Load AlphaRouterConfig from strategy_config.json.
    
    Expects a top-level "alpha_router" block:
    {
        "alpha_router": {
            "enabled": false,
            "allocation_floor": 0.20,
            ...
        }
    }
    
    Args:
        strategy_cfg: Strategy config dict, or None for defaults.
        
    Returns:
        AlphaRouterConfig instance
    """
    if strategy_cfg is None:
        return AlphaRouterConfig()
    
    ar_cfg = strategy_cfg.get("alpha_router", {})
    if not isinstance(ar_cfg, Mapping):
        return AlphaRouterConfig()
    
    # Build config with overrides
    return AlphaRouterConfig(
        enabled=bool(ar_cfg.get("enabled", False)),
        allocation_floor=float(ar_cfg.get("allocation_floor", 0.20)),
        allocation_ceiling=float(ar_cfg.get("allocation_ceiling", 1.00)),
        health_thresholds=dict(ar_cfg.get("health_thresholds", {
            "strong": 0.75,
            "weak": 0.45,
        })),
        regime_penalties=dict(ar_cfg.get("regime_penalties", {
            "LOW": 1.0,
            "NORMAL": 1.0,
            "HIGH": 0.80,
            "CRISIS": 0.50,
        })),
        dd_penalties=dict(ar_cfg.get("dd_penalties", {
            "NORMAL": 1.0,
            "RECOVERY": 0.85,
            "DRAWDOWN": 0.65,
        })),
        router_quality_thresholds=dict(ar_cfg.get("router_quality_thresholds", {
            "good": 0.65,
            "moderate": 0.45,
        })),
        meta_influence=float(ar_cfg.get("meta_influence", 0.10)),
        smoothing_alpha=float(ar_cfg.get("smoothing_alpha", 0.25)),
    )


# ---------------------------------------------------------------------------
# State Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AlphaRouterState:
    """
    Alpha Router state snapshot.
    
    Attributes:
        updated_ts: ISO timestamp of last update
        target_allocation: Final computed allocation ∈ [floor, ceiling]
        raw_components: Dict with intermediate contribution values
        smoothed: Whether smoothing was applied
        prev_allocation: Previous allocation (for EMA)
    """
    
    updated_ts: str = ""
    target_allocation: float = NEUTRAL_ALLOCATION
    raw_components: Dict[str, Any] = field(default_factory=dict)
    smoothed: bool = False
    prev_allocation: float = NEUTRAL_ALLOCATION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "target_allocation": round(self.target_allocation, 4),
            "raw_components": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.raw_components.items()
            },
            "smoothed": self.smoothed,
            "prev_allocation": round(self.prev_allocation, 4),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlphaRouterState":
        """Reconstruct from dictionary."""
        return cls(
            updated_ts=str(data.get("updated_ts", "")),
            target_allocation=float(data.get("target_allocation", NEUTRAL_ALLOCATION)),
            raw_components=dict(data.get("raw_components", {})),
            smoothed=bool(data.get("smoothed", False)),
            prev_allocation=float(data.get("prev_allocation", NEUTRAL_ALLOCATION)),
        )


def create_neutral_state() -> AlphaRouterState:
    """
    Create a neutral alpha router state with full allocation.
    
    Returns:
        AlphaRouterState with allocation = 1.0
    """
    return AlphaRouterState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        target_allocation=NEUTRAL_ALLOCATION,
        raw_components={
            "health_base": NEUTRAL_ALLOCATION,
            "vol_penalty": 1.0,
            "dd_penalty": 1.0,
            "router_penalty": 1.0,
            "meta_adjustment": 0.0,
            "raw_allocation": NEUTRAL_ALLOCATION,
        },
        smoothed=False,
        prev_allocation=NEUTRAL_ALLOCATION,
    )


# ---------------------------------------------------------------------------
# State Loader & Writer
# ---------------------------------------------------------------------------


def load_alpha_router_state(
    path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> Optional[AlphaRouterState]:
    """
    Load alpha router state from file.
    
    Args:
        path: Path to alpha_router_state.json
        
    Returns:
        AlphaRouterState if file exists and is valid, else None
    """
    path = Path(path)
    if not path.exists():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AlphaRouterState.from_dict(data)
    except Exception:
        return None


def write_alpha_router_state(
    state: AlphaRouterState,
    path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> None:
    """
    Write alpha router state to file (atomic write).
    
    This is the ONLY allowed writer for alpha_router_state.json.
    
    Args:
        state: AlphaRouterState to write
        path: Output path (default: logs/state/alpha_router_state.json)
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
        # Fail silently - alpha router state is non-critical
        pass


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def _ema_smooth(prev: float, curr: float, alpha: float) -> float:
    """
    Apply exponential moving average smoothing.
    
    Args:
        prev: Previous value
        curr: Current value
        alpha: Smoothing factor (0 = use prev, 1 = use curr)
        
    Returns:
        Smoothed value
    """
    if alpha <= 0:
        return prev
    if alpha >= 1:
        return curr
    return (1.0 - alpha) * prev + alpha * curr


def _compute_health_base(
    health_score: float,
    cfg: AlphaRouterConfig,
) -> float:
    """
    Compute base allocation from strategy health score.
    
    Logic:
    - If health_score >= strong threshold → 1.0 (full)
    - If health_score <= weak threshold → 0.30 (reduced)
    - Otherwise → linear interpolation
    
    Args:
        health_score: Strategy health score ∈ [0, 1]
        cfg: Alpha router config
        
    Returns:
        Base allocation ∈ [0.30, 1.0]
    """
    strong = cfg.health_thresholds.get("strong", 0.75)
    weak = cfg.health_thresholds.get("weak", 0.45)
    
    if health_score >= strong:
        return 1.0
    elif health_score <= weak:
        return 0.30
    else:
        # Linear interpolation between weak (0.30) and strong (1.0)
        range_health = strong - weak
        if range_health <= 0:
            return 0.65  # midpoint fallback
        position = (health_score - weak) / range_health
        return 0.30 + position * 0.70


def _compute_vol_penalty(
    vol_regime: str,
    cfg: AlphaRouterConfig,
) -> float:
    """
    Get volatility regime penalty multiplier.
    
    Args:
        vol_regime: Volatility regime (LOW/NORMAL/HIGH/CRISIS or lowercase)
        cfg: Alpha router config
        
    Returns:
        Penalty multiplier ∈ (0, 1]
    """
    vol_key = vol_regime.upper() if vol_regime else "NORMAL"
    return cfg.regime_penalties.get(vol_key, 1.0)


def _compute_dd_penalty(
    dd_state: str,
    cfg: AlphaRouterConfig,
) -> float:
    """
    Get drawdown state penalty multiplier.
    
    Args:
        dd_state: Drawdown state (NORMAL/RECOVERY/DRAWDOWN)
        cfg: Alpha router config
        
    Returns:
        Penalty multiplier ∈ (0, 1]
    """
    dd_key = dd_state.upper() if dd_state else "NORMAL"
    return cfg.dd_penalties.get(dd_key, 1.0)


def _compute_router_penalty(
    router_quality: float,
    cfg: AlphaRouterConfig,
) -> float:
    """
    Compute router execution quality penalty.
    
    Logic:
    - quality >= good → 1.0 (no penalty)
    - quality >= moderate → 0.90 (small penalty)
    - quality < moderate → 0.70 (larger penalty)
    
    Args:
        router_quality: Global router quality score ∈ [0, 1]
        cfg: Alpha router config
        
    Returns:
        Penalty multiplier ∈ [0.70, 1.0]
    """
    good = cfg.router_quality_thresholds.get("good", 0.65)
    moderate = cfg.router_quality_thresholds.get("moderate", 0.45)
    
    if router_quality >= good:
        return 1.0
    elif router_quality >= moderate:
        return 0.90
    else:
        return 0.70


def _compute_meta_adjustment(
    meta_state: Optional[Any],
    cfg: AlphaRouterConfig,
) -> float:
    """
    Compute meta-scheduler influenced adjustment.
    
    Looks at:
    - Meta conviction strength (if available)
    - EMA health (from meta_scheduler)
    
    Returns bounded adjustment ∈ [-meta_influence, +meta_influence]
    
    Args:
        meta_state: MetaSchedulerState or None
        cfg: Alpha router config
        
    Returns:
        Adjustment value ∈ [-meta_influence, +meta_influence]
    """
    if meta_state is None:
        return 0.0
    
    max_shift = cfg.meta_influence
    
    # Try to get conviction meta strength
    conviction_state = getattr(meta_state, "conviction_state", None)
    if conviction_state is None:
        return 0.0
    
    global_strength = getattr(conviction_state, "global_strength", 1.0)
    ema_health = getattr(conviction_state, "ema_health", 0.5)
    
    # Deviation from neutral (1.0)
    strength_deviation = global_strength - 1.0
    
    # If meta-scheduler is pushing strength up AND health is good → positive adjustment
    # If meta-scheduler is pulling strength down → negative adjustment
    if strength_deviation > 0 and ema_health > 0.5:
        # Upward nudge
        adjustment = strength_deviation * max_shift
    elif strength_deviation < 0:
        # Downward nudge (scaled by magnitude)
        adjustment = strength_deviation * max_shift
    else:
        adjustment = 0.0
    
    # Clamp to bounds
    return _clamp(adjustment, -max_shift, max_shift)


def _compute_sentinel_x_factor(
    sentinel_x_state: Optional[Dict[str, Any]],
    cfg: AlphaRouterConfig,
) -> tuple[float, str]:
    """
    Compute Sentinel-X regime-based allocation factor.
    
    Uses Sentinel-X primary regime to adjust allocation:
    - TREND_UP: slight boost (1.05)
    - TREND_DOWN: reduce (0.90)
    - MEAN_REVERT: slight reduce (0.95)
    - BREAKOUT: neutral (1.00)
    - CHOPPY: reduce (0.85)
    - CRISIS: strong reduce (0.60)
    
    v7.8_P6: Sentinel-X integration for Alpha Router.
    
    Args:
        sentinel_x_state: Sentinel-X state dict or None
        cfg: Alpha router config
        
    Returns:
        Tuple of (factor, regime_label)
    """
    if sentinel_x_state is None:
        return 1.0, ""
    
    primary_regime = sentinel_x_state.get("primary_regime", "")
    if not primary_regime:
        return 1.0, ""
    
    # Default allocation factors per regime
    allocation_factors = {
        "TREND_UP": 1.05,
        "TREND_DOWN": 0.90,
        "MEAN_REVERT": 0.95,
        "BREAKOUT": 1.00,
        "CHOPPY": 0.85,
        "CRISIS": 0.60,
    }
    
    factor = allocation_factors.get(primary_regime, 1.0)
    return factor, primary_regime


# ---------------------------------------------------------------------------
# Core Allocation Computation
# ---------------------------------------------------------------------------


def compute_target_allocation(
    health: Dict[str, Any],
    meta_state: Optional[Any],
    router_quality: float,
    vol_regime: str,
    dd_state: str,
    cfg: AlphaRouterConfig,
    prev_state: Optional[AlphaRouterState] = None,
    sentinel_x_state: Optional[Dict[str, Any]] = None,
) -> AlphaRouterState:
    """
    Compute target capital allocation based on all input signals.
    
    Components:
    1. Health-based allocation (from strategy health score)
    2. Regime penalties (vol_regime, dd_state multipliers)
    3. Router quality penalty
    4. MetaScheduler influence (small bounded drift)
    5. Sentinel-X regime factor (v7.8_P6)
    6. EMA smoothing (if prev_state available)
    7. Clamp to [floor, ceiling]
    
    Args:
        health: Strategy health dict with "health_score" key
        meta_state: MetaSchedulerState or None
        router_quality: Global router quality ∈ [0, 1]
        vol_regime: Volatility regime label
        dd_state: Drawdown state label
        cfg: AlphaRouterConfig
        prev_state: Previous AlphaRouterState for smoothing
        sentinel_x_state: Sentinel-X state dict (v7.8_P6)
        
    Returns:
        AlphaRouterState with computed target_allocation
    """
    # Extract health score
    health_score = float(health.get("health_score", 0.5))
    
    # Step 1: Health-based allocation
    health_base = _compute_health_base(health_score, cfg)
    
    # Step 2: Regime penalties
    vol_penalty = _compute_vol_penalty(vol_regime, cfg)
    dd_penalty = _compute_dd_penalty(dd_state, cfg)
    
    # Step 3: Router quality penalty
    router_penalty = _compute_router_penalty(router_quality, cfg)
    
    # Step 4: Meta-scheduler adjustment
    meta_adjustment = _compute_meta_adjustment(meta_state, cfg)
    
    # Step 5: Sentinel-X regime factor (v7.8_P6)
    sentinel_x_factor, sentinel_x_regime = _compute_sentinel_x_factor(sentinel_x_state, cfg)
    
    # Combine components
    raw_allocation = health_base * vol_penalty * dd_penalty * router_penalty * sentinel_x_factor
    raw_allocation += meta_adjustment
    
    # Step 6: EMA smoothing
    prev_alloc = NEUTRAL_ALLOCATION
    smoothed = False
    if prev_state is not None:
        prev_alloc = prev_state.target_allocation
        if cfg.smoothing_alpha > 0:
            raw_allocation = _ema_smooth(prev_alloc, raw_allocation, cfg.smoothing_alpha)
            smoothed = True
    
    # Step 7: Clamp to floor/ceiling
    target_allocation = _clamp(raw_allocation, cfg.allocation_floor, cfg.allocation_ceiling)
    
    return AlphaRouterState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        target_allocation=target_allocation,
        raw_components={
            "health_score": health_score,
            "health_base": health_base,
            "vol_regime": vol_regime,
            "vol_penalty": vol_penalty,
            "dd_state": dd_state,
            "dd_penalty": dd_penalty,
            "router_quality": router_quality,
            "router_penalty": router_penalty,
            "meta_adjustment": meta_adjustment,
            "sentinel_x_factor": sentinel_x_factor,  # v7.8_P6
            "sentinel_x_regime": sentinel_x_regime,  # v7.8_P6
            "raw_allocation": raw_allocation,
        },
        smoothed=smoothed,
        prev_allocation=prev_alloc,
    )


def get_alpha_decay_router_adjustment(
    alpha_decay_state: Optional[Dict[str, Any]],
) -> float:
    """
    Get allocation adjustment based on overall alpha health (v7.8_P7).
    
    Args:
        alpha_decay_state: Alpha decay state dict
        
    Returns:
        Multiplier typically in [0.8, 1.0]
    """
    if alpha_decay_state is None:
        return 1.0
    
    avg_survival = alpha_decay_state.get("avg_symbol_survival", 0.5)
    
    # allocation *= (0.8 + 0.2 * avg_survival)
    return 0.8 + 0.2 * avg_survival


def get_cerberus_router_adjustment(
    cerberus_state: Optional[Dict[str, Any]],
) -> float:
    """
    Get allocation adjustment based on Cerberus head state (v7.8_P8).
    
    Uses weighted average of head multipliers to adjust overall allocation.
    
    Args:
        cerberus_state: Cerberus state dict
        
    Returns:
        Multiplier typically in [0.5, 1.5]
    """
    if cerberus_state is None:
        return 1.0
    
    head_state = cerberus_state.get("head_state", {})
    if not head_state:
        return 1.0
    
    heads = head_state.get("heads", {})
    if not heads:
        return 1.0
    
    # Weight TREND and VOL_HARVEST higher for allocation decisions
    weights = {
        "TREND": 0.30,
        "MEAN_REVERT": 0.15,
        "RELATIVE_VALUE": 0.10,
        "CATEGORY": 0.15,
        "VOL_HARVEST": 0.20,
        "EMERGENT_ALPHA": 0.10,
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for head_name, weight in weights.items():
        head_data = heads.get(head_name, {})
        if isinstance(head_data, dict):
            multiplier = head_data.get("multiplier", 1.0)
            weighted_sum += multiplier * weight
            total_weight += weight
    
    if total_weight <= 0:
        return 1.0
    
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Activation Check
# ---------------------------------------------------------------------------


def is_alpha_router_active(
    cfg: AlphaRouterConfig,
    state: Optional[AlphaRouterState] = None,
) -> bool:
    """
    Check if alpha router is active and should be applied.
    
    Args:
        cfg: AlphaRouterConfig
        state: Optional AlphaRouterState (not used for now, reserved)
        
    Returns:
        True if alpha router should be applied
    """
    return cfg.enabled


# ---------------------------------------------------------------------------
# Public API - Views
# ---------------------------------------------------------------------------


def get_target_allocation(
    state: Optional[AlphaRouterState] = None,
    path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> float:
    """
    Get current target allocation from state.
    
    Returns NEUTRAL_ALLOCATION (1.0) if:
    - State is None
    - State file doesn't exist
    - Any error occurs
    
    Args:
        state: Optional pre-loaded state
        path: Path to state file if state not provided
        
    Returns:
        Target allocation ∈ [0, 1]
    """
    if state is not None:
        return state.target_allocation
    
    loaded = load_alpha_router_state(path)
    if loaded is None:
        return NEUTRAL_ALLOCATION
    
    return loaded.target_allocation


def get_allocation_components(
    state: Optional[AlphaRouterState] = None,
    path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> Dict[str, Any]:
    """
    Get allocation components breakdown for dashboard/debugging.
    
    Returns empty dict if state unavailable.
    
    Args:
        state: Optional pre-loaded state
        path: Path to state file if state not provided
        
    Returns:
        Dict with allocation components
    """
    if state is not None:
        return dict(state.raw_components)
    
    loaded = load_alpha_router_state(path)
    if loaded is None:
        return {}
    
    return dict(loaded.raw_components)


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def apply_allocation_to_limits(
    allocation: float,
    max_total_exposure_pct: float,
    max_per_trade_nav_pct: float,
    symbol_cap_pct: float,
) -> tuple[float, float, float]:
    """
    Apply allocation multiplier to sizing limits.
    
    CRITICAL: Allocation only scales DOWN from configured ceilings.
    It never increases limits above their configured values.
    
    Args:
        allocation: Target allocation ∈ [0, 1]
        max_total_exposure_pct: Configured max total exposure
        max_per_trade_nav_pct: Configured max per-trade NAV
        symbol_cap_pct: Configured per-symbol cap
        
    Returns:
        Tuple of (adjusted_exposure, adjusted_per_trade, adjusted_symbol_cap)
    """
    # Clamp allocation to valid range
    alloc = _clamp(allocation, MIN_ALLOCATION, 1.0)
    
    return (
        max_total_exposure_pct * alloc,
        max_per_trade_nav_pct * alloc,
        symbol_cap_pct * alloc,
    )


def load_allocation_for_sizing(
    cfg: Optional[AlphaRouterConfig] = None,
    state: Optional[AlphaRouterState] = None,
    path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> float:
    """
    Load allocation for use in sizing calculations.
    
    Returns NEUTRAL_ALLOCATION if:
    - Alpha router is disabled
    - State unavailable
    
    Args:
        cfg: AlphaRouterConfig (will load if None)
        state: Pre-loaded state (will load if None)
        path: Path to state file
        
    Returns:
        Allocation ∈ [floor, ceiling] or 1.0 if disabled
    """
    if cfg is None:
        cfg = load_alpha_router_config(None)
    
    if not cfg.enabled:
        return NEUTRAL_ALLOCATION
    
    return get_target_allocation(state, path)


# ---------------------------------------------------------------------------
# Orchestration - Run Alpha Router Step
# ---------------------------------------------------------------------------


def run_alpha_router_step(
    health: Dict[str, Any],
    meta_state: Optional[Any],
    router_quality: float,
    vol_regime: str,
    dd_state: str,
    strategy_cfg: Optional[Mapping[str, Any]] = None,
    state_path: Path | str = DEFAULT_ALPHA_ROUTER_PATH,
) -> AlphaRouterState:
    """
    Run one alpha router computation step.
    
    This is the main entry point for the executor to call.
    
    Steps:
    1. Load config
    2. If disabled → return neutral state
    3. Load previous state
    4. Compute new allocation
    5. Write state
    6. Return state
    
    Args:
        health: Strategy health dict with "health_score"
        meta_state: MetaSchedulerState or None
        router_quality: Global router quality ∈ [0, 1]
        vol_regime: Volatility regime label
        dd_state: Drawdown state label
        strategy_cfg: Strategy config dict or None
        state_path: Path to state file
        
    Returns:
        AlphaRouterState (neutral if disabled)
    """
    cfg = load_alpha_router_config(strategy_cfg)
    
    if not cfg.enabled:
        # Return neutral state without writing
        return create_neutral_state()
    
    # Load previous state for smoothing
    prev_state = load_alpha_router_state(state_path)
    
    # Compute new allocation
    new_state = compute_target_allocation(
        health=health,
        meta_state=meta_state,
        router_quality=router_quality,
        vol_regime=vol_regime,
        dd_state=dd_state,
        cfg=cfg,
        prev_state=prev_state,
    )
    
    # Write state
    write_alpha_router_state(new_state, state_path)
    
    return new_state


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "AlphaRouterConfig",
    "load_alpha_router_config",
    # State
    "AlphaRouterState",
    "create_neutral_state",
    "load_alpha_router_state",
    "write_alpha_router_state",
    # Computation
    "compute_target_allocation",
    "is_alpha_router_active",
    # Views
    "get_target_allocation",
    "get_allocation_components",
    # Integration
    "apply_allocation_to_limits",
    "load_allocation_for_sizing",
    # Orchestration
    "run_alpha_router_step",
    # Constants
    "NEUTRAL_ALLOCATION",
    "DEFAULT_ALPHA_ROUTER_PATH",
]
