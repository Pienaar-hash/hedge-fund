"""
Conviction Sizing Engine (v7.7)

Pure deterministic function to compute conviction scores for position sizing.
Uses existing factors (hybrid_score, expectancy, router_quality, vol_regime)
and risk state (dd_state, risk_mode) to produce a conviction multiplier.

This module:
- Reads state surfaces (read-only)
- Never writes to any state file
- Produces a conviction score in [0.0, 1.0]
- Applies sizing multipliers based on conviction band
- Respects DD/risk_mode overrides for safety
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DDState = Literal["NORMAL", "DRAWDOWN", "RECOVERY"]
RiskMode = Literal["OK", "WARN", "DEFENSIVE", "HALTED", "EXTREME"]
VolRegimeLabel = Literal["low", "normal", "high", "crisis"]
ConvictionBand = Literal["very_low", "low", "medium", "high", "very_high"]

# ---------------------------------------------------------------------------
# State Paths
# ---------------------------------------------------------------------------

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
RISK_SNAPSHOT_PATH = STATE_DIR / "risk_snapshot.json"
ROUTER_HEALTH_PATH = STATE_DIR / "router_health.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeCurvesConfig:
    """Configuration for regime-specific conviction curves (v7.7_P6)."""
    
    # Volatility regime multipliers
    volatility: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 1.10,
        "NORMAL": 1.00,
        "HIGH": 0.75,
        "CRISIS": 0.40,
    })
    
    # Drawdown state multipliers
    drawdown: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.00,
        "RECOVERY": 0.85,
        "DRAWDOWN": 0.50,
    })
    
    # Smoothing alpha for EMA (0 = no smoothing, 1 = no memory)
    smoothing_alpha: float = 0.1


def load_regime_curves_config(conviction_cfg: Optional[Mapping[str, Any]] = None) -> RegimeCurvesConfig:
    """Load regime curves config from conviction config block."""
    if conviction_cfg is None:
        return RegimeCurvesConfig()
    
    regime_curves = conviction_cfg.get("regime_curves", {})
    if not isinstance(regime_curves, Mapping):
        return RegimeCurvesConfig()
    
    return RegimeCurvesConfig(
        volatility=dict(regime_curves.get("volatility", RegimeCurvesConfig().volatility)),
        drawdown=dict(regime_curves.get("drawdown", RegimeCurvesConfig().drawdown)),
        smoothing_alpha=float(regime_curves.get("smoothing_alpha", 0.1)),
    )


@dataclass
class ConvictionConfig:
    """Configuration for conviction-weighted sizing."""
    # Default to False for backward compatibility with existing code/tests.
    # Production should explicitly enable via strategy_config.json.
    enabled: bool = False
    
    # Thresholds for conviction bands (score must be >= threshold for band)
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_low": 0.20,
        "low": 0.40,
        "medium": 0.60,
        "high": 0.80,
        "very_high": 0.92,
    })
    
    # Size multipliers per conviction band
    size_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "very_low": 0.3,
        "low": 0.5,
        "medium": 1.0,
        "high": 1.6,
        "very_high": 2.2,
    })
    
    # Drawdown state overrides (multiplier caps)
    dd_overrides: Dict[str, float] = field(default_factory=lambda: {
        "DRAWDOWN": 0.5,
        "RECOVERY": 0.8,
    })
    
    # Risk mode overrides (multiplier caps)
    risk_mode_overrides: Dict[str, float] = field(default_factory=lambda: {
        "DEFENSIVE": 0.5,
        "EXTREME": 0.0,
    })
    
    # Router quality thresholds
    router_quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_for_any_trade": 0.50,
        "min_for_full_size": 0.80,
    })
    
    # v7.7_P3: Category momentum contribution weight (0.0 = disabled)
    category_weight: float = 0.0
    
    # v7.7_P6: Regime-specific conviction curves
    regime_curves: RegimeCurvesConfig = field(default_factory=RegimeCurvesConfig)


# v7.7_P6: State for EMA smoothing of regime-adjusted conviction
_prev_conviction_cache: Dict[str, float] = {}


def clear_conviction_cache() -> None:
    """
    Clear the conviction EMA smoothing cache.
    
    Call this between tests or when resetting state.
    """
    global _prev_conviction_cache
    _prev_conviction_cache.clear()


def load_conviction_config(strategy_config: Optional[Mapping[str, Any]] = None) -> ConvictionConfig:
    """Load conviction config from strategy_config."""
    if strategy_config is None:
        return ConvictionConfig()
    
    conviction_cfg = strategy_config.get("conviction", {})
    if not isinstance(conviction_cfg, Mapping):
        return ConvictionConfig()
    
    return ConvictionConfig(
        enabled=bool(conviction_cfg.get("enabled", False)),  # Default to False for backward compat
        thresholds=dict(conviction_cfg.get("thresholds", ConvictionConfig().thresholds)),
        size_multipliers=dict(conviction_cfg.get("size_multipliers", ConvictionConfig().size_multipliers)),
        dd_overrides=dict(conviction_cfg.get("dd_overrides", ConvictionConfig().dd_overrides)),
        risk_mode_overrides=dict(conviction_cfg.get("risk_mode_overrides", ConvictionConfig().risk_mode_overrides)),
        router_quality_thresholds=dict(conviction_cfg.get("router_quality_thresholds", ConvictionConfig().router_quality_thresholds)),
        category_weight=float(conviction_cfg.get("category_weight", 0.0)),  # v7.7_P3
        regime_curves=load_regime_curves_config(conviction_cfg),  # v7.7_P6
    )


# ---------------------------------------------------------------------------
# Context for Conviction Computation
# ---------------------------------------------------------------------------

@dataclass
class ConvictionContext:
    """Input context for conviction score computation."""
    # Core factors
    hybrid_score: float = 0.0
    expectancy_alpha: float = 0.0
    router_quality: float = 1.0
    trend_strength: float = 0.0
    
    # v7.7_P3: Category momentum factor
    category_momentum: float = 0.0  # In [-1, 1], 0 = neutral
    
    # Regime / risk state
    vol_regime: VolRegimeLabel = "normal"
    dd_state: DDState = "NORMAL"
    risk_mode: RiskMode = "OK"
    
    # v7.8_P1: Meta-scheduler conviction overlay (default 1.0 = no effect)
    meta_strength: float = 1.0
    
    # v7.8_P6: Sentinel-X regime classifier weight (default 1.0 = no effect)
    sentinel_x_weight: float = 1.0
    sentinel_x_regime: str = ""  # Primary regime label from Sentinel-X
    
    # v7.8_P7: Alpha decay conviction adjustment (default 1.0 = no effect)
    alpha_decay_adjustment: float = 1.0
    
    # v7.8_P8: Cerberus multi-strategy conviction multiplier (default 1.0 = no effect)
    cerberus_conviction_multiplier: float = 1.0


@dataclass
class ConvictionResult:
    """Result of conviction score computation."""
    conviction_score: float
    conviction_band: ConvictionBand
    size_multiplier: float
    vetoed: bool = False
    veto_reason: Optional[str] = None
    components: Dict[str, float] = field(default_factory=dict)
    # v7.7_P6: Regime modifiers applied to conviction
    regime_modifiers: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State Loaders (Read-Only)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    """Safely load JSON from path."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_risk_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load risk_snapshot.json (read-only)."""
    return _load_json(path or RISK_SNAPSHOT_PATH)


def load_router_health(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load router_health.json (read-only)."""
    return _load_json(path or ROUTER_HEALTH_PATH)


def get_global_router_quality(router_health: Mapping[str, Any]) -> float:
    """Extract global router quality score from router_health snapshot."""
    # Try nested structure first: router_health.global.quality_score
    rh_block = router_health.get("router_health", {})
    if isinstance(rh_block, Mapping):
        global_block = rh_block.get("global", {})
        if isinstance(global_block, Mapping):
            qs = global_block.get("quality_score")
            if qs is not None:
                try:
                    return float(qs)
                except (TypeError, ValueError):
                    pass
    
    # Fallback: top-level global
    global_block = router_health.get("global", {})
    if isinstance(global_block, Mapping):
        qs = global_block.get("quality_score")
        if qs is not None:
            try:
                return float(qs)
            except (TypeError, ValueError):
                pass
    
    # Default to 1.0 (no penalty) if unavailable
    return 1.0


def get_dd_state(risk_snapshot: Mapping[str, Any]) -> DDState:
    """Extract dd_state from risk_snapshot."""
    dd_state = risk_snapshot.get("dd_state", "NORMAL")
    if dd_state in ("NORMAL", "DRAWDOWN", "RECOVERY"):
        return dd_state  # type: ignore[return-value]
    return "NORMAL"


def get_risk_mode(risk_snapshot: Mapping[str, Any]) -> RiskMode:
    """Extract risk_mode from risk_snapshot."""
    risk_mode = risk_snapshot.get("risk_mode", "OK")
    if risk_mode in ("OK", "WARN", "DEFENSIVE", "HALTED", "EXTREME"):
        return risk_mode  # type: ignore[return-value]
    return "OK"


# ---------------------------------------------------------------------------
# Conviction Score Computation
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def _vol_regime_penalty(vol_regime: VolRegimeLabel) -> float:
    """
    Return penalty factor for volatility regime.
    Low/normal = 1.0, high = 0.85, crisis = 0.6
    
    Note: This is the legacy function. v7.7_P6 uses regime curves instead.
    """
    penalties = {
        "low": 1.0,
        "normal": 1.0,
        "high": 0.85,
        "crisis": 0.6,
    }
    return penalties.get(vol_regime, 1.0)


def get_vol_regime_multiplier(
    vol_regime: VolRegimeLabel,
    regime_curves: RegimeCurvesConfig,
) -> float:
    """
    Get volatility regime multiplier from regime curves config (v7.7_P6).
    
    Args:
        vol_regime: Current volatility regime (low/normal/high/crisis)
        regime_curves: RegimeCurvesConfig with volatility multipliers
        
    Returns:
        Multiplier in range [0.4, 1.1] typically
    """
    # Map lowercase vol_regime to uppercase key
    vol_key = vol_regime.upper() if vol_regime else "NORMAL"
    return regime_curves.volatility.get(vol_key, 1.0)


def get_dd_regime_multiplier(
    dd_state: DDState,
    regime_curves: RegimeCurvesConfig,
) -> float:
    """
    Get drawdown state multiplier from regime curves config (v7.7_P6).
    
    Args:
        dd_state: Current drawdown state (NORMAL/RECOVERY/DRAWDOWN)
        regime_curves: RegimeCurvesConfig with drawdown multipliers
        
    Returns:
        Multiplier in range [0.5, 1.0] typically
    """
    return regime_curves.drawdown.get(dd_state, 1.0)


def _ema_smooth(
    prev_value: float,
    curr_value: float,
    alpha: float,
) -> float:
    """
    Apply exponential moving average smoothing.
    
    Args:
        prev_value: Previous smoothed value
        curr_value: Current raw value
        alpha: Smoothing factor (0 = no smoothing, 1 = no memory)
        
    Returns:
        Smoothed value
    """
    return alpha * curr_value + (1.0 - alpha) * prev_value


def _compute_raw_conviction_score(
    ctx: ConvictionContext,
    cfg: Optional[ConvictionConfig] = None,
) -> float:
    """
    Compute raw conviction score WITHOUT any regime adjustments.
    
    This is the base formula before vol_regime or dd_state penalties.
    
    Formula:
        raw = (0.4 * hybrid_score + 0.25 * expectancy_alpha + 
               0.20 * router_quality + 0.15 * trend_strength)
               + category_weight * category_momentum_bias
    
    Returns:
        Float in [0.0, 1.0]
    """
    if cfg is None:
        cfg = ConvictionConfig()
    
    # Normalize inputs to [0, 1]
    hybrid = _clamp(ctx.hybrid_score, 0.0, 1.0)
    expectancy = _clamp(ctx.expectancy_alpha, 0.0, 1.0)
    router_q = _clamp(ctx.router_quality, 0.0, 1.0)
    trend = _clamp(ctx.trend_strength, 0.0, 1.0)
    
    # Weighted combination
    raw = (
        0.40 * hybrid +
        0.25 * expectancy +
        0.20 * router_q +
        0.15 * trend
    )
    
    # v7.7_P3: Add category momentum contribution (if enabled)
    if cfg.category_weight > 0 and ctx.category_momentum != 0:
        cat_contrib = (ctx.category_momentum + 1.0) / 2.0  # Map [-1,1] to [0,1]
        cat_bias = cfg.category_weight * (cat_contrib - 0.5)  # [-weight/2, +weight/2]
        raw += cat_bias
    
    return _clamp(raw, 0.0, 1.0)


def compute_conviction_score(
    ctx: ConvictionContext,
    cfg: Optional[ConvictionConfig] = None,
) -> float:
    """
    Compute raw conviction score from input context.
    
    Formula:
        raw = (0.4 * hybrid_score + 0.25 * expectancy_alpha + 
               0.20 * router_quality + 0.15 * trend_strength)
        conviction = raw * vol_regime_penalty + category_weight * category_momentum
    
    v7.7_P3: Category momentum contributes a small additive term controlled by
    category_weight config (default 0 = no effect).
    
    NOTE: This function uses the LEGACY vol_regime_penalty. For P6 regime curves,
    use compute_conviction_score_with_regime() instead.
    
    Returns:
        Float in [0.0, 1.0]
    """
    if cfg is None:
        cfg = ConvictionConfig()
    
    # Normalize inputs to [0, 1]
    hybrid = _clamp(ctx.hybrid_score, 0.0, 1.0)
    expectancy = _clamp(ctx.expectancy_alpha, 0.0, 1.0)
    router_q = _clamp(ctx.router_quality, 0.0, 1.0)
    trend = _clamp(ctx.trend_strength, 0.0, 1.0)
    
    # Weighted combination
    raw = (
        0.40 * hybrid +
        0.25 * expectancy +
        0.20 * router_q +
        0.15 * trend
    )
    
    # Apply vol regime penalty (LEGACY)
    vol_penalty = _vol_regime_penalty(ctx.vol_regime)
    conviction = raw * vol_penalty
    
    # v7.7_P3: Add category momentum contribution (if enabled)
    # category_momentum is in [-1, 1], we scale it by category_weight
    # and add it as a bounded term
    if cfg.category_weight > 0 and ctx.category_momentum != 0:
        # Map [-1, 1] to [0, 1] for the contribution (centered at 0.5)
        cat_contrib = (ctx.category_momentum + 1.0) / 2.0  # [0, 1]
        cat_bias = cfg.category_weight * (cat_contrib - 0.5)  # [-weight/2, +weight/2]
        conviction += cat_bias
    
    return _clamp(conviction, 0.0, 1.0)


def compute_conviction_score_with_regime(
    ctx: ConvictionContext,
    cfg: Optional[ConvictionConfig] = None,
    symbol: Optional[str] = None,
) -> tuple[float, Dict[str, Any]]:
    """
    Compute conviction score with regime curve adjustments (v7.7_P6).
    
    This function applies regime-specific multipliers from the config,
    smooths the result using EMA, and returns both the score and
    the applied regime modifiers for transparency.
    
    NOTE: This uses _compute_raw_conviction_score (without legacy vol penalty)
    and applies P6 regime curves instead.
    
    Args:
        ctx: ConvictionContext with all input factors
        cfg: ConvictionConfig (uses defaults if None)
        symbol: Optional symbol for per-symbol smoothing cache
        
    Returns:
        Tuple of (conviction_score, regime_modifiers dict)
    """
    if cfg is None:
        cfg = ConvictionConfig()
    
    # Compute base conviction score WITHOUT legacy vol penalty
    # P6 regime curves will handle vol_regime and dd_state adjustments
    base_score = _compute_raw_conviction_score(ctx, cfg)
    
    # Get regime multipliers from config
    vol_mult = get_vol_regime_multiplier(ctx.vol_regime, cfg.regime_curves)
    dd_mult = get_dd_regime_multiplier(ctx.dd_state, cfg.regime_curves)
    
    # Apply regime multipliers
    adjusted_score = base_score * vol_mult * dd_mult
    
    # v7.8_P1: Apply meta-scheduler conviction strength overlay
    meta_strength = getattr(ctx, "meta_strength", 1.0)
    adjusted_score = adjusted_score * meta_strength
    
    # v7.8_P6: Apply Sentinel-X regime weight if available
    sentinel_x_weight = getattr(ctx, "sentinel_x_weight", 1.0)
    sentinel_x_regime = getattr(ctx, "sentinel_x_regime", "")
    if sentinel_x_weight != 1.0:
        adjusted_score = adjusted_score * sentinel_x_weight
    
    # v7.8_P7: Apply Alpha Decay conviction adjustment if available
    alpha_decay_adj = getattr(ctx, "alpha_decay_adjustment", 1.0)
    if alpha_decay_adj != 1.0:
        adjusted_score = adjusted_score * alpha_decay_adj
    
    # v7.8_P8: Apply Cerberus multi-strategy conviction multiplier if available
    cerberus_mult = getattr(ctx, "cerberus_conviction_multiplier", 1.0)
    if cerberus_mult != 1.0:
        adjusted_score = adjusted_score * cerberus_mult
    
    # Apply EMA smoothing if configured
    alpha = cfg.regime_curves.smoothing_alpha
    cache_key = symbol or "_global"
    prev_score = _prev_conviction_cache.get(cache_key, adjusted_score)
    smoothed_score = _ema_smooth(prev_score, adjusted_score, alpha)
    
    # Update cache
    _prev_conviction_cache[cache_key] = smoothed_score
    
    # Clamp final score
    final_score = _clamp(smoothed_score, 0.0, 1.0)
    
    # Build regime modifiers metadata
    regime_modifiers = {
        "vol_regime": ctx.vol_regime,
        "vol_multiplier": vol_mult,
        "dd_state": ctx.dd_state,
        "dd_multiplier": dd_mult,
        "combined_multiplier": vol_mult * dd_mult,
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "smoothed": alpha < 1.0,
        "smoothing_alpha": alpha,
        "meta_strength": meta_strength,  # v7.8_P1
        "sentinel_x_weight": sentinel_x_weight,  # v7.8_P6
        "sentinel_x_regime": sentinel_x_regime,  # v7.8_P6
        "alpha_decay_adjustment": alpha_decay_adj,  # v7.8_P7
        "cerberus_conviction_multiplier": cerberus_mult,  # v7.8_P8
    }
    
    return final_score, regime_modifiers


def get_conviction_band(score: float, cfg: ConvictionConfig) -> ConvictionBand:
    """
    Map conviction score to a band based on thresholds.
    
    Thresholds are checked from highest to lowest.
    """
    thresholds = cfg.thresholds
    
    if score >= thresholds.get("very_high", 0.92):
        return "very_high"
    if score >= thresholds.get("high", 0.80):
        return "high"
    if score >= thresholds.get("medium", 0.60):
        return "medium"
    if score >= thresholds.get("low", 0.40):
        return "low"
    return "very_low"


def compute_size_multiplier(
    score: float,
    band: ConvictionBand,
    ctx: ConvictionContext,
    cfg: ConvictionConfig,
) -> tuple[float, Optional[str]]:
    """
    Compute final size multiplier from conviction band and apply overrides.
    
    Returns:
        Tuple of (multiplier, veto_reason or None)
    """
    # Base multiplier from band
    base_mult = cfg.size_multipliers.get(band, 1.0)
    
    # Check router quality gate: veto if below min_for_any_trade
    min_for_any = cfg.router_quality_thresholds.get("min_for_any_trade", 0.50)
    if ctx.router_quality < min_for_any:
        return 0.0, f"router_quality={ctx.router_quality:.2f} < min_for_any_trade={min_for_any}"
    
    # Clamp multiplier if router quality below min_for_full_size
    min_for_full = cfg.router_quality_thresholds.get("min_for_full_size", 0.80)
    if ctx.router_quality < min_for_full:
        # Linear interpolation from min_for_any to min_for_full → [0.5, 1.0] of base
        ratio = (ctx.router_quality - min_for_any) / max(min_for_full - min_for_any, 0.01)
        router_scale = 0.5 + 0.5 * _clamp(ratio, 0.0, 1.0)
        base_mult *= router_scale
    
    # Apply DD state override
    dd_override = cfg.dd_overrides.get(ctx.dd_state)
    if dd_override is not None:
        base_mult = min(base_mult, dd_override)
    
    # Apply risk mode override
    risk_override = cfg.risk_mode_overrides.get(ctx.risk_mode)
    if risk_override is not None:
        if risk_override == 0.0:
            return 0.0, f"risk_mode={ctx.risk_mode} → size blocked"
        base_mult = min(base_mult, risk_override)
    
    return base_mult, None


def compute_conviction(
    ctx: ConvictionContext,
    cfg: Optional[ConvictionConfig] = None,
) -> ConvictionResult:
    """
    Main entry point: compute conviction score, band, and size multiplier.
    
    v7.7_P6: Now uses regime curves for vol/DD adjustments and includes
    regime_modifiers in the result for transparency.
    
    Args:
        ctx: ConvictionContext with all input factors
        cfg: ConvictionConfig (uses defaults if None)
    
    Returns:
        ConvictionResult with score, band, multiplier, and potential veto
    """
    if cfg is None:
        cfg = ConvictionConfig()
    
    # v7.7_P6: Compute conviction score with regime curve adjustments
    score, regime_modifiers = compute_conviction_score_with_regime(ctx, cfg)
    
    # Map to band
    band = get_conviction_band(score, cfg)
    
    # Compute size multiplier with overrides
    multiplier, veto_reason = compute_size_multiplier(score, band, ctx, cfg)
    
    return ConvictionResult(
        conviction_score=score,
        conviction_band=band,
        size_multiplier=multiplier,
        vetoed=(multiplier == 0.0),
        veto_reason=veto_reason,
        components={
            "hybrid_score": ctx.hybrid_score,
            "expectancy_alpha": ctx.expectancy_alpha,
            "router_quality": ctx.router_quality,
            "trend_strength": ctx.trend_strength,
            "category_momentum": ctx.category_momentum,  # v7.7_P3
            "vol_regime": ctx.vol_regime,
            "dd_state": ctx.dd_state,
            "risk_mode": ctx.risk_mode,
        },
        regime_modifiers=regime_modifiers,  # v7.7_P6
    )


# ---------------------------------------------------------------------------
# Sizing Integration Helper
# ---------------------------------------------------------------------------

def apply_conviction_to_nav_pct(
    base_nav_pct: float,
    conviction_result: ConvictionResult,
    min_nav_pct: float,
    max_nav_pct: float,
) -> float:
    """
    Apply conviction multiplier to base NAV percentage and clamp to bounds.
    
    Args:
        base_nav_pct: Base per-trade NAV percentage
        conviction_result: Result from compute_conviction()
        min_nav_pct: Minimum allowed NAV pct
        max_nav_pct: Maximum allowed NAV pct
    
    Returns:
        Final NAV percentage, clamped to [min, max], or 0.0 if vetoed
    """
    if conviction_result.vetoed:
        return 0.0
    
    scaled = base_nav_pct * conviction_result.size_multiplier
    return _clamp(scaled, min_nav_pct, max_nav_pct)


# ---------------------------------------------------------------------------
# v7.8_P1: Meta-Scheduler Integration Helper
# ---------------------------------------------------------------------------


def get_meta_conviction_strength() -> float:
    """
    Get meta-scheduler conviction strength overlay.
    
    Returns 1.0 (neutral) if meta-scheduler is disabled or unavailable.
    
    v7.8_P1: Helper for loading meta_strength from meta_scheduler.
    
    Returns:
        Conviction strength multiplier from meta-scheduler
    """
    try:
        from execution.meta_scheduler import (
            load_meta_scheduler_config,
            load_meta_scheduler_state,
            get_conviction_meta_strength,
            is_meta_scheduler_active,
        )
        
        cfg = load_meta_scheduler_config(None)
        if not cfg.enabled:
            return 1.0
        
        state = load_meta_scheduler_state()
        if not is_meta_scheduler_active(cfg, state):
            return 1.0
        
        return get_conviction_meta_strength(state)
    except ImportError:
        return 1.0
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# v7.8_P6: Sentinel-X Integration Helper
# ---------------------------------------------------------------------------


def get_sentinel_x_conviction_weight() -> tuple[float, str]:
    """
    Get Sentinel-X conviction weight overlay.
    
    Returns (1.0, "") (neutral) if Sentinel-X is disabled or unavailable.
    
    v7.8_P6: Helper for loading conviction weight from Sentinel-X.
    
    Returns:
        Tuple of (weight, primary_regime) from Sentinel-X
    """
    try:
        from execution.sentinel_x import (
            load_sentinel_x_config,
            load_sentinel_x_state,
            get_regime_conviction_weight,
        )
        
        cfg = load_sentinel_x_config(None)
        if not cfg.enabled:
            return 1.0, ""
        
        state = load_sentinel_x_state()
        if not state.primary_regime:
            return 1.0, ""
        
        weight = get_regime_conviction_weight(state.primary_regime)
        return weight, state.primary_regime
    except ImportError:
        return 1.0, ""
    except Exception:
        return 1.0, ""


__all__ = [
    # Types
    "DDState",
    "RiskMode",
    "VolRegimeLabel",
    "ConvictionBand",
    # Config
    "ConvictionConfig",
    "RegimeCurvesConfig",  # v7.7_P6
    "load_conviction_config",
    "load_regime_curves_config",  # v7.7_P6
    # Context / Result
    "ConvictionContext",
    "ConvictionResult",
    # State loaders
    "load_risk_snapshot",
    "load_router_health",
    "get_global_router_quality",
    "get_dd_state",
    "get_risk_mode",
    # Computation
    "compute_conviction_score",
    "compute_conviction_score_with_regime",  # v7.7_P6
    "get_vol_regime_multiplier",  # v7.7_P6
    "get_dd_regime_multiplier",  # v7.7_P6
    "get_conviction_band",
    "compute_size_multiplier",
    "compute_conviction",
    "apply_conviction_to_nav_pct",
    # Cache management
    "clear_conviction_cache",  # v7.7_P6
    # v7.8_P1: Meta-scheduler
    "get_meta_conviction_strength",
    # v7.8_P6: Sentinel-X
    "get_sentinel_x_conviction_weight",
]
