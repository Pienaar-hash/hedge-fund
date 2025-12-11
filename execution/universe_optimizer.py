"""
v7.8_P3 — Universe Optimizer (Dynamic Symbol Universe Curator)

Codename: "Curator"

A dynamic symbol whitelist that expands/contracts based on:
- Symbol edge scores (P4)
- Category edges (P3 + P4)
- MetaScheduler trends (P8)
- Regime conditions (P6)
- Alpha Router allocation (P2)
- StrategyHealth (P7)

The optimizer produces:
1. A state surface: logs/state/universe_optimizer.json
2. A runtime artifact: allowed_universe (list of symbols)

The strategy uses this to filter trade candidates BEFORE scoring/sizing.

State Contract:
- Single writer: executor (via this module's write_universe_optimizer_state)
- Surface: logs/state/universe_optimizer.json
- Behaviour when disabled: identical to v7.8_P2 baseline (no filtering)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE_OPTIMIZER_PATH = Path("logs/state/universe_optimizer.json")

# Minimum universe size (safety)
ABSOLUTE_MIN_UNIVERSE = 3

# Default weights for scoring
DEFAULT_WEIGHTS = {
    "edge_score": 0.35,
    "category_score": 0.20,
    "meta_overlay": 0.15,
    "strategy_health": 0.15,
    "allocation_confidence": 0.15,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UniverseOptimizerConfig:
    """
    Configuration for the Universe Optimizer.
    
    Attributes:
        enabled: Whether optimizer is active (default: False for back-compat)
        min_universe_size: Minimum number of symbols to keep (safety floor)
        max_universe_size: Maximum universe size (ceiling)
        base_universe: List of symbols that are always included (core)
        volatility_regime_shrink: Whether to shrink universe on HIGH/CRISIS vol
        drawdown_shrink: Whether to shrink on DRAWDOWN state
        vol_shrink_factors: Multipliers for universe size per vol regime
        dd_shrink_factors: Multipliers for universe size per DD state
        category_diversification_min: Minimum categories to maintain
        score_weights: Weights for combining score components
        score_threshold: Minimum score to be included in universe
        meta_bias_strength: How much MetaScheduler influences scores
        health_bias_strength: How much StrategyHealth influences scores
        smoothing_alpha: EMA alpha for score smoothing (0 = no smoothing)
    """
    
    enabled: bool = False
    min_universe_size: int = 4
    max_universe_size: int = 20
    base_universe: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    volatility_regime_shrink: bool = True
    drawdown_shrink: bool = True
    vol_shrink_factors: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 1.0,
        "NORMAL": 1.0,
        "HIGH": 0.70,
        "CRISIS": 0.40,
    })
    dd_shrink_factors: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "RECOVERY": 0.85,
        "DRAWDOWN": 0.65,
    })
    category_diversification_min: int = 2
    score_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    score_threshold: float = 0.30
    meta_bias_strength: float = 0.10
    health_bias_strength: float = 0.15
    smoothing_alpha: float = 0.20
    # v7.8_P5: Cross-pair bias settings
    use_cross_pair_bias: bool = False
    cross_pair_bias_threshold: float = 0.50
    cross_pair_bias_max: float = 0.10


def load_universe_optimizer_config(
    strategy_cfg: Optional[Mapping[str, Any]] = None,
) -> UniverseOptimizerConfig:
    """
    Load UniverseOptimizerConfig from strategy_config.json.
    
    Expects a top-level "universe_optimizer" block:
    {
        "universe_optimizer": {
            "enabled": false,
            "min_universe_size": 4,
            ...
        }
    }
    
    Args:
        strategy_cfg: Strategy config dict, or None for defaults.
        
    Returns:
        UniverseOptimizerConfig instance
    """
    if strategy_cfg is None:
        return UniverseOptimizerConfig()
    
    uo_cfg = strategy_cfg.get("universe_optimizer", {})
    if not isinstance(uo_cfg, Mapping):
        return UniverseOptimizerConfig()
    
    # Build config with overrides
    return UniverseOptimizerConfig(
        enabled=bool(uo_cfg.get("enabled", False)),
        min_universe_size=int(uo_cfg.get("min_universe_size", 4)),
        max_universe_size=int(uo_cfg.get("max_universe_size", 20)),
        base_universe=list(uo_cfg.get("base_universe", ["BTCUSDT", "ETHUSDT"])),
        volatility_regime_shrink=bool(uo_cfg.get("volatility_regime_shrink", True)),
        drawdown_shrink=bool(uo_cfg.get("drawdown_shrink", True)),
        vol_shrink_factors=dict(uo_cfg.get("vol_shrink_factors", {
            "LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.70, "CRISIS": 0.40,
        })),
        dd_shrink_factors=dict(uo_cfg.get("dd_shrink_factors", {
            "NORMAL": 1.0, "RECOVERY": 0.85, "DRAWDOWN": 0.65,
        })),
        category_diversification_min=int(uo_cfg.get("category_diversification_min", 2)),
        score_weights=dict(uo_cfg.get("score_weights", DEFAULT_WEIGHTS)),
        score_threshold=float(uo_cfg.get("score_threshold", 0.30)),
        meta_bias_strength=float(uo_cfg.get("meta_bias_strength", 0.10)),
        health_bias_strength=float(uo_cfg.get("health_bias_strength", 0.15)),
        smoothing_alpha=float(uo_cfg.get("smoothing_alpha", 0.20)),
        # v7.8_P5: Cross-pair bias settings
        use_cross_pair_bias=bool(uo_cfg.get("use_cross_pair_bias", False)),
        cross_pair_bias_threshold=float(uo_cfg.get("cross_pair_bias_threshold", 0.50)),
        cross_pair_bias_max=float(uo_cfg.get("cross_pair_bias_max", 0.10)),
    )


# ---------------------------------------------------------------------------
# State Dataclass
# ---------------------------------------------------------------------------


@dataclass
class UniverseOptimizerState:
    """
    Universe Optimizer state snapshot.
    
    Attributes:
        updated_ts: ISO timestamp of last update
        allowed_symbols: List of symbols in the optimized universe
        symbol_scores: Dict of symbol → composite score (EMA smoothed)
        category_scores: Dict of category → aggregate score
        total_universe_size: Size of full candidate universe
        effective_max_size: Max size after regime adjustments
        notes: Explanatory notes for debugging
    """
    
    updated_ts: str = ""
    allowed_symbols: List[str] = field(default_factory=list)
    symbol_scores: Dict[str, float] = field(default_factory=dict)
    category_scores: Dict[str, float] = field(default_factory=dict)
    total_universe_size: int = 0
    effective_max_size: int = 0
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "allowed_symbols": list(self.allowed_symbols),
            "symbol_scores": {
                k: round(v, 4) for k, v in self.symbol_scores.items()
            },
            "category_scores": {
                k: round(v, 4) for k, v in self.category_scores.items()
            },
            "total_universe_size": self.total_universe_size,
            "effective_max_size": self.effective_max_size,
            "notes": list(self.notes),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UniverseOptimizerState":
        """Reconstruct from dictionary."""
        return cls(
            updated_ts=str(data.get("updated_ts", "")),
            allowed_symbols=list(data.get("allowed_symbols", [])),
            symbol_scores=dict(data.get("symbol_scores", {})),
            category_scores=dict(data.get("category_scores", {})),
            total_universe_size=int(data.get("total_universe_size", 0)),
            effective_max_size=int(data.get("effective_max_size", 0)),
            notes=list(data.get("notes", [])),
        )


def create_empty_state() -> UniverseOptimizerState:
    """
    Create an empty universe optimizer state.
    
    Returns:
        UniverseOptimizerState with no symbols
    """
    return UniverseOptimizerState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        allowed_symbols=[],
        symbol_scores={},
        category_scores={},
        total_universe_size=0,
        effective_max_size=0,
        notes=["Empty state - optimizer not yet run"],
    )


# ---------------------------------------------------------------------------
# State Loader & Writer
# ---------------------------------------------------------------------------


def load_universe_optimizer_state(
    path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> Optional[UniverseOptimizerState]:
    """
    Load universe optimizer state from file.
    
    Args:
        path: Path to universe_optimizer.json
        
    Returns:
        UniverseOptimizerState if file exists and is valid, else None
    """
    path = Path(path)
    if not path.exists():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return UniverseOptimizerState.from_dict(data)
    except Exception:
        return None


def write_universe_optimizer_state(
    state: UniverseOptimizerState,
    path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> None:
    """
    Write universe optimizer state to file (atomic write).
    
    This is the ONLY allowed writer for universe_optimizer.json.
    
    Args:
        state: UniverseOptimizerState to write
        path: Output path (default: logs/state/universe_optimizer.json)
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
        # Fail silently - universe optimizer state is non-critical
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


def _get_symbol_edge_score(
    symbol: str,
    symbol_edges: Dict[str, Any],
) -> float:
    """
    Get edge score for a symbol from symbol_edges.
    
    Args:
        symbol: Symbol string
        symbol_edges: Symbol edges dict from EdgeInsights
        
    Returns:
        Edge score ∈ [0, 1] or 0.5 if not found
    """
    edge_data = symbol_edges.get(symbol, {})
    if isinstance(edge_data, dict):
        return float(edge_data.get("edge_score", edge_data.get("hybrid_score", 0.5)))
    return 0.5


def _get_symbol_category(
    symbol: str,
    symbol_edges: Dict[str, Any],
) -> str:
    """
    Get category for a symbol.
    
    Args:
        symbol: Symbol string
        symbol_edges: Symbol edges dict
        
    Returns:
        Category string or "OTHER"
    """
    edge_data = symbol_edges.get(symbol, {})
    if isinstance(edge_data, dict):
        return str(edge_data.get("category", "OTHER"))
    return "OTHER"


def _get_category_score(
    category: str,
    category_edges: Dict[str, Any],
) -> float:
    """
    Get score for a category from category_edges.
    
    Args:
        category: Category string
        category_edges: Category edges dict from EdgeInsights
        
    Returns:
        Category score ∈ [0, 1] or 0.5 if not found
    """
    edge_data = category_edges.get(category, {})
    if isinstance(edge_data, dict):
        return float(edge_data.get("edge_score", edge_data.get("momentum_score", 0.5)))
    return 0.5


def _compute_effective_max_size(
    base_max: int,
    vol_regime: str,
    dd_state: str,
    cfg: UniverseOptimizerConfig,
    sentinel_x_state: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Compute effective max universe size after regime adjustments.
    
    Args:
        base_max: Base maximum from config
        vol_regime: Current volatility regime
        dd_state: Current drawdown state
        cfg: Universe optimizer config
        sentinel_x_state: Sentinel-X state dict (v7.8_P6)
        
    Returns:
        Adjusted max size
    """
    effective = float(base_max)
    
    # Apply volatility shrink
    if cfg.volatility_regime_shrink:
        vol_key = vol_regime.upper() if vol_regime else "NORMAL"
        vol_factor = cfg.vol_shrink_factors.get(vol_key, 1.0)
        effective *= vol_factor
    
    # Apply drawdown shrink
    if cfg.drawdown_shrink:
        dd_key = dd_state.upper() if dd_state else "NORMAL"
        dd_factor = cfg.dd_shrink_factors.get(dd_key, 1.0)
        effective *= dd_factor
    
    # v7.8_P6: Apply Sentinel-X regime shrink
    if sentinel_x_state is not None:
        primary_regime = sentinel_x_state.get("primary_regime", "")
        if primary_regime:
            # Shrink factors based on Sentinel-X regime
            sentinel_shrink_factors = {
                "TREND_UP": 1.0,      # No shrink
                "TREND_DOWN": 0.85,   # Slight shrink
                "MEAN_REVERT": 0.90,
                "BREAKOUT": 0.95,
                "CHOPPY": 0.75,       # More shrink in chop
                "CRISIS": 0.60,       # Aggressive shrink
            }
            sentinel_factor = sentinel_shrink_factors.get(primary_regime, 1.0)
            effective *= sentinel_factor
    
    # Ensure minimum
    return max(cfg.min_universe_size, int(effective))


def get_alpha_decay_symbol_penalty(
    symbol: str,
    alpha_decay_state: Optional[Dict[str, Any]],
) -> float:
    """
    Get alpha decay penalty for a symbol (v7.8_P7).
    
    Returns multiplier in [0, 1] where 1 = no penalty.
    """
    if alpha_decay_state is None:
        return 1.0
    
    symbols = alpha_decay_state.get("symbols", {})
    if symbol not in symbols:
        return 1.0
    
    sym_data = symbols[symbol]
    if isinstance(sym_data, dict):
        # Use deterioration_prob to compute penalty
        deterioration = sym_data.get("deterioration_prob", 0.0)
        # penalty_strength = 0.20 (from config default)
        penalty_strength = 0.20
        return max(0.0, 1.0 - penalty_strength * deterioration)
    
    return 1.0


def _get_cerberus_category_multiplier(
    category: str,
    cerberus_state: Optional[Dict[str, Any]],
) -> float:
    """
    Get Cerberus category multiplier for universe scoring (v7.8_P8).
    
    Uses CATEGORY head multiplier from Cerberus state.
    
    Returns multiplier typically in [0.5, 2.0] where 1.0 = neutral.
    """
    if cerberus_state is None:
        return 1.0
    
    head_state = cerberus_state.get("head_state", {})
    if not head_state:
        return 1.0
    
    heads = head_state.get("heads", {})
    category_head = heads.get("CATEGORY", {})
    
    if isinstance(category_head, dict):
        multiplier = category_head.get("multiplier", 1.0)
        # Clamp to reasonable range for scoring
        return max(0.5, min(2.0, multiplier))
    
    return 1.0


# ---------------------------------------------------------------------------
# Symbol Scoring
# ---------------------------------------------------------------------------


def compute_symbol_composite_score(
    symbol: str,
    symbol_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    meta_overlay: float,
    strategy_health: float,
    allocation_confidence: float,
    cfg: UniverseOptimizerConfig,
    alpha_decay_state: Optional[Dict[str, Any]] = None,
    cerberus_state: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute composite score for a symbol.
    
    Combines:
    - Symbol edge score
    - Category score
    - Meta overlay (from MetaScheduler)
    - Strategy health
    - Alpha Router allocation confidence
    
    Args:
        symbol: Symbol string
        symbol_edges: Symbol edges from EdgeInsights
        category_edges: Category edges from EdgeInsights
        meta_overlay: Meta-scheduler overlay value (default 1.0)
        strategy_health: StrategyHealth score
        allocation_confidence: Alpha Router allocation
        cfg: Universe optimizer config
        
    Returns:
        Composite score ∈ [0, 1]
    """
    weights = cfg.score_weights
    
    # Get component scores
    edge_score = _get_symbol_edge_score(symbol, symbol_edges)
    
    category = _get_symbol_category(symbol, symbol_edges)
    category_score = _get_category_score(category, category_edges)
    
    # Normalize meta_overlay and allocation_confidence to [0, 1] range
    # Meta overlay is typically ~1.0 with ±0.15 deviation
    meta_normalized = _clamp((meta_overlay - 0.5) * 2, 0.0, 1.0)
    
    # Allocation confidence is [0, 1] already
    alloc_normalized = _clamp(allocation_confidence, 0.0, 1.0)
    
    # Health is [0, 1] already
    health_normalized = _clamp(strategy_health, 0.0, 1.0)
    
    # Weighted combination
    w_edge = weights.get("edge_score", 0.35)
    w_cat = weights.get("category_score", 0.20)
    w_meta = weights.get("meta_overlay", 0.15)
    w_health = weights.get("strategy_health", 0.15)
    w_alloc = weights.get("allocation_confidence", 0.15)
    
    total_weight = w_edge + w_cat + w_meta + w_health + w_alloc
    if total_weight <= 0:
        return 0.5
    
    composite = (
        w_edge * edge_score
        + w_cat * category_score
        + w_meta * meta_normalized
        + w_health * health_normalized
        + w_alloc * alloc_normalized
    ) / total_weight
    
    # v7.8_P7: Apply alpha decay penalty
    decay_penalty = get_alpha_decay_symbol_penalty(symbol, alpha_decay_state)
    composite = composite * decay_penalty
    
    # v7.8_P8: Apply Cerberus category multiplier if available
    cerberus_mult = _get_cerberus_category_multiplier(category, cerberus_state)
    composite = composite * cerberus_mult
    
    return _clamp(composite, 0.0, 1.0)


def compute_all_symbol_scores(
    candidate_symbols: List[str],
    symbol_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    meta_overlay: float,
    strategy_health: float,
    allocation_confidence: float,
    prev_scores: Dict[str, float],
    cfg: UniverseOptimizerConfig,
    alpha_decay_state: Optional[Dict[str, Any]] = None,
    cerberus_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute and smooth scores for all candidate symbols.
    
    Args:
        candidate_symbols: List of all candidate symbols
        symbol_edges: Symbol edges from EdgeInsights
        category_edges: Category edges from EdgeInsights
        meta_overlay: Meta-scheduler overlay
        strategy_health: Strategy health score
        allocation_confidence: Alpha Router allocation
        prev_scores: Previous EMA scores
        cfg: Universe optimizer config
        alpha_decay_state: Alpha decay state for P7 decay penalties
        cerberus_state: Cerberus state for P8 category multipliers
        
    Returns:
        Dict of symbol → smoothed composite score
    """
    scores = {}
    
    # v7.8_P5: Load cross-pair state for bias if enabled
    cross_pair_boost_fn = None
    if cfg.use_cross_pair_bias:
        try:
            from execution.cross_pair_engine import get_pair_leg_boost, load_cross_pair_state
            cross_pair_state = load_cross_pair_state()
            
            def _cross_pair_boost(sym: str) -> float:
                return get_pair_leg_boost(
                    sym, 
                    cross_pair_state, 
                    cfg.cross_pair_bias_threshold, 
                    cfg.cross_pair_bias_max
                )
            
            cross_pair_boost_fn = _cross_pair_boost
        except ImportError:
            pass
    
    for symbol in candidate_symbols:
        # Compute raw score
        raw_score = compute_symbol_composite_score(
            symbol=symbol,
            symbol_edges=symbol_edges,
            category_edges=category_edges,
            meta_overlay=meta_overlay,
            strategy_health=strategy_health,
            allocation_confidence=allocation_confidence,
            cfg=cfg,
            alpha_decay_state=alpha_decay_state,  # v7.8_P7
            cerberus_state=cerberus_state,  # v7.8_P8
        )
        
        # v7.8_P5: Apply cross-pair bias boost if enabled
        if cross_pair_boost_fn is not None:
            boost = cross_pair_boost_fn(symbol)
            raw_score = _clamp(raw_score + boost, 0.0, 1.0)
        
        # Apply EMA smoothing
        prev = prev_scores.get(symbol, raw_score)
        smoothed = _ema_smooth(prev, raw_score, cfg.smoothing_alpha)
        scores[symbol] = smoothed
    
    return scores


def compute_category_aggregate_scores(
    symbol_scores: Dict[str, float],
    symbol_edges: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute aggregate scores per category.
    
    Args:
        symbol_scores: Dict of symbol → score
        symbol_edges: Symbol edges for category lookup
        
    Returns:
        Dict of category → average score
    """
    category_totals: Dict[str, List[float]] = {}
    
    for symbol, score in symbol_scores.items():
        category = _get_symbol_category(symbol, symbol_edges)
        if category not in category_totals:
            category_totals[category] = []
        category_totals[category].append(score)
    
    return {
        cat: sum(scores) / len(scores) if scores else 0.0
        for cat, scores in category_totals.items()
    }


# ---------------------------------------------------------------------------
# Universe Selection
# ---------------------------------------------------------------------------


def select_optimized_universe(
    symbol_scores: Dict[str, float],
    symbol_edges: Dict[str, Any],
    effective_max_size: int,
    cfg: UniverseOptimizerConfig,
) -> Tuple[List[str], List[str]]:
    """
    Select the optimized universe from scored symbols.
    
    Selection criteria:
    1. Always include base_universe symbols
    2. Rank remaining by score
    3. Take top N up to effective_max_size
    4. Ensure category diversification
    5. Filter by score threshold
    
    Args:
        symbol_scores: Dict of symbol → score
        symbol_edges: Symbol edges for category lookup
        effective_max_size: Max universe size after regime adjustments
        cfg: Universe optimizer config
        
    Returns:
        Tuple of (allowed_symbols, notes)
    """
    notes = []
    
    # Start with base universe (always included)
    selected: Set[str] = set(cfg.base_universe)
    notes.append(f"Base universe: {list(cfg.base_universe)}")
    
    # Get ranked symbols by score (excluding base)
    ranked = sorted(
        [(sym, score) for sym, score in symbol_scores.items() if sym not in selected],
        key=lambda x: x[1],
        reverse=True,
    )
    
    # Track categories for diversification
    category_counts: Dict[str, int] = {}
    for sym in selected:
        cat = _get_symbol_category(sym, symbol_edges)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Add symbols up to effective_max_size
    for sym, score in ranked:
        if len(selected) >= effective_max_size:
            break
        
        # Check score threshold
        if score < cfg.score_threshold:
            continue
        
        # Check category diversification
        cat = _get_symbol_category(sym, symbol_edges)
        cat_count = category_counts.get(cat, 0)
        
        # Prefer adding from underrepresented categories
        selected.add(sym)
        category_counts[cat] = cat_count + 1
    
    # Ensure minimum category diversification
    if len(category_counts) < cfg.category_diversification_min:
        notes.append(f"Category diversification below minimum ({len(category_counts)} < {cfg.category_diversification_min})")
    
    # Ensure minimum universe size
    if len(selected) < cfg.min_universe_size:
        # Add more from ranked list regardless of threshold
        for sym, score in ranked:
            if sym not in selected:
                selected.add(sym)
            if len(selected) >= cfg.min_universe_size:
                break
        notes.append(f"Universe padded to minimum size: {cfg.min_universe_size}")
    
    notes.append(f"Selected {len(selected)} symbols from {len(symbol_scores)} candidates")
    
    # Sort alphabetically for consistency
    return sorted(selected), notes


# ---------------------------------------------------------------------------
# Core Computation
# ---------------------------------------------------------------------------


def compute_optimized_universe(
    candidate_symbols: List[str],
    symbol_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    vol_regime: str,
    dd_state: str,
    meta_overlay: float,
    strategy_health: float,
    allocation_confidence: float,
    cfg: UniverseOptimizerConfig,
    prev_state: Optional[UniverseOptimizerState] = None,
    sentinel_x_state: Optional[Dict[str, Any]] = None,
) -> UniverseOptimizerState:
    """
    Compute the optimized universe.
    
    Args:
        candidate_symbols: Full list of candidate symbols
        symbol_edges: Symbol edges from EdgeInsights
        category_edges: Category edges from EdgeInsights
        vol_regime: Current volatility regime
        dd_state: Current drawdown state
        meta_overlay: Meta-scheduler overlay
        strategy_health: Strategy health score
        allocation_confidence: Alpha Router allocation
        cfg: Universe optimizer config
        prev_state: Previous state for EMA smoothing
        sentinel_x_state: Sentinel-X state dict (v7.8_P6)
        
    Returns:
        UniverseOptimizerState with computed universe
    """
    notes = []
    
    # Get previous scores for smoothing
    prev_scores = {}
    if prev_state is not None:
        prev_scores = dict(prev_state.symbol_scores)
    
    # Compute effective max size based on regime
    effective_max = _compute_effective_max_size(
        base_max=cfg.max_universe_size,
        vol_regime=vol_regime,
        dd_state=dd_state,
        cfg=cfg,
        sentinel_x_state=sentinel_x_state,
    )
    
    # Include sentinel regime in notes if available
    sentinel_regime = ""
    if sentinel_x_state is not None:
        sentinel_regime = sentinel_x_state.get("primary_regime", "")
    notes.append(f"Effective max size: {effective_max} (base: {cfg.max_universe_size}, vol: {vol_regime}, dd: {dd_state}, sentinel: {sentinel_regime})")
    
    # Compute symbol scores
    symbol_scores = compute_all_symbol_scores(
        candidate_symbols=candidate_symbols,
        symbol_edges=symbol_edges,
        category_edges=category_edges,
        meta_overlay=meta_overlay,
        strategy_health=strategy_health,
        allocation_confidence=allocation_confidence,
        prev_scores=prev_scores,
        cfg=cfg,
    )
    
    # Compute category scores
    category_scores = compute_category_aggregate_scores(
        symbol_scores=symbol_scores,
        symbol_edges=symbol_edges,
    )
    
    # Select optimized universe
    allowed_symbols, selection_notes = select_optimized_universe(
        symbol_scores=symbol_scores,
        symbol_edges=symbol_edges,
        effective_max_size=effective_max,
        cfg=cfg,
    )
    notes.extend(selection_notes)
    
    return UniverseOptimizerState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        allowed_symbols=allowed_symbols,
        symbol_scores=symbol_scores,
        category_scores=category_scores,
        total_universe_size=len(candidate_symbols),
        effective_max_size=effective_max,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Activation Check
# ---------------------------------------------------------------------------


def is_universe_optimizer_active(
    cfg: UniverseOptimizerConfig,
) -> bool:
    """
    Check if universe optimizer is active and should be applied.
    
    Args:
        cfg: UniverseOptimizerConfig
        
    Returns:
        True if optimizer should be applied
    """
    return cfg.enabled


# ---------------------------------------------------------------------------
# Public API - Views
# ---------------------------------------------------------------------------


def get_allowed_symbols(
    state: Optional[UniverseOptimizerState] = None,
    path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> Optional[List[str]]:
    """
    Get current allowed symbols from state.
    
    Returns None if:
    - State is None
    - State file doesn't exist
    - Any error occurs
    
    Use None to indicate "no filtering" (optimizer disabled or unavailable).
    
    Args:
        state: Optional pre-loaded state
        path: Path to state file if state not provided
        
    Returns:
        List of allowed symbols or None if unavailable
    """
    if state is not None:
        if state.allowed_symbols:
            return list(state.allowed_symbols)
        return None
    
    loaded = load_universe_optimizer_state(path)
    if loaded is None:
        return None
    
    if loaded.allowed_symbols:
        return list(loaded.allowed_symbols)
    return None


def get_symbol_score(
    symbol: str,
    state: Optional[UniverseOptimizerState] = None,
    path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> Optional[float]:
    """
    Get score for a specific symbol.
    
    Args:
        symbol: Symbol to look up
        state: Optional pre-loaded state
        path: Path to state file
        
    Returns:
        Symbol score or None if not found
    """
    if state is not None:
        return state.symbol_scores.get(symbol)
    
    loaded = load_universe_optimizer_state(path)
    if loaded is None:
        return None
    
    return loaded.symbol_scores.get(symbol)


def is_symbol_allowed(
    symbol: str,
    state: Optional[UniverseOptimizerState] = None,
    path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> bool:
    """
    Check if a symbol is in the allowed universe.
    
    Returns True if:
    - Optimizer is disabled (no state)
    - Symbol is in allowed_symbols
    
    Args:
        symbol: Symbol to check
        state: Optional pre-loaded state
        path: Path to state file
        
    Returns:
        True if symbol is allowed
    """
    allowed = get_allowed_symbols(state, path)
    
    # If no state, allow all (optimizer disabled)
    if allowed is None:
        return True
    
    return symbol in allowed


# ---------------------------------------------------------------------------
# Screener Integration
# ---------------------------------------------------------------------------


def filter_candidates_by_universe(
    candidates: List[str],
    cfg: Optional[UniverseOptimizerConfig] = None,
    state: Optional[UniverseOptimizerState] = None,
    min_size_fallback: int = ABSOLUTE_MIN_UNIVERSE,
) -> Tuple[List[str], bool]:
    """
    Filter candidate symbols by the optimized universe.
    
    Soft-fail logic:
    - If optimizer disabled → return all candidates, filtered=False
    - If universe size < min_size_fallback → return all candidates, filtered=False
    
    Args:
        candidates: List of candidate symbols
        cfg: Universe optimizer config (will load if None)
        state: Pre-loaded state (will load if None)
        min_size_fallback: Minimum universe size before fallback
        
    Returns:
        Tuple of (filtered_candidates, was_filtering_applied)
    """
    if cfg is None:
        cfg = load_universe_optimizer_config(None)
    
    if not cfg.enabled:
        return candidates, False
    
    if state is None:
        state = load_universe_optimizer_state()
    
    if state is None:
        return candidates, False
    
    allowed = state.allowed_symbols
    if not allowed or len(allowed) < min_size_fallback:
        # Fallback to full universe
        return candidates, False
    
    allowed_set = set(allowed)
    filtered = [sym for sym in candidates if sym in allowed_set]
    
    # Safety: if filtering results in too few candidates, fallback
    if len(filtered) < min_size_fallback:
        return candidates, False
    
    return filtered, True


# ---------------------------------------------------------------------------
# Orchestration - Run Universe Optimizer Step
# ---------------------------------------------------------------------------


def run_universe_optimizer_step(
    candidate_symbols: List[str],
    symbol_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    vol_regime: str,
    dd_state: str,
    meta_overlay: float = 1.0,
    strategy_health: float = 0.5,
    allocation_confidence: float = 1.0,
    strategy_cfg: Optional[Mapping[str, Any]] = None,
    state_path: Path | str = DEFAULT_UNIVERSE_OPTIMIZER_PATH,
) -> UniverseOptimizerState:
    """
    Run one universe optimizer computation step.
    
    This is the main entry point for the executor to call.
    
    Steps:
    1. Load config
    2. If disabled → return empty state, don't write
    3. Load previous state
    4. Compute optimized universe
    5. Write state
    6. Return state
    
    Args:
        candidate_symbols: Full list of candidate symbols
        symbol_edges: Symbol edges from EdgeInsights
        category_edges: Category edges from EdgeInsights
        vol_regime: Current volatility regime
        dd_state: Current drawdown state
        meta_overlay: Meta-scheduler overlay (default 1.0)
        strategy_health: Strategy health score (default 0.5)
        allocation_confidence: Alpha Router allocation (default 1.0)
        strategy_cfg: Strategy config dict or None
        state_path: Path to state file
        
    Returns:
        UniverseOptimizerState (empty if disabled)
    """
    cfg = load_universe_optimizer_config(strategy_cfg)
    
    if not cfg.enabled:
        # Return empty state without writing
        return create_empty_state()
    
    # Load previous state for EMA smoothing
    prev_state = load_universe_optimizer_state(state_path)
    
    # Compute optimized universe
    new_state = compute_optimized_universe(
        candidate_symbols=candidate_symbols,
        symbol_edges=symbol_edges,
        category_edges=category_edges,
        vol_regime=vol_regime,
        dd_state=dd_state,
        meta_overlay=meta_overlay,
        strategy_health=strategy_health,
        allocation_confidence=allocation_confidence,
        cfg=cfg,
        prev_state=prev_state,
    )
    
    # Write state
    write_universe_optimizer_state(new_state, state_path)
    
    return new_state


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "UniverseOptimizerConfig",
    "load_universe_optimizer_config",
    # State
    "UniverseOptimizerState",
    "create_empty_state",
    "load_universe_optimizer_state",
    "write_universe_optimizer_state",
    # Computation
    "compute_symbol_composite_score",
    "compute_all_symbol_scores",
    "compute_category_aggregate_scores",
    "select_optimized_universe",
    "compute_optimized_universe",
    "is_universe_optimizer_active",
    # Views
    "get_allowed_symbols",
    "get_symbol_score",
    "is_symbol_allowed",
    # Screener integration
    "filter_candidates_by_universe",
    # Orchestration
    "run_universe_optimizer_step",
    # Constants
    "DEFAULT_UNIVERSE_OPTIMIZER_PATH",
]
