"""
v7.7_P3 — Category Momentum Factor & Sector Rotation.

Provides:
- Symbol → Category mapping
- Category-level performance metrics (returns, IR)
- Per-symbol category_momentum score for factor integration
- Sector rotation bias computation

The category_momentum factor integrates into:
- factor_diagnostics (as a factor in the factor vector)
- hybrid scoring (small weight contribution)
- conviction engine (optional category bias)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY = "OTHER"
DEFAULT_CATEGORIES_PATH = Path("config/symbol_categories.json")


@dataclass
class CategoryConfig:
    """Configuration for category momentum factor."""

    enabled: bool = True
    lookback_bars: int = 48  # Number of bars for momentum calculation
    half_life_bars: int = 24  # Exponential decay half-life
    ir_scale: float = 2.0  # Scale factor for IR → score mapping
    max_abs_score: float = 1.0  # Clamp for category_momentum score
    # For factor integration
    base_factor_weight: float = 0.08  # Default weight in factor composition


@dataclass
class CategoryStats:
    """Performance statistics for a category."""

    name: str
    symbols: List[str] = field(default_factory=list)
    mean_return: float = 0.0
    volatility: float = 0.0
    ir: float = 0.0  # Information ratio (mean_return / volatility)
    total_pnl: float = 0.0  # Cumulative PnL proxy
    momentum_score: float = 0.0  # Normalized score in [-1, 1]


@dataclass
class CategoryMomentumSnapshot:
    """Snapshot of category momentum state."""

    per_symbol: Dict[str, float] = field(default_factory=dict)  # symbol → category_momentum
    category_stats: Dict[str, CategoryStats] = field(default_factory=dict)
    symbol_categories: Dict[str, str] = field(default_factory=dict)  # symbol → category
    updated_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state publishing."""
        return {
            "updated_ts": self.updated_ts,
            "per_symbol": {sym: round(score, 4) for sym, score in self.per_symbol.items()},
            "category_stats": {
                cat: {
                    "symbols": stats.symbols,
                    "mean_return": round(stats.mean_return, 6),
                    "volatility": round(stats.volatility, 6),
                    "ir": round(stats.ir, 4),
                    "total_pnl": round(stats.total_pnl, 4),
                    "momentum_score": round(stats.momentum_score, 4),
                }
                for cat, stats in self.category_stats.items()
            },
            "symbol_categories": self.symbol_categories,
        }


# ---------------------------------------------------------------------------
# Config & Mapping Loading
# ---------------------------------------------------------------------------


def load_category_config(
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> CategoryConfig:
    """
    Load category momentum config from strategy_config.

    Args:
        strategy_config: Strategy config dict, or None to use defaults.

    Returns:
        CategoryConfig instance
    """
    if strategy_config is None:
        return CategoryConfig()

    cat_cfg = strategy_config.get("category_momentum", {})
    if not isinstance(cat_cfg, dict):
        return CategoryConfig()

    return CategoryConfig(
        enabled=bool(cat_cfg.get("enabled", True)),
        lookback_bars=int(cat_cfg.get("lookback_bars", 48)),
        half_life_bars=int(cat_cfg.get("half_life_bars", 24)),
        ir_scale=float(cat_cfg.get("ir_scale", 2.0)),
        max_abs_score=float(cat_cfg.get("max_abs_score", 1.0)),
        base_factor_weight=float(cat_cfg.get("base_factor_weight", 0.08)),
    )


def load_symbol_categories(
    path: Path | str = DEFAULT_CATEGORIES_PATH,
) -> Dict[str, str]:
    """
    Load symbol → category mapping from config file.

    Args:
        path: Path to symbol_categories.json

    Returns:
        Dict mapping symbol to category name
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            return {str(k).upper(): str(v) for k, v in categories.items()}
        return {}
    except (json.JSONDecodeError, IOError):
        return {}


def get_symbol_category(
    symbol: str,
    categories: Dict[str, str],
    default: str = DEFAULT_CATEGORY,
) -> str:
    """
    Get category for a symbol with fallback.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        categories: Symbol → category mapping
        default: Fallback category if not found

    Returns:
        Category name
    """
    return categories.get(symbol.upper(), default)


def get_symbols_by_category(
    categories: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Invert the symbol → category mapping to category → symbols.

    Args:
        categories: Symbol → category mapping

    Returns:
        Dict mapping category to list of symbols
    """
    result: Dict[str, List[str]] = {}
    for symbol, category in categories.items():
        if category not in result:
            result[category] = []
        result[category].append(symbol)
    return result


# ---------------------------------------------------------------------------
# Momentum Computation
# ---------------------------------------------------------------------------


def _compute_ewma_returns(
    returns: np.ndarray,
    half_life: int,
) -> float:
    """
    Compute exponentially weighted mean return.

    Args:
        returns: Array of returns
        half_life: Half-life in bars for exponential weighting

    Returns:
        EWMA of returns
    """
    if len(returns) == 0:
        return 0.0

    # Compute decay factor
    decay = 0.5 ** (1.0 / max(half_life, 1))
    n = len(returns)

    # Weights: most recent = highest weight
    weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
    weights = weights / weights.sum()

    return float(np.dot(returns, weights))


def compute_category_stats(
    symbol_returns: Dict[str, np.ndarray],
    categories: Dict[str, str],
    cfg: CategoryConfig,
) -> Dict[str, CategoryStats]:
    """
    Compute performance statistics for each category.

    Args:
        symbol_returns: Dict mapping symbol to return series (most recent last)
        categories: Symbol → category mapping
        cfg: Category momentum config

    Returns:
        Dict mapping category name to CategoryStats
    """
    # Group symbols by category
    cat_symbols = get_symbols_by_category(categories)

    # Also include symbols from returns that may not be in categories
    for sym in symbol_returns:
        cat = get_symbol_category(sym, categories)
        if cat not in cat_symbols:
            cat_symbols[cat] = []
        if sym not in cat_symbols[cat]:
            cat_symbols[cat].append(sym)

    result: Dict[str, CategoryStats] = {}

    for category, symbols in cat_symbols.items():
        # Collect returns for symbols in this category
        cat_returns: List[np.ndarray] = []
        for sym in symbols:
            if sym in symbol_returns and len(symbol_returns[sym]) > 0:
                cat_returns.append(symbol_returns[sym])

        if not cat_returns:
            # No data for this category
            result[category] = CategoryStats(name=category, symbols=symbols)
            continue

        # Stack and compute category-level metrics
        # Use equal-weighted average of symbol returns
        min_len = min(len(r) for r in cat_returns)
        if min_len == 0:
            result[category] = CategoryStats(name=category, symbols=symbols)
            continue

        # Truncate to common length and stack
        truncated = [r[-min_len:] for r in cat_returns]
        stacked = np.stack(truncated, axis=0)  # shape: (n_symbols, n_bars)

        # Category return = mean across symbols
        category_returns = stacked.mean(axis=0)  # shape: (n_bars,)

        # Compute EWMA mean return
        mean_ret = _compute_ewma_returns(category_returns, cfg.half_life_bars)

        # Compute volatility (std of returns)
        vol = float(np.std(category_returns)) if len(category_returns) > 1 else 0.0

        # Compute IR
        ir = mean_ret / (vol + 1e-9) if vol > 1e-9 else 0.0

        # Total PnL proxy (cumulative return)
        total_pnl = float(np.sum(category_returns))

        result[category] = CategoryStats(
            name=category,
            symbols=symbols,
            mean_return=mean_ret,
            volatility=vol,
            ir=ir,
            total_pnl=total_pnl,
            momentum_score=0.0,  # Will be normalized later
        )

    return result


def normalize_category_momentum(
    category_stats: Dict[str, CategoryStats],
    cfg: CategoryConfig,
) -> Dict[str, CategoryStats]:
    """
    Normalize category IR/returns to momentum scores in [-max_abs, +max_abs].

    Uses cross-sectional z-score of IR, then applies tanh for bounded output.

    Args:
        category_stats: Dict of CategoryStats with raw IR values
        cfg: Category momentum config

    Returns:
        Updated CategoryStats with normalized momentum_score
    """
    if not category_stats:
        return category_stats

    # Collect IR values
    irs = [stats.ir for stats in category_stats.values()]

    if len(irs) < 2:
        # Not enough categories for z-score; use raw IR scaled by ir_scale
        for stats in category_stats.values():
            raw_score = stats.ir * cfg.ir_scale
            stats.momentum_score = max(-cfg.max_abs_score, min(cfg.max_abs_score, np.tanh(raw_score)))
        return category_stats

    # Z-score normalization
    ir_array = np.array(irs)
    mean_ir = float(np.mean(ir_array))
    std_ir = float(np.std(ir_array))

    for stats in category_stats.values():
        if std_ir > 1e-9:
            z = (stats.ir - mean_ir) / std_ir
        else:
            z = 0.0

        # Scale and apply tanh for bounded output
        raw_score = z * cfg.ir_scale
        stats.momentum_score = float(max(-cfg.max_abs_score, min(cfg.max_abs_score, np.tanh(raw_score))))

    return category_stats


def compute_category_momentum(
    symbol_returns: Dict[str, np.ndarray],
    categories: Dict[str, str],
    cfg: Optional[CategoryConfig] = None,
) -> Tuple[Dict[str, float], Dict[str, CategoryStats]]:
    """
    Main entry point: compute category momentum for all symbols.

    Args:
        symbol_returns: Dict mapping symbol to return series (most recent last)
        categories: Symbol → category mapping
        cfg: Category momentum config (uses defaults if None)

    Returns:
        Tuple of:
        - per_symbol: Dict mapping symbol to category_momentum score
        - category_stats: Dict mapping category to CategoryStats
    """
    if cfg is None:
        cfg = CategoryConfig()

    if not cfg.enabled:
        # Return neutral scores
        per_symbol = {sym: 0.0 for sym in symbol_returns}
        return per_symbol, {}

    # Compute category stats
    category_stats = compute_category_stats(symbol_returns, categories, cfg)

    # Normalize momentum scores
    category_stats = normalize_category_momentum(category_stats, cfg)

    # v7.8_P1: Apply meta-scheduler category overlays
    category_stats = _apply_meta_category_overlays(category_stats, cfg)

    # Map category momentum back to symbols
    per_symbol: Dict[str, float] = {}
    for sym in symbol_returns:
        cat = get_symbol_category(sym, categories)
        if cat in category_stats:
            per_symbol[sym] = category_stats[cat].momentum_score
        else:
            per_symbol[sym] = 0.0

    # Also include symbols from categories that may not have returns
    for sym, cat in categories.items():
        if sym not in per_symbol:
            if cat in category_stats:
                per_symbol[sym] = category_stats[cat].momentum_score
            else:
                per_symbol[sym] = 0.0

    return per_symbol, category_stats


def _apply_meta_category_overlays(
    category_stats: Dict[str, CategoryStats],
    cfg: CategoryConfig,
) -> Dict[str, CategoryStats]:
    """
    Apply meta-scheduler category overlays to momentum scores.
    
    v7.8_P1: Slow-timescale category rotation overlays.
    
    Args:
        category_stats: Dict of CategoryStats with momentum scores
        cfg: Category momentum config
        
    Returns:
        Updated CategoryStats with overlays applied
    """
    try:
        from execution.meta_scheduler import (
            load_meta_scheduler_config,
            load_meta_scheduler_state,
            get_category_meta_overlays,
            is_meta_scheduler_active,
        )
        
        meta_cfg = load_meta_scheduler_config(None)
        if not meta_cfg.enabled:
            return category_stats
        
        meta_state = load_meta_scheduler_state()
        if not is_meta_scheduler_active(meta_cfg, meta_state):
            return category_stats
        
        overlays = get_category_meta_overlays(meta_state)
        if not overlays:
            return category_stats
        
        # Apply overlays to momentum scores
        for cat_name, stats in category_stats.items():
            overlay = overlays.get(cat_name, 1.0)
            # Apply overlay and re-clamp
            adjusted_momentum = stats.momentum_score * overlay
            stats.momentum_score = float(max(-cfg.max_abs_score, min(cfg.max_abs_score, adjusted_momentum)))
        
        return category_stats
    except ImportError:
        return category_stats
    except Exception:
        # Don't let meta-scheduler errors break category momentum
        return category_stats


def build_category_momentum_snapshot(
    symbol_returns: Dict[str, np.ndarray],
    categories: Optional[Dict[str, str]] = None,
    cfg: Optional[CategoryConfig] = None,
) -> CategoryMomentumSnapshot:
    """
    Build a complete category momentum snapshot.

    Args:
        symbol_returns: Dict mapping symbol to return series
        categories: Symbol → category mapping (loads from file if None)
        cfg: Category momentum config

    Returns:
        CategoryMomentumSnapshot with all data
    """
    import time

    if categories is None:
        categories = load_symbol_categories()

    if cfg is None:
        cfg = CategoryConfig()

    per_symbol, category_stats = compute_category_momentum(symbol_returns, categories, cfg)

    return CategoryMomentumSnapshot(
        per_symbol=per_symbol,
        category_stats=category_stats,
        symbol_categories=categories,
        updated_ts=time.time(),
    )


# ---------------------------------------------------------------------------
# Sector Rotation Helpers
# ---------------------------------------------------------------------------


def get_category_bias(
    symbol: str,
    category_stats: Dict[str, CategoryStats],
    categories: Dict[str, str],
    boost_threshold: float = 0.3,
    cut_threshold: float = -0.3,
) -> float:
    """
    Get sector rotation bias for a symbol based on its category momentum.

    Returns a value in [-1, 1] that can be used to adjust conviction/sizing.

    Args:
        symbol: Trading pair
        category_stats: Dict of CategoryStats with momentum scores
        categories: Symbol → category mapping
        boost_threshold: Momentum above this → positive bias
        cut_threshold: Momentum below this → negative bias

    Returns:
        Bias value in [-1, 1]
    """
    cat = get_symbol_category(symbol, categories)

    if cat not in category_stats:
        return 0.0

    momentum = category_stats[cat].momentum_score

    if momentum >= boost_threshold:
        # Scale boost: [threshold, 1.0] → [0, 1.0]
        return min(1.0, (momentum - boost_threshold) / (1.0 - boost_threshold + 1e-9))
    elif momentum <= cut_threshold:
        # Scale cut: [-1.0, threshold] → [-1.0, 0]
        return max(-1.0, (momentum - cut_threshold) / (abs(cut_threshold) + 1.0 + 1e-9))
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "CategoryConfig",
    "load_category_config",
    # Mapping
    "load_symbol_categories",
    "get_symbol_category",
    "get_symbols_by_category",
    # Data structures
    "CategoryStats",
    "CategoryMomentumSnapshot",
    # Computation
    "compute_category_stats",
    "normalize_category_momentum",
    "compute_category_momentum",
    "build_category_momentum_snapshot",
    # Rotation helpers
    "get_category_bias",
]
