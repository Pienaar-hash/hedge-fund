"""
v7.8_P7 — Alpha Decay & Survival Curves (Thanatos).

The Mortality Model: Answers "How long does alpha last before it dies?"

This module computes survival probabilities, half-lives, and decay metrics for:
- Individual symbols
- Categories
- Factors
- Overall alpha quality

It is a RESEARCH / GOVERNANCE layer — NOT a trading strategy.
Outputs are used to bias Universe Optimizer, Alpha Router, Factor Diagnostics,
and Conviction Engine when enabled via config.

Key concepts:
- Decay Rate: Slope of log(edge_scores) over time — negative = decaying alpha
- Half-Life: Time for alpha to decay to 50% — ln(2) / |decay_rate|
- Survival Probability: exp(-time / half_life) — probability alpha is still alive
- Deterioration Probability: 1 - survival_prob — probability alpha is dying

State file: logs/state/alpha_decay.json
Single writer: Only executor/intel pipeline may write this file.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ALPHA_DECAY_PATH = Path("logs/state/alpha_decay.json")
DEFAULT_EDGE_INSIGHTS_PATH = Path("logs/state/edge_insights.json")
DEFAULT_FACTOR_DIAGNOSTICS_PATH = Path("logs/state/factor_diagnostics.json")
DEFAULT_SYMBOL_SCORES_PATH = Path("logs/state/symbol_scores_v6.json")
DEFAULT_HYBRID_SCORES_PATH = Path("logs/state/hybrid_scores.json")
DEFAULT_SENTINEL_X_PATH = Path("logs/state/sentinel_x.json")

# Canonical factors for decay tracking
CANONICAL_FACTORS = ["trend", "momentum", "value", "mean_revert", "carry", "quality"]

# EPS for numerical stability
EPS = 1e-10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlphaDecayConfig:
    """Configuration for Alpha Decay Model."""

    enabled: bool = False
    lookback_days: int = 120
    min_samples: int = 30
    smoothing_alpha: float = 0.15
    symbol_half_life_floor: int = 5
    symbol_half_life_ceiling: int = 90
    category_half_life_floor: int = 10
    factor_decay_floor: float = 0.95
    decay_penalty_strength: float = 0.20
    sentinel_x_integration: bool = True
    # Sentinel-X crisis/choppy acceleration
    crisis_half_life_reduction: float = 0.30  # 30% reduction in CRISIS
    choppy_half_life_reduction: float = 0.15  # 15% reduction in CHOPPY
    # Integration hooks
    universe_optimizer_enabled: bool = True
    alpha_router_enabled: bool = True
    factor_diagnostics_enabled: bool = True
    conviction_enabled: bool = True
    # Thresholds
    high_deterioration_threshold: float = 0.60  # Above this = weak alpha
    low_survival_threshold: float = 0.40  # Below this = dying alpha

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlphaDecayConfig":
        """Create config from dictionary."""
        return cls(
            enabled=d.get("enabled", False),
            lookback_days=d.get("lookback_days", 120),
            min_samples=d.get("min_samples", 30),
            smoothing_alpha=d.get("smoothing_alpha", 0.15),
            symbol_half_life_floor=d.get("symbol_half_life_floor", 5),
            symbol_half_life_ceiling=d.get("symbol_half_life_ceiling", 90),
            category_half_life_floor=d.get("category_half_life_floor", 10),
            factor_decay_floor=d.get("factor_decay_floor", 0.95),
            decay_penalty_strength=d.get("decay_penalty_strength", 0.20),
            sentinel_x_integration=d.get("sentinel_x_integration", True),
            crisis_half_life_reduction=d.get("crisis_half_life_reduction", 0.30),
            choppy_half_life_reduction=d.get("choppy_half_life_reduction", 0.15),
            universe_optimizer_enabled=d.get("universe_optimizer_enabled", True),
            alpha_router_enabled=d.get("alpha_router_enabled", True),
            factor_diagnostics_enabled=d.get("factor_diagnostics_enabled", True),
            conviction_enabled=d.get("conviction_enabled", True),
            high_deterioration_threshold=d.get("high_deterioration_threshold", 0.60),
            low_survival_threshold=d.get("low_survival_threshold", 0.40),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "lookback_days": self.lookback_days,
            "min_samples": self.min_samples,
            "smoothing_alpha": self.smoothing_alpha,
            "symbol_half_life_floor": self.symbol_half_life_floor,
            "symbol_half_life_ceiling": self.symbol_half_life_ceiling,
            "category_half_life_floor": self.category_half_life_floor,
            "factor_decay_floor": self.factor_decay_floor,
            "decay_penalty_strength": self.decay_penalty_strength,
            "sentinel_x_integration": self.sentinel_x_integration,
            "crisis_half_life_reduction": self.crisis_half_life_reduction,
            "choppy_half_life_reduction": self.choppy_half_life_reduction,
            "universe_optimizer_enabled": self.universe_optimizer_enabled,
            "alpha_router_enabled": self.alpha_router_enabled,
            "factor_diagnostics_enabled": self.factor_diagnostics_enabled,
            "conviction_enabled": self.conviction_enabled,
            "high_deterioration_threshold": self.high_deterioration_threshold,
            "low_survival_threshold": self.low_survival_threshold,
        }


# ---------------------------------------------------------------------------
# Symbol Decay Stats
# ---------------------------------------------------------------------------


@dataclass
class SymbolDecayStats:
    """Decay statistics for a single symbol."""

    symbol: str
    decay_rate: float  # Slope of log(edge_scores) — negative = decaying
    half_life: float  # Days until alpha halves
    survival_prob: float  # P(alpha still alive)
    deterioration_prob: float  # 1 - survival_prob
    ema_edge_score: float  # Smoothed edge score
    # Additional context
    sample_count: int = 0
    last_edge_score: float = 0.0
    trend_direction: str = "stable"  # "improving", "stable", "declining"
    days_since_peak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "decay_rate": round(self.decay_rate, 6),
            "half_life": round(self.half_life, 2),
            "survival_prob": round(self.survival_prob, 4),
            "deterioration_prob": round(self.deterioration_prob, 4),
            "ema_edge_score": round(self.ema_edge_score, 4),
            "sample_count": self.sample_count,
            "last_edge_score": round(self.last_edge_score, 4),
            "trend_direction": self.trend_direction,
            "days_since_peak": self.days_since_peak,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolDecayStats":
        """Create from dictionary."""
        return cls(
            symbol=d.get("symbol", ""),
            decay_rate=d.get("decay_rate", 0.0),
            half_life=d.get("half_life", 30.0),
            survival_prob=d.get("survival_prob", 0.5),
            deterioration_prob=d.get("deterioration_prob", 0.5),
            ema_edge_score=d.get("ema_edge_score", 0.0),
            sample_count=d.get("sample_count", 0),
            last_edge_score=d.get("last_edge_score", 0.0),
            trend_direction=d.get("trend_direction", "stable"),
            days_since_peak=d.get("days_since_peak", 0),
        )


# ---------------------------------------------------------------------------
# Category Decay Stats
# ---------------------------------------------------------------------------


@dataclass
class CategoryDecayStats:
    """Decay statistics for a category (L1_ALT, MEME, etc.)."""

    category: str
    decay_rate: float
    half_life: float
    survival_prob: float
    deterioration_prob: float
    # Additional context
    symbol_count: int = 0
    avg_symbol_survival: float = 0.5
    weakest_symbol: str = ""
    strongest_symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "decay_rate": round(self.decay_rate, 6),
            "half_life": round(self.half_life, 2),
            "survival_prob": round(self.survival_prob, 4),
            "deterioration_prob": round(self.deterioration_prob, 4),
            "symbol_count": self.symbol_count,
            "avg_symbol_survival": round(self.avg_symbol_survival, 4),
            "weakest_symbol": self.weakest_symbol,
            "strongest_symbol": self.strongest_symbol,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CategoryDecayStats":
        """Create from dictionary."""
        return cls(
            category=d.get("category", ""),
            decay_rate=d.get("decay_rate", 0.0),
            half_life=d.get("half_life", 30.0),
            survival_prob=d.get("survival_prob", 0.5),
            deterioration_prob=d.get("deterioration_prob", 0.5),
            symbol_count=d.get("symbol_count", 0),
            avg_symbol_survival=d.get("avg_symbol_survival", 0.5),
            weakest_symbol=d.get("weakest_symbol", ""),
            strongest_symbol=d.get("strongest_symbol", ""),
        )


# ---------------------------------------------------------------------------
# Factor Decay Stats
# ---------------------------------------------------------------------------


@dataclass
class FactorDecayStats:
    """Decay statistics for a factor (trend, momentum, etc.)."""

    factor: str
    decay_rate: float
    survival_prob: float
    adjusted_factor_weight_multiplier: float  # Applied to factor weights
    # Additional context
    pnl_contribution: float = 0.0
    ir_rolling: float = 0.0
    trend_direction: str = "stable"
    days_positive: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor": self.factor,
            "decay_rate": round(self.decay_rate, 6),
            "survival_prob": round(self.survival_prob, 4),
            "adjusted_factor_weight_multiplier": round(
                self.adjusted_factor_weight_multiplier, 4
            ),
            "pnl_contribution": round(self.pnl_contribution, 4),
            "ir_rolling": round(self.ir_rolling, 4),
            "trend_direction": self.trend_direction,
            "days_positive": self.days_positive,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FactorDecayStats":
        """Create from dictionary."""
        return cls(
            factor=d.get("factor", ""),
            decay_rate=d.get("decay_rate", 0.0),
            survival_prob=d.get("survival_prob", 0.5),
            adjusted_factor_weight_multiplier=d.get(
                "adjusted_factor_weight_multiplier", 1.0
            ),
            pnl_contribution=d.get("pnl_contribution", 0.0),
            ir_rolling=d.get("ir_rolling", 0.0),
            trend_direction=d.get("trend_direction", "stable"),
            days_positive=d.get("days_positive", 0),
        )


# ---------------------------------------------------------------------------
# Alpha Decay State (Main State Object)
# ---------------------------------------------------------------------------


@dataclass
class AlphaDecayState:
    """Complete alpha decay state snapshot."""

    updated_ts: str = ""
    cycle_count: int = 0
    symbols: Dict[str, SymbolDecayStats] = field(default_factory=dict)
    categories: Dict[str, CategoryDecayStats] = field(default_factory=dict)
    factors: Dict[str, FactorDecayStats] = field(default_factory=dict)
    # Aggregate metrics
    avg_symbol_survival: float = 0.5
    avg_category_survival: float = 0.5
    avg_factor_survival: float = 0.5
    overall_alpha_health: float = 0.5  # Composite score
    # Summary lists
    weakest_symbols: List[str] = field(default_factory=list)
    strongest_symbols: List[str] = field(default_factory=list)
    weakest_categories: List[str] = field(default_factory=list)
    weakest_factors: List[str] = field(default_factory=list)
    # Meta information
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "cycle_count": self.cycle_count,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "factors": {k: v.to_dict() for k, v in self.factors.items()},
            "avg_symbol_survival": round(self.avg_symbol_survival, 4),
            "avg_category_survival": round(self.avg_category_survival, 4),
            "avg_factor_survival": round(self.avg_factor_survival, 4),
            "overall_alpha_health": round(self.overall_alpha_health, 4),
            "weakest_symbols": self.weakest_symbols,
            "strongest_symbols": self.strongest_symbols,
            "weakest_categories": self.weakest_categories,
            "weakest_factors": self.weakest_factors,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlphaDecayState":
        """Create from dictionary."""
        symbols = {}
        for k, v in d.get("symbols", {}).items():
            symbols[k] = SymbolDecayStats.from_dict(v)

        categories = {}
        for k, v in d.get("categories", {}).items():
            categories[k] = CategoryDecayStats.from_dict(v)

        factors = {}
        for k, v in d.get("factors", {}).items():
            factors[k] = FactorDecayStats.from_dict(v)

        return cls(
            updated_ts=d.get("updated_ts", ""),
            cycle_count=d.get("cycle_count", 0),
            symbols=symbols,
            categories=categories,
            factors=factors,
            avg_symbol_survival=d.get("avg_symbol_survival", 0.5),
            avg_category_survival=d.get("avg_category_survival", 0.5),
            avg_factor_survival=d.get("avg_factor_survival", 0.5),
            overall_alpha_health=d.get("overall_alpha_health", 0.5),
            weakest_symbols=d.get("weakest_symbols", []),
            strongest_symbols=d.get("strongest_symbols", []),
            weakest_categories=d.get("weakest_categories", []),
            weakest_factors=d.get("weakest_factors", []),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Alpha Decay History (for time-series tracking)
# ---------------------------------------------------------------------------


@dataclass
class AlphaDecayHistory:
    """Historical edge scores for decay estimation."""

    symbol_edges: Dict[str, List[float]] = field(default_factory=dict)
    symbol_timestamps: Dict[str, List[float]] = field(default_factory=dict)
    category_edges: Dict[str, List[float]] = field(default_factory=dict)
    factor_pnl: Dict[str, List[float]] = field(default_factory=dict)
    factor_timestamps: Dict[str, List[float]] = field(default_factory=dict)

    def add_symbol_edge(
        self, symbol: str, edge_score: float, timestamp: float
    ) -> None:
        """Add a symbol edge score observation."""
        if symbol not in self.symbol_edges:
            self.symbol_edges[symbol] = []
            self.symbol_timestamps[symbol] = []
        self.symbol_edges[symbol].append(edge_score)
        self.symbol_timestamps[symbol].append(timestamp)

    def add_factor_pnl(
        self, factor: str, pnl_contrib: float, timestamp: float
    ) -> None:
        """Add a factor PnL contribution observation."""
        if factor not in self.factor_pnl:
            self.factor_pnl[factor] = []
            self.factor_timestamps[factor] = []
        self.factor_pnl[factor].append(pnl_contrib)
        self.factor_timestamps[factor].append(timestamp)

    def get_symbol_series(
        self, symbol: str, max_samples: int = 1000
    ) -> Tuple[List[float], List[float]]:
        """Get edge score and timestamp series for a symbol."""
        edges = self.symbol_edges.get(symbol, [])[-max_samples:]
        times = self.symbol_timestamps.get(symbol, [])[-max_samples:]
        return edges, times

    def get_factor_series(
        self, factor: str, max_samples: int = 1000
    ) -> Tuple[List[float], List[float]]:
        """Get PnL contribution and timestamp series for a factor."""
        pnl = self.factor_pnl.get(factor, [])[-max_samples:]
        times = self.factor_timestamps.get(factor, [])[-max_samples:]
        return pnl, times

    def prune_old_samples(self, max_age_seconds: float) -> None:
        """Remove samples older than max_age_seconds."""
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - max_age_seconds

        for symbol in list(self.symbol_edges.keys()):
            edges = self.symbol_edges[symbol]
            times = self.symbol_timestamps[symbol]
            # Keep only samples after cutoff
            valid = [(e, t) for e, t in zip(edges, times) if t >= cutoff]
            if valid:
                self.symbol_edges[symbol] = [e for e, _ in valid]
                self.symbol_timestamps[symbol] = [t for _, t in valid]
            else:
                del self.symbol_edges[symbol]
                del self.symbol_timestamps[symbol]

        for factor in list(self.factor_pnl.keys()):
            pnl = self.factor_pnl[factor]
            times = self.factor_timestamps[factor]
            valid = [(p, t) for p, t in zip(pnl, times) if t >= cutoff]
            if valid:
                self.factor_pnl[factor] = [p for p, _ in valid]
                self.factor_timestamps[factor] = [t for _, t in valid]
            else:
                del self.factor_pnl[factor]
                del self.factor_timestamps[factor]


# ---------------------------------------------------------------------------
# Decay Estimation Functions
# ---------------------------------------------------------------------------


def compute_decay_rate(
    values: List[float], timestamps: List[float], min_samples: int = 10
) -> float:
    """
    Compute decay rate from time series using log-linear regression.

    Returns:
        Decay rate (negative = decaying, positive = improving)
        Returns 0.0 if insufficient samples or numerical issues.
    """
    if len(values) < min_samples:
        return 0.0

    # Filter out non-positive values for log
    valid_pairs = [
        (v, t) for v, t in zip(values, timestamps) if v > EPS
    ]
    if len(valid_pairs) < min_samples:
        return 0.0

    vals, times = zip(*valid_pairs)
    log_vals = np.log(np.array(vals) + EPS)
    times_arr = np.array(times)

    # Normalize times to days from first observation
    times_days = (times_arr - times_arr[0]) / 86400.0

    if times_days[-1] - times_days[0] < 1.0:
        # Less than a day of data
        return 0.0

    try:
        # Linear regression: log(y) = a + b*t
        # decay_rate = b (negative = decay)
        coeffs = np.polyfit(times_days, log_vals, 1)
        decay_rate = float(coeffs[0])  # Slope

        # Clamp to reasonable range
        decay_rate = max(-1.0, min(1.0, decay_rate))
        return decay_rate
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_half_life(
    decay_rate: float, floor: int = 5, ceiling: int = 90
) -> float:
    """
    Compute half-life from decay rate.

    half_life = ln(2) / |decay_rate|

    Clamped to [floor, ceiling] days.
    """
    if abs(decay_rate) < EPS:
        return float(ceiling)

    half_life = math.log(2) / abs(decay_rate)

    # Clamp to bounds
    half_life = max(float(floor), min(float(ceiling), half_life))
    return half_life


def compute_survival_probability(
    time_elapsed: float, half_life: float
) -> float:
    """
    Compute survival probability.

    survival_prob = exp(-time / half_life)

    Args:
        time_elapsed: Time since alpha started (days)
        half_life: Half-life in days

    Returns:
        Probability in [0, 1]
    """
    if half_life <= EPS:
        return 0.0

    prob = math.exp(-time_elapsed / half_life)
    return max(0.0, min(1.0, prob))


def apply_sentinel_x_acceleration(
    half_life: float,
    regime: str,
    config: AlphaDecayConfig,
) -> float:
    """
    Apply Sentinel-X regime-based half-life acceleration.

    In CRISIS/CHOPPY regimes, decay accelerates (half-life reduces).
    """
    if not config.sentinel_x_integration:
        return half_life

    if regime == "CRISIS":
        reduction = config.crisis_half_life_reduction
        return half_life * (1.0 - reduction)
    elif regime == "CHOPPY":
        reduction = config.choppy_half_life_reduction
        return half_life * (1.0 - reduction)
    else:
        return half_life


def ema_smooth(
    current: float, previous: float, alpha: float
) -> float:
    """Exponential moving average smoothing."""
    return alpha * current + (1.0 - alpha) * previous


def classify_trend_direction(
    decay_rate: float, threshold: float = 0.001
) -> str:
    """Classify trend direction based on decay rate."""
    if decay_rate > threshold:
        return "improving"
    elif decay_rate < -threshold:
        return "declining"
    else:
        return "stable"


# ---------------------------------------------------------------------------
# Factor Decay Weight Multiplier
# ---------------------------------------------------------------------------


def compute_factor_weight_multiplier(
    decay_rate: float,
    survival_prob: float,
    config: AlphaDecayConfig,
) -> float:
    """
    Compute adjusted factor weight multiplier based on decay.

    multiplier = max(floor, 1 - penalty_strength * deterioration_prob)
    """
    deterioration_prob = 1.0 - survival_prob
    multiplier = 1.0 - config.decay_penalty_strength * deterioration_prob
    return max(config.factor_decay_floor, min(1.0, multiplier))


# ---------------------------------------------------------------------------
# Symbol Decay Computation
# ---------------------------------------------------------------------------


def compute_symbol_decay_stats(
    symbol: str,
    edge_scores: List[float],
    timestamps: List[float],
    config: AlphaDecayConfig,
    regime: str = "UNKNOWN",
    prev_ema_edge: float = 0.0,
) -> SymbolDecayStats:
    """
    Compute decay statistics for a single symbol.

    Args:
        symbol: Symbol name
        edge_scores: Historical edge scores
        timestamps: Corresponding timestamps
        config: Alpha decay config
        regime: Current Sentinel-X regime
        prev_ema_edge: Previous EMA edge score for smoothing

    Returns:
        SymbolDecayStats with computed metrics
    """
    if len(edge_scores) < config.min_samples:
        # Insufficient data — return neutral stats
        return SymbolDecayStats(
            symbol=symbol,
            decay_rate=0.0,
            half_life=float(config.symbol_half_life_ceiling),
            survival_prob=0.5,
            deterioration_prob=0.5,
            ema_edge_score=prev_ema_edge,
            sample_count=len(edge_scores),
            last_edge_score=edge_scores[-1] if edge_scores else 0.0,
            trend_direction="stable",
            days_since_peak=0,
        )

    # Compute decay rate
    decay_rate = compute_decay_rate(
        edge_scores, timestamps, min_samples=config.min_samples
    )

    # Compute half-life
    half_life = compute_half_life(
        decay_rate,
        floor=config.symbol_half_life_floor,
        ceiling=config.symbol_half_life_ceiling,
    )

    # Apply Sentinel-X acceleration
    half_life = apply_sentinel_x_acceleration(half_life, regime, config)

    # Compute survival probability
    # Use time since first observation
    time_elapsed_days = (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) > 1 else 0.0
    survival_prob = compute_survival_probability(time_elapsed_days, half_life)

    # EMA edge score
    last_edge = edge_scores[-1] if edge_scores else 0.0
    ema_edge = ema_smooth(last_edge, prev_ema_edge, config.smoothing_alpha)

    # Find days since peak
    peak_idx = int(np.argmax(edge_scores))
    peak_time = timestamps[peak_idx]
    days_since_peak = int((timestamps[-1] - peak_time) / 86400.0)

    return SymbolDecayStats(
        symbol=symbol,
        decay_rate=decay_rate,
        half_life=half_life,
        survival_prob=survival_prob,
        deterioration_prob=1.0 - survival_prob,
        ema_edge_score=ema_edge,
        sample_count=len(edge_scores),
        last_edge_score=last_edge,
        trend_direction=classify_trend_direction(decay_rate),
        days_since_peak=days_since_peak,
    )


# ---------------------------------------------------------------------------
# Category Decay Computation
# ---------------------------------------------------------------------------


def compute_category_decay_stats(
    category: str,
    symbol_stats: Dict[str, SymbolDecayStats],
    category_symbols: List[str],
    config: AlphaDecayConfig,
) -> CategoryDecayStats:
    """
    Compute decay statistics for a category by aggregating symbol stats.

    Args:
        category: Category name
        symbol_stats: All computed symbol decay stats
        category_symbols: Symbols in this category
        config: Alpha decay config

    Returns:
        CategoryDecayStats with aggregated metrics
    """
    # Get stats for symbols in this category
    relevant_stats = [
        symbol_stats[s] for s in category_symbols if s in symbol_stats
    ]

    if not relevant_stats:
        return CategoryDecayStats(
            category=category,
            decay_rate=0.0,
            half_life=float(config.category_half_life_floor),
            survival_prob=0.5,
            deterioration_prob=0.5,
            symbol_count=0,
            avg_symbol_survival=0.5,
        )

    # Aggregate decay rates
    decay_rates = [s.decay_rate for s in relevant_stats]
    avg_decay_rate = float(np.mean(decay_rates))

    # Compute category half-life
    half_life = compute_half_life(
        avg_decay_rate, floor=config.category_half_life_floor, ceiling=180
    )

    # Aggregate survival probabilities
    survival_probs = [s.survival_prob for s in relevant_stats]
    avg_survival = float(np.mean(survival_probs))

    # Find weakest and strongest symbols
    sorted_by_survival = sorted(relevant_stats, key=lambda x: x.survival_prob)
    weakest = sorted_by_survival[0].symbol if sorted_by_survival else ""
    strongest = sorted_by_survival[-1].symbol if sorted_by_survival else ""

    return CategoryDecayStats(
        category=category,
        decay_rate=avg_decay_rate,
        half_life=half_life,
        survival_prob=avg_survival,
        deterioration_prob=1.0 - avg_survival,
        symbol_count=len(relevant_stats),
        avg_symbol_survival=avg_survival,
        weakest_symbol=weakest,
        strongest_symbol=strongest,
    )


# ---------------------------------------------------------------------------
# Factor Decay Computation
# ---------------------------------------------------------------------------


def compute_factor_decay_stats(
    factor: str,
    pnl_contributions: List[float],
    timestamps: List[float],
    config: AlphaDecayConfig,
    ir_rolling: float = 0.0,
) -> FactorDecayStats:
    """
    Compute decay statistics for a factor.

    Args:
        factor: Factor name
        pnl_contributions: Historical PnL contributions
        timestamps: Corresponding timestamps
        config: Alpha decay config
        ir_rolling: Rolling information ratio

    Returns:
        FactorDecayStats with computed metrics
    """
    if len(pnl_contributions) < config.min_samples:
        return FactorDecayStats(
            factor=factor,
            decay_rate=0.0,
            survival_prob=0.5,
            adjusted_factor_weight_multiplier=1.0,
            pnl_contribution=pnl_contributions[-1] if pnl_contributions else 0.0,
            ir_rolling=ir_rolling,
            trend_direction="stable",
            days_positive=0,
        )

    # For factors, use cumulative PnL for decay estimation
    # Shift to positive range for log
    shifted = [p - min(pnl_contributions) + 1.0 for p in pnl_contributions]

    decay_rate = compute_decay_rate(
        shifted, timestamps, min_samples=config.min_samples
    )

    # Time elapsed
    time_elapsed = (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) > 1 else 0.0

    # Half-life (use longer bounds for factors)
    half_life = compute_half_life(decay_rate, floor=10, ceiling=180)
    survival_prob = compute_survival_probability(time_elapsed, half_life)

    # Compute weight multiplier
    multiplier = compute_factor_weight_multiplier(decay_rate, survival_prob, config)

    # Count positive PnL days
    days_positive = sum(1 for p in pnl_contributions if p > 0)

    return FactorDecayStats(
        factor=factor,
        decay_rate=decay_rate,
        survival_prob=survival_prob,
        adjusted_factor_weight_multiplier=multiplier,
        pnl_contribution=pnl_contributions[-1] if pnl_contributions else 0.0,
        ir_rolling=ir_rolling,
        trend_direction=classify_trend_direction(decay_rate),
        days_positive=days_positive,
    )


# ---------------------------------------------------------------------------
# State File I/O
# ---------------------------------------------------------------------------


def load_alpha_decay_state(
    path: Path | str = DEFAULT_ALPHA_DECAY_PATH,
) -> Optional[AlphaDecayState]:
    """Load alpha decay state from file."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return AlphaDecayState.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def save_alpha_decay_state(
    state: AlphaDecayState,
    path: Path | str = DEFAULT_ALPHA_DECAY_PATH,
) -> bool:
    """Save alpha decay state to file."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2))
        return True
    except IOError:
        return False


def load_config_from_strategy(
    strategy_config_path: Path | str = Path("config/strategy_config.json"),
) -> AlphaDecayConfig:
    """Load alpha decay config from strategy_config.json."""
    path = Path(strategy_config_path)
    if not path.exists():
        return AlphaDecayConfig()
    try:
        data = json.loads(path.read_text())
        decay_config = data.get("alpha_decay", {})
        return AlphaDecayConfig.from_dict(decay_config)
    except (json.JSONDecodeError, IOError):
        return AlphaDecayConfig()


# ---------------------------------------------------------------------------
# Input Loaders (Read from existing state surfaces)
# ---------------------------------------------------------------------------


def load_edge_insights(
    path: Path | str = DEFAULT_EDGE_INSIGHTS_PATH,
) -> Dict[str, Any]:
    """Load edge insights state."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_factor_diagnostics(
    path: Path | str = DEFAULT_FACTOR_DIAGNOSTICS_PATH,
) -> Dict[str, Any]:
    """Load factor diagnostics state."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_hybrid_scores(
    path: Path | str = DEFAULT_HYBRID_SCORES_PATH,
) -> Dict[str, Any]:
    """Load hybrid scores state."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_sentinel_x_state(
    path: Path | str = DEFAULT_SENTINEL_X_PATH,
) -> Dict[str, Any]:
    """Load Sentinel-X state."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_symbol_scores(
    path: Path | str = Path("logs/state/symbol_scores_v6.json"),
) -> Dict[str, Any]:
    """Load symbol scores state."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Main Step Function
# ---------------------------------------------------------------------------


def run_alpha_decay_step(
    config: AlphaDecayConfig,
    history: AlphaDecayHistory,
    prev_state: Optional[AlphaDecayState] = None,
    edge_insights: Optional[Dict[str, Any]] = None,
    factor_diagnostics: Optional[Dict[str, Any]] = None,
    hybrid_scores: Optional[Dict[str, Any]] = None,
    sentinel_x_state: Optional[Dict[str, Any]] = None,
    symbol_to_category: Optional[Dict[str, str]] = None,
) -> AlphaDecayState:
    """
    Run one step of the Alpha Decay Model.

    This function:
    1. Reads current edge scores from intel surfaces
    2. Updates history with new observations
    3. Computes decay stats for symbols, categories, factors
    4. Returns new AlphaDecayState

    Args:
        config: Alpha decay configuration
        history: Historical edge score data
        prev_state: Previous state for EMA smoothing
        edge_insights: Current edge insights
        factor_diagnostics: Current factor diagnostics
        hybrid_scores: Current hybrid scores
        sentinel_x_state: Current Sentinel-X state
        symbol_to_category: Symbol to category mapping

    Returns:
        New AlphaDecayState
    """
    now = datetime.now(timezone.utc)
    timestamp = now.timestamp()

    # Get current regime from Sentinel-X
    regime = "UNKNOWN"
    if sentinel_x_state:
        regime = sentinel_x_state.get("primary_regime", "UNKNOWN")

    # Default symbol to category mapping
    if symbol_to_category is None:
        symbol_to_category = {}

    # Previous state for EMA
    prev_symbol_ema = {}
    if prev_state:
        for sym, stats in prev_state.symbols.items():
            prev_symbol_ema[sym] = stats.ema_edge_score

    # -------------------------------------------------------------------
    # 1. Extract symbol edge scores from hybrid scores / edge insights
    # -------------------------------------------------------------------
    symbol_edges_current: Dict[str, float] = {}

    # From hybrid_scores
    if hybrid_scores:
        scores = hybrid_scores.get("scores", {})
        for sym, data in scores.items():
            if isinstance(data, dict):
                score = data.get("hybrid_score", 0.0)
            else:
                score = float(data) if data else 0.0
            symbol_edges_current[sym] = score

    # From edge_insights symbol_edges
    if edge_insights:
        symbol_data = edge_insights.get("symbol_edges", {})
        for sym, data in symbol_data.items():
            if isinstance(data, dict):
                score = data.get("hybrid_score", data.get("score", 0.0))
                if sym not in symbol_edges_current:
                    symbol_edges_current[sym] = score

    # Update history with current observations
    for sym, edge_score in symbol_edges_current.items():
        history.add_symbol_edge(sym, edge_score, timestamp)

    # -------------------------------------------------------------------
    # 2. Extract factor PnL contributions
    # -------------------------------------------------------------------
    if factor_diagnostics:
        factors_data = factor_diagnostics.get("factors", {})
        for factor, data in factors_data.items():
            if isinstance(data, dict):
                pnl_contrib = data.get("pnl_contribution", 0.0)
                history.add_factor_pnl(factor, pnl_contrib, timestamp)

    # -------------------------------------------------------------------
    # 3. Compute symbol decay stats
    # -------------------------------------------------------------------
    symbol_stats: Dict[str, SymbolDecayStats] = {}

    for sym in history.symbol_edges.keys():
        edges, times = history.get_symbol_series(sym)
        prev_ema = prev_symbol_ema.get(sym, 0.0)

        stats = compute_symbol_decay_stats(
            symbol=sym,
            edge_scores=edges,
            timestamps=times,
            config=config,
            regime=regime,
            prev_ema_edge=prev_ema,
        )
        symbol_stats[sym] = stats

    # -------------------------------------------------------------------
    # 4. Compute category decay stats
    # -------------------------------------------------------------------
    # Group symbols by category
    categories: Dict[str, List[str]] = {}
    for sym, cat in symbol_to_category.items():
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(sym)

    # Also infer categories from edge_insights
    if edge_insights:
        cat_edges = edge_insights.get("category_edges", {})
        for cat in cat_edges.keys():
            if cat not in categories:
                categories[cat] = []

    category_stats: Dict[str, CategoryDecayStats] = {}
    for cat, syms in categories.items():
        stats = compute_category_decay_stats(cat, symbol_stats, syms, config)
        category_stats[cat] = stats

    # -------------------------------------------------------------------
    # 5. Compute factor decay stats
    # -------------------------------------------------------------------
    factor_stats: Dict[str, FactorDecayStats] = {}

    # Get IR from factor diagnostics
    factor_ir: Dict[str, float] = {}
    if factor_diagnostics:
        factors_data = factor_diagnostics.get("factors", {})
        for factor, data in factors_data.items():
            if isinstance(data, dict):
                factor_ir[factor] = data.get("ir_rolling", 0.0)

    for factor in history.factor_pnl.keys():
        pnl, times = history.get_factor_series(factor)
        ir = factor_ir.get(factor, 0.0)

        stats = compute_factor_decay_stats(
            factor=factor,
            pnl_contributions=pnl,
            timestamps=times,
            config=config,
            ir_rolling=ir,
        )
        factor_stats[factor] = stats

    # Also include canonical factors with neutral stats if not tracked
    for factor in CANONICAL_FACTORS:
        if factor not in factor_stats:
            factor_stats[factor] = FactorDecayStats(
                factor=factor,
                decay_rate=0.0,
                survival_prob=0.5,
                adjusted_factor_weight_multiplier=1.0,
            )

    # -------------------------------------------------------------------
    # 6. Compute aggregate metrics
    # -------------------------------------------------------------------
    avg_symbol_survival = 0.5
    if symbol_stats:
        avg_symbol_survival = float(
            np.mean([s.survival_prob for s in symbol_stats.values()])
        )

    avg_category_survival = 0.5
    if category_stats:
        avg_category_survival = float(
            np.mean([c.survival_prob for c in category_stats.values()])
        )

    avg_factor_survival = 0.5
    if factor_stats:
        avg_factor_survival = float(
            np.mean([f.survival_prob for f in factor_stats.values()])
        )

    # Overall alpha health = weighted average
    overall_alpha_health = (
        0.5 * avg_symbol_survival +
        0.3 * avg_category_survival +
        0.2 * avg_factor_survival
    )

    # -------------------------------------------------------------------
    # 7. Build summary lists
    # -------------------------------------------------------------------
    # Weakest/strongest symbols (by survival prob)
    sorted_symbols = sorted(
        symbol_stats.values(), key=lambda x: x.survival_prob
    )
    weakest_symbols = [s.symbol for s in sorted_symbols[:5]]
    strongest_symbols = [s.symbol for s in sorted_symbols[-5:]][::-1]

    # Weakest categories
    sorted_categories = sorted(
        category_stats.values(), key=lambda x: x.survival_prob
    )
    weakest_categories = [c.category for c in sorted_categories[:3]]

    # Weakest factors
    sorted_factors = sorted(
        factor_stats.values(), key=lambda x: x.survival_prob
    )
    weakest_factors = [f.factor for f in sorted_factors[:3]]

    # -------------------------------------------------------------------
    # 8. Build state
    # -------------------------------------------------------------------
    cycle_count = (prev_state.cycle_count + 1) if prev_state else 1

    state = AlphaDecayState(
        updated_ts=now.isoformat(),
        cycle_count=cycle_count,
        symbols=symbol_stats,
        categories=category_stats,
        factors=factor_stats,
        avg_symbol_survival=avg_symbol_survival,
        avg_category_survival=avg_category_survival,
        avg_factor_survival=avg_factor_survival,
        overall_alpha_health=overall_alpha_health,
        weakest_symbols=weakest_symbols,
        strongest_symbols=strongest_symbols,
        weakest_categories=weakest_categories,
        weakest_factors=weakest_factors,
        meta={
            "sentinel_primary_regime": regime,
            "universe_size": len(symbol_stats),
            "category_count": len(category_stats),
            "factor_count": len(factor_stats),
            "history_symbol_count": len(history.symbol_edges),
            "config_enabled": config.enabled,
        },
    )

    return state


# ---------------------------------------------------------------------------
# Integration Helpers (for downstream consumers)
# ---------------------------------------------------------------------------


def get_symbol_decay_penalty(
    symbol: str,
    decay_state: Optional[AlphaDecayState],
    config: AlphaDecayConfig,
) -> float:
    """
    Get decay penalty multiplier for a symbol.

    Used by Universe Optimizer to bias down decaying symbols.

    Returns:
        Multiplier in [0, 1] where 1 = no penalty
    """
    if decay_state is None or not config.enabled:
        return 1.0

    if symbol not in decay_state.symbols:
        return 1.0

    stats = decay_state.symbols[symbol]
    deterioration = stats.deterioration_prob

    # penalty = 1 - penalty_strength * deterioration
    multiplier = 1.0 - config.decay_penalty_strength * deterioration
    return max(0.0, min(1.0, multiplier))


def get_alpha_router_adjustment(
    decay_state: Optional[AlphaDecayState],
    config: AlphaDecayConfig,
) -> float:
    """
    Get allocation adjustment based on overall alpha health.

    Used by Alpha Router to scale allocations.

    Returns:
        Multiplier typically in [0.8, 1.0]
    """
    if decay_state is None or not config.enabled:
        return 1.0

    avg_survival = decay_state.avg_symbol_survival

    # allocation *= (0.8 + 0.2 * avg_survival)
    return 0.8 + 0.2 * avg_survival


def get_factor_decay_multipliers(
    decay_state: Optional[AlphaDecayState],
    config: AlphaDecayConfig,
) -> Dict[str, float]:
    """
    Get factor weight multipliers based on decay.

    Used by Factor Diagnostics to adjust factor weights.

    Returns:
        Dict of factor -> multiplier
    """
    if decay_state is None or not config.enabled:
        return {}

    multipliers = {}
    for factor, stats in decay_state.factors.items():
        multipliers[factor] = stats.adjusted_factor_weight_multiplier

    return multipliers


def get_conviction_decay_adjustment(
    decay_state: Optional[AlphaDecayState],
    config: AlphaDecayConfig,
) -> float:
    """
    Get conviction adjustment based on overall alpha decay.

    Used by Conviction Engine to scale conviction.

    Returns:
        Multiplier typically in [0.85, 1.0]
    """
    if decay_state is None or not config.enabled:
        return 1.0

    health = decay_state.overall_alpha_health

    # Scale conviction: 0.85 + 0.15 * health
    return 0.85 + 0.15 * health


def get_alpha_decay_summary(
    decay_state: Optional[AlphaDecayState],
) -> Dict[str, Any]:
    """
    Get summary for EdgeInsights integration.

    Returns dict with weakest/strongest lists.
    """
    if decay_state is None:
        return {}

    return {
        "weakest_symbols": decay_state.weakest_symbols,
        "strongest_symbols": decay_state.strongest_symbols,
        "weakest_categories": decay_state.weakest_categories,
        "weakest_factors": decay_state.weakest_factors,
        "overall_alpha_health": decay_state.overall_alpha_health,
        "avg_symbol_survival": decay_state.avg_symbol_survival,
    }


# ---------------------------------------------------------------------------
# Full Run (for executor integration)
# ---------------------------------------------------------------------------


_HISTORY_CACHE: Optional[AlphaDecayHistory] = None


def get_or_create_history() -> AlphaDecayHistory:
    """Get or create history cache."""
    global _HISTORY_CACHE
    if _HISTORY_CACHE is None:
        _HISTORY_CACHE = AlphaDecayHistory()
    return _HISTORY_CACHE


def run_alpha_decay_full(
    symbol_to_category: Optional[Dict[str, str]] = None,
) -> Optional[AlphaDecayState]:
    """
    Full alpha decay step reading from all state surfaces.

    This is the main entry point for executor integration.

    Returns:
        AlphaDecayState if enabled, None otherwise
    """
    # Load config
    config = load_config_from_strategy()
    if not config.enabled:
        return None

    # Load inputs
    edge_insights = load_edge_insights()
    factor_diagnostics = load_factor_diagnostics()
    hybrid_scores = load_hybrid_scores()
    sentinel_x_state = load_sentinel_x_state()

    # Load previous state
    prev_state = load_alpha_decay_state()

    # Get history
    history = get_or_create_history()

    # Prune old samples (keep lookback_days worth)
    max_age = config.lookback_days * 86400
    history.prune_old_samples(max_age)

    # Run step
    state = run_alpha_decay_step(
        config=config,
        history=history,
        prev_state=prev_state,
        edge_insights=edge_insights,
        factor_diagnostics=factor_diagnostics,
        hybrid_scores=hybrid_scores,
        sentinel_x_state=sentinel_x_state,
        symbol_to_category=symbol_to_category,
    )

    # Save state
    save_alpha_decay_state(state)

    return state


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "AlphaDecayConfig",
    # Data classes
    "SymbolDecayStats",
    "CategoryDecayStats",
    "FactorDecayStats",
    "AlphaDecayState",
    "AlphaDecayHistory",
    # Core functions
    "compute_decay_rate",
    "compute_half_life",
    "compute_survival_probability",
    "apply_sentinel_x_acceleration",
    "ema_smooth",
    "classify_trend_direction",
    "compute_factor_weight_multiplier",
    "compute_symbol_decay_stats",
    "compute_category_decay_stats",
    "compute_factor_decay_stats",
    # State I/O
    "load_alpha_decay_state",
    "save_alpha_decay_state",
    "load_config_from_strategy",
    # Input loaders
    "load_edge_insights",
    "load_factor_diagnostics",
    "load_hybrid_scores",
    "load_sentinel_x_state",
    "load_symbol_scores",
    # Main step
    "run_alpha_decay_step",
    "run_alpha_decay_full",
    "get_or_create_history",
    # Integration helpers
    "get_symbol_decay_penalty",
    "get_alpha_router_adjustment",
    "get_factor_decay_multipliers",
    "get_conviction_decay_adjustment",
    "get_alpha_decay_summary",
]
