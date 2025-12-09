"""
v7.5_C2/C3 — Factor Diagnostics Engine.

Provides:
- Factor vector normalization (z-score, minmax)
- Cross-factor covariance and correlation matrices
- Factor volatilities
- Gram-Schmidt orthogonalization (v7.5_C3)
- Auto-weighting based on vol/IR (v7.5_C3)
- Diagnostic snapshots for dashboard consumption

This module is analysis-only and does NOT affect trading decisions directly,
but factor weights can be consumed by hybrid_score when enabled.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from execution.intel.symbol_score_v6 import FactorVector, build_factor_vector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_FACTORS = [
    "trend",
    "carry",
    "rv_momentum",
    "router_quality",
    "expectancy",
    "vol_regime",
]


@dataclass
class FactorDiagnosticsConfig:
    """Configuration for factor diagnostics."""

    enabled: bool = True
    factors: List[str] = field(default_factory=lambda: DEFAULT_FACTORS.copy())
    normalization_mode: str = "zscore"  # "zscore" or "minmax"
    covariance_lookback_days: int = 30
    pnl_attribution_lookback_days: int = 14
    max_abs_zscore: float = 3.0


@dataclass
class OrthogonalizationConfig:
    """Configuration for factor orthogonalization (v7.5_C3)."""

    enabled: bool = True
    method: str = "gram_schmidt"  # "gram_schmidt" or "none"


@dataclass
class AutoWeightingConfig:
    """Configuration for factor auto-weighting (v7.5_C3)."""

    enabled: bool = True
    mode: str = "vol_inverse_ir"  # "equal", "vol_inverse", "ir_only", "vol_inverse_ir"
    min_weight: float = 0.05
    max_weight: float = 0.40
    normalize_to_one: bool = True
    smoothing_alpha: float = 0.2  # EWMA smoothing for weights


@dataclass
class FactorWeights:
    """Per-factor weights computed from IR/vol."""

    weights: Dict[str, float] = field(default_factory=dict)  # factor_name -> weight

    def to_dict(self) -> Dict[str, float]:
        """Convert to JSON-serializable dict."""
        return dict(self.weights)


@dataclass
class OrthogonalizedFactorVectors:
    """Orthogonalized factor values per symbol."""

    per_symbol: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # symbol -> factor_name -> ortho_value
    norms: Dict[str, float] = field(default_factory=dict)
    dot_products: Dict[str, Dict[str, float]] = field(default_factory=dict)
    degenerate: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "per_symbol": self.per_symbol,
            "norms": self.norms,
            "dot_products": self.dot_products,
            "degenerate": self.degenerate,
        }


@dataclass
class FactorWeightsSnapshot:
    """Complete factor weights snapshot for state publishing."""

    weights: Dict[str, float] = field(default_factory=dict)
    factor_vols: Dict[str, float] = field(default_factory=dict)
    factor_ir: Dict[str, float] = field(default_factory=dict)
    mode: str = "vol_inverse_ir"
    updated_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "weights": self.weights,
            "factor_vols": self.factor_vols,
            "factor_ir": self.factor_ir,
            "mode": self.mode,
            "updated_ts": self.updated_ts,
        }


def load_factor_diagnostics_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> FactorDiagnosticsConfig:
    """
    Load factor diagnostics config from strategy_config.

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        FactorDiagnosticsConfig instance
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return FactorDiagnosticsConfig()

    cfg = strategy_config.get("factor_diagnostics", {})
    if not isinstance(cfg, dict):
        return FactorDiagnosticsConfig()

    return FactorDiagnosticsConfig(
        enabled=bool(cfg.get("enabled", True)),
        factors=list(cfg.get("factors", DEFAULT_FACTORS)),
        normalization_mode=str(cfg.get("normalization_mode", "zscore")),
        covariance_lookback_days=int(cfg.get("covariance_lookback_days", 30)),
        pnl_attribution_lookback_days=int(cfg.get("pnl_attribution_lookback_days", 14)),
        max_abs_zscore=float(cfg.get("max_abs_zscore", 3.0)),
    )


def load_orthogonalization_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> OrthogonalizationConfig:
    """
    Load orthogonalization config from strategy_config.

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        OrthogonalizationConfig instance
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return OrthogonalizationConfig(enabled=False)

    fd_cfg = strategy_config.get("factor_diagnostics", {})
    if not isinstance(fd_cfg, dict):
        return OrthogonalizationConfig(enabled=False)

    ortho_cfg = fd_cfg.get("orthogonalization", {})
    if not isinstance(ortho_cfg, dict) or not ortho_cfg:
        # No orthogonalization config = disabled
        return OrthogonalizationConfig(enabled=False)

    return OrthogonalizationConfig(
        enabled=bool(ortho_cfg.get("enabled", True)),
        method=str(ortho_cfg.get("method", "gram_schmidt")),
    )


def load_auto_weighting_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> AutoWeightingConfig:
    """
    Load auto-weighting config from strategy_config.

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        AutoWeightingConfig instance
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return AutoWeightingConfig(enabled=False)

    fd_cfg = strategy_config.get("factor_diagnostics", {})
    if not isinstance(fd_cfg, dict):
        return AutoWeightingConfig(enabled=False)

    aw_cfg = fd_cfg.get("auto_weighting", {})
    if not isinstance(aw_cfg, dict) or not aw_cfg:
        # No auto_weighting config = disabled
        return AutoWeightingConfig(enabled=False)

    return AutoWeightingConfig(
        enabled=bool(aw_cfg.get("enabled", True)),
        mode=str(aw_cfg.get("mode", "vol_inverse_ir")),
        min_weight=float(aw_cfg.get("min_weight", 0.05)),
        max_weight=float(aw_cfg.get("max_weight", 0.40)),
        normalize_to_one=bool(aw_cfg.get("normalize_to_one", True)),
        smoothing_alpha=float(aw_cfg.get("smoothing_alpha", 0.2)),
    )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class NormalizedFactorVector:
    """Factor vector with normalized values."""

    symbol: str
    factors: Dict[str, float] = field(default_factory=dict)  # Normalized values
    direction: str = "LONG"


@dataclass
class FactorCovarianceSnapshot:
    """Covariance and correlation analysis of factors."""

    factors: List[str]
    covariance: np.ndarray  # shape (F, F)
    correlation: np.ndarray  # shape (F, F)
    factor_vols: Dict[str, float] = field(default_factory=dict)
    lookback_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "factors": self.factors,
            "covariance_matrix": self.covariance.tolist(),
            "correlation_matrix": self.correlation.tolist(),
            "factor_vols": self.factor_vols,
            "lookback_days": self.lookback_days,
        }


@dataclass
class FactorDiagnosticsSnapshot:
    """Complete factor diagnostics snapshot including orthogonalization and weights (v7.5_C3)."""

    per_symbol: Dict[str, NormalizedFactorVector] = field(default_factory=dict)
    covariance: Optional[FactorCovarianceSnapshot] = None
    orthogonalized: Optional[OrthogonalizedFactorVectors] = None  # v7.5_C3
    factor_weights: Optional[FactorWeightsSnapshot] = None  # v7.5_C3
    orthogonalization_enabled: bool = False  # v7.5_C3
    auto_weighting_enabled: bool = False  # v7.5_C3
    updated_ts: float = 0.0
    config: Optional[FactorDiagnosticsConfig] = None
    normalization_coeffs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_factors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    factor_ir: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    pnl_attribution: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "updated_ts": self.updated_ts,
            "raw_factors": self.raw_factors,
            "per_symbol": {
                sym: {
                    "factors": vec.factors,
                    "direction": vec.direction,
                }
                for sym, vec in self.per_symbol.items()
            },
            "normalization_coeffs": self.normalization_coeffs,
            "covariance": self.covariance.to_dict() if self.covariance else None,
            "orthogonalized": self.orthogonalized.to_dict() if self.orthogonalized else None,
            "factor_ir": self.factor_ir,
            "weights": self.weights,
            "factor_weights": self.factor_weights.to_dict()
            if self.factor_weights
            else {"weights": self.weights, "factor_ir": self.factor_ir},
            "orthogonalization_enabled": self.orthogonalization_enabled,
            "auto_weighting_enabled": self.auto_weighting_enabled,
            "pnl_attribution": self.pnl_attribution,
            "config": {
                "enabled": self.config.enabled,
                "factors": self.config.factors,
                "normalization_mode": self.config.normalization_mode,
                "max_abs_zscore": self.config.max_abs_zscore,
                "covariance_lookback_days": self.config.covariance_lookback_days,
                "pnl_attribution_lookback_days": self.config.pnl_attribution_lookback_days,
            }
            if self.config
            else None,
        }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_factor_vectors(
    vectors: List[FactorVector],
    factor_names: List[str],
    mode: str = "zscore",
    max_abs_zscore: float = 3.0,
    return_stats: bool = False,
) -> List[NormalizedFactorVector] | Tuple[List[NormalizedFactorVector], Dict[str, Dict[str, float]]]:
    """
    Normalize factor values per factor across symbols.

    Args:
        vectors: List of raw FactorVector objects
        factor_names: Factor names to normalize
        mode: "zscore" or "minmax"
        max_abs_zscore: Clamp z-scores to ±this value

    Returns:
        List of NormalizedFactorVector with normalized values, and optionally normalization coefficients
    """
    if not vectors:
        return ([], {}) if return_stats else []

    # Collect factor values across all symbols
    factor_values: Dict[str, List[float]] = {f: [] for f in factor_names}
    for vec in vectors:
        for factor in factor_names:
            val = vec.factors.get(factor, 0.0)
            factor_values[factor].append(float(val) if val is not None else 0.0)

    # Compute normalization stats per factor
    norm_stats: Dict[str, Dict[str, float]] = {}
    for factor in factor_names:
        vals = factor_values[factor]
        if not vals:
            norm_stats[factor] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
            continue

        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        norm_stats[factor] = {
            "mean": mean,
            "std": std if std > 1e-9 else 1.0,  # Avoid division by zero
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val if max_val != min_val else 1.0,
        }

    # Normalize each vector
    result: List[NormalizedFactorVector] = []
    for vec in vectors:
        normalized_factors: Dict[str, float] = {}

        for factor in factor_names:
            raw_val = float(vec.factors.get(factor, 0.0) or 0.0)
            stats = norm_stats[factor]

            if mode == "zscore":
                z = (raw_val - stats["mean"]) / stats["std"]
                # Clamp to ±max_abs_zscore
                normalized_factors[factor] = max(-max_abs_zscore, min(max_abs_zscore, z))
            elif mode == "minmax":
                # Scale to [0, 1]
                range_val = stats["range"]
                if range_val > 1e-9:
                    normalized_factors[factor] = (raw_val - stats["min"]) / range_val
                else:
                    normalized_factors[factor] = 0.0
            else:
                # Default: no normalization
                normalized_factors[factor] = raw_val

        result.append(
            NormalizedFactorVector(
                symbol=vec.symbol,
                factors=normalized_factors,
                direction=vec.direction,
            )
        )

    if return_stats:
        return result, norm_stats
    return result


# ---------------------------------------------------------------------------
# Covariance & Correlation
# ---------------------------------------------------------------------------


def compute_factor_covariance(
    vectors: List[FactorVector],
    factor_names: List[str],
    lookback_days: int = 0,
) -> FactorCovarianceSnapshot:
    """
    Build factor covariance and correlation matrices across symbols.

    Args:
        vectors: List of FactorVector objects
        factor_names: Factor names to include

    Returns:
        FactorCovarianceSnapshot with covariance and correlation matrices
    """
    n_symbols = len(vectors)
    n_factors = len(factor_names)

    if n_symbols < 2 or n_factors < 1:
        # Not enough data for covariance
        return FactorCovarianceSnapshot(
            factors=factor_names,
            covariance=np.zeros((n_factors, n_factors)),
            correlation=np.eye(n_factors),
            factor_vols={f: 0.0 for f in factor_names},
            lookback_days=lookback_days,
        )

    # Build matrix X of shape (N_symbols, F)
    X = np.zeros((n_symbols, n_factors))
    for i, vec in enumerate(vectors):
        for j, factor in enumerate(factor_names):
            X[i, j] = float(vec.factors.get(factor, 0.0) or 0.0)

    # Compute covariance matrix
    # Using np.cov with rowvar=False (columns are variables, rows are observations)
    cov_matrix = np.cov(X, rowvar=False)

    # Handle case where cov returns scalar (single factor)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[float(cov_matrix)]])
    elif cov_matrix.ndim == 1:
        cov_matrix = cov_matrix.reshape((n_factors, n_factors))

    # Compute factor volatilities (sqrt of diagonal)
    factor_vols: Dict[str, float] = {}
    for j, factor in enumerate(factor_names):
        var = cov_matrix[j, j]
        factor_vols[factor] = float(np.sqrt(max(0.0, var)))

    # Compute correlation matrix
    corr_matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(n_factors):
            vol_i = factor_vols[factor_names[i]]
            vol_j = factor_vols[factor_names[j]]
            if vol_i > 1e-9 and vol_j > 1e-9:
                corr_matrix[i, j] = cov_matrix[i, j] / (vol_i * vol_j)
            elif i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = 0.0

    return FactorCovarianceSnapshot(
        factors=factor_names,
        covariance=cov_matrix,
        correlation=corr_matrix,
        factor_vols=factor_vols,
        lookback_days=lookback_days,
    )


# ---------------------------------------------------------------------------
# Factor Orthogonalization (v7.5_C3)
# ---------------------------------------------------------------------------


def orthogonalize_factors(
    factor_vectors: List[FactorVector],
    factor_names: List[str],
) -> OrthogonalizedFactorVectors:
    """
    Applies Gram-Schmidt orthogonalization to factor columns across symbols.

    This removes redundancy between correlated factors. Think of:
    - For each factor f, we have a column vector X_f over symbols.
    - Combine all into matrix X of shape (N_symbols, F).
    - Apply Gram-Schmidt on columns to produce orthogonal columns Q.

    Args:
        factor_vectors: List of FactorVector objects
        factor_names: Factor names to orthogonalize

    Returns:
        OrthogonalizedFactorVectors with per-symbol orthogonalized values
    """
    n_symbols = len(factor_vectors)
    n_factors = len(factor_names)

    if n_symbols < 1 or n_factors < 1:
        return OrthogonalizedFactorVectors(per_symbol={})

    # Build matrix X with shape (N_symbols, F)
    X = np.zeros((n_symbols, n_factors))
    symbols: List[str] = []
    for i, vec in enumerate(factor_vectors):
        symbols.append(vec.symbol)
        for j, factor in enumerate(factor_names):
            X[i, j] = float(vec.factors.get(factor, 0.0) or 0.0)

    # Apply deterministic Gram-Schmidt orthogonalization on columns.
    # We keep the original scale for the leading factor and only remove
    # projections for later factors (no unit-norm normalization) so single
    # factors remain unchanged.
    Q = np.zeros_like(X, dtype=float)
    norms: Dict[str, float] = {}
    degenerate: List[str] = []
    for j in range(n_factors):
        v = X[:, j].astype(float).copy()
        # Subtract projections onto previous orthogonal vectors
        for k in range(j):
            q_k = Q[:, k]
            denom = np.dot(q_k, q_k)
            if denom > 1e-12:
                proj = (np.dot(v, q_k) / denom) * q_k
                v = v - proj
        norm = np.linalg.norm(v)
        norms[factor_names[j]] = float(norm)
        if norm > 1e-9:
            Q[:, j] = v
        else:
            # Degenerate / collinear factor -> keep zeros for stability
            degenerate.append(factor_names[j])
            Q[:, j] = np.zeros_like(v)

    # Map back to per-symbol dict
    result: Dict[str, Dict[str, float]] = {}
    for i, symbol in enumerate(symbols):
        result[symbol] = {}
        for j, factor_name in enumerate(factor_names):
            result[symbol][factor_name] = float(Q[i, j])

    # Pairwise dot products for diagnostics
    dot_products: Dict[str, Dict[str, float]] = {}
    for i, fi in enumerate(factor_names):
        dot_products[fi] = {}
        for j, fj in enumerate(factor_names):
            dot_products[fi][fj] = float(np.dot(Q[:, i], Q[:, j]))

    return OrthogonalizedFactorVectors(
        per_symbol=result,
        norms=norms,
        dot_products=dot_products,
        degenerate=degenerate,
    )


# ---------------------------------------------------------------------------
# Factor Auto-Weighting (v7.5_C3)
# ---------------------------------------------------------------------------


def compute_factor_ir(
    factor_pnl: Dict[str, float],
    factor_vols: Dict[str, float],
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Compute per-factor 'information ratio' (Sharpe-like metric).

    IR_f = factor_pnl[f] / (factor_vols[f] + eps)

    Args:
        factor_pnl: Per-factor PnL attribution
        factor_vols: Per-factor volatilities
        eps: Small constant to avoid division by zero

    Returns:
        Dict mapping factor name to IR value (signed)
    """
    result: Dict[str, float] = {}
    for factor in factor_pnl:
        pnl = float(factor_pnl.get(factor, 0.0))
        vol = float(factor_vols.get(factor, 0.0))
        result[factor] = pnl / (vol + eps)
    return result


def compute_factor_ir_from_vectors(
    vectors: List[NormalizedFactorVector],
    factor_names: List[str],
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Compute per-factor IR using cross-sectional mean/std of normalized vectors.

    Args:
        vectors: Normalized factor vectors
        factor_names: Factor names to include
        eps: Small constant to avoid division by zero

    Returns:
        Dict mapping factor name to IR (mean / std)
    """
    ir: Dict[str, float] = {}
    if not vectors:
        return ir

    for factor in factor_names:
        values = [float(vec.factors.get(factor, 0.0) or 0.0) for vec in vectors]
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        ir[factor] = mean / (std + eps) if std > eps else 0.0

    return ir


def compute_raw_factor_weights(
    mode: str,
    factor_names: List[str],
    factor_vols: Dict[str, float],
    factor_ir: Dict[str, float],
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Compute raw (unnormalized) factor weights based on mode.

    Modes:
    - "equal": all factors same weight (1.0)
    - "vol_inverse": weights proportional to 1/vol
    - "ir_only": weights proportional to IR
    - "vol_inverse_ir": weights proportional to IR/vol (recommended)

    Args:
        mode: Weighting mode
        factor_names: Factor names to weight
        factor_vols: Per-factor volatilities
        factor_ir: Per-factor IR values
        eps: Small constant to avoid division by zero

    Returns:
        Dict mapping factor name to raw weight (may be negative if IR negative)
    """
    result: Dict[str, float] = {}

    for factor in factor_names:
        vol = float(factor_vols.get(factor, 0.0))
        ir = float(factor_ir.get(factor, 0.0))

        if mode == "equal":
            result[factor] = 1.0
        elif mode == "vol_inverse":
            result[factor] = 1.0 / (vol + eps)
        elif mode == "ir_only":
            result[factor] = ir
        elif mode == "vol_inverse_ir":
            result[factor] = ir / (vol + eps)
        else:
            # Default to equal
            result[factor] = 1.0

    return result


def normalize_factor_weights(
    raw_weights: Dict[str, float],
    min_weight: float,
    max_weight: float,
    normalize_to_one: bool,
) -> FactorWeights:
    """
    Normalize and clamp factor weights.

    Steps:
    1. Convert to absolute values
    2. Normalize sum to 1 if requested
    3. Clamp each weight to [min_weight, max_weight]
    4. Iteratively renormalize while respecting clamps

    Args:
        raw_weights: Raw weight values (may be negative)
        min_weight: Minimum per-factor weight
        max_weight: Maximum per-factor weight
        normalize_to_one: Whether weights should sum to 1

    Returns:
        FactorWeights with normalized, clamped weights
    """
    if not raw_weights:
        return FactorWeights(weights={})

    # Step 1: Take absolute values
    abs_weights = {f: abs(w) for f, w in raw_weights.items()}

    # Step 2: Check if all zeros
    total = sum(abs_weights.values())
    if total < 1e-12:
        # All zeros: assign equal weights
        n = len(abs_weights)
        equal_weight = 1.0 / n if n > 0 else 0.0
        abs_weights = {f: equal_weight for f in abs_weights}
        total = 1.0

    # Step 3: Initial normalization if requested
    if normalize_to_one and total > 1e-12:
        abs_weights = {f: w / total for f, w in abs_weights.items()}

    # Step 4: Clamp with iterative normalization
    # We iteratively clamp and renormalize to converge on a solution
    # that respects both clamps and sum-to-one constraint
    if normalize_to_one:
        weights = dict(abs_weights)
        max_iterations = 10
        
        for _ in range(max_iterations):
            # Clamp weights
            clamped_at_min = set()
            clamped_at_max = set()
            
            for f in weights:
                if weights[f] <= min_weight:
                    weights[f] = min_weight
                    clamped_at_min.add(f)
                elif weights[f] >= max_weight:
                    weights[f] = max_weight
                    clamped_at_max.add(f)
            
            # Calculate how much weight is fixed vs free
            fixed_weight = sum(weights[f] for f in clamped_at_min | clamped_at_max)
            free_factors = [f for f in weights if f not in clamped_at_min and f not in clamped_at_max]
            
            if not free_factors:
                # All clamped, just normalize what we have
                total = sum(weights.values())
                if total > 1e-12:
                    weights = {f: w / total for f, w in weights.items()}
                break
            
            # Distribute remaining weight among free factors
            remaining_weight = 1.0 - fixed_weight
            if remaining_weight <= 0:
                # Need to scale down clamped weights
                break
            
            free_total = sum(weights[f] for f in free_factors)
            if free_total > 1e-12:
                scale = remaining_weight / free_total
                for f in free_factors:
                    weights[f] *= scale
            
            # Check if converged
            total = sum(weights.values())
            if abs(total - 1.0) < 1e-6:
                break
        
        return FactorWeights(weights=weights)
    else:
        # Just clamp, no normalization
        clamped = {f: max(min_weight, min(max_weight, w)) for f, w in abs_weights.items()}
        return FactorWeights(weights=clamped)


def smooth_factor_weights(
    prev: Optional[FactorWeights],
    current: FactorWeights,
    alpha: float,
) -> FactorWeights:
    """
    Apply EWMA smoothing to factor weights for stability.

    w_new = alpha * current + (1 - alpha) * prev

    Args:
        prev: Previous weights (None if first time)
        current: Current computed weights
        alpha: Smoothing factor (0 = all prev, 1 = all current)

    Returns:
        Smoothed FactorWeights
    """
    if prev is None or not prev.weights:
        return current

    smoothed: Dict[str, float] = {}
    all_factors = set(current.weights.keys()) | set(prev.weights.keys())

    for factor in all_factors:
        curr_w = current.weights.get(factor, 0.0)
        prev_w = prev.weights.get(factor, 0.0)
        smoothed[factor] = alpha * curr_w + (1 - alpha) * prev_w

    return FactorWeights(weights=smoothed)


def build_factor_weights_snapshot(
    factor_cov: FactorCovarianceSnapshot,
    factor_pnl: Dict[str, float],
    auto_weight_cfg: AutoWeightingConfig,
    prev_weights: Optional[FactorWeights] = None,
    factor_ir_override: Optional[Dict[str, float]] = None,
) -> FactorWeightsSnapshot:
    """
    Compute final per-factor weights with IR/vol-based weighting.

    Args:
        factor_cov: Covariance snapshot with factor_vols
        factor_pnl: Per-factor PnL attribution
        auto_weight_cfg: Auto-weighting configuration
        prev_weights: Previous weights for EWMA smoothing

    Returns:
        FactorWeightsSnapshot with computed weights
    """
    factor_vols = factor_cov.factor_vols
    factor_names = factor_cov.factors

    # Compute factor IR
    factor_ir = factor_ir_override or compute_factor_ir(
        factor_pnl=factor_pnl,
        factor_vols=factor_vols,
    )

    # Compute raw weights
    raw_weights = compute_raw_factor_weights(
        mode=auto_weight_cfg.mode,
        factor_names=factor_names,
        factor_vols=factor_vols,
        factor_ir=factor_ir,
    )

    # Normalize and clamp
    normalized = normalize_factor_weights(
        raw_weights=raw_weights,
        min_weight=auto_weight_cfg.min_weight,
        max_weight=auto_weight_cfg.max_weight,
        normalize_to_one=auto_weight_cfg.normalize_to_one,
    )

    # Apply EWMA smoothing
    smoothed = smooth_factor_weights(
        prev=prev_weights,
        current=normalized,
        alpha=auto_weight_cfg.smoothing_alpha,
    )

    return FactorWeightsSnapshot(
        weights=smoothed.weights,
        factor_vols=factor_vols,
        factor_ir=factor_ir,
        mode=auto_weight_cfg.mode,
        updated_ts=time.time(),
    )


# ---------------------------------------------------------------------------
# Diagnostic Snapshot Builder
# ---------------------------------------------------------------------------


def build_factor_diagnostics_snapshot(
    factor_vectors: List[FactorVector],
    cfg: FactorDiagnosticsConfig | None = None,
    factor_pnl: Dict[str, float] | None = None,
    prev_weights: FactorWeights | None = None,
    strategy_config: Mapping[str, Any] | None = None,
) -> FactorDiagnosticsSnapshot:
    """
    Produce normalized per-symbol factor vectors + covariance snapshot.

    v7.5_C3: Also computes orthogonalized vectors and auto-weights if enabled.

    Args:
        factor_vectors: List of raw FactorVector objects from hybrid scoring
        cfg: Factor diagnostics configuration
        factor_pnl: Per-factor PnL for weight computation (optional)
        prev_weights: Previous factor weights for EWMA smoothing (optional)
        strategy_config: Full strategy config for loading ortho/weight configs

    Returns:
        Complete FactorDiagnosticsSnapshot
    """
    if cfg is None:
        cfg = load_factor_diagnostics_config(strategy_config)

    # Load C3 configs
    ortho_cfg = load_orthogonalization_config(strategy_config)
    auto_weight_cfg = load_auto_weighting_config(strategy_config)

    if not cfg.enabled or not factor_vectors:
        ts = time.time()
        return FactorDiagnosticsSnapshot(
            per_symbol={},
            covariance=None,
            orthogonalized=None,
            factor_weights=None,
            orthogonalization_enabled=ortho_cfg.enabled,
            auto_weighting_enabled=auto_weight_cfg.enabled,
            updated_ts=ts,
            config=cfg,
            normalization_coeffs={},
            raw_factors={},
            factor_ir={},
            weights={},
            pnl_attribution={"window_days": cfg.pnl_attribution_lookback_days, "by_factor": factor_pnl or {}},
        )

    ts = time.time()

    # Capture raw factor surface keyed by symbol:direction for determinism
    raw_factors: Dict[str, Dict[str, Any]] = {}
    for vec in factor_vectors:
        key = f"{vec.symbol}:{vec.direction}"
        raw_factors[key] = {
            "factors": {name: float(vec.factors.get(name, 0.0) or 0.0) for name in cfg.factors},
            "direction": vec.direction,
            "regime": getattr(vec, "regime", "normal"),
        }

    # Normalize factor vectors (returns both normalized vectors and stats)
    normalized, norm_stats = normalize_factor_vectors(
        vectors=factor_vectors,
        factor_names=cfg.factors,
        mode=cfg.normalization_mode,
        max_abs_zscore=cfg.max_abs_zscore,
        return_stats=True,
    )

    # Build per-symbol mapping
    per_symbol: Dict[str, NormalizedFactorVector] = {}
    for nvec in normalized:
        # Key by symbol:direction for uniqueness
        key = f"{nvec.symbol}:{nvec.direction}"
        per_symbol[key] = nvec

    # Compute covariance using normalized vectors
    covariance = compute_factor_covariance(
        vectors=normalized,  # type: ignore[arg-type]
        factor_names=cfg.factors,
        lookback_days=cfg.covariance_lookback_days,
    )

    # v7.5_C3: Orthogonalization (now orthonormalized)
    orthogonalized: Optional[OrthogonalizedFactorVectors] = None
    if ortho_cfg.enabled and ortho_cfg.method == "gram_schmidt":
        orthogonalized = orthogonalize_factors(
            factor_vectors=normalized,  # type: ignore[arg-type]
            factor_names=cfg.factors,
        )

    # Compute IR both from pnl (if available) and observed factors
    factor_ir_vectors = compute_factor_ir_from_vectors(
        vectors=normalized,
        factor_names=cfg.factors,
    )
    factor_ir_pnl = compute_factor_ir(
        factor_pnl=factor_pnl or {},
        factor_vols=covariance.factor_vols if covariance else {},
    )
    factor_ir = factor_ir_pnl or factor_ir_vectors

    # v7.5_C3: Auto-weighting
    factor_weights_snapshot: Optional[FactorWeightsSnapshot] = None
    if auto_weight_cfg.enabled and covariance is not None:
        # Use provided factor_pnl or default to zeros
        pnl = factor_pnl if factor_pnl else {f: 0.0 for f in cfg.factors}
        factor_weights_snapshot = build_factor_weights_snapshot(
            factor_cov=covariance,
            factor_pnl=pnl,
            auto_weight_cfg=auto_weight_cfg,
            prev_weights=prev_weights,
            factor_ir_override=factor_ir,
        )

    # Best-effort weights surface (even if auto_weighting disabled)
    weights = factor_weights_snapshot.weights if factor_weights_snapshot else {}

    pnl_attribution = {
        "window_days": cfg.pnl_attribution_lookback_days,
        "by_factor": factor_pnl or {},
        "factor_ir": factor_ir,
        "weights": weights,
    }
    if orthogonalized:
        pnl_attribution["orthogonalized_summary"] = {
            "degenerate": orthogonalized.degenerate,
            "norms": orthogonalized.norms,
        }

    return FactorDiagnosticsSnapshot(
        per_symbol=per_symbol,
        covariance=covariance,
        orthogonalized=orthogonalized,
        factor_weights=factor_weights_snapshot,
        orthogonalization_enabled=ortho_cfg.enabled,
        auto_weighting_enabled=auto_weight_cfg.enabled,
        updated_ts=ts,
        config=cfg,
        normalization_coeffs=norm_stats,
        raw_factors=raw_factors,
        factor_ir=factor_ir,
        weights=weights,
        pnl_attribution=pnl_attribution,
    )


def extract_factor_vectors_from_hybrid_results(
    hybrid_results: List[Dict[str, Any]],
) -> List[FactorVector]:
    """
    Extract FactorVector objects from hybrid scoring results.

    Args:
        hybrid_results: List of hybrid_score() result dicts

    Returns:
        List of FactorVector objects
    """
    vectors: List[FactorVector] = []

    for result in hybrid_results:
        symbol = result.get("symbol", "")
        direction = result.get("direction", "LONG")
        hybrid_score = float(result.get("hybrid_score", 0.0))
        regime = result.get("regime", "normal")

        # Get factor_vector if present, otherwise build from components
        factor_dict = result.get("factor_vector")
        if not factor_dict:
            components = result.get("components", {})
            rv_momentum = result.get("rv_momentum", {})
            router_quality = result.get("router_quality", {})

            factor_dict = {
                "trend": float(components.get("trend", 0.0)),
                "carry": float(components.get("carry", 0.0)),
                "expectancy": float(components.get("expectancy", 0.0)),
                "router": float(components.get("router", 0.0)),
                "rv_momentum": float(rv_momentum.get("score", 0.0) if rv_momentum else 0.0),
                "router_quality": float(
                    router_quality.get("score", 0.0) if router_quality else 0.0
                ),
                "vol_regime": 1.0 if regime == "normal" else 0.5,
            }

        vectors.append(
            build_factor_vector(
                symbol=symbol,
                components=factor_dict,
                hybrid_score=hybrid_score,
                direction=direction,
                regime=regime or "normal",
            )
        )

    return vectors


# ---------------------------------------------------------------------------
# State File I/O
# ---------------------------------------------------------------------------

DEFAULT_FACTOR_DIAGNOSTICS_PATH = Path("logs/state/factor_diagnostics.json")


def load_factor_diagnostics_state(
    path: Path | str = DEFAULT_FACTOR_DIAGNOSTICS_PATH,
) -> Dict[str, Any]:
    """Load factor diagnostics state from file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def write_factor_diagnostics_state(
    snapshot: FactorDiagnosticsSnapshot | Dict[str, Any],
    path: Path | str = DEFAULT_FACTOR_DIAGNOSTICS_PATH,
) -> None:
    """Write factor diagnostics state to file."""
    path = Path(path)
    if not path.name.endswith(".json"):
        path = path / "factor_diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
            json.dump(payload, f, indent=2)
        tmp.replace(path)
    except Exception:
        pass


__all__ = [
    # Config
    "FactorDiagnosticsConfig",
    "OrthogonalizationConfig",
    "AutoWeightingConfig",
    "load_factor_diagnostics_config",
    "load_orthogonalization_config",
    "load_auto_weighting_config",
    # Data structures
    "NormalizedFactorVector",
    "FactorCovarianceSnapshot",
    "FactorDiagnosticsSnapshot",
    "FactorWeights",
    "FactorWeightsSnapshot",
    "OrthogonalizedFactorVectors",
    # Normalization & Covariance
    "normalize_factor_vectors",
    "compute_factor_covariance",
    # Orthogonalization (v7.5_C3)
    "orthogonalize_factors",
    # Auto-weighting (v7.5_C3)
    "compute_factor_ir",
    "compute_factor_ir_from_vectors",
    "compute_raw_factor_weights",
    "normalize_factor_weights",
    "smooth_factor_weights",
    "build_factor_weights_snapshot",
    # Snapshot builders
    "build_factor_diagnostics_snapshot",
    "extract_factor_vectors_from_hybrid_results",
    # State I/O
    "load_factor_diagnostics_state",
    "write_factor_diagnostics_state",
]
