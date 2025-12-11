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
    "category_momentum",
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
class AdaptiveConfig:
    """Configuration for adaptive IR/PnL-based weight biasing (v7.7_P2)."""

    enabled: bool = False  # Default disabled for backward compatibility
    ir_boost_threshold: float = 0.25  # IR above this gets positive bias
    ir_cut_threshold: float = 0.0  # IR below this gets negative bias
    pnl_min_for_boost: float = 0.0  # PnL must be >= this for boost
    max_shift: float = 0.10  # Maximum bias per factor


@dataclass
class FactorRegimeCurvesConfig:
    """Configuration for regime-specific factor weight curves (v7.7_P6)."""

    # Volatility regime multipliers for factor weights
    volatility: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 1.05,
        "NORMAL": 1.00,
        "HIGH": 0.90,
        "CRISIS": 0.70,
    })

    # Drawdown state multipliers for factor weights
    drawdown: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.00,
        "RECOVERY": 0.90,
        "DRAWDOWN": 0.65,
    })


@dataclass
class FactorPerformance:
    """Per-factor performance metrics for adaptive weighting (v7.7_P2)."""

    name: str
    ir: float = 0.0  # Information ratio for the factor
    pnl_contrib: float = 0.0  # Contribution to PnL (normalized or absolute)


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
    # v7.7_P2: Adaptive weighting metadata
    adaptive_enabled: bool = False
    adaptive_bias: Dict[str, float] = field(default_factory=dict)
    # v7.7_P6: Regime modifiers metadata
    regime_modifiers: Dict[str, Any] = field(default_factory=dict)
    # v7.8_P1: Meta-scheduler overlay metadata
    meta_overlay_enabled: bool = False
    meta_overlay: Dict[str, float] = field(default_factory=dict)
    # v7.8_P6: Sentinel-X overlay metadata
    sentinel_x_overlay_enabled: bool = False
    sentinel_x_overlay: Dict[str, float] = field(default_factory=dict)
    sentinel_x_regime: str = ""
    # v7.8_P7: Alpha decay overlay metadata
    alpha_decay_overlay_enabled: bool = False
    alpha_decay_multipliers: Dict[str, float] = field(default_factory=dict)
    # v7.8_P8: Cerberus multi-strategy overlay metadata
    cerberus_overlay_enabled: bool = False
    cerberus_overlay: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "weights": self.weights,
            "factor_vols": self.factor_vols,
            "factor_ir": self.factor_ir,
            "mode": self.mode,
            "updated_ts": self.updated_ts,
            "adaptive_enabled": self.adaptive_enabled,
            "adaptive_bias": self.adaptive_bias,
            "regime_modifiers": self.regime_modifiers,  # v7.7_P6
            "meta_overlay_enabled": self.meta_overlay_enabled,  # v7.8_P1
            "meta_overlay": self.meta_overlay,  # v7.8_P1
            "sentinel_x_overlay_enabled": self.sentinel_x_overlay_enabled,  # v7.8_P6
            "sentinel_x_overlay": self.sentinel_x_overlay,  # v7.8_P6
            "sentinel_x_regime": self.sentinel_x_regime,  # v7.8_P6
            "alpha_decay_overlay_enabled": self.alpha_decay_overlay_enabled,  # v7.8_P7
            "alpha_decay_multipliers": self.alpha_decay_multipliers,  # v7.8_P7
            "cerberus_overlay_enabled": self.cerberus_overlay_enabled,  # v7.8_P8
            "cerberus_overlay": self.cerberus_overlay,  # v7.8_P8
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


def load_adaptive_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> AdaptiveConfig:
    """
    Load adaptive IR/PnL-based weight biasing config from strategy_config (v7.7_P2).

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        AdaptiveConfig instance
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return AdaptiveConfig(enabled=False)

    fd_cfg = strategy_config.get("factor_diagnostics", {})
    if not isinstance(fd_cfg, dict):
        return AdaptiveConfig(enabled=False)

    aw_cfg = fd_cfg.get("auto_weighting", {})
    if not isinstance(aw_cfg, dict):
        return AdaptiveConfig(enabled=False)

    adaptive_cfg = aw_cfg.get("adaptive", {})
    if not isinstance(adaptive_cfg, dict) or not adaptive_cfg:
        # No adaptive config = disabled
        return AdaptiveConfig(enabled=False)

    return AdaptiveConfig(
        enabled=bool(adaptive_cfg.get("enabled", False)),  # Default disabled for safety
        ir_boost_threshold=float(adaptive_cfg.get("ir_boost_threshold", 0.25)),
        ir_cut_threshold=float(adaptive_cfg.get("ir_cut_threshold", 0.0)),
        pnl_min_for_boost=float(adaptive_cfg.get("pnl_min_for_boost", 0.0)),
        max_shift=float(adaptive_cfg.get("max_shift", 0.10)),
    )


def load_factor_regime_curves_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> FactorRegimeCurvesConfig:
    """
    Load regime-specific factor weight curves config (v7.7_P6).

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        FactorRegimeCurvesConfig instance
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return FactorRegimeCurvesConfig()

    fd_cfg = strategy_config.get("factor_diagnostics", {})
    if not isinstance(fd_cfg, dict):
        return FactorRegimeCurvesConfig()

    aw_cfg = fd_cfg.get("auto_weighting", {})
    if not isinstance(aw_cfg, dict):
        return FactorRegimeCurvesConfig()

    regime_curves_cfg = aw_cfg.get("regime_curves", {})
    if not isinstance(regime_curves_cfg, dict) or not regime_curves_cfg:
        # No regime curves config = use defaults
        return FactorRegimeCurvesConfig()

    return FactorRegimeCurvesConfig(
        volatility=dict(regime_curves_cfg.get("volatility", FactorRegimeCurvesConfig().volatility)),
        drawdown=dict(regime_curves_cfg.get("drawdown", FactorRegimeCurvesConfig().drawdown)),
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


# ---------------------------------------------------------------------------
# Adaptive Factor Outcome Weighting (v7.7_P2)
# ---------------------------------------------------------------------------


def compute_factor_performance(
    factor_ir: Dict[str, float],
    factor_pnl: Dict[str, float],
) -> Dict[str, FactorPerformance]:
    """
    Derive per-factor performance metrics from IR and PnL attribution.

    Args:
        factor_ir: Per-factor IR values (information ratio)
        factor_pnl: Per-factor PnL contribution (by_factor from pnl_attribution)

    Returns:
        Dict mapping factor name to FactorPerformance
    """
    result: Dict[str, FactorPerformance] = {}

    # Combine keys from both dicts
    all_factors = set(factor_ir.keys()) | set(factor_pnl.keys())

    for factor in all_factors:
        ir = float(factor_ir.get(factor, 0.0))
        pnl = float(factor_pnl.get(factor, 0.0))
        result[factor] = FactorPerformance(name=factor, ir=ir, pnl_contrib=pnl)

    return result


def compute_adaptive_weight_bias(
    base_weights: Dict[str, float],
    factor_perf: Dict[str, FactorPerformance],
    cfg: AdaptiveConfig,
) -> Dict[str, float]:
    """
    Compute per-factor bias based on IR/PnL outcomes (v7.7_P2).

    Returns bias in [-max_shift, +max_shift] to be added to base_weights
    before re-normalization.

    Rules:
    - If factor.ir >= ir_boost_threshold AND factor.pnl_contrib >= pnl_min_for_boost:
      positive bias (up to +max_shift)
    - If factor.ir < ir_cut_threshold OR factor.pnl_contrib < 0:
      negative bias (down to -max_shift)
    - Otherwise: zero bias

    The magnitude is proportional to how far the IR is from the threshold.

    Args:
        base_weights: Current factor weights (before adaptive adjustment)
        factor_perf: Per-factor performance metrics
        cfg: Adaptive weighting configuration

    Returns:
        Dict mapping factor name to bias value in [-max_shift, +max_shift]
    """
    bias: Dict[str, float] = {}

    for factor in base_weights:
        if factor not in factor_perf:
            # Missing performance data = no bias
            bias[factor] = 0.0
            continue

        perf = factor_perf[factor]
        ir = perf.ir
        pnl = perf.pnl_contrib

        # Compute bias direction and magnitude
        factor_bias = 0.0

        # Case 1: Boost - high IR AND positive PnL
        if ir >= cfg.ir_boost_threshold and pnl >= cfg.pnl_min_for_boost:
            # Scale bias by how much IR exceeds threshold (up to 2x threshold = full boost)
            ir_excess = ir - cfg.ir_boost_threshold
            # Normalize: if IR is 2x the boost threshold, get full max_shift
            scale = min(1.0, ir_excess / max(cfg.ir_boost_threshold, 0.01))
            factor_bias = cfg.max_shift * scale

        # Case 2: Cut - low IR OR negative PnL
        elif ir < cfg.ir_cut_threshold or pnl < 0:
            # For negative PnL, apply proportional cut (more negative = bigger cut)
            if pnl < 0:
                # Scale by magnitude of loss relative to a "significant" loss threshold
                # Use -1.0 as a reference for "full cut"
                pnl_scale = min(1.0, abs(pnl))
                factor_bias = -cfg.max_shift * max(0.3, pnl_scale)
            else:
                # Low IR but non-negative PnL: modest cut
                ir_deficit = cfg.ir_cut_threshold - ir
                scale = min(1.0, ir_deficit / max(abs(cfg.ir_cut_threshold) + 0.1, 0.1))
                factor_bias = -cfg.max_shift * scale * 0.5

        # Clamp to bounds
        bias[factor] = max(-cfg.max_shift, min(cfg.max_shift, factor_bias))

    return bias


def apply_adaptive_bias_to_weights(
    base_weights: Dict[str, float],
    bias: Dict[str, float],
    min_weight: float,
    max_weight: float,
    normalize_to_one: bool,
) -> Dict[str, float]:
    """
    Apply adaptive bias to base weights with clamping and optional normalization.

    Args:
        base_weights: Base factor weights
        bias: Per-factor bias values
        min_weight: Minimum allowed weight per factor
        max_weight: Maximum allowed weight per factor
        normalize_to_one: Whether to normalize sum to 1.0

    Returns:
        Adjusted weights
    """
    # Add bias to base weights
    adjusted = {}
    for factor, base_w in base_weights.items():
        factor_bias = bias.get(factor, 0.0)
        adjusted[factor] = base_w + factor_bias

    # Clamp to bounds
    for factor in adjusted:
        adjusted[factor] = max(min_weight, min(max_weight, adjusted[factor]))

    # Normalize if requested
    if normalize_to_one:
        total = sum(adjusted.values())
        if total > 1e-12:
            adjusted = {f: w / total for f, w in adjusted.items()}

            # After normalization, may need to re-clamp iteratively
            # (same logic as normalize_factor_weights)
            for _ in range(10):
                clamped_at_bound = False
                fixed_weight = 0.0
                free_factors = []

                for f in adjusted:
                    if adjusted[f] <= min_weight:
                        adjusted[f] = min_weight
                        fixed_weight += min_weight
                        clamped_at_bound = True
                    elif adjusted[f] >= max_weight:
                        adjusted[f] = max_weight
                        fixed_weight += max_weight
                        clamped_at_bound = True
                    else:
                        free_factors.append(f)

                if not clamped_at_bound or not free_factors:
                    break

                remaining = 1.0 - fixed_weight
                free_total = sum(adjusted[f] for f in free_factors)
                if free_total > 1e-12 and remaining > 0:
                    scale = remaining / free_total
                    for f in free_factors:
                        adjusted[f] *= scale

    return adjusted


def get_vol_regime_from_snapshot(risk_snapshot: Mapping[str, Any] | None) -> str:
    """
    Extract volatility regime from risk_snapshot (v7.7_P6).

    Args:
        risk_snapshot: Risk snapshot dict or None

    Returns:
        Vol regime string (LOW/NORMAL/HIGH/CRISIS), defaults to NORMAL
    """
    if not risk_snapshot:
        return "NORMAL"
    vol_regime = risk_snapshot.get("vol_regime", "normal")
    if isinstance(vol_regime, str):
        return vol_regime.upper()
    return "NORMAL"


def get_dd_state_from_snapshot(risk_snapshot: Mapping[str, Any] | None) -> str:
    """
    Extract drawdown state from risk_snapshot (v7.7_P6).

    Args:
        risk_snapshot: Risk snapshot dict or None

    Returns:
        DD state string (NORMAL/RECOVERY/DRAWDOWN), defaults to NORMAL
    """
    if not risk_snapshot:
        return "NORMAL"
    dd_state = risk_snapshot.get("dd_state", "NORMAL")
    if isinstance(dd_state, str):
        return dd_state.upper()
    return "NORMAL"


def apply_regime_curves_to_weights(
    weights: Dict[str, float],
    vol_regime: str,
    dd_state: str,
    regime_curves: FactorRegimeCurvesConfig,
    min_weight: float,
    max_weight: float,
    normalize_to_one: bool,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply regime-specific multipliers to factor weights (v7.7_P6).

    Args:
        weights: Base factor weights
        vol_regime: Current volatility regime (LOW/NORMAL/HIGH/CRISIS)
        dd_state: Current drawdown state (NORMAL/RECOVERY/DRAWDOWN)
        regime_curves: Regime curves configuration
        min_weight: Minimum allowed weight per factor
        max_weight: Maximum allowed weight per factor
        normalize_to_one: Whether to normalize sum to 1.0

    Returns:
        Tuple of (adjusted_weights, regime_modifiers metadata)
    """
    # Get multipliers
    vol_mult = regime_curves.volatility.get(vol_regime.upper(), 1.0)
    dd_mult = regime_curves.drawdown.get(dd_state.upper(), 1.0)
    combined_mult = vol_mult * dd_mult

    # Apply multiplier to all weights
    adjusted = {f: w * combined_mult for f, w in weights.items()}

    # Clamp to bounds
    for factor in adjusted:
        adjusted[factor] = max(min_weight, min(max_weight, adjusted[factor]))

    # Normalize if requested
    if normalize_to_one:
        total = sum(adjusted.values())
        if total > 1e-12:
            adjusted = {f: w / total for f, w in adjusted.items()}

    # Build regime modifiers metadata
    regime_modifiers = {
        "vol_regime": vol_regime,
        "vol_multiplier": vol_mult,
        "dd_state": dd_state,
        "dd_multiplier": dd_mult,
        "combined_multiplier": combined_mult,
    }

    return adjusted, regime_modifiers


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
    adaptive_cfg: Optional[AdaptiveConfig] = None,
    regime_curves_cfg: Optional[FactorRegimeCurvesConfig] = None,
    risk_snapshot: Optional[Mapping[str, Any]] = None,
) -> FactorWeightsSnapshot:
    """
    Compute final per-factor weights with IR/vol-based weighting.

    v7.7_P2: Now supports adaptive IR/PnL-based weight biasing when adaptive_cfg is
    provided and enabled.
    
    v7.7_P6: Now supports regime-specific weight curves when regime_curves_cfg is
    provided and risk_snapshot contains vol_regime/dd_state.

    Args:
        factor_cov: Covariance snapshot with factor_vols
        factor_pnl: Per-factor PnL attribution
        auto_weight_cfg: Auto-weighting configuration
        prev_weights: Previous weights for EWMA smoothing
        factor_ir_override: Optional IR values to use instead of computing
        adaptive_cfg: Optional adaptive weighting config (v7.7_P2)
        regime_curves_cfg: Optional regime curves config (v7.7_P6)
        risk_snapshot: Optional risk snapshot for regime context (v7.7_P6)

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

    # v7.7_P2: Apply adaptive bias if enabled
    adaptive_bias: Dict[str, float] = {}
    adaptive_enabled = adaptive_cfg is not None and adaptive_cfg.enabled

    if adaptive_enabled and adaptive_cfg is not None:
        # Compute factor performance from IR and PnL
        factor_perf = compute_factor_performance(
            factor_ir=factor_ir,
            factor_pnl=factor_pnl,
        )

        # Compute adaptive bias
        adaptive_bias = compute_adaptive_weight_bias(
            base_weights=normalized.weights,
            factor_perf=factor_perf,
            cfg=adaptive_cfg,
        )

        # Apply bias with clamping and normalization
        adjusted_weights = apply_adaptive_bias_to_weights(
            base_weights=normalized.weights,
            bias=adaptive_bias,
            min_weight=auto_weight_cfg.min_weight,
            max_weight=auto_weight_cfg.max_weight,
            normalize_to_one=auto_weight_cfg.normalize_to_one,
        )
        normalized = FactorWeights(weights=adjusted_weights)

    # v7.7_P6: Apply regime curves if enabled
    regime_modifiers: Dict[str, Any] = {}
    if regime_curves_cfg is not None:
        vol_regime = get_vol_regime_from_snapshot(risk_snapshot)
        dd_state = get_dd_state_from_snapshot(risk_snapshot)
        
        adjusted_weights, regime_modifiers = apply_regime_curves_to_weights(
            weights=normalized.weights,
            vol_regime=vol_regime,
            dd_state=dd_state,
            regime_curves=regime_curves_cfg,
            min_weight=auto_weight_cfg.min_weight,
            max_weight=auto_weight_cfg.max_weight,
            normalize_to_one=auto_weight_cfg.normalize_to_one,
        )
        normalized = FactorWeights(weights=adjusted_weights)

    # v7.8_P1: Apply meta-scheduler overlay if enabled
    meta_overlay: Dict[str, float] = {}
    meta_overlay_enabled = False
    try:
        from execution.meta_scheduler import (
            load_meta_scheduler_config,
            load_meta_scheduler_state,
            get_factor_meta_weights,
            is_meta_scheduler_active,
        )
        # Load meta-scheduler config from strategy_config if available
        # Note: strategy_config is passed to build_factor_diagnostics_snapshot,
        # but not directly here. We load it fresh for safety.
        meta_cfg = load_meta_scheduler_config(None)  # Will use defaults
        meta_state = load_meta_scheduler_state()
        
        if meta_cfg.enabled and is_meta_scheduler_active(meta_cfg, meta_state):
            meta_overlay = get_factor_meta_weights(meta_state)
            if meta_overlay:
                meta_overlay_enabled = True
                # Apply meta overlays as multipliers
                meta_adjusted = {}
                for factor_name, w in normalized.weights.items():
                    overlay = meta_overlay.get(factor_name, 1.0)
                    meta_adjusted[factor_name] = w * overlay
                
                # Re-normalize after applying meta overlays
                normalized = normalize_factor_weights(
                    raw_weights=meta_adjusted,
                    min_weight=auto_weight_cfg.min_weight,
                    max_weight=auto_weight_cfg.max_weight,
                    normalize_to_one=auto_weight_cfg.normalize_to_one,
                )
    except ImportError:
        # meta_scheduler not available - skip
        pass
    except Exception:
        # Don't let meta-scheduler errors break factor diagnostics
        pass

    # v7.8_P6: Apply Sentinel-X regime overlay if enabled
    sentinel_x_overlay: Dict[str, float] = {}
    sentinel_x_overlay_enabled = False
    sentinel_x_regime = ""
    try:
        from execution.sentinel_x import (
            load_sentinel_x_config,
            load_sentinel_x_state,
            get_factor_regime_weights,
        )
        
        sentinel_cfg = load_sentinel_x_config(None)
        
        if sentinel_cfg.enabled:
            sentinel_state = load_sentinel_x_state()
            sentinel_x_regime = sentinel_state.primary_regime
            
            if sentinel_x_regime:
                sentinel_x_overlay = get_factor_regime_weights(sentinel_x_regime)
                if sentinel_x_overlay:
                    sentinel_x_overlay_enabled = True
                    # Apply sentinel overlays as multipliers
                    sentinel_adjusted = {}
                    for factor_name, w in normalized.weights.items():
                        overlay = sentinel_x_overlay.get(factor_name, 1.0)
                        sentinel_adjusted[factor_name] = w * overlay
                    
                    # Re-normalize after applying Sentinel-X overlays
                    normalized = normalize_factor_weights(
                        raw_weights=sentinel_adjusted,
                        min_weight=auto_weight_cfg.min_weight,
                        max_weight=auto_weight_cfg.max_weight,
                        normalize_to_one=auto_weight_cfg.normalize_to_one,
                    )
    except ImportError:
        # sentinel_x not available - skip
        pass
    except Exception:
        # Don't let sentinel_x errors break factor diagnostics
        pass

    # v7.8_P8: Apply Cerberus multi-strategy overlay if enabled
    cerberus_overlay: Dict[str, float] = {}
    cerberus_overlay_enabled = False
    try:
        from execution.cerberus_router import (
            load_cerberus_config,
            load_cerberus_state,
            get_cerberus_factor_weight_overlay,
        )
        
        cerberus_cfg = load_cerberus_config(strategy_config=None)
        
        if cerberus_cfg.enabled:
            cerberus_state = load_cerberus_state()
            if cerberus_state:
                cerberus_overlay_enabled = True
                # Get per-factor overlays from Cerberus head multipliers
                cerberus_adjusted = {}
                for factor_name, w in normalized.weights.items():
                    overlay = get_cerberus_factor_weight_overlay(factor_name, cerberus_state, cerberus_cfg)
                    cerberus_overlay[factor_name] = overlay
                    cerberus_adjusted[factor_name] = w * overlay
                
                # Re-normalize after applying Cerberus overlays
                normalized = normalize_factor_weights(
                    raw_weights=cerberus_adjusted,
                    min_weight=auto_weight_cfg.min_weight,
                    max_weight=auto_weight_cfg.max_weight,
                    normalize_to_one=auto_weight_cfg.normalize_to_one,
                )
    except ImportError:
        # cerberus_router not available - skip
        pass
    except Exception:
        # Don't let cerberus errors break factor diagnostics
        pass

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
        adaptive_enabled=adaptive_enabled,
        adaptive_bias=adaptive_bias,
        regime_modifiers=regime_modifiers,  # v7.7_P6
        meta_overlay_enabled=meta_overlay_enabled,  # v7.8_P1
        meta_overlay=meta_overlay,  # v7.8_P1
        sentinel_x_overlay_enabled=sentinel_x_overlay_enabled,  # v7.8_P6
        sentinel_x_overlay=sentinel_x_overlay,  # v7.8_P6
        sentinel_x_regime=sentinel_x_regime,  # v7.8_P6
        cerberus_overlay_enabled=cerberus_overlay_enabled,  # v7.8_P8
        cerberus_overlay=cerberus_overlay,  # v7.8_P8
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
    # v7.7_P2: Load adaptive config
    adaptive_cfg = load_adaptive_config(strategy_config)

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
            adaptive_cfg=adaptive_cfg,  # v7.7_P2
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
                # v7.7_P3: Category momentum (default to 0 if not provided)
                "category_momentum": float(components.get("category_momentum", 0.0)),
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
    "AdaptiveConfig",  # v7.7_P2
    "load_factor_diagnostics_config",
    "load_orthogonalization_config",
    "load_auto_weighting_config",
    "load_adaptive_config",  # v7.7_P2
    # Data structures
    "NormalizedFactorVector",
    "FactorCovarianceSnapshot",
    "FactorDiagnosticsSnapshot",
    "FactorWeights",
    "FactorWeightsSnapshot",
    "OrthogonalizedFactorVectors",
    "FactorPerformance",  # v7.7_P2
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
    # Adaptive weighting (v7.7_P2)
    "compute_factor_performance",
    "compute_adaptive_weight_bias",
    "apply_adaptive_bias_to_weights",
    # Snapshot builders
    "build_factor_diagnostics_snapshot",
    "extract_factor_vectors_from_hybrid_results",
    # State I/O
    "load_factor_diagnostics_state",
    "write_factor_diagnostics_state",
]
