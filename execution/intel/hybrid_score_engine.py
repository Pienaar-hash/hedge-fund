"""
Hybrid scoring engine with factor governance (weights, normalization, modifiers).

Canonical formula:
    hybrid_score = (Σ weight_i * normalized_factor_i) * vol_regime_modifier * router_quality_modifier

Factors are normalized using factor_diagnostics config, weights are clamped/smoothed,
and volatility/router quality modifiers are applied for routing-aware governance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from execution.factor_diagnostics import (
    FactorVector,
    load_factor_diagnostics_config,
    load_auto_weighting_config,
    load_orthogonalization_config,
    normalize_factor_vectors,
    orthogonalize_factors,
)
from execution.router_metrics import load_router_quality_config
from execution.utils.vol import get_hybrid_weight_modifiers


@dataclass
class HybridFactorPayload:
    """Raw factors for a single symbol/direction prior to normalization."""

    symbol: str
    direction: str
    regime: str
    factors: Dict[str, float]
    router_quality_score: float = 0.0
    vol_regime_label: str = "normal"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_weights(
    weights: Dict[str, float],
    min_weight: float,
    max_weight: float,
) -> Dict[str, float]:
    clamped = {k: _clamp(v, min_weight, max_weight) for k, v in weights.items()}
    total = sum(clamped.values())
    if total <= 1e-9:
        n = len(clamped) or 1
        return {k: 1.0 / n for k in clamped}
    return {k: v / total for k, v in clamped.items()}


def _smooth_weights(
    raw: Dict[str, float],
    prev: Optional[Mapping[str, float]],
    alpha: float,
) -> Dict[str, float]:
    if not prev:
        return dict(raw)
    smoothed: Dict[str, float] = {}
    for k, v in raw.items():
        smoothed[k] = alpha * v + (1 - alpha) * float(prev.get(k, 0.0))
    return smoothed


def compute_hybrid_scores(
    payloads: Sequence[HybridFactorPayload],
    *,
    factor_weights_raw: Optional[Mapping[str, float]] = None,
    prev_factor_weights: Optional[Mapping[str, float]] = None,
    strategy_config: Optional[Mapping[str, Any]] = None,
    max_abs_score: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Compute hybrid scores for a batch of factor payloads.

    Args:
        payloads: Iterable of HybridFactorPayload with raw factors
        factor_weights_raw: Optional raw factor weights (per factor)
        prev_factor_weights: Optional previous weights for smoothing
        strategy_config: Strategy config for factor/vol/router settings
        max_abs_score: Clamp for final hybrid score (default 1.0)

    Returns:
        List of result dicts aligned with input payload order.
    """
    cfg_fd = load_factor_diagnostics_config(strategy_config)
    ortho_cfg = load_orthogonalization_config(strategy_config)
    aw_cfg = load_auto_weighting_config(strategy_config)
    rq_cfg = load_router_quality_config(strategy_config)

    factors = cfg_fd.factors or ["trend", "carry", "expectancy", "router_quality", "rv_momentum", "vol_regime"]

    # Build factor vectors for normalization/orthogonalization
    factor_vectors: List[FactorVector] = []
    for p in payloads:
        vector_factors = {f: float(p.factors.get(f, 0.0) or 0.0) for f in factors}
        factor_vectors.append(
            FactorVector(
                symbol=p.symbol,
                factors=vector_factors,
                hybrid_score=0.0,
                direction=p.direction,
                regime=p.regime or "normal",
            )
        )

    normalized_vectors = normalize_factor_vectors(
        vectors=factor_vectors,
        factor_names=factors,
        mode=cfg_fd.normalization_mode,
        max_abs_zscore=cfg_fd.max_abs_zscore,
    )

    orthogonalized: Dict[str, Dict[str, float]] = {}
    if ortho_cfg.enabled and ortho_cfg.method == "gram_schmidt":
        ortho = orthogonalize_factors(factor_vectors=factor_vectors, factor_names=factors)
        orthogonalized = ortho.per_symbol if ortho else {}

    # Prepare weights with smoothing + clamp + renormalize
    if factor_weights_raw:
        raw_weights = {f: float(factor_weights_raw.get(f, 0.0) or 0.0) for f in factors}
    else:
        # default equal weights
        raw_weights = {f: 1.0 / len(factors) for f in factors}
    smoothed_weights = _smooth_weights(raw_weights, prev_factor_weights, aw_cfg.smoothing_alpha)
    weights = _normalize_weights(smoothed_weights, aw_cfg.min_weight, aw_cfg.max_weight)

    results: List[Dict[str, Any]] = []
    for norm_vec, raw_payload in zip(normalized_vectors, payloads):
        # Use orthogonalized factors when available; fall back to normalized factors
        ortho_map = orthogonalized.get(raw_payload.symbol, {})
        factors_for_score: Dict[str, float] = {}
        for name in factors:
            base_val = norm_vec.factors.get(name, 0.0)
            if ortho_map and name in ortho_map:
                base_val = ortho_map.get(name, base_val)
            factors_for_score[name] = float(base_val)

        # Volatility regime modifiers per factor (carry/expectancy/router; others default 1.0)
        vol_mods_cfg = get_hybrid_weight_modifiers(raw_payload.vol_regime_label or "normal", strategy_config)
        vol_mods = {
            "carry": vol_mods_cfg.carry,
            "expectancy": vol_mods_cfg.expectancy,
            "router_quality": vol_mods_cfg.router,
            "router": vol_mods_cfg.router,  # compatibility alias
        }
        vol_mod_values = [v for k, v in vol_mods.items() if k in factors]
        vol_regime_modifier = sum(vol_mod_values) / len(vol_mod_values) if vol_mod_values else 1.0

        weighted_sum = 0.0
        for name in factors:
            val = factors_for_score.get(name, 0.0)
            val *= vol_mods.get(name, 1.0)
            w = weights.get(name, 0.0)
            weighted_sum += w * val

        # Router quality multiplier
        rq_score = float(raw_payload.router_quality_score or 0.0)
        rq_multiplier = 1.0
        if rq_cfg.enabled:
            if rq_score <= rq_cfg.low_quality_threshold:
                rq_multiplier = rq_cfg.low_quality_hybrid_multiplier
            elif rq_score >= rq_cfg.high_quality_threshold:
                rq_multiplier = rq_cfg.high_quality_hybrid_multiplier

        hybrid = weighted_sum * vol_regime_modifier * rq_multiplier
        hybrid = _clamp(hybrid, -abs(max_abs_score), abs(max_abs_score))

        passes_emission = hybrid >= rq_cfg.min_for_emission

        results.append(
            {
                "symbol": raw_payload.symbol,
                "direction": raw_payload.direction,
                "regime": raw_payload.regime,
                "hybrid_score": hybrid,
                "passes_emission": passes_emission,
                "vol_regime_modifier": vol_regime_modifier,
                "router_quality": {
                    "score": rq_score,
                    "multiplier": rq_multiplier,
                },
                "weights": dict(weights),
                "factor_vector": factors_for_score,
                "factors": factors_for_score,
            }
        )

    return results


__all__ = [
    "HybridFactorPayload",
    "compute_hybrid_scores",
]
