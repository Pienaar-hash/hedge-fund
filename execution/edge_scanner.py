"""
v7.7_P4 — EdgeScanner Surface & Research Layer.

Provides a unified, structured view of:
- Factor edges (IR, PnL contribution, weights)
- Symbol edges (hybrid scores, conviction, recent PnL)
- Category edges (momentum, IR, aggregate PnL)
- Regime context (vol regime, DD state, risk mode)
- Router execution quality context
- v7.7_P7: Strategy Health Score & Alpha Attribution Index
- v7.8_P1: Meta-Learning Weight Scheduler integration

This module is RESEARCH-ONLY and does NOT influence execution.
It reads from existing state surfaces and produces a single
research snapshot: logs/state/edge_insights.json

Single writer rule: Only executor/intel pipeline may write this file.
v7.8_P1: Also writes logs/state/meta_scheduler.json (single writer).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDGE_INSIGHTS_PATH = Path("logs/state/edge_insights.json")
DEFAULT_FACTOR_DIAGNOSTICS_PATH = Path("logs/state/factor_diagnostics.json")
DEFAULT_SYMBOL_SCORES_PATH = Path("logs/state/symbol_scores_v6.json")
DEFAULT_ROUTER_HEALTH_PATH = Path("logs/state/router_health.json")
DEFAULT_RISK_SNAPSHOT_PATH = Path("logs/state/risk_snapshot.json")

TOP_N = 3  # Number of top/bottom entries to include in summaries


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class EdgeSummary:
    """Summary of top and bottom edges across factors, symbols, and categories."""

    top_factors: List[Dict[str, Any]] = field(default_factory=list)
    weak_factors: List[Dict[str, Any]] = field(default_factory=list)
    top_symbols: List[Dict[str, Any]] = field(default_factory=list)
    weak_symbols: List[Dict[str, Any]] = field(default_factory=list)
    top_categories: List[Dict[str, Any]] = field(default_factory=list)
    weak_categories: List[Dict[str, Any]] = field(default_factory=list)
    regime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_factors": self.top_factors,
            "weak_factors": self.weak_factors,
            "top_symbols": self.top_symbols,
            "weak_symbols": self.weak_symbols,
            "top_categories": self.top_categories,
            "weak_categories": self.weak_categories,
            "regime": self.regime,
        }


# ---------------------------------------------------------------------------
# v7.7_P7: Strategy Health Score
# ---------------------------------------------------------------------------


@dataclass
class StrategyHealth:
    """
    Strategy Health Score & Alpha Attribution Index (v7.7_P7).

    A single, interpretable, research-grade score representing:
    - Factor quality
    - Symbol quality
    - Category quality
    - Router health
    - Regime alignment
    - Stability of hybrid scoring

    This is the hedge fund's "System Health Metric" — used internally,
    shown on dashboards, and eventually linked to auto-throttling, alerts,
    and risk overlays in v7.8–v8.0.
    """

    health_score: float = 0.0
    factor_health: Dict[str, Any] = field(default_factory=dict)
    symbol_health: Dict[str, Any] = field(default_factory=dict)
    category_health: Dict[str, Any] = field(default_factory=dict)
    regime_alignment: Dict[str, Any] = field(default_factory=dict)
    execution_quality: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "health_score": round(self.health_score, 4),
            "factor_health": self.factor_health,
            "symbol_health": self.symbol_health,
            "category_health": self.category_health,
            "regime_alignment": self.regime_alignment,
            "execution_quality": self.execution_quality,
            "notes": self.notes,
        }


@dataclass
class EdgeInsights:
    """Complete edge insights snapshot."""

    updated_ts: str = ""
    edge_summary: EdgeSummary = field(default_factory=EdgeSummary)
    factor_edges: Dict[str, Any] = field(default_factory=dict)
    symbol_edges: Dict[str, Any] = field(default_factory=dict)
    category_edges: Dict[str, Any] = field(default_factory=dict)
    config_echo: Dict[str, Any] = field(default_factory=dict)
    # v7.7_P6: Regime adjustments applied to conviction and factor weights
    regime_adjustments: Dict[str, Any] = field(default_factory=dict)
    # v7.7_P7: Strategy health score and alpha attribution
    strategy_health: Optional[StrategyHealth] = None
    # v7.8_P5: Cross-pair statistical arbitrage edges
    pair_edges: Optional[Dict[str, Any]] = None
    # v7.8_P6: Sentinel-X regime classifier
    sentinel_x: Optional[Dict[str, Any]] = None
    # v7.8_P7: Alpha decay summary
    alpha_decay_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "updated_ts": self.updated_ts,
            "edge_summary": self.edge_summary.to_dict(),
            "factor_edges": self.factor_edges,
            "symbol_edges": self.symbol_edges,
            "category_edges": self.category_edges,
            "config_echo": self.config_echo,
            "regime_adjustments": self.regime_adjustments,  # v7.7_P6
        }
        # v7.7_P7: Include strategy_health if available
        if self.strategy_health is not None:
            result["strategy_health"] = self.strategy_health.to_dict()
        # v7.8_P5: Include pair_edges if available
        if self.pair_edges is not None:
            result["pair_edges"] = self.pair_edges
        # v7.8_P6: Include sentinel_x if available
        if self.sentinel_x is not None:
            result["sentinel_x"] = self.sentinel_x
        # v7.8_P7: Include alpha_decay_summary if available
        if self.alpha_decay_summary is not None:
            result["alpha_decay_summary"] = self.alpha_decay_summary
        return result


@dataclass
class EdgeScannerConfig:
    """Configuration for edge scanner."""

    enabled: bool = True
    top_n: int = TOP_N
    factor_ir_threshold: float = 0.1  # Min IR to be considered "edge"
    symbol_score_threshold: float = 0.3  # Min hybrid score for edge
    category_momentum_threshold: float = 0.1  # Min category momentum for edge


# ---------------------------------------------------------------------------
# State Loaders (Read-Only)
# ---------------------------------------------------------------------------


def load_factor_diagnostics(
    path: Path | str = DEFAULT_FACTOR_DIAGNOSTICS_PATH,
) -> Dict[str, Any]:
    """
    Load factor diagnostics state (read-only).

    Returns empty dict if file doesn't exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_symbol_scores(
    path: Path | str = DEFAULT_SYMBOL_SCORES_PATH,
) -> Dict[str, Any]:
    """
    Load symbol scores state (read-only).

    Returns empty dict if file doesn't exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_router_health(
    path: Path | str = DEFAULT_ROUTER_HEALTH_PATH,
) -> Dict[str, Any]:
    """
    Load router health state (read-only).

    Returns empty dict if file doesn't exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_risk_snapshot(
    path: Path | str = DEFAULT_RISK_SNAPSHOT_PATH,
) -> Dict[str, Any]:
    """
    Load risk snapshot state (read-only).

    Returns empty dict if file doesn't exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_category_momentum_snapshot(
    path: Path | str | None = None,
    symbol_scores: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Load category momentum data.

    First attempts to load from dedicated state file (if exists).
    Falls back to extracting from symbol_scores if available.
    Returns empty dict otherwise.
    """
    # Try dedicated category momentum file first
    if path:
        p = Path(path)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, IOError):
                pass

    # Fall back to extracting from symbol_scores
    if symbol_scores:
        # Category momentum may be embedded in per-symbol data
        cat_data: Dict[str, Any] = {}
        per_symbol = symbol_scores.get("per_symbol", {})
        for sym, data in per_symbol.items():
            if isinstance(data, dict):
                cat_mom = data.get("category_momentum", 0.0)
                category = data.get("category", "OTHER")
                if category not in cat_data:
                    cat_data[category] = {
                        "symbols": [],
                        "momentum_scores": [],
                        "mean_return": 0.0,
                        "ir": 0.0,
                    }
                cat_data[category]["symbols"].append(sym)
                cat_data[category]["momentum_scores"].append(cat_mom)

        # Compute aggregates
        for cat, info in cat_data.items():
            scores = info["momentum_scores"]
            if scores:
                info["avg_momentum"] = float(np.mean(scores))
            else:
                info["avg_momentum"] = 0.0

        return {"category_stats": cat_data}

    return {}


# ---------------------------------------------------------------------------
# Edge Computation Pipeline
# ---------------------------------------------------------------------------


def _zscore(values: List[float]) -> List[float]:
    """Compute z-scores for a list of values."""
    if not values or len(values) < 2:
        return [0.0] * len(values)
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-9:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def compute_factor_edges(
    factor_diagnostics: Dict[str, Any],
    config: EdgeScannerConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute factor edge scores from factor diagnostics.

    Returns:
        Tuple of (factor_edges dict, top_factors list, weak_factors list)
    """
    factor_edges: Dict[str, Any] = {}

    # Extract factor-level metrics
    weights = factor_diagnostics.get("weights", {})
    if isinstance(weights, dict):
        # May be nested under "smoothed" or "raw"
        weights = weights.get("smoothed", weights.get("raw", weights))

    factor_ir = factor_diagnostics.get("factor_ir", {})
    pnl_attribution = factor_diagnostics.get("pnl_attribution", {})
    factor_vols = factor_diagnostics.get("factor_volatilities", {})

    # Collect all factors
    all_factors = set()
    all_factors.update(weights.keys() if isinstance(weights, dict) else [])
    all_factors.update(factor_ir.keys() if isinstance(factor_ir, dict) else [])
    all_factors.update(pnl_attribution.keys() if isinstance(pnl_attribution, dict) else [])

    if not all_factors:
        return {}, [], []

    # Build per-factor metrics
    factor_list: List[Dict[str, Any]] = []
    for f in all_factors:
        ir = float(factor_ir.get(f, 0.0)) if isinstance(factor_ir, dict) else 0.0
        weight = float(weights.get(f, 0.0)) if isinstance(weights, dict) else 0.0
        pnl = float(pnl_attribution.get(f, 0.0)) if isinstance(pnl_attribution, dict) else 0.0
        vol = float(factor_vols.get(f, 0.0)) if isinstance(factor_vols, dict) else 0.0

        factor_list.append({
            "factor": f,
            "ir": round(ir, 4),
            "weight": round(weight, 4),
            "pnl_contrib": round(pnl, 6),
            "volatility": round(vol, 6),
        })

    # Compute z-scores for edge scoring
    irs = [f["ir"] for f in factor_list]
    pnls = [f["pnl_contrib"] for f in factor_list]
    weights_list = [f["weight"] for f in factor_list]

    ir_z = _zscore(irs)
    pnl_z = _zscore(pnls)
    weight_z = _zscore(weights_list)

    # Compute composite edge score
    for i, f in enumerate(factor_list):
        edge_score = (ir_z[i] + pnl_z[i] + weight_z[i]) / 3.0
        f["edge_score"] = round(edge_score, 4)
        f["ir_z"] = round(ir_z[i], 4)
        f["pnl_z"] = round(pnl_z[i], 4)
        f["weight_z"] = round(weight_z[i], 4)
        factor_edges[f["factor"]] = f

    # Sort by edge score
    sorted_factors = sorted(factor_list, key=lambda x: x["edge_score"], reverse=True)

    top_factors = sorted_factors[: config.top_n]
    weak_factors = sorted_factors[-config.top_n :] if len(sorted_factors) > config.top_n else []

    # Reverse weak factors so worst is first
    weak_factors = list(reversed(weak_factors))

    return factor_edges, top_factors, weak_factors


def compute_symbol_edges(
    symbol_scores: Dict[str, Any],
    config: EdgeScannerConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute symbol edge scores from symbol scores.

    Returns:
        Tuple of (symbol_edges dict, top_symbols list, weak_symbols list)
    """
    symbol_edges: Dict[str, Any] = {}

    per_symbol = symbol_scores.get("per_symbol", {})
    if not per_symbol:
        return {}, [], []

    symbol_list: List[Dict[str, Any]] = []
    for sym, data in per_symbol.items():
        if not isinstance(data, dict):
            continue

        hybrid_score = float(data.get("hybrid_score", 0.0))
        conviction = float(data.get("conviction", data.get("conviction_score", 0.5)))
        recent_pnl = float(data.get("recent_pnl", data.get("pnl", 0.0)))
        category = data.get("category", "OTHER")
        direction = data.get("direction", data.get("side", "NEUTRAL"))

        symbol_list.append({
            "symbol": sym,
            "hybrid_score": round(hybrid_score, 4),
            "conviction": round(conviction, 4),
            "recent_pnl": round(recent_pnl, 6),
            "category": category,
            "direction": direction,
        })

    if not symbol_list:
        return {}, [], []

    # Compute z-scores
    scores = [s["hybrid_score"] for s in symbol_list]
    pnls = [s["recent_pnl"] for s in symbol_list]
    convictions = [s["conviction"] for s in symbol_list]

    score_z = _zscore(scores)
    pnl_z = _zscore(pnls)
    conv_z = _zscore(convictions)

    # Composite edge score
    for i, s in enumerate(symbol_list):
        edge_score = (score_z[i] + pnl_z[i] + conv_z[i]) / 3.0
        s["edge_score"] = round(edge_score, 4)
        symbol_edges[s["symbol"]] = s

    # Sort by edge score
    sorted_symbols = sorted(symbol_list, key=lambda x: x["edge_score"], reverse=True)

    top_symbols = sorted_symbols[: config.top_n]
    weak_symbols = sorted_symbols[-config.top_n :] if len(sorted_symbols) > config.top_n else []
    weak_symbols = list(reversed(weak_symbols))

    return symbol_edges, top_symbols, weak_symbols


def compute_category_edges(
    category_momentum: Dict[str, Any],
    config: EdgeScannerConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute category edge scores from category momentum data.

    Returns:
        Tuple of (category_edges dict, top_categories list, weak_categories list)
    """
    category_edges: Dict[str, Any] = {}

    category_stats = category_momentum.get("category_stats", {})
    if not category_stats:
        return {}, [], []

    category_list: List[Dict[str, Any]] = []
    for cat, stats in category_stats.items():
        if not isinstance(stats, dict):
            continue

        ir = float(stats.get("ir", 0.0))
        momentum = float(stats.get("momentum_score", stats.get("avg_momentum", 0.0)))
        total_pnl = float(stats.get("total_pnl", 0.0))
        mean_return = float(stats.get("mean_return", 0.0))
        volatility = float(stats.get("volatility", 0.0))
        symbols = stats.get("symbols", [])

        # Sharpe proxy = mean_return / volatility (if volatility > 0)
        sharpe = mean_return / (volatility + 1e-9) if volatility > 1e-9 else 0.0

        category_list.append({
            "category": cat,
            "ir": round(ir, 4),
            "momentum": round(momentum, 4),
            "total_pnl": round(total_pnl, 6),
            "sharpe_proxy": round(sharpe, 4),
            "symbol_count": len(symbols) if isinstance(symbols, list) else 0,
        })

    if not category_list:
        return {}, [], []

    # Z-score for edge scoring
    irs = [c["ir"] for c in category_list]
    moms = [c["momentum"] for c in category_list]
    pnls = [c["total_pnl"] for c in category_list]

    ir_z = _zscore(irs)
    mom_z = _zscore(moms)
    pnl_z = _zscore(pnls)

    for i, c in enumerate(category_list):
        edge_score = (ir_z[i] + mom_z[i] + pnl_z[i]) / 3.0
        c["edge_score"] = round(edge_score, 4)
        category_edges[c["category"]] = c

    sorted_cats = sorted(category_list, key=lambda x: x["edge_score"], reverse=True)

    top_categories = sorted_cats[: config.top_n]
    weak_categories = sorted_cats[-config.top_n :] if len(sorted_cats) > config.top_n else []
    weak_categories = list(reversed(weak_categories))

    return category_edges, top_categories, weak_categories


def extract_regime_context(
    risk_snapshot: Dict[str, Any],
    router_health: Dict[str, Any],
    symbol_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract regime context from risk/router/symbol state.

    Returns:
        Dict with dd_state, risk_mode, vol_regime, router_quality
    """
    regime: Dict[str, Any] = {}

    # From risk snapshot
    regime["dd_state"] = risk_snapshot.get("dd_state", "normal")
    regime["risk_mode"] = risk_snapshot.get("risk_mode", "normal")
    regime["current_dd_pct"] = round(float(risk_snapshot.get("current_dd_pct", 0.0)), 4)
    regime["portfolio_var"] = round(float(risk_snapshot.get("portfolio_var", 0.0)), 4)
    regime["portfolio_cvar"] = round(float(risk_snapshot.get("portfolio_cvar", 0.0)), 4)

    # Vol regime from symbol_scores global or individual symbols
    global_data = symbol_scores.get("global", {})
    vol_regime = global_data.get("vol_regime", global_data.get("vol_regime_label", "normal"))
    regime["vol_regime"] = vol_regime

    # Router quality from router_health
    global_router = router_health.get("global", router_health)
    regime["router_quality"] = round(float(global_router.get("quality_score", 0.8)), 4)
    regime["avg_slippage_bps"] = round(float(global_router.get("avg_slippage_bps", 0.0)), 2)
    regime["maker_rate"] = round(float(global_router.get("maker_rate", 0.0)), 4)

    return regime


def _extract_regime_adjustments(
    factor_diagnostics: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    regime: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract regime adjustment metadata for transparency (v7.7_P6).

    Gathers regime modifiers from factor weights snapshot and computes
    conviction regime multipliers for display.

    Args:
        factor_diagnostics: Factor diagnostics state dict
        risk_snapshot: Risk snapshot state dict
        regime: Already-extracted regime context

    Returns:
        Dict with conviction and factor_weights regime adjustments
    """
    result: Dict[str, Any] = {}

    # Get current regime values
    vol_regime = str(regime.get("vol_regime", "normal")).upper()
    dd_state = str(regime.get("dd_state", "normal")).upper()

    # Default regime curves for conviction (from conviction_engine defaults)
    conviction_vol_curves = {
        "LOW": 1.10,
        "NORMAL": 1.00,
        "HIGH": 0.75,
        "CRISIS": 0.40,
    }
    conviction_dd_curves = {
        "NORMAL": 1.00,
        "RECOVERY": 0.85,
        "DRAWDOWN": 0.50,
    }

    conv_vol_mult = conviction_vol_curves.get(vol_regime, 1.0)
    conv_dd_mult = conviction_dd_curves.get(dd_state, 1.0)

    result["conviction"] = {
        "vol_regime": vol_regime,
        "vol_multiplier": conv_vol_mult,
        "dd_state": dd_state,
        "dd_multiplier": conv_dd_mult,
        "combined_multiplier": round(conv_vol_mult * conv_dd_mult, 4),
    }

    # Extract factor weights regime modifiers if available
    factor_weights = factor_diagnostics.get("factor_weights", {})
    if isinstance(factor_weights, dict):
        fw_regime_mods = factor_weights.get("regime_modifiers", {})
        if fw_regime_mods:
            result["factor_weights"] = fw_regime_mods
        else:
            # Use defaults if not present
            factor_vol_curves = {
                "LOW": 1.05,
                "NORMAL": 1.00,
                "HIGH": 0.90,
                "CRISIS": 0.70,
            }
            factor_dd_curves = {
                "NORMAL": 1.00,
                "RECOVERY": 0.90,
                "DRAWDOWN": 0.65,
            }
            fw_vol_mult = factor_vol_curves.get(vol_regime, 1.0)
            fw_dd_mult = factor_dd_curves.get(dd_state, 1.0)
            result["factor_weights"] = {
                "vol_regime": vol_regime,
                "vol_multiplier": fw_vol_mult,
                "dd_state": dd_state,
                "dd_multiplier": fw_dd_mult,
                "combined_multiplier": round(fw_vol_mult * fw_dd_mult, 4),
            }

    return result


# ---------------------------------------------------------------------------
# v7.7_P7: Strategy Health Score Computation
# ---------------------------------------------------------------------------

# Default weights for health score components
DEFAULT_HEALTH_WEIGHTS = {
    "factor_quality": 0.30,
    "symbol_quality": 0.25,
    "category_quality": 0.15,
    "regime_alignment": 0.20,
    "execution_quality": 0.10,
}


def _compute_factor_health(
    factor_edges: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute factor health from factor edge scores.

    Returns:
        Tuple of (quality_score, health_dict)
    """
    if not factor_edges:
        return 0.5, {
            "mean_edge": 0.0,
            "pct_negative": 0.0,
            "strength_label": "unknown",
            "factor_count": 0,
        }

    edge_scores = [f.get("edge_score", 0.0) for f in factor_edges.values()]
    irs = [f.get("ir", 0.0) for f in factor_edges.values()]

    if not edge_scores:
        return 0.5, {"mean_edge": 0.0, "pct_negative": 0.0, "strength_label": "unknown", "factor_count": 0}

    mean_edge = float(np.mean(edge_scores))
    mean_ir = float(np.mean(irs)) if irs else 0.0
    negative_count = sum(1 for e in edge_scores if e < 0)
    pct_negative = negative_count / len(edge_scores) if edge_scores else 0.0

    # Quality score: normalize mean_edge from [-2, 2] to [0, 1], with penalty for negative factors
    # Base quality from normalized edge score
    normalized_edge = (mean_edge + 2.0) / 4.0  # Map [-2, 2] → [0, 1]
    normalized_edge = max(0.0, min(1.0, normalized_edge))

    # Apply penalty if >40% of factors have negative edge
    penalty = 0.0
    if pct_negative > 0.40:
        penalty = 0.15 * (pct_negative - 0.40) / 0.60  # Max ~0.15 penalty at 100% negative

    quality = max(0.0, normalized_edge - penalty)

    # Strength label
    if quality >= 0.7 and pct_negative < 0.25:
        strength_label = "strong"
    elif quality >= 0.4 or pct_negative < 0.50:
        strength_label = "mixed"
    else:
        strength_label = "weak"

    return quality, {
        "mean_edge": round(mean_edge, 4),
        "mean_ir": round(mean_ir, 4),
        "pct_negative": round(pct_negative, 4),
        "strength_label": strength_label,
        "factor_count": len(edge_scores),
    }


def _compute_symbol_health(
    symbol_edges: Dict[str, Any],
    top_n: int = 3,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute symbol health from symbol edge scores.

    Returns:
        Tuple of (quality_score, health_dict)
    """
    if not symbol_edges:
        return 0.5, {
            "mean_edge": 0.0,
            "top_contributors": [],
            "draggers": [],
            "symbol_count": 0,
        }

    # Extract scores, weighted by conviction
    entries = []
    for sym, data in symbol_edges.items():
        if isinstance(data, dict):
            edge_score = data.get("edge_score", 0.0)
            conviction = data.get("conviction", 0.5)
            hybrid_score = data.get("hybrid_score", 0.0)
            entries.append({
                "symbol": sym,
                "edge_score": edge_score,
                "conviction": conviction,
                "hybrid_score": hybrid_score,
                "weighted_score": edge_score * (0.5 + 0.5 * conviction),  # Conviction-weighted
            })

    if not entries:
        return 0.5, {"mean_edge": 0.0, "top_contributors": [], "draggers": [], "symbol_count": 0}

    # Sort by weighted score
    sorted_entries = sorted(entries, key=lambda x: x["weighted_score"], reverse=True)

    mean_edge = float(np.mean([e["edge_score"] for e in entries]))
    mean_weighted = float(np.mean([e["weighted_score"] for e in entries]))

    # Normalize mean weighted from [-2, 2] to [0, 1]
    quality = (mean_weighted + 2.0) / 4.0
    quality = max(0.0, min(1.0, quality))

    top_contributors = [
        {"symbol": e["symbol"], "edge_score": round(e["edge_score"], 4), "conviction": round(e["conviction"], 4)}
        for e in sorted_entries[:top_n]
    ]
    draggers = [
        {"symbol": e["symbol"], "edge_score": round(e["edge_score"], 4), "conviction": round(e["conviction"], 4)}
        for e in sorted_entries[-top_n:]
    ]
    draggers = list(reversed(draggers))

    return quality, {
        "mean_edge": round(mean_edge, 4),
        "mean_weighted": round(mean_weighted, 4),
        "top_contributors": top_contributors,
        "draggers": draggers,
        "symbol_count": len(entries),
    }


def _compute_category_health(
    category_edges: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute category health from category edge scores.

    Returns:
        Tuple of (quality_score, health_dict)
    """
    if not category_edges:
        return 0.5, {
            "mean_edge": 0.0,
            "strongest_category": None,
            "weakest_category": None,
            "category_count": 0,
        }

    entries = []
    for cat, data in category_edges.items():
        if isinstance(data, dict):
            edge_score = data.get("edge_score", 0.0)
            momentum = data.get("momentum", 0.0)
            ir = data.get("ir", 0.0)
            entries.append({
                "category": cat,
                "edge_score": edge_score,
                "momentum": momentum,
                "ir": ir,
            })

    if not entries:
        return 0.5, {"mean_edge": 0.0, "strongest_category": None, "weakest_category": None, "category_count": 0}

    sorted_entries = sorted(entries, key=lambda x: x["edge_score"], reverse=True)
    mean_edge = float(np.mean([e["edge_score"] for e in entries]))

    # Normalize
    quality = (mean_edge + 2.0) / 4.0
    quality = max(0.0, min(1.0, quality))

    strongest = sorted_entries[0] if sorted_entries else None
    weakest = sorted_entries[-1] if sorted_entries else None

    return quality, {
        "mean_edge": round(mean_edge, 4),
        "strongest_category": strongest["category"] if strongest else None,
        "strongest_edge": round(strongest["edge_score"], 4) if strongest else 0.0,
        "weakest_category": weakest["category"] if weakest else None,
        "weakest_edge": round(weakest["edge_score"], 4) if weakest else 0.0,
        "category_count": len(entries),
    }


def _compute_regime_alignment(
    regime: Dict[str, Any],
    regime_adjustments: Dict[str, Any],
    factor_health: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute regime alignment score.

    Measures how well the system is adapting to the current regime:
    - In HIGH/CRISIS vol, expect trend/momentum factors to be suppressed
    - In DRAWDOWN, expect overall conviction to be reduced
    - Misalignment = strong trend factors in crisis, or no suppression in drawdown

    Returns:
        Tuple of (alignment_score, alignment_dict)
    """
    vol_regime = str(regime.get("vol_regime", "normal")).upper()
    dd_state = str(regime.get("dd_state", "normal")).upper()

    # Get conviction suppression from regime adjustments
    conviction_adj = regime_adjustments.get("conviction", {})
    conv_combined = conviction_adj.get("combined_multiplier", 1.0)

    factor_adj = regime_adjustments.get("factor_weights", {})
    factor_combined = factor_adj.get("combined_multiplier", 1.0)

    # Expected behavior by regime
    expected_multipliers = {
        # (vol_regime, dd_state) -> expected_range for conviction
        ("CRISIS", "DRAWDOWN"): (0.15, 0.35),  # Should be heavily suppressed
        ("CRISIS", "NORMAL"): (0.35, 0.50),
        ("HIGH", "DRAWDOWN"): (0.30, 0.50),
        ("HIGH", "NORMAL"): (0.70, 0.85),
        ("NORMAL", "DRAWDOWN"): (0.45, 0.60),
        ("NORMAL", "RECOVERY"): (0.80, 0.95),
        ("NORMAL", "NORMAL"): (0.95, 1.05),
        ("LOW", "NORMAL"): (1.00, 1.15),
    }

    key = (vol_regime, dd_state)
    expected_range = expected_multipliers.get(key, (0.9, 1.1))

    # Check if actual multiplier is within expected range
    in_range = expected_range[0] <= conv_combined <= expected_range[1]
    if in_range:
        alignment = 1.0
    else:
        # Distance from range
        if conv_combined < expected_range[0]:
            distance = expected_range[0] - conv_combined
        else:
            distance = conv_combined - expected_range[1]
        # Penalize based on distance (max ~0.3 penalty)
        alignment = max(0.0, 1.0 - distance * 2)

    # Bonus for strong factors in good regime, penalty for strong factors in crisis
    factor_strength = factor_health.get("strength_label", "mixed")
    if vol_regime in ("CRISIS", "HIGH") and factor_strength == "strong":
        # Potential misalignment: strong factors in volatile regime
        alignment *= 0.85
    elif vol_regime in ("LOW", "NORMAL") and factor_strength == "weak":
        # Weak factors when regime is favorable
        alignment *= 0.90

    return alignment, {
        "vol_regime": vol_regime,
        "dd_state": dd_state,
        "conviction_multiplier": round(conv_combined, 4),
        "factor_multiplier": round(factor_combined, 4),
        "expected_range": [round(expected_range[0], 2), round(expected_range[1], 2)],
        "in_expected_range": in_range,
        "alignment_score": round(alignment, 4),
    }


def _compute_execution_quality(
    regime: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute execution quality from router health metrics.

    Returns:
        Tuple of (quality_score, quality_dict)
    """
    router_quality = regime.get("router_quality", 0.8)
    avg_slippage = regime.get("avg_slippage_bps", 0.0)
    maker_rate = regime.get("maker_rate", 0.0)

    # Base quality from router
    quality = float(router_quality)

    # Slippage penalty: >10 bps → minor penalty, >30 bps → significant
    if avg_slippage > 30:
        quality *= 0.85
    elif avg_slippage > 10:
        quality *= 0.95

    # Maker rate bonus: higher maker rate = better quality
    if maker_rate > 0.7:
        quality = min(1.0, quality * 1.05)
    elif maker_rate < 0.3:
        quality *= 0.95

    quality = max(0.0, min(1.0, quality))

    # Quality bucket
    if quality >= 0.8:
        bucket = "excellent"
    elif quality >= 0.6:
        bucket = "good"
    elif quality >= 0.4:
        bucket = "degraded"
    else:
        bucket = "poor"

    return quality, {
        "router_quality": round(float(router_quality), 4),
        "avg_slippage_bps": round(avg_slippage, 2),
        "maker_rate": round(maker_rate, 4),
        "quality_bucket": bucket,
        "adjusted_quality": round(quality, 4),
    }


def compute_strategy_health(
    factor_edges: Dict[str, Any],
    symbol_edges: Dict[str, Any],
    category_edges: Dict[str, Any],
    regime: Dict[str, Any],
    regime_adjustments: Dict[str, Any],
    weights: Dict[str, float] | None = None,
) -> StrategyHealth:
    """
    Compute the Strategy Health Score (v7.7_P7).

    Aggregates:
    - Factor edge quality (30%)
    - Symbol edge quality (25%)
    - Category rotation quality (15%)
    - Regime alignment (20%)
    - Execution quality (10%)

    Args:
        factor_edges: Factor edge scores dict
        symbol_edges: Symbol edge scores dict
        category_edges: Category edge scores dict
        regime: Regime context dict
        regime_adjustments: Regime adjustments dict (from P6)
        weights: Optional custom weights for components

    Returns:
        StrategyHealth with aggregate score and component breakdowns
    """
    if weights is None:
        weights = DEFAULT_HEALTH_WEIGHTS

    notes: List[str] = []

    # Compute component health scores
    factor_quality, factor_health = _compute_factor_health(factor_edges)
    symbol_quality, symbol_health = _compute_symbol_health(symbol_edges)
    category_quality, category_health = _compute_category_health(category_edges)
    regime_alignment_score, regime_alignment = _compute_regime_alignment(
        regime, regime_adjustments, factor_health
    )
    execution_quality_score, execution_quality = _compute_execution_quality(regime)

    # Aggregate health score
    health_score = (
        weights.get("factor_quality", 0.30) * factor_quality +
        weights.get("symbol_quality", 0.25) * symbol_quality +
        weights.get("category_quality", 0.15) * category_quality +
        weights.get("regime_alignment", 0.20) * regime_alignment_score +
        weights.get("execution_quality", 0.10) * execution_quality_score
    )
    health_score = max(0.0, min(1.0, health_score))

    # Generate notes based on component scores
    if factor_health.get("strength_label") == "strong":
        notes.append("Factors strong")
    elif factor_health.get("strength_label") == "weak":
        notes.append("Factors weak — review factor weights")

    if factor_health.get("pct_negative", 0) > 0.5:
        notes.append(f"Warning: {factor_health['pct_negative']:.0%} of factors have negative edge")

    if symbol_health.get("symbol_count", 0) < 5:
        notes.append("Limited symbol coverage")

    if category_health.get("category_count", 0) < 3:
        notes.append("Limited category diversification")

    if not regime_alignment.get("in_expected_range", True):
        notes.append(f"Regime adjustment outside expected range for {regime_alignment.get('vol_regime')}/{regime_alignment.get('dd_state')}")

    if execution_quality.get("quality_bucket") == "degraded":
        notes.append("Router quality degrading")
    elif execution_quality.get("quality_bucket") == "poor":
        notes.append("Router quality poor — check execution")

    if execution_quality.get("avg_slippage_bps", 0) > 20:
        notes.append(f"Elevated slippage: {execution_quality['avg_slippage_bps']:.1f} bps")

    # Overall health assessment
    if health_score >= 0.75:
        notes.insert(0, "✅ System healthy")
    elif health_score >= 0.50:
        notes.insert(0, "⚠️ System fair — monitor closely")
    else:
        notes.insert(0, "🚨 System stressed — consider risk reduction")

    return StrategyHealth(
        health_score=health_score,
        factor_health=factor_health,
        symbol_health=symbol_health,
        category_health=category_health,
        regime_alignment=regime_alignment,
        execution_quality=execution_quality,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Snapshot Builder
# ---------------------------------------------------------------------------


def build_edge_insights_snapshot(
    factor_diagnostics: Dict[str, Any] | None = None,
    symbol_scores: Dict[str, Any] | None = None,
    router_health: Dict[str, Any] | None = None,
    risk_snapshot: Dict[str, Any] | None = None,
    category_momentum: Dict[str, Any] | None = None,
    config: EdgeScannerConfig | None = None,
) -> EdgeInsights:
    """
    Build a complete EdgeInsights snapshot from existing state surfaces.

    All inputs are optional; will load from default paths if not provided.
    This function is READ-ONLY and does not modify any state.

    Args:
        factor_diagnostics: Factor diagnostics state dict
        symbol_scores: Symbol scores state dict
        router_health: Router health state dict
        risk_snapshot: Risk snapshot state dict
        category_momentum: Category momentum state dict
        config: Edge scanner config

    Returns:
        EdgeInsights snapshot
    """
    if config is None:
        config = EdgeScannerConfig()

    # Load from files if not provided
    if factor_diagnostics is None:
        factor_diagnostics = load_factor_diagnostics()
    if symbol_scores is None:
        symbol_scores = load_symbol_scores()
    if router_health is None:
        router_health = load_router_health()
    if risk_snapshot is None:
        risk_snapshot = load_risk_snapshot()
    if category_momentum is None:
        category_momentum = load_category_momentum_snapshot(symbol_scores=symbol_scores)

    # Compute edges
    factor_edges, top_factors, weak_factors = compute_factor_edges(factor_diagnostics, config)
    symbol_edges, top_symbols, weak_symbols = compute_symbol_edges(symbol_scores, config)
    category_edges, top_categories, weak_categories = compute_category_edges(category_momentum, config)

    # Extract regime context
    regime = extract_regime_context(risk_snapshot, router_health, symbol_scores)

    # Build summary
    edge_summary = EdgeSummary(
        top_factors=top_factors,
        weak_factors=weak_factors,
        top_symbols=top_symbols,
        weak_symbols=weak_symbols,
        top_categories=top_categories,
        weak_categories=weak_categories,
        regime=regime,
    )

    # Config echo for reproducibility
    config_echo = {
        "top_n": config.top_n,
        "factor_ir_threshold": config.factor_ir_threshold,
        "symbol_score_threshold": config.symbol_score_threshold,
        "category_momentum_threshold": config.category_momentum_threshold,
        "source_files": {
            "factor_diagnostics": str(DEFAULT_FACTOR_DIAGNOSTICS_PATH),
            "symbol_scores": str(DEFAULT_SYMBOL_SCORES_PATH),
            "router_health": str(DEFAULT_ROUTER_HEALTH_PATH),
            "risk_snapshot": str(DEFAULT_RISK_SNAPSHOT_PATH),
        },
    }

    # v7.7_P6: Extract regime adjustments from factor_diagnostics if available
    regime_adjustments = _extract_regime_adjustments(factor_diagnostics, risk_snapshot, regime)

    # v7.7_P7: Compute strategy health score
    strategy_health = compute_strategy_health(
        factor_edges=factor_edges,
        symbol_edges=symbol_edges,
        category_edges=category_edges,
        regime=regime,
        regime_adjustments=regime_adjustments,
    )

    return EdgeInsights(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        edge_summary=edge_summary,
        factor_edges=factor_edges,
        symbol_edges=symbol_edges,
        category_edges=category_edges,
        config_echo=config_echo,
        regime_adjustments=regime_adjustments,  # v7.7_P6
        strategy_health=strategy_health,  # v7.7_P7
    )


# ---------------------------------------------------------------------------
# State Writer
# ---------------------------------------------------------------------------


def write_edge_insights(
    snapshot: EdgeInsights,
    path: Path | str = DEFAULT_EDGE_INSIGHTS_PATH,
) -> None:
    """
    Write edge insights snapshot to file.

    This is the ONLY allowed writer for edge_insights.json.
    Must be called from executor/intel pipeline only.

    Args:
        snapshot: EdgeInsights snapshot to write
        path: Output path (default: logs/state/edge_insights.json)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tmp = path.with_suffix(".tmp")
        payload = snapshot.to_dict()
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(path)
    except Exception:
        # Fail silently - research surface, not critical
        pass


def load_edge_insights(
    path: Path | str = DEFAULT_EDGE_INSIGHTS_PATH,
) -> Dict[str, Any]:
    """
    Load edge insights state (read-only).

    Returns empty dict if file doesn't exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# v7.8_P1: Meta-Scheduler Integration
# ---------------------------------------------------------------------------


def update_meta_scheduler(
    edge_insights: EdgeInsights,
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Update meta-scheduler state based on edge insights.
    
    This is the ONLY place that writes meta_scheduler.json.
    Called after building EdgeInsights snapshot.
    
    v7.8_P1: Meta-Learning Weight Scheduler integration.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        strategy_config: Strategy config for loading meta_scheduler config
    """
    try:
        from execution.meta_scheduler import (
            MetaSchedulerConfig,
            load_meta_scheduler_config,
            load_meta_scheduler_state,
            write_meta_scheduler_state,
            meta_learning_step,
        )
    except ImportError:
        # meta_scheduler not available - skip silently
        return
    
    # Load config
    cfg = load_meta_scheduler_config(strategy_config)
    
    # If disabled, don't update
    if not cfg.enabled:
        return
    
    # Load previous state
    prev_state = load_meta_scheduler_state()
    
    # Extract data for learning step
    factor_edges = edge_insights.factor_edges
    category_edges = edge_insights.category_edges
    strategy_health = None
    if edge_insights.strategy_health is not None:
        strategy_health = edge_insights.strategy_health.to_dict()
    
    # Perform learning step
    new_state = meta_learning_step(
        cfg=cfg,
        prev_state=prev_state,
        factor_edges=factor_edges,
        category_edges=category_edges,
        strategy_health=strategy_health,
    )
    
    # Write updated state
    write_meta_scheduler_state(new_state)


def update_universe_optimizer(
    edge_insights: EdgeInsights,
    strategy_config: Optional[dict] = None,
    risk_snapshot: Optional[dict] = None,
    alpha_router_allocation: Optional[float] = None,
) -> None:
    """
    Update universe optimizer state based on edge insights.
    
    This is the ONLY place that writes universe_optimizer.json.
    Called after meta_scheduler update in the intel loop.
    
    v7.8_P3: Universe Optimizer (Curator) integration.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        strategy_config: Strategy config for loading universe_optimizer config
        risk_snapshot: Optional risk snapshot for DD state
        alpha_router_allocation: Optional allocation from alpha_router
    """
    try:
        from execution.universe_optimizer import (
            load_universe_optimizer_config,
            run_universe_optimizer_step,
        )
    except ImportError:
        # universe_optimizer not available - skip silently
        return
    
    # Load config to check if enabled
    cfg = load_universe_optimizer_config(strategy_config)
    
    # If disabled, don't update
    if not cfg.enabled:
        return
    
    # Build symbol edges dict
    symbol_edges: dict[str, Any] = {}
    if edge_insights.symbol_edges:
        for sym, edge in edge_insights.symbol_edges.items():
            if isinstance(edge, dict):
                symbol_edges[sym] = edge
            elif hasattr(edge, "edge"):
                symbol_edges[sym] = {"edge": edge.edge}
            else:
                symbol_edges[sym] = {"edge": 0.0}
    
    # Build category edges dict
    category_edges: dict[str, Any] = {}
    if edge_insights.category_edges:
        for cat, edge in edge_insights.category_edges.items():
            if isinstance(edge, dict):
                category_edges[cat] = edge
            elif hasattr(edge, "edge"):
                category_edges[cat] = {"edge": edge.edge}
            else:
                category_edges[cat] = {"edge": 0.0}
    
    # Extract vol regime from regime_adjustments
    vol_regime = "NORMAL"
    if edge_insights.regime_adjustments:
        vol_regime = edge_insights.regime_adjustments.get("vol_regime", "NORMAL")
    
    # Extract DD state from risk snapshot
    dd_state = "NORMAL"
    if risk_snapshot:
        dd_state = risk_snapshot.get("dd_state", "NORMAL")
    
    # Extract meta overlay if available
    meta_overlay = 1.0
    try:
        from execution.meta_scheduler import load_meta_scheduler_state
        meta_state = load_meta_scheduler_state()
        if meta_state and hasattr(meta_state, "conviction_state"):
            if hasattr(meta_state.conviction_state, "global_strength"):
                meta_overlay = meta_state.conviction_state.global_strength
    except ImportError:
        pass
    
    # Extract strategy health score
    health_score = 0.5
    if edge_insights.strategy_health is not None:
        if hasattr(edge_insights.strategy_health, "composite_score"):
            health_score = edge_insights.strategy_health.composite_score
        elif isinstance(edge_insights.strategy_health, dict):
            health_score = edge_insights.strategy_health.get("composite_score", 0.5)
    
    # Get candidate symbols from symbol edges
    candidate_symbols = list(symbol_edges.keys())
    
    # Run optimizer step (handles loading prev state, writing new state)
    run_universe_optimizer_step(
        candidate_symbols=candidate_symbols,
        symbol_edges=symbol_edges,
        category_edges=category_edges,
        vol_regime=vol_regime,
        dd_state=dd_state,
        meta_overlay=meta_overlay,
        strategy_health=health_score,
        allocation_confidence=alpha_router_allocation or 1.0,
        strategy_cfg=strategy_config,
    )


# ---------------------------------------------------------------------------
# v7.8_P4: Alpha Miner Integration
# ---------------------------------------------------------------------------

_ALPHA_MINER_CYCLE_COUNT = 0  # Module-level counter for throttling


def update_alpha_miner(
    edge_insights: EdgeInsights,
    strategy_config: Optional[dict] = None,
    force_run: bool = False,
) -> None:
    """
    Update alpha miner state to discover new candidate symbols.
    
    This is the ONLY place that writes alpha_miner.json.
    Called after universe_optimizer update in the intel loop.
    
    Alpha miner runs every N cycles (expensive operation).
    
    v7.8_P4: Autonomous Alpha Miner (Prospector) integration.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        strategy_config: Strategy config for loading alpha_miner config
        force_run: If True, run regardless of cycle counter
    """
    global _ALPHA_MINER_CYCLE_COUNT
    _ALPHA_MINER_CYCLE_COUNT += 1
    
    try:
        from execution.alpha_miner import (
            load_alpha_miner_config,
            run_alpha_miner_step,
            should_run_miner,
            load_alpha_miner_state,
        )
    except ImportError:
        # alpha_miner not available - skip silently
        return
    
    # Load config to check if enabled
    cfg = load_alpha_miner_config()
    
    # If disabled, don't update
    if not cfg.enabled:
        return
    
    # Check if we should run this cycle (throttled)
    prev_state = load_alpha_miner_state()
    if not force_run and not should_run_miner(
        _ALPHA_MINER_CYCLE_COUNT,
        cfg,
        prev_state,
    ):
        return
    
    # Run miner step (expensive - scans all exchange symbols)
    # Note: fetch_orderbook=False for performance, enable for higher quality scores
    run_alpha_miner_step(
        config=cfg,
        fetch_orderbook=False,  # Expensive - enable for production
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# v7.8_P5: Cross-Pair Engine Integration
# ---------------------------------------------------------------------------

_CROSS_PAIR_CYCLE_COUNT = 0  # Module-level counter for throttling


def update_cross_pair_engine(
    edge_insights: EdgeInsights,
    strategy_config: Optional[dict] = None,
    force_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Update cross-pair engine state to detect statistical mispricings.
    
    This is the ONLY place that writes cross_pair_edges.json.
    Called after alpha_miner update in the intel loop.
    
    Cross-pair engine runs every N cycles (moderate cost).
    
    v7.8_P5: Cross-Pair Statistical Arbitrage Engine (Crossfire) integration.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        strategy_config: Strategy config for loading cross_pair_engine config
        force_run: If True, run regardless of cycle counter
        
    Returns:
        Pair edges for EdgeInsights if run, else None
    """
    global _CROSS_PAIR_CYCLE_COUNT
    _CROSS_PAIR_CYCLE_COUNT += 1
    
    try:
        from execution.cross_pair_engine import (
            load_cross_pair_config,
            run_cross_pair_scan,
            should_run_cross_pair,
            get_pair_edges_for_insights,
        )
    except ImportError:
        # cross_pair_engine not available - skip silently
        return None
    
    # Load config to check if enabled
    cfg = load_cross_pair_config(strategy_config)
    
    # If disabled, don't update
    if not cfg.enabled:
        return None
    
    # Check if we should run this cycle (throttled)
    if not force_run and not should_run_cross_pair(
        _CROSS_PAIR_CYCLE_COUNT,
        cfg,
    ):
        # Return cached pair edges from previous run
        try:
            return get_pair_edges_for_insights()
        except Exception:
            return None
    
    # Run cross-pair scan
    state = run_cross_pair_scan(
        config=cfg,
        strategy_cfg=strategy_config,
        dry_run=False,
    )
    
    # Return pair edges for EdgeInsights
    return get_pair_edges_for_insights(state)


# ---------------------------------------------------------------------------
# v7.8_P6: Sentinel-X Integration
# ---------------------------------------------------------------------------

_SENTINEL_X_CYCLE_COUNT = 0  # Module-level counter for throttling


def update_sentinel_x(
    edge_insights: EdgeInsights,
    prices: Optional[List[float]] = None,
    strategy_config: Optional[dict] = None,
    current_dd: float = 0.0,
    force_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Update Sentinel-X regime classifier state.
    
    This is the ONLY place in edge_scanner that calls sentinel_x.
    Called after cross_pair_engine update in the intel loop.
    
    Sentinel-X runs every N cycles (moderate cost).
    
    v7.8_P6: Sentinel-X Hybrid ML Market Regime Classifier integration.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        prices: BTC/index price series (newest last)
        strategy_config: Strategy config for loading sentinel_x config
        current_dd: Current portfolio drawdown for crisis override
        force_run: If True, run regardless of cycle counter
        
    Returns:
        Sentinel-X state dict for EdgeInsights if run, else None
    """
    global _SENTINEL_X_CYCLE_COUNT
    _SENTINEL_X_CYCLE_COUNT += 1
    
    try:
        from execution.sentinel_x import (
            load_sentinel_x_config,
            load_sentinel_x_state,
            run_sentinel_x_step,
            get_sentinel_x_for_insights,
            should_run_sentinel_x,
        )
    except ImportError:
        # sentinel_x not available - skip silently
        return None
    
    # Load config to check if enabled
    cfg = load_sentinel_x_config(strategy_config)
    
    # If disabled, don't update
    if not cfg.enabled:
        return None
    
    # Check if we should run this cycle (throttled)
    if not force_run and not should_run_sentinel_x(
        _SENTINEL_X_CYCLE_COUNT,
        cfg,
    ):
        # Return cached state from previous run
        try:
            return get_sentinel_x_for_insights()
        except Exception:
            return None
    
    # Need price data to run
    if not prices or len(prices) < 20:
        # Try to return cached state
        try:
            return get_sentinel_x_for_insights()
        except Exception:
            return None
    
    # Run Sentinel-X step
    state = run_sentinel_x_step(
        prices=prices,
        cfg=cfg,
        strategy_cfg=strategy_config,
        current_dd=current_dd,
        dry_run=False,
    )
    
    if state is None:
        return None
    
    # Return sentinel_x dict for EdgeInsights
    return get_sentinel_x_for_insights(state)


# ---------------------------------------------------------------------------
# Alpha Decay Integration (v7.8_P7)
# ---------------------------------------------------------------------------

_ALPHA_DECAY_CYCLE_COUNT = 0


def load_alpha_decay_state() -> Optional[Dict[str, Any]]:
    """
    Load alpha decay state from state file.
    
    Returns:
        Alpha decay state dict or None if not available
    """
    path = Path("logs/state/alpha_decay.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def update_alpha_decay(
    edge_insights: EdgeInsights,
    strategy_config: Optional[dict] = None,
    symbol_to_category: Optional[Dict[str, str]] = None,
    force_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Update Alpha Decay state (v7.8_P7).
    
    This computes decay rates, half-lives, and survival probabilities
    for symbols, categories, and factors based on edge score trends.
    
    Args:
        edge_insights: Complete EdgeInsights snapshot
        strategy_config: Strategy config for loading alpha_decay config
        symbol_to_category: Symbol to category mapping
        force_run: If True, run regardless of cycle counter
        
    Returns:
        Alpha decay summary dict for EdgeInsights if run, else None
    """
    global _ALPHA_DECAY_CYCLE_COUNT
    _ALPHA_DECAY_CYCLE_COUNT += 1
    
    try:
        from execution.alpha_decay import (
            load_config_from_strategy,
            load_alpha_decay_state as load_decay_state,
            get_or_create_history,
            run_alpha_decay_step,
            save_alpha_decay_state,
            get_alpha_decay_summary,
            load_edge_insights as load_edge_insights_for_decay,
            load_factor_diagnostics as load_fd_for_decay,
            load_hybrid_scores,
            load_sentinel_x_state,
        )
    except ImportError:
        return None
    
    # Load config
    cfg = load_config_from_strategy()
    
    if not cfg.enabled:
        return None
    
    # Check run interval (default 10 cycles)
    run_interval = 10
    if strategy_config:
        decay_cfg = strategy_config.get("alpha_decay", {})
        run_interval = decay_cfg.get("run_interval_cycles", 10)
    
    if not force_run and (_ALPHA_DECAY_CYCLE_COUNT % run_interval) != 0:
        # Return cached state
        cached = load_decay_state()
        if cached:
            return get_alpha_decay_summary(cached)
        return None
    
    # Load inputs
    edge_insights_data = load_edge_insights_for_decay()
    factor_diag = load_fd_for_decay()
    hybrid_scores = load_hybrid_scores()
    sentinel_x = load_sentinel_x_state()
    
    # Load previous state
    prev_state = load_decay_state()
    
    # Get history
    history = get_or_create_history()
    
    # Prune old samples
    max_age = cfg.lookback_days * 86400
    history.prune_old_samples(max_age)
    
    # Run step
    from execution.alpha_decay import AlphaDecayState
    state = run_alpha_decay_step(
        config=cfg,
        history=history,
        prev_state=AlphaDecayState.from_dict(prev_state) if prev_state else None,
        edge_insights=edge_insights_data,
        factor_diagnostics=factor_diag,
        hybrid_scores=hybrid_scores,
        sentinel_x_state=sentinel_x,
        symbol_to_category=symbol_to_category,
    )
    
    # Save state
    save_alpha_decay_state(state)
    
    # Return summary for EdgeInsights
    return get_alpha_decay_summary(state)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "EdgeSummary",
    "EdgeInsights",
    "EdgeScannerConfig",
    "StrategyHealth",  # v7.7_P7
    # State loaders
    "load_factor_diagnostics",
    "load_symbol_scores",
    "load_router_health",
    "load_risk_snapshot",
    "load_category_momentum_snapshot",
    "load_edge_insights",
    # Computation
    "compute_factor_edges",
    "compute_symbol_edges",
    "compute_category_edges",
    "extract_regime_context",
    "compute_strategy_health",  # v7.7_P7
    # Snapshot builder
    "build_edge_insights_snapshot",
    # Writer
    "write_edge_insights",
    # v7.8_P1: Meta-Scheduler
    "update_meta_scheduler",
    # v7.8_P3: Universe Optimizer
    "update_universe_optimizer",
    # v7.8_P4: Alpha Miner
    "update_alpha_miner",
    # v7.8_P5: Cross-Pair Engine
    "update_cross_pair_engine",
    # v7.8_P6: Sentinel-X
    "update_sentinel_x",
    # v7.8_P7: Alpha Decay
    "update_alpha_decay",
    "load_alpha_decay_state",
]
