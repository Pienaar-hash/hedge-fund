"""
Cerberus: Multi-Strategy Portfolio Router — v7.8_P8

The Three-Headed Allocator that dynamically weights strategy "heads":

1. TREND          — Directional momentum / trend following
2. MEAN_REVERT    — Mean reversion / reversal
3. RELATIVE_VALUE — Crossfire pair trades
4. CATEGORY       — Category rotation / sector tilt
5. VOL_HARVEST    — Volatility harvesting (position sizing)
6. EMERGENT_ALPHA — Prospector / Universe expansion

Cerberus allocates *intelligence-level weights*, not trade-level orders.
Initially, routing is purely internal, adjusting:
- Factor weights
- Conviction weights
- Universe scoring weights
- Alpha router allocation weights

Architecture:
    Sentinel-X Regime  ──┐
    Alpha Decay        ──┼──► Cerberus ──► Head Multipliers ──► Downstream Overlays
    MetaScheduler      ──┤                      ↓
    Edge Scores        ──┤        logs/state/cerberus_state.json
    Strategy Health    ──┘

This module is RESEARCH-ONLY and does NOT directly place trades.
It reads from intel surfaces and produces head multipliers for downstream modules.

Single writer rule: Only executor/intel pipeline may write cerberus_state.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/cerberus_state.json")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")

_LOG = logging.getLogger(__name__)

# Canonical strategy heads
STRATEGY_HEADS = [
    "TREND",
    "MEAN_REVERT",
    "RELATIVE_VALUE",
    "CATEGORY",
    "VOL_HARVEST",
    "EMERGENT_ALPHA",
]

# Default weights for signal scoring
DEFAULT_SIGNAL_WEIGHTS = {
    "regime": 0.25,
    "decay": 0.15,
    "meta": 0.15,
    "edge": 0.20,
    "health": 0.15,
    "universe": 0.05,
    "rv": 0.05,
}

# Safety bounds
MIN_MULTIPLIER = 0.10
MAX_MULTIPLIER = 3.00


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CerberusConfig:
    """Configuration for the Cerberus Multi-Strategy Router."""

    enabled: bool = False
    strategy_heads: Dict[str, float] = field(default_factory=lambda: {
        "TREND": 1.0,
        "MEAN_REVERT": 1.0,
        "RELATIVE_VALUE": 1.0,
        "CATEGORY": 1.0,
        "VOL_HARVEST": 1.0,
        "EMERGENT_ALPHA": 1.0,
    })
    learning_rate: float = 0.05
    bounds: Dict[str, float] = field(default_factory=lambda: {
        "min": 0.25,
        "max": 2.0,
    })
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "TREND_UP": {
            "TREND": 1.3,
            "MEAN_REVERT": 0.7,
            "RELATIVE_VALUE": 1.0,
            "CATEGORY": 1.2,
            "VOL_HARVEST": 1.0,
            "EMERGENT_ALPHA": 1.0,
        },
        "TREND_DOWN": {
            "TREND": 1.2,
            "MEAN_REVERT": 0.8,
            "RELATIVE_VALUE": 1.1,
            "CATEGORY": 1.0,
            "VOL_HARVEST": 0.9,
            "EMERGENT_ALPHA": 0.8,
        },
        "MEAN_REVERT": {
            "TREND": 0.7,
            "MEAN_REVERT": 1.3,
            "RELATIVE_VALUE": 1.1,
            "CATEGORY": 1.0,
            "VOL_HARVEST": 1.0,
            "EMERGENT_ALPHA": 0.9,
        },
        "BREAKOUT": {
            "TREND": 1.4,
            "MEAN_REVERT": 0.8,
            "RELATIVE_VALUE": 1.0,
            "CATEGORY": 1.1,
            "VOL_HARVEST": 1.1,
            "EMERGENT_ALPHA": 1.1,
        },
        "CHOPPY": {
            "TREND": 0.8,
            "MEAN_REVERT": 1.1,
            "RELATIVE_VALUE": 1.2,
            "CATEGORY": 0.9,
            "VOL_HARVEST": 0.8,
            "EMERGENT_ALPHA": 0.7,
        },
        "CRISIS": {
            "TREND": 0.6,
            "MEAN_REVERT": 0.8,
            "RELATIVE_VALUE": 1.4,
            "CATEGORY": 1.0,
            "VOL_HARVEST": 0.5,
            "EMERGENT_ALPHA": 0.4,
        },
    })
    sentinel_x_integration: bool = True
    decay_integration: bool = True
    meta_scheduler_integration: bool = True
    signal_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGNAL_WEIGHTS))
    smoothing_alpha: float = 0.20
    run_interval_cycles: int = 5

    def __post_init__(self) -> None:
        """Validate and normalize config values."""
        # Ensure all canonical heads exist
        for head in STRATEGY_HEADS:
            if head not in self.strategy_heads:
                self.strategy_heads[head] = 1.0

        # Clamp bounds
        if self.bounds.get("min", 0.25) < MIN_MULTIPLIER:
            self.bounds["min"] = MIN_MULTIPLIER
        if self.bounds.get("max", 2.0) > MAX_MULTIPLIER:
            self.bounds["max"] = MAX_MULTIPLIER

        # Clamp learning rate
        if self.learning_rate < 0.001:
            self.learning_rate = 0.001
        if self.learning_rate > 0.5:
            self.learning_rate = 0.5


def load_cerberus_config(
    config_path: Path | str | None = None,
    strategy_config: Mapping[str, Any] | None = None,
) -> CerberusConfig:
    """
    Load Cerberus configuration from strategy_config.json.

    Args:
        config_path: Path to strategy_config.json
        strategy_config: Pre-loaded strategy config dict

    Returns:
        CerberusConfig instance
    """
    if strategy_config is None:
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        if cfg_path.exists():
            try:
                strategy_config = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, IOError):
                return CerberusConfig()
        else:
            return CerberusConfig()

    cerb_cfg = strategy_config.get("cerberus_router", {})
    if not isinstance(cerb_cfg, Mapping):
        return CerberusConfig()

    return CerberusConfig(
        enabled=bool(cerb_cfg.get("enabled", False)),
        strategy_heads=dict(cerb_cfg.get("strategy_heads", CerberusConfig().strategy_heads)),
        learning_rate=float(cerb_cfg.get("learning_rate", 0.05)),
        bounds=dict(cerb_cfg.get("bounds", {"min": 0.25, "max": 2.0})),
        regime_weights=dict(cerb_cfg.get("regime_weights", CerberusConfig().regime_weights)),
        sentinel_x_integration=bool(cerb_cfg.get("sentinel_x_integration", True)),
        decay_integration=bool(cerb_cfg.get("decay_integration", True)),
        meta_scheduler_integration=bool(cerb_cfg.get("meta_scheduler_integration", True)),
        signal_weights=dict(cerb_cfg.get("signal_weights", DEFAULT_SIGNAL_WEIGHTS)),
        smoothing_alpha=float(cerb_cfg.get("smoothing_alpha", 0.20)),
        run_interval_cycles=int(cerb_cfg.get("run_interval_cycles", 5)),
    )


# ---------------------------------------------------------------------------
# State Structures
# ---------------------------------------------------------------------------


@dataclass
class HeadMetrics:
    """
    Metrics for a single strategy head.

    Tracks multiplier, EMA score, signal components, and trend direction.
    """

    multiplier: float = 1.0
    ema_score: float = 0.5
    signal_score: float = 0.5
    regime_component: float = 1.0
    decay_component: float = 1.0
    meta_component: float = 1.0
    edge_component: float = 0.5
    health_component: float = 0.5
    trend_direction: str = "flat"  # "up", "down", "flat"
    last_update_ts: str = ""
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "multiplier": round(self.multiplier, 4),
            "ema_score": round(self.ema_score, 4),
            "signal_score": round(self.signal_score, 4),
            "regime_component": round(self.regime_component, 4),
            "decay_component": round(self.decay_component, 4),
            "meta_component": round(self.meta_component, 4),
            "edge_component": round(self.edge_component, 4),
            "health_component": round(self.health_component, 4),
            "trend_direction": self.trend_direction,
            "last_update_ts": self.last_update_ts,
            "sample_count": self.sample_count,
        }


@dataclass
class StrategyHeadState:
    """
    State for all strategy heads.
    """

    heads: Dict[str, HeadMetrics] = field(default_factory=dict)
    mean_multiplier: float = 1.0
    normalized: bool = True

    def __post_init__(self) -> None:
        """Ensure all canonical heads exist."""
        for head in STRATEGY_HEADS:
            if head not in self.heads:
                self.heads[head] = HeadMetrics()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "heads": {k: v.to_dict() for k, v in self.heads.items()},
            "mean_multiplier": round(self.mean_multiplier, 4),
            "normalized": self.normalized,
        }


@dataclass
class CerberusState:
    """
    Full Cerberus state for persistence.
    """

    updated_ts: str = ""
    cycle_count: int = 0
    head_state: StrategyHeadState = field(default_factory=StrategyHeadState)
    regime: str = "UNKNOWN"
    regime_probs: Dict[str, float] = field(default_factory=dict)
    overall_health: float = 0.5
    avg_decay_survival: float = 1.0
    meta_scheduler_active: bool = False
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "cycle_count": self.cycle_count,
            "head_state": self.head_state.to_dict(),
            "regime": self.regime,
            "regime_probs": {k: round(v, 4) for k, v in self.regime_probs.items()},
            "overall_health": round(self.overall_health, 4),
            "avg_decay_survival": round(self.avg_decay_survival, 4),
            "meta_scheduler_active": self.meta_scheduler_active,
            "notes": self.notes[-10:] if len(self.notes) > 10 else self.notes,
            "errors": self.errors[-10:] if len(self.errors) > 10 else self.errors,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Input Signal Loaders
# ---------------------------------------------------------------------------


def load_sentinel_x_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load Sentinel-X regime state."""
    p = Path(path or "logs/state/sentinel_x.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_alpha_decay_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load alpha decay state."""
    p = Path(path or "logs/state/alpha_decay.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_meta_scheduler_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load meta-scheduler state."""
    p = Path(path or "logs/state/meta_scheduler.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_edge_insights_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load edge insights state."""
    p = Path(path or "logs/state/edge_insights.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_alpha_miner_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load alpha miner (Prospector) state."""
    p = Path(path or "logs/state/alpha_miner.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_cross_pair_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load cross-pair engine (Crossfire) state."""
    p = Path(path or "logs/state/cross_pair_edges.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_universe_optimizer_state(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Load universe optimizer state."""
    p = Path(path or "logs/state/universe_optimizer.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_cerberus_state(
    path: Path | str | None = None,
) -> Optional[CerberusState]:
    """
    Load existing Cerberus state from file.

    Returns None if file doesn't exist or is invalid.
    """
    p = Path(path or DEFAULT_STATE_PATH)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return _parse_cerberus_state(data)
    except (json.JSONDecodeError, IOError):
        return None


def _parse_cerberus_state(data: Dict[str, Any]) -> CerberusState:
    """Parse dict into CerberusState."""
    head_state_data = data.get("head_state", {})
    heads_data = head_state_data.get("heads", {})

    heads = {}
    for head_name, metrics_data in heads_data.items():
        if isinstance(metrics_data, dict):
            heads[head_name] = HeadMetrics(
                multiplier=float(metrics_data.get("multiplier", 1.0)),
                ema_score=float(metrics_data.get("ema_score", 0.5)),
                signal_score=float(metrics_data.get("signal_score", 0.5)),
                regime_component=float(metrics_data.get("regime_component", 1.0)),
                decay_component=float(metrics_data.get("decay_component", 1.0)),
                meta_component=float(metrics_data.get("meta_component", 1.0)),
                edge_component=float(metrics_data.get("edge_component", 0.5)),
                health_component=float(metrics_data.get("health_component", 0.5)),
                trend_direction=str(metrics_data.get("trend_direction", "flat")),
                last_update_ts=str(metrics_data.get("last_update_ts", "")),
                sample_count=int(metrics_data.get("sample_count", 0)),
            )

    head_state = StrategyHeadState(
        heads=heads,
        mean_multiplier=float(head_state_data.get("mean_multiplier", 1.0)),
        normalized=bool(head_state_data.get("normalized", True)),
    )

    return CerberusState(
        updated_ts=str(data.get("updated_ts", "")),
        cycle_count=int(data.get("cycle_count", 0)),
        head_state=head_state,
        regime=str(data.get("regime", "UNKNOWN")),
        regime_probs=dict(data.get("regime_probs", {})),
        overall_health=float(data.get("overall_health", 0.5)),
        avg_decay_survival=float(data.get("avg_decay_survival", 1.0)),
        meta_scheduler_active=bool(data.get("meta_scheduler_active", False)),
        notes=list(data.get("notes", [])),
        errors=list(data.get("errors", [])),
        meta=dict(data.get("meta", {})),
    )


def write_cerberus_state(
    state: CerberusState,
    path: Path | str | None = None,
) -> bool:
    """
    Write Cerberus state to file.

    Returns True if successful.
    """
    p = Path(path or DEFAULT_STATE_PATH)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state.to_dict(), indent=2))
        return True
    except IOError as e:
        _LOG.error(f"Failed to write cerberus state: {e}")
        return False


# ---------------------------------------------------------------------------
# Signal Extraction
# ---------------------------------------------------------------------------


@dataclass
class CerberusSignals:
    """
    Aggregated signals for Cerberus decision-making.
    """

    # Regime
    regime: str = "UNKNOWN"
    regime_probs: Dict[str, float] = field(default_factory=dict)

    # Decay
    avg_symbol_survival: float = 1.0
    avg_factor_survival: float = 1.0
    overall_alpha_health: float = 0.5

    # Meta-scheduler
    meta_active: bool = False
    meta_factor_weights: Dict[str, float] = field(default_factory=dict)
    meta_conviction_strength: float = 1.0
    meta_category_overlays: Dict[str, float] = field(default_factory=dict)

    # Edge insights
    strategy_health_score: float = 0.5
    top_factors: List[str] = field(default_factory=list)
    weak_factors: List[str] = field(default_factory=list)
    factor_edges: Dict[str, float] = field(default_factory=dict)
    category_edges: Dict[str, float] = field(default_factory=dict)

    # Crossfire (RV)
    crossfire_active: bool = False
    avg_pair_edge: float = 0.0
    pairs_eligible: int = 0

    # Alpha Miner (Prospector)
    prospector_active: bool = False
    candidates_count: int = 0
    avg_candidate_score: float = 0.0

    # Universe
    universe_size: int = 0
    effective_max_size: int = 0
    category_diversity: int = 0


def extract_cerberus_signals(
    sentinel_x: Dict[str, Any],
    alpha_decay: Dict[str, Any],
    meta_scheduler: Dict[str, Any],
    edge_insights: Dict[str, Any],
    cross_pair: Dict[str, Any],
    alpha_miner: Dict[str, Any],
    universe_optimizer: Dict[str, Any],
) -> CerberusSignals:
    """
    Extract signals from all intel surfaces.

    Args:
        sentinel_x: Sentinel-X regime state
        alpha_decay: Alpha decay state
        meta_scheduler: Meta-scheduler state
        edge_insights: Edge insights state
        cross_pair: Cross-pair edges state
        alpha_miner: Alpha miner state
        universe_optimizer: Universe optimizer state

    Returns:
        CerberusSignals with all extracted signals
    """
    signals = CerberusSignals()

    # Regime from Sentinel-X
    if sentinel_x:
        signals.regime = sentinel_x.get("primary_regime", "UNKNOWN")
        signals.regime_probs = dict(sentinel_x.get("smoothed_probs", {}))

    # Alpha decay
    if alpha_decay:
        signals.avg_symbol_survival = float(alpha_decay.get("avg_symbol_survival", 1.0))
        signals.avg_factor_survival = float(alpha_decay.get("avg_factor_survival", 1.0))
        signals.overall_alpha_health = float(alpha_decay.get("overall_alpha_health", 0.5))

    # Meta-scheduler
    if meta_scheduler:
        stats = meta_scheduler.get("stats", {})
        signals.meta_active = bool(stats.get("learning_active", False))
        factor_state = meta_scheduler.get("factor_state", {})
        signals.meta_factor_weights = {
            k: float(v.get("meta_weight", 1.0))
            for k, v in factor_state.items()
            if isinstance(v, dict)
        }
        conviction_state = meta_scheduler.get("conviction_state", {})
        signals.meta_conviction_strength = float(conviction_state.get("global_strength", 1.0))
        category_state = meta_scheduler.get("category_state", {})
        signals.meta_category_overlays = {
            k: float(v.get("overlay", 1.0))
            for k, v in category_state.items()
            if isinstance(v, dict)
        }

    # Edge insights
    if edge_insights:
        health = edge_insights.get("strategy_health", {})
        if isinstance(health, dict):
            signals.strategy_health_score = float(health.get("health_score", 0.5))

        summary = edge_insights.get("edge_summary", {})
        if isinstance(summary, dict):
            signals.top_factors = [
                f.get("factor", "") for f in summary.get("top_factors", [])
                if isinstance(f, dict)
            ]
            signals.weak_factors = [
                f.get("factor", "") for f in summary.get("weak_factors", [])
                if isinstance(f, dict)
            ]

        factor_edges = edge_insights.get("factor_edges", {})
        if isinstance(factor_edges, dict):
            signals.factor_edges = {
                k: float(v.get("ir", 0.0) if isinstance(v, dict) else 0.0)
                for k, v in factor_edges.items()
            }

        category_edges = edge_insights.get("category_edges", {})
        if isinstance(category_edges, dict):
            signals.category_edges = {
                k: float(v.get("avg_momentum", 0.0) if isinstance(v, dict) else 0.0)
                for k, v in category_edges.items()
            }

    # Crossfire
    if cross_pair:
        signals.crossfire_active = cross_pair.get("pairs_eligible", 0) > 0
        signals.pairs_eligible = int(cross_pair.get("pairs_eligible", 0))
        pair_edges = cross_pair.get("pair_edges", {})
        if pair_edges:
            edge_scores = [
                float(v.get("edge_score", 0.0))
                for v in pair_edges.values()
                if isinstance(v, dict)
            ]
            if edge_scores:
                signals.avg_pair_edge = sum(edge_scores) / len(edge_scores)

    # Alpha Miner
    if alpha_miner:
        candidates = alpha_miner.get("candidates", [])
        signals.prospector_active = len(candidates) > 0
        signals.candidates_count = len(candidates)
        if candidates:
            scores = [float(c.get("score", 0.0)) for c in candidates if isinstance(c, dict)]
            if scores:
                signals.avg_candidate_score = sum(scores) / len(scores)

    # Universe optimizer
    if universe_optimizer:
        signals.universe_size = int(universe_optimizer.get("total_universe_size", 0))
        signals.effective_max_size = int(universe_optimizer.get("effective_max_size", 0))
        category_scores = universe_optimizer.get("category_scores", {})
        signals.category_diversity = len(category_scores)

    return signals


# ---------------------------------------------------------------------------
# Head Signal Scoring
# ---------------------------------------------------------------------------


def compute_regime_multiplier(
    head: str,
    regime: str,
    config: CerberusConfig,
) -> float:
    """
    Get regime-based multiplier for a strategy head.

    Returns 1.0 if regime not found.
    """
    regime_weights = config.regime_weights.get(regime, {})
    return float(regime_weights.get(head, 1.0))


def compute_decay_component(
    head: str,
    signals: CerberusSignals,
) -> float:
    """
    Compute decay-based component for a head.

    Uses alpha decay survival to penalize deteriorating edges.
    """
    # Map heads to relevant decay metrics
    if head == "TREND":
        # Trend depends on overall factor survival
        return signals.avg_factor_survival
    elif head == "MEAN_REVERT":
        # Mean reversion benefits when trend decays
        return 1.0 - (1.0 - signals.overall_alpha_health) * 0.5
    elif head == "RELATIVE_VALUE":
        # RV is more robust to decay
        return min(1.0, signals.overall_alpha_health + 0.2)
    elif head == "CATEGORY":
        # Category rotation depends on symbol survival
        return signals.avg_symbol_survival
    elif head == "VOL_HARVEST":
        # Vol harvesting is decay-neutral
        return 1.0
    elif head == "EMERGENT_ALPHA":
        # Emergent alpha penalized when overall alpha is dying
        return signals.overall_alpha_health
    else:
        return 1.0


def compute_meta_component(
    head: str,
    signals: CerberusSignals,
) -> float:
    """
    Compute meta-scheduler component for a head.

    Uses meta-scheduler overlays to adjust head weights.
    """
    if not signals.meta_active:
        return 1.0

    # Map heads to meta-scheduler signals
    if head == "TREND":
        # Trend boosted by trend factors
        trend_weight = signals.meta_factor_weights.get("trend", 1.0)
        return trend_weight
    elif head == "MEAN_REVERT":
        # Mean reversion uses conviction strength inverse
        return 2.0 - signals.meta_conviction_strength if signals.meta_conviction_strength > 0 else 1.0
    elif head == "CATEGORY":
        # Category uses average category overlay
        if signals.meta_category_overlays:
            avg_overlay = sum(signals.meta_category_overlays.values()) / len(signals.meta_category_overlays)
            return avg_overlay
        return 1.0
    else:
        return 1.0


def compute_edge_component(
    head: str,
    signals: CerberusSignals,
) -> float:
    """
    Compute edge-based component for a head.

    Uses factor/category edges to score head strength.
    """
    if head == "TREND":
        # Trend edge from trend factor
        return min(1.0, max(0.0, signals.factor_edges.get("trend", 0.5) / 2.0 + 0.5))
    elif head == "MEAN_REVERT":
        # Mean revert edge (inverse of trend)
        trend_edge = signals.factor_edges.get("trend", 0.5)
        return 1.0 - min(1.0, max(0.0, trend_edge))
    elif head == "RELATIVE_VALUE":
        # RV edge from crossfire
        return signals.avg_pair_edge if signals.crossfire_active else 0.5
    elif head == "CATEGORY":
        # Category edge from category momentum
        if signals.category_edges:
            avg_edge = sum(abs(v) for v in signals.category_edges.values()) / len(signals.category_edges)
            return min(1.0, avg_edge)
        return 0.5
    elif head == "VOL_HARVEST":
        # Vol harvest is stronger in high vol regimes
        return 1.0 if signals.regime in ("BREAKOUT", "CRISIS") else 0.5
    elif head == "EMERGENT_ALPHA":
        # Emergent alpha from prospector
        return signals.avg_candidate_score if signals.prospector_active else 0.3
    else:
        return 0.5


def compute_health_component(
    signals: CerberusSignals,
) -> float:
    """
    Compute strategy health component.

    All heads use the same health score.
    """
    return signals.strategy_health_score


def compute_universe_component(
    head: str,
    signals: CerberusSignals,
) -> float:
    """
    Compute universe confidence component.

    Based on universe size and diversity.
    """
    if signals.effective_max_size > 0:
        utilization = signals.universe_size / signals.effective_max_size
    else:
        utilization = 0.5

    diversity_bonus = min(0.2, signals.category_diversity * 0.05)
    return min(1.0, utilization + diversity_bonus)


def compute_rv_component(
    head: str,
    signals: CerberusSignals,
) -> float:
    """
    Compute relative value component (only for RV head).
    """
    if head != "RELATIVE_VALUE":
        return 0.5

    if not signals.crossfire_active:
        return 0.3

    # Score based on eligible pairs and average edge
    pairs_score = min(1.0, signals.pairs_eligible / 5.0)  # Normalize to 5 pairs
    return (pairs_score + signals.avg_pair_edge) / 2.0


def compute_head_signal_score(
    head: str,
    signals: CerberusSignals,
    config: CerberusConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite signal score for a strategy head.

    Returns:
        Tuple of (signal_score, component_dict)
    """
    weights = config.signal_weights

    # Compute components
    regime_mult = compute_regime_multiplier(head, signals.regime, config)
    decay_comp = compute_decay_component(head, signals) if config.decay_integration else 1.0
    meta_comp = compute_meta_component(head, signals) if config.meta_scheduler_integration else 1.0
    edge_comp = compute_edge_component(head, signals)
    health_comp = compute_health_component(signals)
    universe_comp = compute_universe_component(head, signals)
    rv_comp = compute_rv_component(head, signals)

    # Weighted sum (regime is multiplicative, others are additive)
    base_score = (
        weights.get("decay", 0.15) * decay_comp +
        weights.get("meta", 0.15) * meta_comp +
        weights.get("edge", 0.20) * edge_comp +
        weights.get("health", 0.15) * health_comp +
        weights.get("universe", 0.05) * universe_comp +
        weights.get("rv", 0.05) * rv_comp
    )

    # Apply regime multiplier
    signal_score = base_score * regime_mult * weights.get("regime", 0.25) + base_score * (1.0 - weights.get("regime", 0.25))

    # Clamp to [0, 1]
    signal_score = max(0.0, min(1.0, signal_score))

    components = {
        "regime_component": regime_mult,
        "decay_component": decay_comp,
        "meta_component": meta_comp,
        "edge_component": edge_comp,
        "health_component": health_comp,
        "universe_component": universe_comp,
        "rv_component": rv_comp,
    }

    return signal_score, components


# ---------------------------------------------------------------------------
# Multiplier Update Logic
# ---------------------------------------------------------------------------


def update_head_multiplier(
    prev_multiplier: float,
    signal_score: float,
    learning_rate: float,
    bounds: Dict[str, float],
) -> float:
    """
    Update head multiplier using exponential smoothing.

    Formula: new = prev + lr * (signal_score * 2 - prev)

    This moves multiplier toward 2*signal_score (so 0.5 score → 1.0 multiplier).
    """
    target = signal_score * 2.0  # Map [0, 1] to [0, 2]
    new_multiplier = prev_multiplier + learning_rate * (target - prev_multiplier)

    # Clamp to bounds
    min_mult = bounds.get("min", 0.25)
    max_mult = bounds.get("max", 2.0)
    return max(min_mult, min(max_mult, new_multiplier))


def normalize_multipliers(
    multipliers: Dict[str, float],
) -> Dict[str, float]:
    """
    Normalize multipliers so mean = 1.0.

    This preserves relative weights while keeping overall budget neutral.
    """
    if not multipliers:
        return multipliers

    mean = sum(multipliers.values()) / len(multipliers)
    if mean <= 0:
        return multipliers

    return {k: v / mean for k, v in multipliers.items()}


def compute_trend_direction(
    current_score: float,
    prev_ema: float,
    threshold: float = 0.05,
) -> str:
    """
    Determine trend direction based on EMA change.
    """
    diff = current_score - prev_ema
    if diff > threshold:
        return "up"
    elif diff < -threshold:
        return "down"
    else:
        return "flat"


# ---------------------------------------------------------------------------
# Main Step Function
# ---------------------------------------------------------------------------


def run_cerberus_step(
    config: CerberusConfig,
    prev_state: Optional[CerberusState] = None,
    sentinel_x: Optional[Dict[str, Any]] = None,
    alpha_decay: Optional[Dict[str, Any]] = None,
    meta_scheduler: Optional[Dict[str, Any]] = None,
    edge_insights: Optional[Dict[str, Any]] = None,
    cross_pair: Optional[Dict[str, Any]] = None,
    alpha_miner: Optional[Dict[str, Any]] = None,
    universe_optimizer: Optional[Dict[str, Any]] = None,
) -> CerberusState:
    """
    Run one Cerberus step to update head multipliers.

    Args:
        config: CerberusConfig
        prev_state: Previous CerberusState (or None for cold start)
        sentinel_x: Sentinel-X state dict
        alpha_decay: Alpha decay state dict
        meta_scheduler: Meta-scheduler state dict
        edge_insights: Edge insights state dict
        cross_pair: Cross-pair edges state dict
        alpha_miner: Alpha miner state dict
        universe_optimizer: Universe optimizer state dict

    Returns:
        Updated CerberusState
    """
    if not config.enabled:
        # Return neutral state when disabled
        state = CerberusState()
        state.updated_ts = datetime.now(timezone.utc).isoformat()
        state.notes.append("Cerberus disabled - returning neutral multipliers")
        return state

    # Extract signals
    signals = extract_cerberus_signals(
        sentinel_x=sentinel_x or {},
        alpha_decay=alpha_decay or {},
        meta_scheduler=meta_scheduler or {},
        edge_insights=edge_insights or {},
        cross_pair=cross_pair or {},
        alpha_miner=alpha_miner or {},
        universe_optimizer=universe_optimizer or {},
    )

    # Initialize state
    now = datetime.now(timezone.utc).isoformat()
    cycle_count = (prev_state.cycle_count + 1) if prev_state else 1

    state = CerberusState(
        updated_ts=now,
        cycle_count=cycle_count,
        regime=signals.regime,
        regime_probs=signals.regime_probs,
        overall_health=signals.strategy_health_score,
        avg_decay_survival=signals.avg_symbol_survival,
        meta_scheduler_active=signals.meta_active,
    )

    notes: List[str] = []
    errors: List[str] = []

    # Get previous head state
    prev_heads = (prev_state.head_state.heads if prev_state else {})

    # Update each head
    new_heads: Dict[str, HeadMetrics] = {}
    for head in STRATEGY_HEADS:
        prev_metrics = prev_heads.get(head, HeadMetrics())

        # Compute signal score
        try:
            signal_score, components = compute_head_signal_score(head, signals, config)
        except Exception as e:
            errors.append(f"{head}: {e}")
            signal_score = 0.5
            components = {}

        # Update EMA score
        alpha = config.smoothing_alpha
        new_ema = alpha * signal_score + (1 - alpha) * prev_metrics.ema_score

        # Update multiplier
        new_multiplier = update_head_multiplier(
            prev_multiplier=prev_metrics.multiplier,
            signal_score=signal_score,
            learning_rate=config.learning_rate,
            bounds=config.bounds,
        )

        # Compute trend direction
        trend_dir = compute_trend_direction(signal_score, prev_metrics.ema_score)

        new_heads[head] = HeadMetrics(
            multiplier=new_multiplier,
            ema_score=new_ema,
            signal_score=signal_score,
            regime_component=components.get("regime_component", 1.0),
            decay_component=components.get("decay_component", 1.0),
            meta_component=components.get("meta_component", 1.0),
            edge_component=components.get("edge_component", 0.5),
            health_component=components.get("health_component", 0.5),
            trend_direction=trend_dir,
            last_update_ts=now,
            sample_count=prev_metrics.sample_count + 1,
        )

        # Add notes for significant changes
        mult_change = new_multiplier - prev_metrics.multiplier
        if abs(mult_change) > 0.1:
            direction = "boosted" if mult_change > 0 else "reduced"
            notes.append(f"{head} {direction}: {prev_metrics.multiplier:.2f} → {new_multiplier:.2f}")

    # Normalize multipliers
    raw_multipliers = {h: m.multiplier for h, m in new_heads.items()}
    normalized_multipliers = normalize_multipliers(raw_multipliers)

    # Apply normalized multipliers back
    for head, norm_mult in normalized_multipliers.items():
        new_heads[head].multiplier = norm_mult

    # Compute mean multiplier (should be ~1.0 after normalization)
    mean_mult = sum(m.multiplier for m in new_heads.values()) / len(new_heads) if new_heads else 1.0

    state.head_state = StrategyHeadState(
        heads=new_heads,
        mean_multiplier=mean_mult,
        normalized=True,
    )

    # Add regime note
    if signals.regime != "UNKNOWN":
        notes.append(f"Regime: {signals.regime}")

    state.notes = notes
    state.errors = errors
    state.meta = {
        "config_version": "v7.8_P8",
        "heads_count": len(STRATEGY_HEADS),
        "learning_rate": config.learning_rate,
        "bounds": config.bounds,
    }

    return state


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def get_cerberus_head_multiplier(
    head: str,
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get multiplier for a specific strategy head.

    Returns 1.0 if Cerberus is disabled or state unavailable.
    """
    if config and not config.enabled:
        return 1.0

    if state is None:
        return 1.0

    metrics = state.head_state.heads.get(head)
    if metrics is None:
        return 1.0

    return metrics.multiplier


def get_cerberus_all_multipliers(
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> Dict[str, float]:
    """
    Get all head multipliers as a dict.

    Returns all 1.0 if Cerberus is disabled or state unavailable.
    """
    if config and not config.enabled:
        return {h: 1.0 for h in STRATEGY_HEADS}

    if state is None:
        return {h: 1.0 for h in STRATEGY_HEADS}

    return {
        head: metrics.multiplier
        for head, metrics in state.head_state.heads.items()
    }


def get_cerberus_factor_weight_overlay(
    factor: str,
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get factor weight overlay based on head multipliers.

    Maps factors to heads:
    - trend → TREND
    - mean_revert → MEAN_REVERT
    - rv_momentum → RELATIVE_VALUE
    - category_momentum → CATEGORY
    - vol_regime → VOL_HARVEST
    """
    factor_to_head = {
        "trend": "TREND",
        "carry": "TREND",  # Carry aligns with trend
        "rv_momentum": "RELATIVE_VALUE",
        "expectancy": "TREND",  # Expectancy aligns with trend
        "vol_regime": "VOL_HARVEST",
        "category_momentum": "CATEGORY",
        "router_quality": "TREND",  # Router quality affects trend
    }

    head = factor_to_head.get(factor, "TREND")
    return get_cerberus_head_multiplier(head, state, config)


def get_cerberus_conviction_multiplier(
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get conviction multiplier based on overall head state.

    Returns average of TREND and MEAN_REVERT multipliers.
    """
    trend = get_cerberus_head_multiplier("TREND", state, config)
    mr = get_cerberus_head_multiplier("MEAN_REVERT", state, config)
    return (trend + mr) / 2.0


def get_cerberus_universe_category_multiplier(
    category: str,
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get category multiplier for universe scoring.

    Uses CATEGORY head multiplier as base.
    """
    return get_cerberus_head_multiplier("CATEGORY", state, config)


def get_cerberus_alpha_router_adjustment(
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get alpha router allocation adjustment based on head state.

    Returns weighted average of all heads.
    """
    multipliers = get_cerberus_all_multipliers(state, config)
    if not multipliers:
        return 1.0

    # Weight TREND and VOL_HARVEST higher for allocation
    weights = {
        "TREND": 0.30,
        "MEAN_REVERT": 0.15,
        "RELATIVE_VALUE": 0.10,
        "CATEGORY": 0.15,
        "VOL_HARVEST": 0.20,
        "EMERGENT_ALPHA": 0.10,
    }

    weighted_sum = sum(multipliers.get(h, 1.0) * w for h, w in weights.items())
    total_weight = sum(weights.values())
    return weighted_sum / total_weight if total_weight > 0 else 1.0


def get_cerberus_crossfire_multiplier(
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get crossfire (RV) influence multiplier.
    """
    return get_cerberus_head_multiplier("RELATIVE_VALUE", state, config)


def get_cerberus_prospector_multiplier(
    state: Optional[CerberusState] = None,
    config: Optional[CerberusConfig] = None,
) -> float:
    """
    Get prospector (emergent alpha) acceptance multiplier.
    """
    return get_cerberus_head_multiplier("EMERGENT_ALPHA", state, config)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "STRATEGY_HEADS",
    "DEFAULT_STATE_PATH",
    # Config
    "CerberusConfig",
    "load_cerberus_config",
    # State
    "HeadMetrics",
    "StrategyHeadState",
    "CerberusState",
    "CerberusSignals",
    # I/O
    "load_cerberus_state",
    "write_cerberus_state",
    # Signals
    "extract_cerberus_signals",
    "load_sentinel_x_state",
    "load_alpha_decay_state",
    "load_meta_scheduler_state",
    "load_edge_insights_state",
    "load_alpha_miner_state",
    "load_cross_pair_state",
    "load_universe_optimizer_state",
    # Core
    "run_cerberus_step",
    "compute_head_signal_score",
    "update_head_multiplier",
    "normalize_multipliers",
    # Integration helpers
    "get_cerberus_head_multiplier",
    "get_cerberus_all_multipliers",
    "get_cerberus_factor_weight_overlay",
    "get_cerberus_conviction_multiplier",
    "get_cerberus_universe_category_multiplier",
    "get_cerberus_alpha_router_adjustment",
    "get_cerberus_crossfire_multiplier",
    "get_cerberus_prospector_multiplier",
]
