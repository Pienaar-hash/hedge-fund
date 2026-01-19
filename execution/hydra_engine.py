"""
Hydra: Multi-Strategy Execution Engine — v7.9_P1

Promotes Cerberus' 6 heads from "weights in intel-space" to first-class strategy tracks:

1. TREND          — Directional momentum / trend following
2. MEAN_REVERT    — Mean reversion / reversal
3. RELATIVE_VALUE — Crossfire pair trades
4. CATEGORY       — Category rotation / sector tilt
5. VOL_HARVEST    — Volatility harvesting (position sizing)
6. EMERGENT_ALPHA — Prospector / Universe expansion

Hydra provides:
- Per-head signals → intents
- Per-head risk budgets
- Central router that merges/filters conflicting intents
- Single unified order stream obeying existing risk & router rails
- Per-head PnL, exposure, and trade attribution

Architecture:
    Head Generators (per-head)  ──┐
                                  │
    Cerberus Multipliers       ──┼──► Hydra Engine ──► Merged Intents ──► Risk/Router
                                  │
    Intel Surfaces             ──┘

Critical rails:
- Hydra is config-gated. When disabled, behavior is identical to v7.8.
- No breaking changes to risk veto engine, router, or execution contracts.
- Existing single-strategy flow remains the fallback path.

Single writer rule: Only executor may write hydra_state.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/hydra_state.json")
DEFAULT_INTENT_LOG_PATH = Path("logs/hydra/hydra_intents.jsonl")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")

_LOG = logging.getLogger(__name__)

# Canonical strategy heads (same as Cerberus)
STRATEGY_HEADS = [
    "TREND",
    "MEAN_REVERT",
    "RELATIVE_VALUE",
    "CATEGORY",
    "VOL_HARVEST",
    "EMERGENT_ALPHA",
]

# Default head configurations
DEFAULT_HEAD_CONFIGS = {
    "TREND": {
        "enabled": True,
        "max_nav_pct": 0.50,
        "max_gross_nav_pct": 0.75,
        "max_positions": 12,
        "priority": 100,
        "direction": "both",
    },
    "MEAN_REVERT": {
        "enabled": True,
        "max_nav_pct": 0.25,
        "max_gross_nav_pct": 0.35,
        "max_positions": 8,
        "priority": 80,
        "direction": "both",
    },
    "RELATIVE_VALUE": {
        "enabled": True,
        "max_nav_pct": 0.30,
        "max_gross_nav_pct": 0.40,
        "max_positions": 10,
        "priority": 90,
        "direction": "both",
    },
    "CATEGORY": {
        "enabled": True,
        "max_nav_pct": 0.20,
        "max_gross_nav_pct": 0.30,
        "max_positions": 8,
        "priority": 60,
        "direction": "both",
    },
    "VOL_HARVEST": {
        "enabled": True,
        "max_nav_pct": 0.20,
        "max_gross_nav_pct": 0.30,
        "max_positions": 8,
        "priority": 70,
        "direction": "both",
    },
    "EMERGENT_ALPHA": {
        "enabled": True,
        "max_nav_pct": 0.15,
        "max_gross_nav_pct": 0.25,
        "max_positions": 6,
        "priority": 50,
        "direction": "both",
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class HydraHeadConfig:
    """Configuration for a single strategy head."""

    name: str
    enabled: bool = True
    max_nav_pct: float = 0.30
    max_gross_nav_pct: float = 0.40
    max_positions: int = 10
    priority: int = 50
    direction: str = "both"  # "long", "short", "both"

    def __post_init__(self) -> None:
        """Validate and normalize config values."""
        # Clamp percentages
        self.max_nav_pct = max(0.0, min(1.0, self.max_nav_pct))
        self.max_gross_nav_pct = max(0.0, min(1.0, self.max_gross_nav_pct))
        # Clamp positions
        self.max_positions = max(1, min(50, self.max_positions))
        # Clamp priority
        self.priority = max(0, min(200, self.priority))
        # Normalize direction
        if self.direction not in ("long", "short", "both"):
            self.direction = "both"


@dataclass
class HydraIntentLimits:
    """Limits for intent generation and merging."""

    max_intents_per_cycle: int = 64
    max_symbol_heads: int = 3


@dataclass
class HydraConflictResolution:
    """Configuration for conflict resolution between heads."""

    allow_netting: bool = True
    prefer_higher_priority: bool = True
    prefer_higher_score: bool = True
    max_head_disagreement: int = 3


@dataclass
class HydraConfig:
    """Configuration for the Hydra Multi-Strategy Execution Engine."""

    enabled: bool = False
    heads: Dict[str, HydraHeadConfig] = field(default_factory=dict)
    intent_limits: HydraIntentLimits = field(default_factory=HydraIntentLimits)
    conflict_resolution: HydraConflictResolution = field(
        default_factory=HydraConflictResolution
    )

    def __post_init__(self) -> None:
        """Ensure all canonical heads exist with defaults."""
        for head_name in STRATEGY_HEADS:
            if head_name not in self.heads:
                defaults = DEFAULT_HEAD_CONFIGS.get(head_name, {})
                self.heads[head_name] = HydraHeadConfig(
                    name=head_name,
                    enabled=defaults.get("enabled", True),
                    max_nav_pct=defaults.get("max_nav_pct", 0.30),
                    max_gross_nav_pct=defaults.get("max_gross_nav_pct", 0.40),
                    max_positions=defaults.get("max_positions", 10),
                    priority=defaults.get("priority", 50),
                    direction=defaults.get("direction", "both"),
                )

    def get_enabled_heads(self) -> List[str]:
        """Return list of enabled head names."""
        return [h for h, cfg in self.heads.items() if cfg.enabled]


@dataclass
class HydraIntent:
    """Intent generated by a single strategy head."""

    head: str
    symbol: str
    side: str  # "long" / "short"
    nav_pct: float
    score: float  # 0–1, head-specific signal strength
    rationale: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate intent values."""
        self.nav_pct = max(0.0, min(1.0, self.nav_pct))
        self.score = max(0.0, min(1.0, self.score))
        if self.side not in ("long", "short"):
            raise ValueError(f"Invalid side: {self.side}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "head": self.head,
            "symbol": self.symbol,
            "side": self.side,
            "nav_pct": round(self.nav_pct, 6),
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class HydraMergedIntent:
    """Merged intent after conflict resolution across heads."""

    symbol: str
    net_side: str  # "long" / "short" / "flat"
    nav_pct: float
    heads: List[str]  # Contributing heads
    head_contributions: Dict[str, float]  # head -> nav_pct contribution
    score: float
    rationale: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate merged intent values."""
        self.nav_pct = max(0.0, min(1.0, self.nav_pct))
        self.score = max(0.0, min(1.0, self.score))
        if self.net_side not in ("long", "short", "flat"):
            self.net_side = "flat"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "net_side": self.net_side,
            "nav_pct": round(self.nav_pct, 6),
            "heads": self.heads,
            "head_contributions": {
                k: round(v, 6) for k, v in self.head_contributions.items()
            },
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class HydraHeadBudget:
    """Budget tracking for a single head."""

    name: str
    max_nav_pct: float
    used_nav_pct: float = 0.0
    position_count: int = 0
    max_positions: int = 10

    @property
    def remaining_nav_pct(self) -> float:
        """Remaining NAV budget."""
        return max(0.0, self.max_nav_pct - self.used_nav_pct)

    @property
    def remaining_positions(self) -> int:
        """Remaining position slots."""
        return max(0, self.max_positions - self.position_count)

    def can_allocate(self, nav_pct: float) -> bool:
        """Check if allocation is possible within budget."""
        return (
            self.remaining_nav_pct >= nav_pct
            and self.remaining_positions > 0
        )


@dataclass
class HydraState:
    """State snapshot of the Hydra engine."""

    updated_ts: str = ""
    head_budgets: Dict[str, float] = field(default_factory=dict)
    head_usage: Dict[str, float] = field(default_factory=dict)
    head_positions: Dict[str, int] = field(default_factory=dict)
    merged_intents: List[Dict[str, Any]] = field(default_factory=list)
    cycle_count: int = 0
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "head_budgets": self.head_budgets,
            "head_usage": self.head_usage,
            "head_positions": self.head_positions,
            "merged_intents": self.merged_intents,
            "cycle_count": self.cycle_count,
            "notes": self.notes,
            "errors": self.errors,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HydraState":
        """Create from dictionary."""
        return cls(
            updated_ts=data.get("updated_ts", ""),
            head_budgets=data.get("head_budgets", {}),
            head_usage=data.get("head_usage", {}),
            head_positions=data.get("head_positions", {}),
            merged_intents=data.get("merged_intents", []),
            cycle_count=data.get("cycle_count", 0),
            notes=data.get("notes", []),
            errors=data.get("errors", []),
            meta=data.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_hydra_config(
    config_path: Path | str | None = None,
    strategy_config: Mapping[str, Any] | None = None,
) -> HydraConfig:
    """
    Load Hydra configuration from strategy_config.json.

    Args:
        config_path: Path to strategy_config.json
        strategy_config: Pre-loaded strategy config dict

    Returns:
        HydraConfig instance
    """
    if strategy_config is None:
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        if cfg_path.exists():
            try:
                strategy_config = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, IOError):
                return HydraConfig()
        else:
            return HydraConfig()

    hydra_cfg = strategy_config.get("hydra_execution", {})
    if not hydra_cfg:
        return HydraConfig()

    # Parse heads
    heads: Dict[str, HydraHeadConfig] = {}
    heads_raw = hydra_cfg.get("heads", {})
    for head_name in STRATEGY_HEADS:
        head_raw = heads_raw.get(head_name, DEFAULT_HEAD_CONFIGS.get(head_name, {}))
        heads[head_name] = HydraHeadConfig(
            name=head_name,
            enabled=head_raw.get("enabled", True),
            max_nav_pct=head_raw.get("max_nav_pct", 0.30),
            max_gross_nav_pct=head_raw.get("max_gross_nav_pct", 0.40),
            max_positions=head_raw.get("max_positions", 10),
            priority=head_raw.get("priority", 50),
            direction=head_raw.get("direction", "both"),
        )

    # Parse intent limits
    limits_raw = hydra_cfg.get("intent_limits", {})
    intent_limits = HydraIntentLimits(
        max_intents_per_cycle=limits_raw.get("max_intents_per_cycle", 64),
        max_symbol_heads=limits_raw.get("max_symbol_heads", 3),
    )

    # Parse conflict resolution
    conflict_raw = hydra_cfg.get("conflict_resolution", {})
    conflict_resolution = HydraConflictResolution(
        allow_netting=conflict_raw.get("allow_netting", True),
        prefer_higher_priority=conflict_raw.get("prefer_higher_priority", True),
        prefer_higher_score=conflict_raw.get("prefer_higher_score", True),
        max_head_disagreement=conflict_raw.get("max_head_disagreement", 3),
    )

    return HydraConfig(
        enabled=hydra_cfg.get("enabled", False),
        heads=heads,
        intent_limits=intent_limits,
        conflict_resolution=conflict_resolution,
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_hydra_state(state_path: Path | str | None = None) -> HydraState:
    """
    Load Hydra state from disk.

    Args:
        state_path: Path to hydra_state.json

    Returns:
        HydraState instance (empty if file missing/invalid)
    """
    path = Path(state_path or DEFAULT_STATE_PATH)
    if not path.exists():
        return HydraState()

    try:
        data = json.loads(path.read_text())
        return HydraState.from_dict(data)
    except (json.JSONDecodeError, IOError) as e:
        _LOG.warning("Failed to load hydra state: %s", e)
        return HydraState()


def save_hydra_state(state: HydraState, state_path: Path | str | None = None) -> bool:
    """
    Save Hydra state to disk.

    Args:
        state: HydraState to save
        state_path: Path to hydra_state.json

    Returns:
        True if successful
    """
    path = Path(state_path or DEFAULT_STATE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2))
        return True
    except IOError as e:
        _LOG.error("Failed to save hydra state: %s", e)
        return False


# ---------------------------------------------------------------------------
# Intent Logging
# ---------------------------------------------------------------------------


def log_hydra_intent(
    intent: HydraMergedIntent,
    log_path: Path | str | None = None,
) -> bool:
    """
    Append a merged intent to the JSONL log.

    Args:
        intent: HydraMergedIntent to log
        log_path: Path to hydra_intents.jsonl

    Returns:
        True if successful
    """
    path = Path(log_path or DEFAULT_INTENT_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            line = json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                **intent.to_dict(),
            })
            f.write(line + "\n")
        return True
    except IOError as e:
        _LOG.error("Failed to log hydra intent: %s", e)
        return False


def log_hydra_intents(
    intents: List[HydraMergedIntent],
    log_path: Path | str | None = None,
) -> int:
    """
    Append multiple merged intents to the JSONL log.

    Args:
        intents: List of HydraMergedIntent to log
        log_path: Path to hydra_intents.jsonl

    Returns:
        Number of successfully logged intents
    """
    path = Path(log_path or DEFAULT_INTENT_LOG_PATH)
    logged = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            ts = datetime.now(timezone.utc).isoformat()
            for intent in intents:
                line = json.dumps({"ts": ts, **intent.to_dict()})
                f.write(line + "\n")
                logged += 1
    except IOError as e:
        _LOG.error("Failed to log hydra intents: %s", e)
    return logged


# ---------------------------------------------------------------------------
# Head Generators (Adapters)
# ---------------------------------------------------------------------------


def generate_trend_intents(
    symbols: List[str],
    hybrid_scores: Dict[str, float],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.02,
) -> List[HydraIntent]:
    """
    Generate TREND head intents from hybrid trend/carry scores.

    Args:
        symbols: Universe of symbols to consider
        hybrid_scores: symbol -> hybrid score (-1 to 1)
        cerberus_multiplier: Cerberus TREND head multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size as NAV fraction

    Returns:
        List of HydraIntent for TREND head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []
    direction = head_cfg.direction

    for symbol in symbols:
        score = hybrid_scores.get(symbol, 0.0)
        if abs(score) < 0.1:  # Minimum threshold
            continue

        # Determine side
        if score > 0:
            if direction == "short":
                continue
            side = "long"
        else:
            if direction == "long":
                continue
            side = "short"

        # Scale NAV by score magnitude and Cerberus multiplier
        raw_nav_pct = base_nav_pct * abs(score) * cerberus_multiplier
        nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / len(symbols))

        intents.append(HydraIntent(
            head="TREND",
            symbol=symbol,
            side=side,
            nav_pct=nav_pct,
            score=abs(score),
            rationale=f"Trend score={score:.3f}, cerb_mult={cerberus_multiplier:.2f}",
        ))

    return intents


def generate_mean_revert_intents(
    symbols: List[str],
    zscore_map: Dict[str, float],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.015,
    zscore_threshold: float = 1.5,
) -> List[HydraIntent]:
    """
    Generate MEAN_REVERT head intents from z-score data.

    Args:
        symbols: Universe of symbols
        zscore_map: symbol -> current z-score
        cerberus_multiplier: Cerberus MEAN_REVERT multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size
        zscore_threshold: Minimum |z-score| for entry

    Returns:
        List of HydraIntent for MEAN_REVERT head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []
    direction = head_cfg.direction

    for symbol in symbols:
        zscore = zscore_map.get(symbol, 0.0)
        if abs(zscore) < zscore_threshold:
            continue

        # Mean revert: short when high, long when low
        if zscore > zscore_threshold:
            if direction == "long":
                continue
            side = "short"
        else:
            if direction == "short":
                continue
            side = "long"

        # Score based on z-score magnitude
        score = min(1.0, abs(zscore) / 3.0)
        raw_nav_pct = base_nav_pct * score * cerberus_multiplier
        nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / len(symbols))

        intents.append(HydraIntent(
            head="MEAN_REVERT",
            symbol=symbol,
            side=side,
            nav_pct=nav_pct,
            score=score,
            rationale=f"Z-score={zscore:.2f}, cerb_mult={cerberus_multiplier:.2f}",
        ))

    return intents


def generate_relative_value_intents(
    pair_edges: List[Dict[str, Any]],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.02,
    min_edge_score: float = 0.3,
) -> List[HydraIntent]:
    """
    Generate RELATIVE_VALUE intents from Crossfire pair edges.

    Args:
        pair_edges: List of pair edge dicts from cross_pair_engine
        cerberus_multiplier: Cerberus RELATIVE_VALUE multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size per leg
        min_edge_score: Minimum edge score threshold

    Returns:
        List of HydraIntent for RELATIVE_VALUE head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []
    direction = head_cfg.direction

    for edge in pair_edges:
        score = edge.get("edge_score", 0.0)
        if score < min_edge_score:
            continue

        # Extract pair info
        long_symbol = edge.get("long_symbol", "")
        short_symbol = edge.get("short_symbol", "")

        if not long_symbol or not short_symbol:
            continue

        # Generate pair of intents
        raw_nav_pct = base_nav_pct * score * cerberus_multiplier
        nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / 10)  # Cap per pair

        if direction != "short":
            intents.append(HydraIntent(
                head="RELATIVE_VALUE",
                symbol=long_symbol,
                side="long",
                nav_pct=nav_pct,
                score=score,
                rationale=f"RV pair: long {long_symbol} vs {short_symbol}, edge={score:.2f}",
            ))

        if direction != "long":
            intents.append(HydraIntent(
                head="RELATIVE_VALUE",
                symbol=short_symbol,
                side="short",
                nav_pct=nav_pct,
                score=score,
                rationale=f"RV pair: short {short_symbol} vs {long_symbol}, edge={score:.2f}",
            ))

    return intents


def generate_category_intents(
    category_scores: Dict[str, float],
    symbol_categories: Dict[str, str],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.015,
    min_category_score: float = 0.2,
) -> List[HydraIntent]:
    """
    Generate CATEGORY intents from category momentum scores.

    Args:
        category_scores: category -> momentum score (-1 to 1)
        symbol_categories: symbol -> category mapping
        cerberus_multiplier: Cerberus CATEGORY multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size
        min_category_score: Minimum category score threshold

    Returns:
        List of HydraIntent for CATEGORY head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []
    direction = head_cfg.direction

    for symbol, category in symbol_categories.items():
        cat_score = category_scores.get(category, 0.0)
        if abs(cat_score) < min_category_score:
            continue

        # Determine side based on category momentum
        if cat_score > 0:
            if direction == "short":
                continue
            side = "long"
        else:
            if direction == "long":
                continue
            side = "short"

        score = abs(cat_score)
        raw_nav_pct = base_nav_pct * score * cerberus_multiplier
        nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / 10)

        intents.append(HydraIntent(
            head="CATEGORY",
            symbol=symbol,
            side=side,
            nav_pct=nav_pct,
            score=score,
            rationale=f"Category={category}, score={cat_score:.2f}, cerb_mult={cerberus_multiplier:.2f}",
        ))

    return intents


def generate_vol_harvest_intents(
    symbols: List[str],
    vol_targets: Dict[str, float],
    realized_vols: Dict[str, float],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.02,
    vol_ratio_threshold: float = 0.7,
) -> List[HydraIntent]:
    """
    Generate VOL_HARVEST intents from volatility targeting signals.

    When realized vol < target vol, size up; when realized vol > target, size down.

    Args:
        symbols: Universe of symbols
        vol_targets: symbol -> target volatility
        realized_vols: symbol -> realized volatility
        cerberus_multiplier: Cerberus VOL_HARVEST multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size
        vol_ratio_threshold: Min vol ratio to act

    Returns:
        List of HydraIntent for VOL_HARVEST head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []

    for symbol in symbols:
        target_vol = vol_targets.get(symbol, 0.0)
        realized_vol = realized_vols.get(symbol, 0.0)

        if target_vol <= 0 or realized_vol <= 0:
            continue

        # Vol ratio determines sizing adjustment
        vol_ratio = target_vol / realized_vol
        if vol_ratio < vol_ratio_threshold or vol_ratio > (1.0 / vol_ratio_threshold):
            # Generate a neutral "sizing" intent
            score = min(1.0, abs(vol_ratio - 1.0))
            raw_nav_pct = base_nav_pct * vol_ratio * cerberus_multiplier
            nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / len(symbols))

            # Vol harvest is direction-agnostic; inherit direction from other heads
            # For now, generate as "long" placeholder
            intents.append(HydraIntent(
                head="VOL_HARVEST",
                symbol=symbol,
                side="long",  # Direction to be refined by other heads
                nav_pct=nav_pct,
                score=score,
                rationale=f"Vol ratio={vol_ratio:.2f}, cerb_mult={cerberus_multiplier:.2f}",
            ))

    return intents


def generate_emergent_alpha_intents(
    universe_scores: Dict[str, float],
    alpha_miner_signals: Dict[str, Dict[str, Any]],
    cerberus_multiplier: float,
    head_cfg: HydraHeadConfig,
    nav_usd: float,
    base_nav_pct: float = 0.01,
    min_alpha_score: float = 0.4,
) -> List[HydraIntent]:
    """
    Generate EMERGENT_ALPHA intents from Prospector/Universe optimizer.

    Args:
        universe_scores: symbol -> universe optimizer score
        alpha_miner_signals: symbol -> alpha miner signal dict
        cerberus_multiplier: Cerberus EMERGENT_ALPHA multiplier
        head_cfg: Head configuration
        nav_usd: Current NAV in USD
        base_nav_pct: Base position size
        min_alpha_score: Minimum alpha score threshold

    Returns:
        List of HydraIntent for EMERGENT_ALPHA head
    """
    if not head_cfg.enabled:
        return []

    intents: List[HydraIntent] = []
    direction = head_cfg.direction

    for symbol, uni_score in universe_scores.items():
        alpha_signal = alpha_miner_signals.get(symbol, {})
        alpha_score = alpha_signal.get("alpha_score", 0.0)

        # Combined score
        combined = (uni_score + alpha_score) / 2.0
        if abs(combined) < min_alpha_score:
            continue

        # Determine side
        if combined > 0:
            if direction == "short":
                continue
            side = "long"
        else:
            if direction == "long":
                continue
            side = "short"

        score = abs(combined)
        raw_nav_pct = base_nav_pct * score * cerberus_multiplier
        nav_pct = min(raw_nav_pct, head_cfg.max_nav_pct / 10)

        intents.append(HydraIntent(
            head="EMERGENT_ALPHA",
            symbol=symbol,
            side=side,
            nav_pct=nav_pct,
            score=score,
            rationale=f"Universe={uni_score:.2f}, alpha={alpha_score:.2f}, combined={combined:.2f}",
        ))

    return intents


# ---------------------------------------------------------------------------
# Budget Enforcement
# ---------------------------------------------------------------------------


def enforce_head_budgets(
    intents: List[HydraIntent],
    head_configs: Dict[str, HydraHeadConfig],
) -> Tuple[List[HydraIntent], Dict[str, float]]:
    """
    Enforce per-head NAV budget constraints by scaling down if needed.

    Args:
        intents: All raw intents from all heads
        head_configs: Per-head configuration

    Returns:
        (scaled_intents, head_usage) where head_usage[head] = total NAV used
    """
    # Group intents by head
    by_head: Dict[str, List[HydraIntent]] = {}
    for intent in intents:
        by_head.setdefault(intent.head, []).append(intent)

    scaled_intents: List[HydraIntent] = []
    head_usage: Dict[str, float] = {}

    for head, head_intents in by_head.items():
        cfg = head_configs.get(head)
        if not cfg:
            continue

        # Sum NAV for this head
        total_nav = sum(i.nav_pct for i in head_intents)
        head_usage[head] = total_nav

        if total_nav <= cfg.max_nav_pct:
            # Within budget
            scaled_intents.extend(head_intents)
        else:
            # Scale down proportionally
            scale = cfg.max_nav_pct / total_nav
            for intent in head_intents:
                scaled = HydraIntent(
                    head=intent.head,
                    symbol=intent.symbol,
                    side=intent.side,
                    nav_pct=intent.nav_pct * scale,
                    score=intent.score,
                    rationale=intent.rationale + f" [scaled {scale:.2f}]",
                    timestamp=intent.timestamp,
                )
                scaled_intents.append(scaled)
            head_usage[head] = cfg.max_nav_pct

    return scaled_intents, head_usage


# ---------------------------------------------------------------------------
# Conflict Resolution
# ---------------------------------------------------------------------------


def resolve_symbol_conflict(
    symbol: str,
    intents: List[HydraIntent],
    head_configs: Dict[str, HydraHeadConfig],
    conflict_cfg: HydraConflictResolution,
) -> HydraMergedIntent:
    """
    Resolve conflicting intents for a single symbol.

    Args:
        symbol: The symbol with multiple intents
        intents: All intents for this symbol
        head_configs: Per-head configuration
        conflict_cfg: Conflict resolution config

    Returns:
        HydraMergedIntent representing the resolved position
    """
    if not intents:
        return HydraMergedIntent(
            symbol=symbol,
            net_side="flat",
            nav_pct=0.0,
            heads=[],
            head_contributions={},
            score=0.0,
            rationale="No intents",
        )

    # Separate by side
    longs = [i for i in intents if i.side == "long"]
    shorts = [i for i in intents if i.side == "short"]

    long_nav = sum(i.nav_pct for i in longs)
    short_nav = sum(i.nav_pct for i in shorts)

    # Check for conflict
    if longs and shorts:
        # Count disagreeing heads
        disagreement_count = min(len(set(i.head for i in longs)), len(set(i.head for i in shorts)))

        if disagreement_count > conflict_cfg.max_head_disagreement:
            # Too much disagreement -> go flat
            return HydraMergedIntent(
                symbol=symbol,
                net_side="flat",
                nav_pct=0.0,
                heads=list(set(i.head for i in intents)),
                head_contributions={},
                score=0.0,
                rationale=f"Excessive disagreement: {disagreement_count} heads",
            )

        if conflict_cfg.allow_netting:
            # Net the positions
            net_nav = long_nav - short_nav
            if abs(net_nav) < 0.001:
                net_side = "flat"
                final_nav = 0.0
            elif net_nav > 0:
                net_side = "long"
                final_nav = net_nav
            else:
                net_side = "short"
                final_nav = abs(net_nav)
        else:
            # Winner takes all based on priority/score
            if conflict_cfg.prefer_higher_priority:
                long_priority = max(
                    (head_configs.get(i.head, HydraHeadConfig(name=i.head)).priority for i in longs),
                    default=0,
                )
                short_priority = max(
                    (head_configs.get(i.head, HydraHeadConfig(name=i.head)).priority for i in shorts),
                    default=0,
                )
                if long_priority > short_priority:
                    net_side = "long"
                    final_nav = long_nav
                elif short_priority > long_priority:
                    net_side = "short"
                    final_nav = short_nav
                else:
                    # Tie: use score
                    long_score = max((i.score for i in longs), default=0)
                    short_score = max((i.score for i in shorts), default=0)
                    if long_score >= short_score:
                        net_side = "long"
                        final_nav = long_nav
                    else:
                        net_side = "short"
                        final_nav = short_nav
            elif conflict_cfg.prefer_higher_score:
                long_score = max((i.score for i in longs), default=0)
                short_score = max((i.score for i in shorts), default=0)
                if long_score >= short_score:
                    net_side = "long"
                    final_nav = long_nav
                else:
                    net_side = "short"
                    final_nav = short_nav
            else:
                # Default: larger NAV wins
                if long_nav >= short_nav:
                    net_side = "long"
                    final_nav = long_nav
                else:
                    net_side = "short"
                    final_nav = short_nav
    else:
        # No conflict
        if longs:
            net_side = "long"
            final_nav = long_nav
        else:
            net_side = "short"
            final_nav = short_nav

    # Build head contributions
    head_contributions: Dict[str, float] = {}
    for intent in intents:
        contribution = intent.nav_pct if intent.side == net_side else -intent.nav_pct
        head_contributions[intent.head] = head_contributions.get(intent.head, 0.0) + contribution

    # Compute weighted average score
    all_scores = [i.score * i.nav_pct for i in intents]
    all_navs = [i.nav_pct for i in intents]
    avg_score = sum(all_scores) / max(sum(all_navs), 0.0001)

    return HydraMergedIntent(
        symbol=symbol,
        net_side=net_side,
        nav_pct=final_nav,
        heads=list(set(i.head for i in intents)),
        head_contributions=head_contributions,
        score=avg_score,
        rationale=f"Merged from {len(intents)} intents: {', '.join(set(i.head for i in intents))}",
    )


def apply_symbol_head_limit(
    merged: HydraMergedIntent,
    max_symbol_heads: int,
    head_configs: Dict[str, HydraHeadConfig],
) -> HydraMergedIntent:
    """
    Limit the number of heads contributing to a single symbol.

    Drops lowest-priority/score contributions when over limit.

    Args:
        merged: The merged intent
        max_symbol_heads: Maximum heads per symbol
        head_configs: Per-head configuration

    Returns:
        Updated HydraMergedIntent with limited heads
    """
    if len(merged.heads) <= max_symbol_heads:
        return merged

    # Sort heads by priority (descending) then by absolute contribution
    sorted_heads = sorted(
        merged.heads,
        key=lambda h: (
            head_configs.get(h, HydraHeadConfig(name=h)).priority,
            abs(merged.head_contributions.get(h, 0.0)),
        ),
        reverse=True,
    )

    # Keep top N heads
    kept_heads = sorted_heads[:max_symbol_heads]
    kept_contributions = {h: merged.head_contributions[h] for h in kept_heads if h in merged.head_contributions}

    # Recalculate NAV
    net_nav = sum(kept_contributions.values())
    if net_nav < 0:
        net_side = "short"
        final_nav = abs(net_nav)
    elif net_nav > 0:
        net_side = "long"
        final_nav = net_nav
    else:
        net_side = "flat"
        final_nav = 0.0

    return HydraMergedIntent(
        symbol=merged.symbol,
        net_side=net_side,
        nav_pct=final_nav,
        heads=kept_heads,
        head_contributions=kept_contributions,
        score=merged.score,
        rationale=merged.rationale + f" [limited to {max_symbol_heads} heads]",
        timestamp=merged.timestamp,
    )


# ---------------------------------------------------------------------------
# Core Routing Function
# ---------------------------------------------------------------------------


def hydra_route_intents(
    cfg: HydraConfig,
    all_intents: List[HydraIntent],
) -> Tuple[List[HydraMergedIntent], Dict[str, float]]:
    """
    Core Hydra routing: merge intents across heads with budget and conflict resolution.

    Steps:
    1. Enforce per-head budget constraints
    2. Group by symbol
    3. Resolve conflicts per symbol
    4. Apply symbol-level head limits
    5. Rank and trim to max_intents_per_cycle

    Args:
        cfg: HydraConfig
        all_intents: All raw intents from all heads

    Returns:
        (merged_intents, head_usage) where head_usage[head] = NAV used
    """
    if not all_intents:
        return [], {}

    # Step 1: Enforce per-head budgets
    scaled_intents, head_usage = enforce_head_budgets(all_intents, cfg.heads)

    # Step 2: Group by symbol
    by_symbol: Dict[str, List[HydraIntent]] = {}
    for intent in scaled_intents:
        by_symbol.setdefault(intent.symbol, []).append(intent)

    # Step 3: Resolve conflicts per symbol
    merged_intents: List[HydraMergedIntent] = []
    for symbol, intents in by_symbol.items():
        merged = resolve_symbol_conflict(
            symbol=symbol,
            intents=intents,
            head_configs=cfg.heads,
            conflict_cfg=cfg.conflict_resolution,
        )
        # Skip flat positions
        if merged.net_side != "flat" and merged.nav_pct > 0:
            merged_intents.append(merged)

    # Step 4: Apply symbol-level head limits
    merged_intents = [
        apply_symbol_head_limit(m, cfg.intent_limits.max_symbol_heads, cfg.heads)
        for m in merged_intents
    ]

    # Filter out any that became flat after limiting
    merged_intents = [m for m in merged_intents if m.net_side != "flat" and m.nav_pct > 0]

    # Step 5: Rank and trim
    # Sort by score (descending), then by nav_pct (descending)
    merged_intents.sort(key=lambda m: (m.score, m.nav_pct), reverse=True)

    # Trim to limit
    if len(merged_intents) > cfg.intent_limits.max_intents_per_cycle:
        merged_intents = merged_intents[: cfg.intent_limits.max_intents_per_cycle]

    return merged_intents, head_usage


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_hydra_step(
    cfg: HydraConfig,
    cerberus_multipliers: Dict[str, float],
    symbols: List[str],
    hybrid_scores: Dict[str, float],
    zscore_map: Dict[str, float],
    pair_edges: List[Dict[str, Any]],
    category_scores: Dict[str, float],
    symbol_categories: Dict[str, str],
    vol_targets: Dict[str, float],
    realized_vols: Dict[str, float],
    universe_scores: Dict[str, float],
    alpha_miner_signals: Dict[str, Dict[str, Any]],
    nav_usd: float,
    cycle_count: int = 0,
    state_path: Path | str | None = None,
    log_path: Path | str | None = None,
) -> Tuple[List[HydraMergedIntent], HydraState]:
    """
    Run one Hydra cycle: generate intents, route, and produce merged output.

    Args:
        cfg: HydraConfig
        cerberus_multipliers: head -> multiplier from Cerberus
        symbols: Universe of symbols
        hybrid_scores: symbol -> hybrid trend/carry score
        zscore_map: symbol -> z-score for mean reversion
        pair_edges: Crossfire pair edges
        category_scores: category -> momentum score
        symbol_categories: symbol -> category
        vol_targets: symbol -> target vol
        realized_vols: symbol -> realized vol
        universe_scores: symbol -> universe optimizer score
        alpha_miner_signals: symbol -> alpha miner signal dict
        nav_usd: Current NAV in USD
        cycle_count: Current execution cycle
        state_path: Path to save state
        log_path: Path to log intents

    Returns:
        (merged_intents, hydra_state)
    """
    if not cfg.enabled:
        # Return empty state when disabled
        state = HydraState(
            updated_ts=datetime.now(timezone.utc).isoformat(),
            head_budgets={h: cfg.heads[h].max_nav_pct for h in STRATEGY_HEADS if h in cfg.heads},
            head_usage={h: 0.0 for h in STRATEGY_HEADS},
            head_positions={h: 0 for h in STRATEGY_HEADS},
            merged_intents=[],
            cycle_count=cycle_count,
            notes=["Hydra disabled"],
        )
        return [], state

    notes: List[str] = []
    errors: List[str] = []
    all_intents: List[HydraIntent] = []

    # Generate intents from each head
    try:
        trend_intents = generate_trend_intents(
            symbols=symbols,
            hybrid_scores=hybrid_scores,
            cerberus_multiplier=cerberus_multipliers.get("TREND", 1.0),
            head_cfg=cfg.heads["TREND"],
            nav_usd=nav_usd,
        )
        all_intents.extend(trend_intents)
        notes.append(f"TREND: {len(trend_intents)} intents")
    except Exception as e:
        errors.append(f"TREND generator error: {e}")

    try:
        mr_intents = generate_mean_revert_intents(
            symbols=symbols,
            zscore_map=zscore_map,
            cerberus_multiplier=cerberus_multipliers.get("MEAN_REVERT", 1.0),
            head_cfg=cfg.heads["MEAN_REVERT"],
            nav_usd=nav_usd,
        )
        all_intents.extend(mr_intents)
        notes.append(f"MEAN_REVERT: {len(mr_intents)} intents")
    except Exception as e:
        errors.append(f"MEAN_REVERT generator error: {e}")

    try:
        rv_intents = generate_relative_value_intents(
            pair_edges=pair_edges,
            cerberus_multiplier=cerberus_multipliers.get("RELATIVE_VALUE", 1.0),
            head_cfg=cfg.heads["RELATIVE_VALUE"],
            nav_usd=nav_usd,
        )
        all_intents.extend(rv_intents)
        notes.append(f"RELATIVE_VALUE: {len(rv_intents)} intents")
    except Exception as e:
        errors.append(f"RELATIVE_VALUE generator error: {e}")

    try:
        cat_intents = generate_category_intents(
            category_scores=category_scores,
            symbol_categories=symbol_categories,
            cerberus_multiplier=cerberus_multipliers.get("CATEGORY", 1.0),
            head_cfg=cfg.heads["CATEGORY"],
            nav_usd=nav_usd,
        )
        all_intents.extend(cat_intents)
        notes.append(f"CATEGORY: {len(cat_intents)} intents")
    except Exception as e:
        errors.append(f"CATEGORY generator error: {e}")

    try:
        vol_intents = generate_vol_harvest_intents(
            symbols=symbols,
            vol_targets=vol_targets,
            realized_vols=realized_vols,
            cerberus_multiplier=cerberus_multipliers.get("VOL_HARVEST", 1.0),
            head_cfg=cfg.heads["VOL_HARVEST"],
            nav_usd=nav_usd,
        )
        all_intents.extend(vol_intents)
        notes.append(f"VOL_HARVEST: {len(vol_intents)} intents")
    except Exception as e:
        errors.append(f"VOL_HARVEST generator error: {e}")

    try:
        alpha_intents = generate_emergent_alpha_intents(
            universe_scores=universe_scores,
            alpha_miner_signals=alpha_miner_signals,
            cerberus_multiplier=cerberus_multipliers.get("EMERGENT_ALPHA", 1.0),
            head_cfg=cfg.heads["EMERGENT_ALPHA"],
            nav_usd=nav_usd,
        )
        all_intents.extend(alpha_intents)
        notes.append(f"EMERGENT_ALPHA: {len(alpha_intents)} intents")
    except Exception as e:
        errors.append(f"EMERGENT_ALPHA generator error: {e}")

    notes.append(f"Total raw intents: {len(all_intents)}")

    # Route intents
    merged_intents, head_usage = hydra_route_intents(cfg, all_intents)
    notes.append(f"Merged intents: {len(merged_intents)}")

    # Count positions per head
    head_positions: Dict[str, int] = {h: 0 for h in STRATEGY_HEADS}
    for merged in merged_intents:
        for head in merged.heads:
            head_positions[head] = head_positions.get(head, 0) + 1

    # Build state
    state = HydraState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        head_budgets={h: cfg.heads[h].max_nav_pct for h in STRATEGY_HEADS if h in cfg.heads},
        head_usage=head_usage,
        head_positions=head_positions,
        merged_intents=[m.to_dict() for m in merged_intents],
        cycle_count=cycle_count,
        notes=notes,
        errors=errors,
        meta={
            "total_raw_intents": len(all_intents),
            "enabled_heads": cfg.get_enabled_heads(),
        },
    )

    # Save state
    if state_path is not None or DEFAULT_STATE_PATH.parent.exists():
        save_hydra_state(state, state_path)

    # Log intents
    if log_path is not None or DEFAULT_INTENT_LOG_PATH.parent.exists():
        log_hydra_intents(merged_intents, log_path)

    return merged_intents, state


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def get_hydra_nav_allocation(
    symbol: str,
    hydra_state: HydraState | None,
) -> Tuple[float, str, List[str]]:
    """
    Get Hydra's NAV allocation for a symbol.

    Args:
        symbol: Symbol to look up
        hydra_state: Current Hydra state

    Returns:
        (nav_pct, side, heads) tuple, or (0.0, "flat", []) if not found
    """
    if not hydra_state or not hydra_state.merged_intents:
        return 0.0, "flat", []

    for intent in hydra_state.merged_intents:
        if intent.get("symbol") == symbol:
            return (
                intent.get("nav_pct", 0.0),
                intent.get("net_side", "flat"),
                intent.get("heads", []),
            )

    return 0.0, "flat", []


def get_hydra_head_exposure(
    head: str,
    hydra_state: HydraState | None,
) -> Tuple[float, int]:
    """
    Get a head's current exposure and position count.

    Args:
        head: Head name
        hydra_state: Current Hydra state

    Returns:
        (usage_nav_pct, position_count)
    """
    if not hydra_state:
        return 0.0, 0

    usage = hydra_state.head_usage.get(head, 0.0)
    positions = hydra_state.head_positions.get(head, 0)
    return usage, positions


def is_hydra_enabled(strategy_config: Mapping[str, Any] | None = None) -> bool:
    """
    Check if Hydra is enabled in config.

    Args:
        strategy_config: Pre-loaded strategy config

    Returns:
        True if enabled
    """
    cfg = load_hydra_config(strategy_config=strategy_config)
    return cfg.enabled


def hydra_merged_intent_to_execution_intent(
    merged: HydraMergedIntent | Dict[str, Any],
    nav_usd: float,
    price: float,
) -> Dict[str, Any]:
    """
    Convert a Hydra merged intent to an execution intent dict.

    This adapter maps Hydra's output to the existing execution pipeline format.

    Args:
        merged: HydraMergedIntent or dict representation
        nav_usd: Current NAV in USD
        price: Current price for the symbol

    Returns:
        Execution intent dict compatible with existing pipeline
    """
    if isinstance(merged, HydraMergedIntent):
        data = merged.to_dict()
    else:
        data = merged

    symbol = data.get("symbol", "")
    net_side = data.get("net_side", "flat")
    nav_pct = data.get("nav_pct", 0.0)
    heads = data.get("heads", [])
    head_contributions = data.get("head_contributions", {})
    score = data.get("score", 0.0)

    if net_side == "flat" or nav_pct <= 0:
        return {}

    # Calculate notional
    notional_usd = nav_usd * nav_pct

    # Calculate quantity (simplified)
    qty = notional_usd / price if price > 0 else 0.0

    return {
        "symbol": symbol,
        "side": net_side.upper(),  # "LONG" / "SHORT"
        "qty": qty,
        "notional_usd": notional_usd,
        "nav_pct": nav_pct,
        "score": score,
        "strategy_heads": heads,
        "head_contributions": head_contributions,
        "source": "hydra",
    }


def get_hydra_throttled_budgets(
    cfg: HydraConfig,
    pnl_state: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Get per-head NAV budgets with Hydra PnL throttle applied.

    Args:
        cfg: HydraConfig
        pnl_state: HydraPnlState from hydra_pnl module (optional)

    Returns:
        Dict of head -> throttled budget (nav_pct)
    """
    # Get base budgets from config
    base_budgets = {h: cfg.heads[h].max_nav_pct for h in STRATEGY_HEADS if h in cfg.heads}

    # Apply PnL throttle if available
    try:
        from execution.hydra_pnl import apply_pnl_throttle_to_hydra_budgets
        return apply_pnl_throttle_to_hydra_budgets(base_budgets, pnl_state)
    except ImportError:
        return base_budgets


def get_hydra_head_active_status(
    cfg: HydraConfig,
    pnl_state: Optional[Any] = None,
) -> Dict[str, bool]:
    """
    Get active/enabled status for each head considering kill switches.

    Args:
        cfg: HydraConfig
        pnl_state: HydraPnlState from hydra_pnl module (optional)

    Returns:
        Dict of head -> active (True = enabled and not killed)
    """
    result = {}
    for head in STRATEGY_HEADS:
        if head not in cfg.heads:
            result[head] = False
            continue

        # Check config enable
        if not cfg.heads[head].enabled:
            result[head] = False
            continue

        # Check kill switch
        try:
            from execution.hydra_pnl import is_head_active
            result[head] = is_head_active(head, pnl_state)
        except ImportError:
            result[head] = True

    return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "STRATEGY_HEADS",
    "DEFAULT_STATE_PATH",
    "DEFAULT_INTENT_LOG_PATH",
    # Data classes
    "HydraHeadConfig",
    "HydraIntentLimits",
    "HydraConflictResolution",
    "HydraConfig",
    "HydraIntent",
    "HydraMergedIntent",
    "HydraHeadBudget",
    "HydraState",
    # Config loading
    "load_hydra_config",
    # State I/O
    "load_hydra_state",
    "save_hydra_state",
    # Logging
    "log_hydra_intent",
    "log_hydra_intents",
    # Head generators
    "generate_trend_intents",
    "generate_mean_revert_intents",
    "generate_relative_value_intents",
    "generate_category_intents",
    "generate_vol_harvest_intents",
    "generate_emergent_alpha_intents",
    # Budget enforcement
    "enforce_head_budgets",
    # Conflict resolution
    "resolve_symbol_conflict",
    "apply_symbol_head_limit",
    # Core routing
    "hydra_route_intents",
    # Pipeline
    "run_hydra_step",
    # Integration helpers
    "get_hydra_nav_allocation",
    "get_hydra_head_exposure",
    "get_hydra_throttled_budgets",
    "get_hydra_head_active_status",
    "is_hydra_enabled",
    "hydra_merged_intent_to_execution_intent",
]
