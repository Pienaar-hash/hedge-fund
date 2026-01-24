"""
Dataset Registry — Single source of truth for dataset admission state.

This module provides the centralized lookup for dataset admission status.
All dataset state checks must go through this module.

Reference: docs/DATASET_ADMISSION_GATE.md
Reference: docs/DATASET_ROLLBACK_CLAUSE.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

__all__ = [
    "DatasetState",
    "DatasetTier",
    "DatasetInfo",
    "get_dataset_state",
    "get_dataset_tier",
    "get_dataset_info",
    "is_production_eligible",
    "is_observe_only",
    "requires_cycle_boundary_rollback",
    "reload_registry",
]


class DatasetState(str, Enum):
    """Dataset admission states per DATASET_ADMISSION_GATE.md"""
    REJECTED = "REJECTED"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    PRODUCTION_ELIGIBLE = "PRODUCTION_ELIGIBLE"


class DatasetTier(str, Enum):
    """Dataset influence tiers per DATASET_ROLLBACK_CLAUSE.md"""
    EXISTENTIAL = "EXISTENTIAL"      # Cannot revoke, only substitute
    AUTHORITATIVE = "AUTHORITATIVE"  # Cannot revoke mid-cycle
    ADVISORY = "ADVISORY"            # May downgrade any time
    OBSERVATIONAL = "OBSERVATIONAL"  # May revoke any time
    UNKNOWN = "UNKNOWN"              # Not in registry


# Tier classification based on dataset_admission.json audit
_TIER_MAP: Dict[str, DatasetTier] = {
    # EXISTENTIAL: Loss = system halt
    "binance_futures_klines": DatasetTier.EXISTENTIAL,
    "binance_futures_positions": DatasetTier.EXISTENTIAL,
    "binance_futures_balance": DatasetTier.EXISTENTIAL,
    "binance_futures_fills": DatasetTier.EXISTENTIAL,
    # AUTHORITATIVE: Defines regime
    "sentinel_x_features": DatasetTier.AUTHORITATIVE,
    # ADVISORY: Influences signals/sizing
    "binance_futures_orderbook": DatasetTier.ADVISORY,
    "symbol_scores_v6": DatasetTier.ADVISORY,
    "expectancy_v6": DatasetTier.ADVISORY,
    "router_health": DatasetTier.ADVISORY,
    # OBSERVATIONAL: Display/diagnostic only
    "coingecko_prices": DatasetTier.OBSERVATIONAL,
    "polymarket_snapshot": DatasetTier.OBSERVATIONAL,
    "regime_pressure": DatasetTier.OBSERVATIONAL,
    "factor_diagnostics": DatasetTier.OBSERVATIONAL,
    "offchain_holdings": DatasetTier.OBSERVATIONAL,
}


@dataclass(frozen=True)
class DatasetInfo:
    """Immutable dataset admission info."""
    dataset_id: str
    state: DatasetState
    tier: DatasetTier
    grandfathered: bool
    influences_regime: bool
    influences_signals: bool
    influences_sizing: bool
    influences_exits: bool
    replay_deterministic: bool

    @property
    def influences_decisions(self) -> bool:
        """True if dataset influences any live decision."""
        return (
            self.influences_regime
            or self.influences_signals
            or self.influences_sizing
            or self.influences_exits
        )

    @property
    def can_rollback_immediately(self) -> bool:
        """True if dataset can be rolled back without waiting for cycle boundary."""
        return self.tier in (DatasetTier.ADVISORY, DatasetTier.OBSERVATIONAL)

    @property
    def requires_substitution_for_rollback(self) -> bool:
        """True if rollback requires pre-defined fallback."""
        return self.tier == DatasetTier.EXISTENTIAL


def _config_path() -> str:
    """Path to dataset admission registry."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "dataset_admission.json",
    )


@lru_cache(maxsize=1)
def _load_registry() -> Dict[str, Any]:
    """Load and cache the dataset admission registry."""
    path = _config_path()
    if not os.path.exists(path):
        return {"datasets": {}}
    with open(path, "r") as f:
        return json.load(f)


def reload_registry() -> None:
    """Force reload of the registry (e.g., after config change)."""
    _load_registry.cache_clear()


def _get_dataset_entry(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get raw entry from registry."""
    registry = _load_registry()
    return registry.get("datasets", {}).get(dataset_id)


def get_dataset_state(dataset_id: str) -> DatasetState:
    """
    Get the admission state of a dataset.
    
    Returns REJECTED if dataset is not in registry.
    """
    entry = _get_dataset_entry(dataset_id)
    if entry is None:
        return DatasetState.REJECTED
    state_str = entry.get("state", "REJECTED")
    try:
        return DatasetState(state_str)
    except ValueError:
        return DatasetState.REJECTED


def get_dataset_tier(dataset_id: str) -> DatasetTier:
    """
    Get the influence tier of a dataset.
    
    Returns UNKNOWN if dataset is not classified.
    """
    return _TIER_MAP.get(dataset_id, DatasetTier.UNKNOWN)


def get_dataset_info(dataset_id: str) -> DatasetInfo:
    """
    Get full dataset admission info.
    
    Returns minimal info with REJECTED state if not in registry.
    """
    entry = _get_dataset_entry(dataset_id)
    
    if entry is None:
        return DatasetInfo(
            dataset_id=dataset_id,
            state=DatasetState.REJECTED,
            tier=DatasetTier.UNKNOWN,
            grandfathered=False,
            influences_regime=False,
            influences_signals=False,
            influences_sizing=False,
            influences_exits=False,
            replay_deterministic=False,
        )
    
    influences = entry.get("influences", {})
    criteria = entry.get("criteria", {})
    replay = criteria.get("replay_determinism", {})
    
    return DatasetInfo(
        dataset_id=dataset_id,
        state=get_dataset_state(dataset_id),
        tier=get_dataset_tier(dataset_id),
        grandfathered=entry.get("grandfathered", False),
        influences_regime=influences.get("regime", False),
        influences_signals=influences.get("signals", False),
        influences_sizing=influences.get("sizing", False),
        influences_exits=influences.get("exits", False),
        replay_deterministic=replay.get("verified", False),
    )


def is_production_eligible(dataset_id: str) -> bool:
    """Check if dataset is eligible for production use."""
    return get_dataset_state(dataset_id) == DatasetState.PRODUCTION_ELIGIBLE


def is_observe_only(dataset_id: str) -> bool:
    """Check if dataset is observe-only (logged but not used for decisions)."""
    return get_dataset_state(dataset_id) == DatasetState.OBSERVE_ONLY


def requires_cycle_boundary_rollback(dataset_id: str) -> bool:
    """Check if dataset rollback must wait for cycle boundary."""
    tier = get_dataset_tier(dataset_id)
    return tier in (DatasetTier.EXISTENTIAL, DatasetTier.AUTHORITATIVE)
