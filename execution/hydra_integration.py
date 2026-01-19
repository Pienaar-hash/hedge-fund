"""
Hydra Executor Integration — v7.9_P1

Provides integration hooks for the Hydra Multi-Strategy Execution Engine
into the existing execution pipeline.

This module acts as a bridge between Hydra and the existing executor,
allowing Hydra to be enabled/disabled via config without modifying
the core executor logic.

Usage:
    from execution.hydra_integration import run_hydra_pipeline, is_hydra_enabled

    if is_hydra_enabled(strategy_config):
        merged_intents, hydra_state = run_hydra_pipeline(...)
    else:
        # Use existing single-strategy flow
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from execution.hydra_engine import (
    HydraMergedIntent,
    HydraState,
    load_hydra_config,
    load_hydra_state,
    run_hydra_step,
    hydra_merged_intent_to_execution_intent,
    is_hydra_enabled,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_hydra_pipeline(
    strategy_config: Mapping[str, Any],
    cerberus_multipliers: Dict[str, float],
    symbols: List[str],
    nav_usd: float,
    cycle_count: int = 0,
    # Intel surfaces (optional - stubbed if not provided)
    hybrid_scores: Optional[Dict[str, float]] = None,
    zscore_map: Optional[Dict[str, float]] = None,
    pair_edges: Optional[List[Dict[str, Any]]] = None,
    category_scores: Optional[Dict[str, float]] = None,
    symbol_categories: Optional[Dict[str, str]] = None,
    vol_targets: Optional[Dict[str, float]] = None,
    realized_vols: Optional[Dict[str, float]] = None,
    universe_scores: Optional[Dict[str, float]] = None,
    alpha_miner_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    state_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Tuple[List[HydraMergedIntent], HydraState]:
    """
    Run the Hydra pipeline if enabled, otherwise return empty results.

    This is the main entry point for executor integration.

    Args:
        strategy_config: Strategy configuration dict
        cerberus_multipliers: head -> multiplier from Cerberus
        symbols: Universe of symbols
        nav_usd: Current NAV in USD
        cycle_count: Current execution cycle
        hybrid_scores: symbol -> hybrid trend/carry score
        zscore_map: symbol -> z-score for mean reversion
        pair_edges: Crossfire pair edges
        category_scores: category -> momentum score
        symbol_categories: symbol -> category
        vol_targets: symbol -> target vol
        realized_vols: symbol -> realized vol
        universe_scores: symbol -> universe optimizer score
        alpha_miner_signals: symbol -> alpha miner signal dict
        state_path: Path to save state
        log_path: Path to log intents

    Returns:
        (merged_intents, hydra_state) - empty if Hydra disabled
    """
    cfg = load_hydra_config(strategy_config=strategy_config)

    if not cfg.enabled:
        _LOG.debug("Hydra disabled, skipping pipeline")
        return [], HydraState()

    # Provide empty defaults for missing intel surfaces
    hybrid_scores = hybrid_scores or {}
    zscore_map = zscore_map or {}
    pair_edges = pair_edges or []
    category_scores = category_scores or {}
    symbol_categories = symbol_categories or {}
    vol_targets = vol_targets or {}
    realized_vols = realized_vols or {}
    universe_scores = universe_scores or {}
    alpha_miner_signals = alpha_miner_signals or {}

    # Run Hydra step
    merged_intents, hydra_state = run_hydra_step(
        cfg=cfg,
        cerberus_multipliers=cerberus_multipliers,
        symbols=symbols,
        hybrid_scores=hybrid_scores,
        zscore_map=zscore_map,
        pair_edges=pair_edges,
        category_scores=category_scores,
        symbol_categories=symbol_categories,
        vol_targets=vol_targets,
        realized_vols=realized_vols,
        universe_scores=universe_scores,
        alpha_miner_signals=alpha_miner_signals,
        nav_usd=nav_usd,
        cycle_count=cycle_count,
        state_path=state_path,
        log_path=log_path,
    )

    _LOG.info(
        "Hydra pipeline complete: %d merged intents from %d heads",
        len(merged_intents),
        len([h for h in cfg.get_enabled_heads()]),
    )

    return merged_intents, hydra_state


def convert_hydra_intents_to_execution(
    merged_intents: List[HydraMergedIntent],
    nav_usd: float,
    prices: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Convert Hydra merged intents to execution intent dicts.

    Args:
        merged_intents: List of HydraMergedIntent
        nav_usd: Current NAV in USD
        prices: symbol -> current price

    Returns:
        List of execution intent dicts
    """
    execution_intents = []

    for merged in merged_intents:
        if isinstance(merged, dict):
            symbol = merged.get("symbol", "")
        else:
            symbol = merged.symbol

        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue

        intent = hydra_merged_intent_to_execution_intent(merged, nav_usd, price)
        if intent:
            execution_intents.append(intent)

    return execution_intents


def get_hydra_attribution_for_order(
    symbol: str,
    side: str,
    hydra_state: Optional[HydraState] = None,
) -> Dict[str, Any]:
    """
    Get Hydra attribution metadata for an order.

    This should be added to order logs for PnL attribution.

    Args:
        symbol: Symbol being traded
        side: Order side ("long" / "short")
        hydra_state: Current Hydra state

    Returns:
        Attribution dict with strategy_heads and head_contributions
    """
    if not hydra_state or not hydra_state.merged_intents:
        return {"strategy_heads": [], "head_contributions": {}, "source": "legacy"}

    # Find matching intent
    for intent in hydra_state.merged_intents:
        if isinstance(intent, dict):
            intent_symbol = intent.get("symbol", "")
            intent_side = intent.get("net_side", "")
            heads = intent.get("heads", [])
            contributions = intent.get("head_contributions", {})
        else:
            intent_symbol = intent.symbol
            intent_side = intent.net_side
            heads = intent.heads
            contributions = intent.head_contributions

        if intent_symbol == symbol and intent_side.lower() == side.lower():
            return {
                "strategy_heads": heads,
                "head_contributions": contributions,
                "source": "hydra",
            }

    return {"strategy_heads": [], "head_contributions": {}, "source": "legacy"}


def merge_with_single_strategy_intents(
    hydra_intents: List[Dict[str, Any]],
    legacy_intents: List[Dict[str, Any]],
    prefer_hydra: bool = True,
) -> List[Dict[str, Any]]:
    """
    Merge Hydra intents with legacy single-strategy intents.

    When both Hydra and legacy produce intents for the same symbol,
    this function resolves which to use.

    Args:
        hydra_intents: Intents from Hydra pipeline
        legacy_intents: Intents from legacy single-strategy pipeline
        prefer_hydra: If True, Hydra wins on conflict

    Returns:
        Merged list of intents
    """
    if not hydra_intents:
        return legacy_intents

    if not legacy_intents:
        return hydra_intents

    # Build symbol set from Hydra
    hydra_symbols = {i.get("symbol") for i in hydra_intents}

    # Filter legacy intents
    if prefer_hydra:
        filtered_legacy = [i for i in legacy_intents if i.get("symbol") not in hydra_symbols]
        return hydra_intents + filtered_legacy
    else:
        legacy_symbols = {i.get("symbol") for i in legacy_intents}
        filtered_hydra = [i for i in hydra_intents if i.get("symbol") not in legacy_symbols]
        return legacy_intents + filtered_hydra


# ---------------------------------------------------------------------------
# Head Contributions Persistence (v7.9_P2)
# ---------------------------------------------------------------------------


def persist_head_contributions_for_position(
    symbol: str,
    side: str,
    head_contributions: Dict[str, float],
    strategy_heads: Optional[List[str]] = None,
) -> bool:
    """
    Persist head_contributions when a Hydra position is opened.
    
    This should be called alongside register_position_tp_sl() when
    a new position is created from a Hydra intent.
    
    Args:
        symbol: Trading symbol
        side: Position side ("LONG" / "SHORT")
        head_contributions: {"TREND": 0.7, "CATEGORY": 0.3, ...}
        strategy_heads: List of contributing head names
        
    Returns:
        True if persisted successfully
    """
    try:
        from execution.position_tp_sl_registry import get_position_tp_sl, _make_key, _load_registry, _save_registry, _TP_SL_REGISTRY
        
        _load_registry()
        key = _make_key(symbol, side)
        
        if key in _TP_SL_REGISTRY:
            # Update existing entry
            _TP_SL_REGISTRY[key]["head_contributions"] = head_contributions
            if strategy_heads:
                _TP_SL_REGISTRY[key]["strategy_heads"] = strategy_heads
            _save_registry()
            _LOG.debug("Updated head_contributions for %s: %s", key, list(head_contributions.keys()))
        else:
            # Create minimal entry for attribution (no TP/SL yet)
            _TP_SL_REGISTRY[key] = {
                "symbol": symbol.upper(),
                "position_side": side.upper(),
                "head_contributions": head_contributions,
                "strategy_heads": strategy_heads or list(head_contributions.keys()),
                "source": "hydra",
                "registered_at": __import__("time").time(),
            }
            _save_registry()
            _LOG.debug("Created head_contributions entry for %s: %s", key, list(head_contributions.keys()))
        
        return True
    except Exception as e:
        _LOG.warning("Failed to persist head_contributions for %s:%s: %s", symbol, side, e)
        return False


def get_head_contributions_for_position(symbol: str, side: str) -> Dict[str, float]:
    """
    Retrieve head_contributions for a position.
    
    Args:
        symbol: Trading symbol
        side: Position side ("LONG" / "SHORT")
        
    Returns:
        Dict of head -> weight, or empty dict if not found
    """
    try:
        from execution.position_tp_sl_registry import get_head_contributions
        return get_head_contributions(symbol, side)
    except ImportError:
        return {}


def get_all_position_head_contributions() -> Dict[str, Dict[str, float]]:
    """
    Get head contributions for all open positions.
    
    Returns:
        Dict of "SYMBOL:SIDE" -> head_contributions
    """
    try:
        from execution.position_tp_sl_registry import get_all_head_contributions
        return get_all_head_contributions()
    except ImportError:
        return {}


def enrich_fill_with_head_contributions(fill: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a fill record with head_contributions from the registry.
    
    Args:
        fill: Fill dict with symbol, side fields
        
    Returns:
        Fill dict with head_contributions added (if available)
    """
    symbol = fill.get("symbol", "")
    side = fill.get("side", fill.get("positionSide", ""))
    
    if not symbol or not side:
        return fill
    
    contributions = get_head_contributions_for_position(symbol, side.upper())
    if contributions:
        fill["head_contributions"] = contributions
        fill["source"] = "hydra"
    else:
        fill["head_contributions"] = {}
        fill["source"] = "legacy"
    
    return fill


def build_head_contributions_map_for_positions(
    positions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Build symbol -> head_contributions map for a list of positions.
    
    Used by hydra_pnl.update_unrealized_pnl() to attribute unrealized PnL.
    
    Args:
        positions: List of position dicts
        
    Returns:
        symbol -> head_contributions dict
    """
    result = {}
    for pos in positions:
        symbol = pos.get("symbol", "")
        side = pos.get("positionSide") or pos.get("side", "")
        if not side:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
            side = "LONG" if qty > 0 else "SHORT"
        
        contributions = get_head_contributions_for_position(symbol, side.upper())
        if contributions:
            result[symbol] = contributions
    
    return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "run_hydra_pipeline",
    "convert_hydra_intents_to_execution",
    "get_hydra_attribution_for_order",
    "merge_with_single_strategy_intents",
    "is_hydra_enabled",
    "load_hydra_config",
    "load_hydra_state",
    # Head contributions persistence (v7.9_P2)
    "persist_head_contributions_for_position",
    "get_head_contributions_for_position",
    "get_all_position_head_contributions",
    "enrich_fill_with_head_contributions",
    "build_head_contributions_map_for_positions",
]
