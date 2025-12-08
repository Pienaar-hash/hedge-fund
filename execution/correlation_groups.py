"""
Correlation Groups Module

Provides utilities for computing correlation-aware position exposure.
Maps symbols to correlation groups and calculates group-level NAV exposure.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Set

from execution.risk_loader import CorrelationGroupsConfig


def build_symbol_to_groups_index(
    corr_cfg: CorrelationGroupsConfig,
) -> Dict[str, Set[str]]:
    """
    Build an index mapping each symbol to the set of groups it belongs to.

    A symbol can belong to multiple groups (overlapping groups are allowed).

    Args:
        corr_cfg: The correlation groups configuration.

    Returns:
        Dict mapping symbol (uppercase) -> set of group names.
    """
    symbol_to_groups: Dict[str, Set[str]] = {}

    for group_name, group_cfg in corr_cfg.groups.items():
        for symbol in group_cfg.symbols:
            sym_upper = str(symbol).upper()
            if sym_upper not in symbol_to_groups:
                symbol_to_groups[sym_upper] = set()
            symbol_to_groups[sym_upper].add(group_name)

    return symbol_to_groups


def _get_position_notional(position: Any) -> float:
    """
    Extract absolute notional USD from a position object.

    Handles both dict-like positions and objects with attributes.
    """
    try:
        if isinstance(position, Mapping):
            # Try common field names for notional
            notional = position.get("notional") or position.get("notional_usd")
            if notional is not None:
                return abs(float(notional))
            # Fallback: compute from positionAmt * markPrice
            amt = position.get("positionAmt") or position.get("amt") or 0
            price = position.get("markPrice") or position.get("entryPrice") or 0
            return abs(float(amt) * float(price))
        else:
            # Object with attributes
            notional = getattr(position, "notional", None) or getattr(position, "notional_usd", None)
            if notional is not None:
                return abs(float(notional))
            amt = getattr(position, "positionAmt", None) or getattr(position, "amt", 0)
            price = getattr(position, "markPrice", None) or getattr(position, "entryPrice", 0)
            return abs(float(amt or 0) * float(price or 0))
    except Exception:
        return 0.0


def _get_position_symbol(position: Any) -> str:
    """Extract symbol from a position object."""
    try:
        if isinstance(position, Mapping):
            return str(position.get("symbol") or "").upper()
        else:
            return str(getattr(position, "symbol", "") or "").upper()
    except Exception:
        return ""


def compute_group_exposure_nav_pct(
    positions: Iterable[Any],
    nav_total_usd: float,
    corr_cfg: CorrelationGroupsConfig,
) -> Dict[str, float]:
    """
    Compute gross NAV percentage exposure for each correlation group.

    Gross exposure is the sum of absolute notional of all positions whose
    symbol is in that group, divided by nav_total_usd.

    Args:
        positions: Iterable of position objects (dicts or objects with symbol/notional).
        nav_total_usd: Total NAV in USD.
        corr_cfg: Correlation groups configuration.

    Returns:
        Mapping of group_name -> gross_nav_pct (0..1).
        Returns all zeros if nav_total_usd <= 0.
    """
    if nav_total_usd <= 0:
        return {group_name: 0.0 for group_name in corr_cfg.groups}

    # Build symbol -> groups index
    symbol_to_groups = build_symbol_to_groups_index(corr_cfg)

    # Accumulate notional per group
    group_notional: Dict[str, float] = {group_name: 0.0 for group_name in corr_cfg.groups}

    for position in positions:
        symbol = _get_position_symbol(position)
        if not symbol:
            continue

        notional = _get_position_notional(position)
        if notional <= 0:
            continue

        # Add to all groups this symbol belongs to
        groups = symbol_to_groups.get(symbol, set())
        for group_name in groups:
            group_notional[group_name] += notional

    # Convert to NAV percentage
    return {
        group_name: notional / nav_total_usd
        for group_name, notional in group_notional.items()
    }


def compute_hypothetical_group_exposure_nav_pct(
    positions: Iterable[Any],
    nav_total_usd: float,
    corr_cfg: CorrelationGroupsConfig,
    order_symbol: str,
    order_notional_usd: float,
) -> Dict[str, float]:
    """
    Compute group exposure assuming a proposed order is added.

    Returns group_name -> gross_nav_pct, with the proposed order's absolute
    notional added to all groups containing order_symbol.

    Args:
        positions: Current positions.
        nav_total_usd: Total NAV in USD.
        corr_cfg: Correlation groups configuration.
        order_symbol: Symbol of the proposed order.
        order_notional_usd: Notional USD of the proposed order (can be negative).

    Returns:
        Mapping of group_name -> hypothetical gross_nav_pct (0..1).
    """
    if nav_total_usd <= 0:
        return {group_name: 0.0 for group_name in corr_cfg.groups}

    # Start with current exposures (as absolute notional, not percentage)
    symbol_to_groups = build_symbol_to_groups_index(corr_cfg)
    group_notional: Dict[str, float] = {group_name: 0.0 for group_name in corr_cfg.groups}

    for position in positions:
        symbol = _get_position_symbol(position)
        if not symbol:
            continue

        notional = _get_position_notional(position)
        if notional <= 0:
            continue

        groups = symbol_to_groups.get(symbol, set())
        for group_name in groups:
            group_notional[group_name] += notional

    # Add the proposed order's notional to relevant groups
    order_sym = str(order_symbol).upper()
    order_abs_notional = abs(float(order_notional_usd))

    groups_for_order = symbol_to_groups.get(order_sym, set())
    for group_name in groups_for_order:
        group_notional[group_name] += order_abs_notional

    # Convert to NAV percentage
    return {
        group_name: notional / nav_total_usd
        for group_name, notional in group_notional.items()
    }


def get_groups_for_symbol(
    symbol: str,
    corr_cfg: CorrelationGroupsConfig,
) -> Set[str]:
    """
    Get the set of group names a symbol belongs to.

    Args:
        symbol: The symbol to look up.
        corr_cfg: Correlation groups configuration.

    Returns:
        Set of group names (may be empty if symbol is not in any group).
    """
    symbol_to_groups = build_symbol_to_groups_index(corr_cfg)
    return symbol_to_groups.get(str(symbol).upper(), set())


__all__ = [
    "build_symbol_to_groups_index",
    "compute_group_exposure_nav_pct",
    "compute_hypothetical_group_exposure_nav_pct",
    "get_groups_for_symbol",
]
