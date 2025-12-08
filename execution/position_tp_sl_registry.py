"""
Position TP/SL Registry (v7.3-alpha2)

Tracks TP/SL levels for open positions. Persists to logs/state/position_tp_sl.json.
The exit scanner reads this registry to determine which positions have active TP/SL levels.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger("position_tp_sl")

# Default path for TP/SL registry
_STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
_REGISTRY_PATH = _STATE_DIR / "position_tp_sl.json"

# In-memory cache of TP/SL levels
_TP_SL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRY_LOADED = False


def _make_key(symbol: str, position_side: str) -> str:
    """Create registry key from symbol and position side."""
    return f"{symbol.upper()}:{position_side.upper()}"


def _load_registry() -> None:
    """Load registry from disk if not already loaded."""
    global _TP_SL_REGISTRY, _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    
    try:
        if _REGISTRY_PATH.exists():
            data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _TP_SL_REGISTRY = data.get("entries", {})
                LOG.debug("[tp_sl_registry] loaded %d entries", len(_TP_SL_REGISTRY))
    except Exception as exc:
        LOG.warning("[tp_sl_registry] load failed: %s", exc)
        _TP_SL_REGISTRY = {}
    
    _REGISTRY_LOADED = True


def _save_registry() -> None:
    """Persist registry to disk."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": _TP_SL_REGISTRY,
            "updated_at": __import__("time").time(),
        }
        tmp = _REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(_REGISTRY_PATH)
    except Exception as exc:
        LOG.warning("[tp_sl_registry] save failed: %s", exc)


def register_position_tp_sl(
    symbol: str,
    position_side: str,
    take_profit_price: Optional[float],
    stop_loss_price: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register TP/SL levels for a position.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        position_side: "LONG" or "SHORT"
        take_profit_price: TP price level
        stop_loss_price: SL price level
        metadata: Additional metadata (sl_atr_mult, tp_atr_mult, etc.)
    """
    _load_registry()
    
    if take_profit_price is None and stop_loss_price is None:
        LOG.debug("[tp_sl_registry] skipping registration for %s:%s (no TP/SL)", symbol, position_side)
        return
    
    key = _make_key(symbol, position_side)
    entry = {
        "symbol": symbol.upper(),
        "position_side": position_side.upper(),
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "strategy": "vol_target",
        "enable_tp_sl": True,
        "registered_at": __import__("time").time(),
    }
    
    if metadata:
        entry.update({
            "sl_atr_mult": metadata.get("sl_atr_mult"),
            "tp_atr_mult": metadata.get("tp_atr_mult"),
            "min_rr": metadata.get("min_rr"),
            "reward_risk": metadata.get("reward_risk"),
        })
    
    _TP_SL_REGISTRY[key] = entry
    _save_registry()
    LOG.info(
        "[tp_sl_registry] registered %s:%s tp=%s sl=%s",
        symbol, position_side, take_profit_price, stop_loss_price
    )


def unregister_position_tp_sl(symbol: str, position_side: str) -> None:
    """
    Remove TP/SL registration for a closed position.
    
    Args:
        symbol: Trading symbol
        position_side: "LONG" or "SHORT"
    """
    _load_registry()
    
    key = _make_key(symbol, position_side)
    if key in _TP_SL_REGISTRY:
        del _TP_SL_REGISTRY[key]
        _save_registry()
        LOG.info("[tp_sl_registry] unregistered %s:%s", symbol, position_side)


def get_position_tp_sl(symbol: str, position_side: str) -> Optional[Dict[str, Any]]:
    """
    Get TP/SL entry for a position.
    
    Returns:
        Dict with tp_sl data or None if not registered
    """
    _load_registry()
    key = _make_key(symbol, position_side)
    return _TP_SL_REGISTRY.get(key)


def get_all_tp_sl_positions() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered TP/SL positions.
    
    Returns:
        Dict mapping keys to tp_sl entries
    """
    _load_registry()
    return dict(_TP_SL_REGISTRY)


def enrich_positions_with_tp_sl(positions: list) -> list:
    """
    Enrich a list of position dicts with tp_sl data from the registry.
    
    Args:
        positions: List of position dicts from exchange
        
    Returns:
        List of positions with tp_sl blocks added where registered
    """
    _load_registry()
    
    enriched = []
    for pos in positions:
        pos_copy = dict(pos)
        symbol = pos.get("symbol", "").upper()
        side = pos.get("positionSide") or pos.get("side", "")
        if not side:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
            side = "LONG" if qty > 0 else "SHORT"
        side = side.upper()
        
        key = _make_key(symbol, side)
        tp_sl = _TP_SL_REGISTRY.get(key)
        if tp_sl:
            pos_copy["tp_sl"] = tp_sl
        
        enriched.append(pos_copy)
    
    return enriched


def cleanup_stale_entries(active_positions: list) -> int:
    """
    Remove registry entries for positions that no longer exist.
    
    Args:
        active_positions: List of currently open positions
        
    Returns:
        Number of stale entries removed
    """
    _load_registry()
    
    # Build set of active position keys
    active_keys = set()
    for pos in active_positions:
        symbol = pos.get("symbol", "").upper()
        side = pos.get("positionSide") or pos.get("side", "")
        if not side:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
            if qty == 0:
                continue
            side = "LONG" if qty > 0 else "SHORT"
        side = side.upper()
        qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
        if abs(qty) > 0:
            active_keys.add(_make_key(symbol, side))
    
    # Find and remove stale entries
    stale_keys = [k for k in _TP_SL_REGISTRY.keys() if k not in active_keys]
    for key in stale_keys:
        del _TP_SL_REGISTRY[key]
        LOG.info("[tp_sl_registry] cleaned stale entry %s", key)
    
    if stale_keys:
        _save_registry()
    
    return len(stale_keys)


def seed_missing_entries(
    active_positions: list,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    atr_fallback_pct: float = 0.007,
) -> int:
    """
    Seed registry entries for positions that don't have TP/SL registered.
    
    Used at startup to ensure all open positions have exit protection.
    Computes TP/SL using ATR-based estimates when actual ATR is unavailable.
    
    Args:
        active_positions: List of currently open positions from exchange
        sl_atr_mult: ATR multiplier for stop loss (default 2.0)
        tp_atr_mult: ATR multiplier for take profit (default 3.0)
        atr_fallback_pct: Fallback ATR as percentage of entry price (default 0.7%)
        
    Returns:
        Number of new entries seeded
    """
    _load_registry()
    
    seeded = 0
    for pos in active_positions:
        symbol = pos.get("symbol", "").upper()
        side = pos.get("positionSide") or pos.get("side", "")
        if not side:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
            if qty == 0:
                continue
            side = "LONG" if qty > 0 else "SHORT"
        side = side.upper()
        
        qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
        if abs(qty) == 0:
            continue
        
        key = _make_key(symbol, side)
        if key in _TP_SL_REGISTRY:
            # Already registered
            continue
        
        entry_price = float(pos.get("entryPrice") or pos.get("entry_price") or 0)
        if entry_price <= 0:
            LOG.warning("[tp_sl_registry] cannot seed %s — invalid entry price", key)
            continue
        
        # Estimate ATR as fallback percentage of entry price
        atr_estimate = entry_price * atr_fallback_pct
        
        # Compute TP/SL based on position direction
        if side == "LONG":
            sl_price = entry_price - (atr_estimate * sl_atr_mult)
            tp_price = entry_price + (atr_estimate * tp_atr_mult)
        else:
            sl_price = entry_price + (atr_estimate * sl_atr_mult)
            tp_price = entry_price - (atr_estimate * tp_atr_mult)
        
        entry = {
            "symbol": symbol,
            "position_side": side,
            "entry_price": entry_price,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "qty": abs(qty),
            "enable_tp_sl": True,
            "created_at": __import__("time").time(),
            "source": "startup_seed",
            "metadata": {
                "atr_estimate": atr_estimate,
                "sl_atr_mult": sl_atr_mult,
                "tp_atr_mult": tp_atr_mult,
            },
        }
        
        _TP_SL_REGISTRY[key] = entry
        LOG.info(
            "[tp_sl_registry] seeded %s: entry=%.4f sl=%.4f tp=%.4f",
            key, entry_price, sl_price, tp_price,
        )
        seeded += 1
    
    if seeded > 0:
        _save_registry()
        LOG.info("[tp_sl_registry] startup seed complete: %d entries added", seeded)
    
    return seeded


def sync_registry_with_positions(
    active_positions: list,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
) -> dict:
    """
    Full registry sync: remove stale entries AND seed missing ones.
    
    Call this at executor startup to ensure registry is consistent with exchange state.
    
    Args:
        active_positions: List of currently open positions from exchange
        sl_atr_mult: ATR multiplier for stop loss
        tp_atr_mult: ATR multiplier for take profit
        
    Returns:
        Dict with 'stale_removed' and 'new_seeded' counts
    """
    stale = cleanup_stale_entries(active_positions)
    seeded = seed_missing_entries(active_positions, sl_atr_mult, tp_atr_mult)
    
    return {"stale_removed": stale, "new_seeded": seeded}
