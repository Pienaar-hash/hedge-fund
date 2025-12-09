"""
Position Ledger (v7.4_C3)

Canonical source of truth for open positions + TP/SL state.
Merges positions_state.json with position_tp_sl.json into unified PositionLedgerEntry objects.

The ledger:
- Reads positions from positions_state.json (raw exchange snapshot)
- Reads TP/SL from position_tp_sl.json (registry)
- Merges into PositionLedgerEntry with consistent schema
- Never mutates positions_state.json (executor's domain)
- Writes only to position_tp_sl.json when TP/SL changes
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

LOG = logging.getLogger("position_ledger")

Side = Literal["LONG", "SHORT"]

# Default paths
_STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
_POSITIONS_STATE_PATH = _STATE_DIR / "positions_state.json"
_POSITIONS_PATH = _STATE_DIR / "positions.json"
_TP_SL_REGISTRY_PATH = _STATE_DIR / "position_tp_sl.json"

# Default ATR params for seeding TP/SL
DEFAULT_SL_ATR_MULT = 2.0
DEFAULT_TP_ATR_MULT = 3.0
DEFAULT_ATR_FALLBACK_PCT = 0.007  # 0.7% of entry price


@dataclass
class PositionTP_SL:
    """TP/SL levels for a position."""
    tp: Optional[Decimal] = None
    sl: Optional[Decimal] = None


@dataclass
class PositionLedgerEntry:
    """Canonical per-position object."""
    symbol: str
    side: Side
    entry_price: Decimal
    qty: Decimal
    tp_sl: PositionTP_SL = field(default_factory=PositionTP_SL)
    created_ts: Optional[float] = None
    updated_ts: Optional[float] = None


@dataclass
class LedgerReconciliationReport:
    """Mismatch summary between positions_state, ledger, and TP/SL registry."""

    missing_ledger_positions: List[str] = field(default_factory=list)
    ghost_ledger_entries: List[str] = field(default_factory=list)
    missing_tp_sl_entries: List[str] = field(default_factory=list)
    stale_tp_sl_entries: List[str] = field(default_factory=list)
    position_keys: Set[str] = field(default_factory=set)

    @property
    def has_mismatch(self) -> bool:
        return bool(
            self.missing_ledger_positions
            or self.ghost_ledger_entries
            or self.missing_tp_sl_entries
            or self.stale_tp_sl_entries
        )

    def breakdown_counts(self) -> Dict[str, int]:
        return {
            "missing_ledger_positions": len(self.missing_ledger_positions),
            "ghost_ledger_entries": len(self.ghost_ledger_entries),
            "missing_tp_sl_entries": len(self.missing_tp_sl_entries),
            "stale_tp_sl_entries": len(self.stale_tp_sl_entries),
        }


def _to_decimal(val: Any) -> Optional[Decimal]:
    """Convert value to Decimal, returning None if invalid."""
    if val is None:
        return None
    try:
        return Decimal(str(val))
    except Exception:
        return None


def _safe_float(val: Any) -> Optional[float]:
    """Convert value to float, returning None if invalid."""
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _normalize_side(pos: Dict[str, Any]) -> Optional[Side]:
    """
    Extract normalized side from position dict.
    Handles various schema formats (positionSide, side, inferred from qty).
    """
    side = pos.get("positionSide") or pos.get("position_side") or pos.get("side")
    if side:
        side_upper = str(side).upper()
        if side_upper in ("LONG", "BUY"):
            return "LONG"
        if side_upper in ("SHORT", "SELL"):
            return "SHORT"
    
    # Infer from qty
    qty = _safe_float(pos.get("qty") or pos.get("positionAmt") or pos.get("amount") or 0)
    if qty is not None and qty != 0:
        return "LONG" if qty > 0 else "SHORT"
    
    return None


def _normalize_entry_price(pos: Dict[str, Any]) -> Optional[Decimal]:
    """Extract entry price from various field names."""
    for key in ("entry_price", "entryPrice", "avg_entry_price", "avgEntryPrice"):
        val = pos.get(key)
        if val is not None:
            dec = _to_decimal(val)
            if dec is not None and dec > 0:
                return dec
    return None


def _normalize_qty(pos: Dict[str, Any]) -> Optional[Decimal]:
    """Extract absolute qty from various field names."""
    for key in ("qty", "positionAmt", "amount", "position_amt"):
        val = pos.get(key)
        if val is not None:
            dec = _to_decimal(val)
            if dec is not None:
                return abs(dec)
    return None


def _make_key(symbol: str, side: Side) -> str:
    """Create registry key from symbol and position side."""
    return f"{symbol.upper()}:{side.upper()}"
def _iter_positions_keys(positions_state: Dict[str, Any]) -> Set[str]:
    """Extract position keys (SYMBOL:SIDE) from a positions_state payload."""
    keys: Set[str] = set()
    for pos in _extract_positions_list(positions_state):
        symbol = (pos.get("symbol") or "").upper()
        if not symbol:
            continue
        side = _normalize_side(pos)
        if not side:
            continue
        qty = _normalize_qty(pos)
        entry_price = _normalize_entry_price(pos)
        if qty is None or qty == 0 or entry_price is None or entry_price <= 0:
            continue
        keys.add(_make_key(symbol, side))
    return keys


# -----------------------------------------------------------------------------
# Low-level file I/O
# -----------------------------------------------------------------------------

def load_positions_state(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Low-level: read positions_state.json or positions.json as dict.
    Returns raw structure from file.
    """
    base = state_dir or _STATE_DIR
    for path in (base / "positions_state.json", base / "positions.json"):
        try:
            if path.exists() and path.stat().st_size > 0:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            LOG.debug("[ledger] failed to load %s: %s", path, exc)
    return {}


def load_tp_sl_registry(state_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Low-level: read position_tp_sl.json; returns entries dict.
    Empty dict if missing or invalid.
    """
    base = state_dir or _STATE_DIR
    path = base / "position_tp_sl.json"
    try:
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("entries", {}) if "entries" in data else data
    except Exception as exc:
        LOG.debug("[ledger] failed to load tp_sl registry: %s", exc)
    return {}


def save_tp_sl_registry(
    registry: Dict[str, Dict[str, Any]],
    state_dir: Optional[Path] = None,
) -> None:
    """Write position_tp_sl.json (atomic write)."""
    base = state_dir or _STATE_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / "position_tp_sl.json"
    
    payload = {
        "entries": registry,
        "updated_at": time.time(),
    }
    
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        LOG.warning("[ledger] save_tp_sl_registry failed: %s", exc)


# -----------------------------------------------------------------------------
# Ledger API
# -----------------------------------------------------------------------------

def _extract_positions_list(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract list of positions from various state structures."""
    # Try common keys
    for key in ("positions", "open_positions", "data"):
        val = state.get(key)
        if isinstance(val, list):
            return val
    
    # If state itself is a list wrapped in dict
    if isinstance(state, list):
        return state
    
    # If state has symbol keys directly (symbol -> position)
    if state and all(isinstance(v, dict) for v in state.values()):
        # Check if values look like positions
        first_val = next(iter(state.values()), {})
        if "symbol" in first_val or "qty" in first_val or "entry_price" in first_val:
            return list(state.values())
    
    return []


def build_position_ledger(state_dir: Optional[Path] = None) -> Dict[str, PositionLedgerEntry]:
    """
    Merge positions_state.json + position_tp_sl.json into PositionLedgerEntry objects.

    Rules:
    - Only symbols with non-zero qty and valid entry_price are included.
    - If TP/SL exists in registry, attach it.
    - If missing, tp_sl = PositionTP_SL(tp=None, sl=None); can be seeded later.

    Returns:
        Dict mapping "SYMBOL:SIDE" keys to PositionLedgerEntry objects.
    """
    pos_state = load_positions_state(state_dir)
    tp_sl_registry = load_tp_sl_registry(state_dir)
    
    positions = _extract_positions_list(pos_state)
    ledger: Dict[str, PositionLedgerEntry] = {}
    
    for pos in positions:
        symbol = (pos.get("symbol") or "").upper()
        if not symbol:
            continue
        
        side = _normalize_side(pos)
        if side is None:
            continue
        
        qty = _normalize_qty(pos)
        if qty is None or qty == 0:
            continue
        
        entry_price = _normalize_entry_price(pos)
        if entry_price is None or entry_price <= 0:
            continue
        
        key = _make_key(symbol, side)
        
        # Extract timestamps
        created_ts = _safe_float(pos.get("created_ts") or pos.get("updateTime"))
        updated_ts = _safe_float(pos.get("updated_ts") or pos.get("updateTime"))
        
        # Build TP/SL from registry
        tp_sl_entry = tp_sl_registry.get(key, {})
        tp = _to_decimal(tp_sl_entry.get("take_profit_price") or tp_sl_entry.get("tp"))
        sl = _to_decimal(tp_sl_entry.get("stop_loss_price") or tp_sl_entry.get("sl"))
        
        # Also check if tp_sl is embedded in position
        if tp is None and sl is None:
            embedded = pos.get("tp_sl", {})
            if embedded:
                tp = _to_decimal(embedded.get("take_profit_price") or embedded.get("tp"))
                sl = _to_decimal(embedded.get("stop_loss_price") or embedded.get("sl"))
        
        ledger[key] = PositionLedgerEntry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            qty=qty,
            tp_sl=PositionTP_SL(tp=tp, sl=sl),
            created_ts=created_ts,
            updated_ts=updated_ts,
        )
    
    return ledger


# -----------------------------------------------------------------------------
# TP/SL Helpers
# -----------------------------------------------------------------------------

def upsert_tp_sl(
    symbol: str,
    side: Side,
    entry_price: Decimal,
    tp: Optional[Decimal],
    sl: Optional[Decimal],
    state_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update or create TP/SL for a given symbol in the registry file.
    """
    registry = load_tp_sl_registry(state_dir)
    key = _make_key(symbol, side)
    
    entry = registry.get(key, {})
    entry.update({
        "symbol": symbol.upper(),
        "position_side": side.upper(),
        "entry_price": float(entry_price),
        "take_profit_price": float(tp) if tp is not None else None,
        "stop_loss_price": float(sl) if sl is not None else None,
        "enable_tp_sl": True,
        "updated_at": time.time(),
    })
    
    if metadata:
        entry["metadata"] = metadata
    
    registry[key] = entry
    save_tp_sl_registry(registry, state_dir)
    
    LOG.info("[ledger] upsert %s: tp=%s sl=%s", key, tp, sl)


def delete_tp_sl(symbol: str, side: Optional[Side] = None, state_dir: Optional[Path] = None) -> None:
    """
    Remove TP/SL when position is fully closed.
    If side is None, removes both LONG and SHORT entries for the symbol.
    """
    registry = load_tp_sl_registry(state_dir)
    removed = []
    
    if side:
        key = _make_key(symbol, side)
        if key in registry:
            del registry[key]
            removed.append(key)
    else:
        # Remove both sides
        for s in ("LONG", "SHORT"):
            key = _make_key(symbol, s)
            if key in registry:
                del registry[key]
                removed.append(key)
    
    if removed:
        save_tp_sl_registry(registry, state_dir)
        LOG.info("[ledger] deleted tp_sl entries: %s", removed)


# -----------------------------------------------------------------------------
# High-level Sync
# -----------------------------------------------------------------------------

def _compute_tp_sl_for_position(
    side: Side,
    entry_price: Decimal,
    sl_atr_mult: float = DEFAULT_SL_ATR_MULT,
    tp_atr_mult: float = DEFAULT_TP_ATR_MULT,
    atr_pct: float = DEFAULT_ATR_FALLBACK_PCT,
) -> tuple[Decimal, Decimal]:
    """Compute TP/SL based on ATR estimate."""
    atr_estimate = entry_price * Decimal(str(atr_pct))
    
    if side == "LONG":
        sl = entry_price - (atr_estimate * Decimal(str(sl_atr_mult)))
        tp = entry_price + (atr_estimate * Decimal(str(tp_atr_mult)))
    else:  # SHORT
        sl = entry_price + (atr_estimate * Decimal(str(sl_atr_mult)))
        tp = entry_price - (atr_estimate * Decimal(str(tp_atr_mult)))
    
    return tp, sl


def sync_ledger_with_positions(
    seed_missing: bool = True,
    remove_stale: bool = True,
    sl_atr_mult: float = DEFAULT_SL_ATR_MULT,
    tp_atr_mult: float = DEFAULT_TP_ATR_MULT,
    atr_pct: float = DEFAULT_ATR_FALLBACK_PCT,
    state_dir: Optional[Path] = None,
) -> Dict[str, PositionLedgerEntry]:
    """
    High-level operation:
    - Rebuild ledger from current positions_state.json.
    - Optionally seed TP/SL for missing entries (using ATR-based logic).
    - Remove TP/SL entries for symbols no longer in positions.
    - Return the in-memory ledger.

    Args:
        seed_missing: If True, generate TP/SL for positions without registry entries.
        remove_stale: If True, remove registry entries for closed positions.
        sl_atr_mult: ATR multiplier for stop loss.
        tp_atr_mult: ATR multiplier for take profit.
        atr_pct: Fallback ATR as percentage of entry price.
        state_dir: Override state directory.

    Returns:
        Dict mapping keys to PositionLedgerEntry objects (updated ledger).
    """
    registry = load_tp_sl_registry(state_dir)
    ledger = build_position_ledger(state_dir)
    
    active_keys = set(ledger.keys())
    registry_keys = set(registry.keys())
    
    # Remove stale registry entries
    if remove_stale:
        stale_keys = registry_keys - active_keys
        for key in stale_keys:
            del registry[key]
            LOG.info("[ledger] removed stale registry entry: %s", key)
        if stale_keys:
            save_tp_sl_registry(registry, state_dir)
    
    # Seed missing entries
    seeded = 0
    if seed_missing:
        for key, entry in ledger.items():
            if key not in registry:
                tp, sl = _compute_tp_sl_for_position(
                    entry.side,
                    entry.entry_price,
                    sl_atr_mult,
                    tp_atr_mult,
                    atr_pct,
                )
                registry[key] = {
                    "symbol": entry.symbol,
                    "position_side": entry.side,
                    "entry_price": float(entry.entry_price),
                    "take_profit_price": float(tp),
                    "stop_loss_price": float(sl),
                    "qty": float(entry.qty),
                    "enable_tp_sl": True,
                    "created_at": time.time(),
                    "source": "ledger_sync",
                    "metadata": {
                        "sl_atr_mult": sl_atr_mult,
                        "tp_atr_mult": tp_atr_mult,
                        "atr_pct": atr_pct,
                    },
                }
                
                # Update the ledger entry with the new TP/SL
                entry.tp_sl = PositionTP_SL(tp=tp, sl=sl)
                
                LOG.info(
                    "[ledger] seeded %s: entry=%.4f tp=%.4f sl=%.4f",
                    key, entry.entry_price, tp, sl,
                )
                seeded += 1
        
        if seeded > 0:
            save_tp_sl_registry(registry, state_dir)
            LOG.info("[ledger] sync complete: %d entries seeded", seeded)
    
    return ledger


def get_ledger_summary(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get a summary of ledger state for debugging/dashboard.
    """
    ledger = build_position_ledger(state_dir)
    registry = load_tp_sl_registry(state_dir)
    
    num_positions = len(ledger)
    num_with_tp_sl = sum(
        1 for e in ledger.values()
        if e.tp_sl.tp is not None or e.tp_sl.sl is not None
    )
    num_registry = len(registry)
    
    return {
        "num_positions": num_positions,
        "num_with_tp_sl": num_with_tp_sl,
        "num_registry_entries": num_registry,
        "consistency": "ok" if num_positions == 0 or num_with_tp_sl == num_positions else "partial",
        "symbols": list(ledger.keys()),
    }


def ledger_to_dict(ledger: Dict[str, PositionLedgerEntry]) -> Dict[str, Dict[str, Any]]:
    """
    Convert ledger entries to JSON-serializable dict for state publishing.
    """
    return {
        key: {
            "symbol": entry.symbol,
            "side": entry.side,
            "qty": float(entry.qty),
            "entry_price": float(entry.entry_price),
            "tp": float(entry.tp_sl.tp) if entry.tp_sl.tp is not None else None,
            "sl": float(entry.tp_sl.sl) if entry.tp_sl.sl is not None else None,
            "created_ts": entry.created_ts,
            "updated_ts": entry.updated_ts,
        }
        for key, entry in ledger.items()
    }


def build_positions_ledger_state(
    ledger: Dict[str, PositionLedgerEntry],
    *,
    updated_at: Optional[str] = None,
    tp_sl_levels: Optional[Dict[str, Any]] = None,
    state_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build a serializable snapshot of the position ledger.

    Args:
        ledger: Mapping of ledger entries keyed by "SYMBOL:SIDE".
        updated_at: Optional ISO timestamp override.
        tp_sl_levels: Optional precomputed TP/SL levels map.
    """
    entries_list: List[Dict[str, Any]] = []
    for key, entry in ledger.items():
        entries_list.append(
            {
                "position_key": key,
                "symbol": entry.symbol,
                "side": entry.side,
                "qty": float(entry.qty),
                "entry_price": float(entry.entry_price),
                "tp": float(entry.tp_sl.tp) if entry.tp_sl.tp is not None else None,
                "sl": float(entry.tp_sl.sl) if entry.tp_sl.sl is not None else None,
                "created_ts": entry.created_ts,
                "updated_ts": entry.updated_ts,
            }
        )

    if tp_sl_levels is None:
        try:
            registry = load_tp_sl_registry(state_dir)
            tp_sl_levels = {
                key: {
                    "tp": entry.get("take_profit_price") or entry.get("tp"),
                    "sl": entry.get("stop_loss_price") or entry.get("sl"),
                }
                for key, entry in registry.items()
                if isinstance(entry, dict)
            }
        except Exception:
            tp_sl_levels = {}

    return {
        "updated_at": updated_at or datetime.now(timezone.utc).isoformat(),
        "updated_ts": updated_at or datetime.now(timezone.utc).isoformat(),
        "entries": entries_list,
        "tp_sl_levels": tp_sl_levels or {},
        "metadata": {
            "version": "v1",
            "entry_count": len(entries_list),
        },
    }


# -----------------------------------------------------------------------------
# Reconciliation (report-only)
# -----------------------------------------------------------------------------


def reconcile_ledger_and_registry(
    positions_state: Dict[str, Any],
    ledger: Dict[str, PositionLedgerEntry],
    tp_sl_registry: Dict[str, Any],
) -> LedgerReconciliationReport:
    """
    Pure reconciliation between live positions, ledger entries, and TP/SL registry.

    Returns:
        LedgerReconciliationReport with mismatch lists and counts; no side effects.
    """
    position_keys = _iter_positions_keys(positions_state)
    ledger_keys = set(ledger.keys())
    registry_keys = set(tp_sl_registry.keys()) if isinstance(tp_sl_registry, dict) else set()

    missing_ledger = sorted(position_keys - ledger_keys)
    ghost_ledger = sorted(ledger_keys - position_keys)

    missing_tp_sl: List[str] = []
    for key, entry in ledger.items():
        tp_sl = entry.tp_sl
        has_levels = tp_sl.tp is not None or tp_sl.sl is not None
        reg_entry = tp_sl_registry.get(key) if isinstance(tp_sl_registry, dict) else None
        reg_has_levels = False
        if isinstance(reg_entry, dict):
            reg_has_levels = (reg_entry.get("take_profit_price") is not None) or (
                reg_entry.get("stop_loss_price") is not None
            ) or (reg_entry.get("tp") is not None) or (reg_entry.get("sl") is not None)
        if (not has_levels or not reg_has_levels) and key in ledger_keys:
            missing_tp_sl.append(key)

    stale_tp_sl = sorted(registry_keys - position_keys)

    return LedgerReconciliationReport(
        missing_ledger_positions=missing_ledger,
        ghost_ledger_entries=ghost_ledger,
        missing_tp_sl_entries=sorted(set(missing_tp_sl)),
        stale_tp_sl_entries=stale_tp_sl,
        position_keys=position_keys,
    )
