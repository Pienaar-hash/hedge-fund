"""
Exit Scanner for Vol-Target TP/SL (v7.3-alpha2)

Scans open positions for TP/SL level crossings and emits reduceOnly exit intents.
Uses position_tp_sl_registry for TP/SL level lookup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from execution.diagnostics_metrics import (
    record_exit_scan_run,
    record_exit_trigger,
    update_exit_pipeline_status,
)

LOG = logging.getLogger("exit_scanner")


class ExitReason(str, Enum):
    """Reason for exit trigger."""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"


@dataclass
class ExitCandidate:
    """Represents a position that has hit TP or SL level."""
    symbol: str
    position_side: str  # "LONG" / "SHORT"
    qty: float
    exit_reason: ExitReason
    trigger_price: float
    tp_price: Optional[float]
    sl_price: Optional[float]
    entry_price: Optional[float] = None
    strategy: str = "vol_target"


def _get_side(position: Dict[str, Any]) -> Optional[str]:
    """
    Extract position side from position dict.
    Adapts to different schema formats (positionSide, side, or inferred from qty).
    """
    side = position.get("positionSide") or position.get("side")
    if not side:
        qty = float(position.get("qty") or position.get("positionAmt") or 0)
        if qty != 0:
            side = "LONG" if qty > 0 else "SHORT"
    return side.upper() if side else None


def _get_last_price(symbol: str, price_map: Dict[str, float]) -> Optional[float]:
    """Get last price for symbol from price map."""
    return price_map.get(symbol)


def _load_exit_pipeline_config() -> Dict[str, Any]:
    """Load exit pipeline diagnostics config from strategy_config.json."""
    try:
        with open("config/strategy_config.json", "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except Exception:
        return {}
    try:
        diag = cfg.get("diagnostics") or {}
        exit_cfg = diag.get("exit_pipeline") if isinstance(diag, dict) else {}
        return exit_cfg if isinstance(exit_cfg, dict) else {}
    except Exception:
        return {}


def _pnl_pct(entry_price: float, last_price: float, side: str) -> Optional[float]:
    try:
        if entry_price <= 0 or last_price <= 0:
            return None
        if side.upper() in ("LONG", "BUY"):
            return (last_price - entry_price) / entry_price
        if side.upper() in ("SHORT", "SELL"):
            return (entry_price - last_price) / entry_price
    except Exception:
        return None
    return None


def _build_exit_candidate(
    position: Dict[str, Any],
    last_price: float,
    tp_sl_entry: Optional[Dict[str, Any]] = None,
) -> Optional[ExitCandidate]:
    """
    Check if position has hit TP or SL level and build ExitCandidate if so.
    
    Args:
        position: Position dict from exchange
        last_price: Current market price
        tp_sl_entry: TP/SL entry from registry (optional, uses position.tp_sl if not provided)
        
    Returns:
        ExitCandidate if TP or SL hit, None otherwise
    """
    # Get TP/SL from registry entry or fallback to position dict
    if tp_sl_entry:
        tp_price = tp_sl_entry.get("take_profit_price")
        sl_price = tp_sl_entry.get("stop_loss_price")
        entry_price = tp_sl_entry.get("entry_price")
    else:
        tp_sl = position.get("tp_sl") or {}
        if not tp_sl or not tp_sl.get("enable_tp_sl", True):
            return None
        tp_price = tp_sl.get("take_profit_price")
        sl_price = tp_sl.get("stop_loss_price")
        entry_price = tp_sl.get("entry_price")

    if not tp_price and not sl_price:
        return None

    side = _get_side(position)
    if side is None:
        return None

    qty = float(position.get("qty") or position.get("positionAmt") or 0.0)
    if qty == 0.0:
        return None

    symbol = position.get("symbol")
    if not symbol:
        return None

    last = float(last_price)

    # Long exits: TP when price >= TP, SL when price <= SL
    if side in ("LONG", "BUY"):
        if tp_price and last >= tp_price:
            return ExitCandidate(
                symbol=symbol,
                position_side=side,
                qty=abs(qty),
                exit_reason=ExitReason.TAKE_PROFIT,
                trigger_price=last,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_price=entry_price,
            )
        if sl_price and last <= sl_price:
            return ExitCandidate(
                symbol=symbol,
                position_side=side,
                qty=abs(qty),
                exit_reason=ExitReason.STOP_LOSS,
                trigger_price=last,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_price=entry_price,
            )

    # Short exits: TP when price <= TP, SL when price >= SL
    if side in ("SHORT", "SELL"):
        if tp_price and last <= tp_price:
            return ExitCandidate(
                symbol=symbol,
                position_side=side,
                qty=abs(qty),
                exit_reason=ExitReason.TAKE_PROFIT,
                trigger_price=last,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_price=entry_price,
            )
        if sl_price and last >= sl_price:
            return ExitCandidate(
                symbol=symbol,
                position_side=side,
                qty=abs(qty),
                exit_reason=ExitReason.STOP_LOSS,
                trigger_price=last,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_price=entry_price,
            )

    return None


def scan_tp_sl_exits(
    positions: List[Dict[str, Any]],
    price_map: Dict[str, float],
) -> List[ExitCandidate]:
    """
    Scan positions for TP/SL level crossings using the position ledger.
    
    V7.4_C3: Uses build_position_ledger() as canonical source for positions + TP/SL.
    Falls back to legacy registry if ledger unavailable.
    
    Given a list of positions (e.g. from exchange) and a symbol→price map,
    return a list of ExitCandidate objects representing TP/SL hits for vol_target positions.
    
    Args:
        positions: List of position dicts from exchange (used for fallback only)
        price_map: Dict mapping symbol to current price
        
    Returns:
        List of ExitCandidate objects for positions that hit TP or SL
    """
    results: List[ExitCandidate] = []
    try:
        record_exit_scan_run()
    except Exception:
        pass
    exit_cfg = _load_exit_pipeline_config()
    underwater_threshold = float(exit_cfg.get("tp_sl_underwater_threshold_pct", -0.02) or -0.02)
    underwater_threshold_pct = underwater_threshold

    # Try ledger-based approach first (C3)
    try:
        from execution.position_ledger import build_position_ledger
        from execution.position_tp_sl_registry import get_all_tp_sl_positions

        ledger = build_position_ledger()
        registry = get_all_tp_sl_positions()
        open_positions = list(ledger.values())
        open_positions_count = len(open_positions)

        registered_keys = set(registry.keys()) if isinstance(registry, dict) else set()
        tp_sl_registered_count = 0
        tp_sl_missing_count = 0
        underwater_without_tp_sl_count = 0

        for entry in open_positions:
            key = f"{entry.symbol}:{entry.side}"
            has_tp_sl = key in registered_keys
            if has_tp_sl:
                tp_sl_registered_count += 1
            else:
                tp_sl_missing_count += 1
                last_price = _get_last_price(entry.symbol, price_map)
                if last_price is not None:
                    pnl_pct_val = _pnl_pct(float(entry.entry_price), float(last_price), entry.side)
                    if pnl_pct_val is not None and pnl_pct_val <= underwater_threshold_pct:
                        underwater_without_tp_sl_count += 1

        update_exit_pipeline_status(
            open_positions_count=open_positions_count,
            tp_sl_registered_count=tp_sl_registered_count,
            tp_sl_missing_count=tp_sl_missing_count,
            underwater_without_tp_sl_count=underwater_without_tp_sl_count,
        )

        for key, entry in ledger.items():
            symbol = entry.symbol
            side = entry.side
            
            last_price = _get_last_price(symbol, price_map)
            if last_price is None:
                LOG.debug("[exit_scanner] no price for %s", symbol)
                continue
            
            tp = float(entry.tp_sl.tp) if entry.tp_sl.tp is not None else None
            sl = float(entry.tp_sl.sl) if entry.tp_sl.sl is not None else None
            
            if tp is None and sl is None:
                # No TP/SL registered for this position
                continue
            
            # Build a minimal position dict for _build_exit_candidate
            pos_dict = {
                "symbol": symbol,
                "positionSide": side,
                "qty": float(entry.qty),
            }
            
            tp_sl_entry = {
                "take_profit_price": tp,
                "stop_loss_price": sl,
                "entry_price": float(entry.entry_price),
            }

            candidate = _build_exit_candidate(pos_dict, last_price, tp_sl_entry)
            if candidate:
                LOG.info(
                    "[exit_scanner] %s %s hit %s price=%.4f level=%.4f",
                    symbol,
                    side,
                    candidate.exit_reason.value.upper(),
                    last_price,
                    candidate.tp_price if candidate.exit_reason == ExitReason.TAKE_PROFIT else candidate.sl_price,
                )
                try:
                    record_exit_trigger()
                except Exception:
                    pass
                results.append(candidate)
        
        return results
    
    except ImportError:
        LOG.debug("[exit_scanner] position_ledger not available, using legacy registry")
    except Exception as exc:
        LOG.warning("[exit_scanner] ledger-based scan failed, falling back to registry: %s", exc)
    
    # Fallback to legacy registry-based approach
    try:
        from execution.position_tp_sl_registry import get_position_tp_sl
    except ImportError:
        LOG.warning("[exit_scanner] position_tp_sl_registry not available")
        return results
    
    for pos in positions:
        symbol = pos.get("symbol")
        if not symbol:
            continue

        side = _get_side(pos)
        if not side:
            continue

        # Look up TP/SL from registry
        tp_sl_entry = get_position_tp_sl(symbol, side)
        if not tp_sl_entry:
            continue

        last_price = _get_last_price(symbol, price_map)
        if last_price is None:
            LOG.debug("[exit_scanner] no price for %s", symbol)
            continue

        candidate = _build_exit_candidate(pos, last_price, tp_sl_entry)
        if candidate:
            LOG.info(
                "[exit_scanner] %s %s hit %s price=%.4f level=%.4f",
                symbol,
                side,
                candidate.exit_reason.value.upper(),
                last_price,
                candidate.tp_price if candidate.exit_reason == ExitReason.TAKE_PROFIT else candidate.sl_price,
            )
            try:
                record_exit_trigger()
            except Exception:
                pass
            results.append(candidate)
    
    return results


def build_exit_intent(candidate: ExitCandidate) -> Dict[str, Any]:
    """
    Build a reduceOnly exit intent from an ExitCandidate.
    
    Args:
        candidate: ExitCandidate from scan_tp_sl_exits
        
    Returns:
        Intent dict ready for executor
    """
    import time
    from datetime import datetime, timezone

    # For closing a LONG position, we SELL. For SHORT, we BUY.
    close_side = "SELL" if candidate.position_side == "LONG" else "BUY"
    exit_block = {
        "reason": candidate.exit_reason.value,
        "trigger_price": candidate.trigger_price,
        "tp_price": candidate.tp_price,
        "sl_price": candidate.sl_price,
        "entry_price": candidate.entry_price,
        "source_strategy": candidate.strategy,
    }
    metadata = {
        "strategy": "vol_target_exit",
        "exit_reason": candidate.exit_reason.value,
        "trigger_price": candidate.trigger_price,
        "tp_price": candidate.tp_price,
        "sl_price": candidate.sl_price,
        "entry_price": candidate.entry_price,
        "exit": exit_block,
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": candidate.symbol,
        "signal": close_side,
        "reduceOnly": True,
        "positionSide": candidate.position_side,
        "quantity": candidate.qty,
        "metadata": metadata,
        "generated_at": time.time(),
    }
