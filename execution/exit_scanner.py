"""
Exit Scanner for Doctrine-Based Exits and TP/SL Seatbelt (v7.X-doctrine)

v7.X_DOCTRINE: Exit precedence is:
  1. CRISIS_OVERRIDE (immediate) - Sentinel-X crisis detection
  2. REGIME_FLIP (immediate) - Regime changed against position
  3. STRUCTURAL_FAILURE (stepped) - Alpha decay / thesis broken
  4. TIME_STOP (patient) - Position exceeded max holding period
  5. SEATBELT (immediate) - Stop-loss emergency protection only

TP/SL is now a "seatbelt" only (catastrophe protection), not primary exit logic.
Positions die when the THESIS dies, not when arbitrary TP/SL levels hit.

Legacy v7.3-alpha2 behavior preserved in scan_tp_sl_exits() for backward compat.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from execution.diagnostics_metrics import (
    record_exit_scan_run,
    record_exit_trigger,
    update_exit_pipeline_status,
)

# Cycle statistics (non-invasive, observational only)
try:
    from execution.cycle_statistics import record_exit_event as _record_exit_stat
    _CYCLE_STATS_AVAILABLE = True
except ImportError:
    _CYCLE_STATS_AVAILABLE = False
    def _record_exit_stat(*args, **kwargs): pass

LOG = logging.getLogger("exit_scanner")

# ---------------------------------------------------------------------------
# Doctrine Exit Integration
# ---------------------------------------------------------------------------
try:
    from execution.doctrine_kernel import (
        ExitReason as DoctrineExitReason,
        ExitUrgency,
        ExitDecision,
        PositionSnapshot,
        RegimeSnapshot,
        AlphaHealthSnapshot,
        ExecutionSnapshot,
        doctrine_exit_verdict,
        build_regime_snapshot_from_state,
        log_doctrine_event,
    )
    _DOCTRINE_AVAILABLE = True
except ImportError:
    _DOCTRINE_AVAILABLE = False
    LOG.warning("[exit_scanner] doctrine_kernel not available - TP/SL seatbelt only mode")


# Legacy ExitReason for backward compatibility with TP/SL seatbelt
class SeatbeltExitReason(str, Enum):
    """Legacy reason for TP/SL seatbelt trigger - emergency protection only."""
    TAKE_PROFIT = "tp"  # Kept for legacy compat, but TP is NOT primary exit
    STOP_LOSS = "sl"    # Seatbelt: catastrophe protection


# Alias for backward compatibility
ExitReason = SeatbeltExitReason


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


@dataclass
class DoctrineExitCandidate:
    """
    Doctrine-based exit candidate.
    
    Doctrine exits are thesis-based, not pain-based:
    - CRISIS_OVERRIDE: Sentinel-X detected crisis
    - REGIME_FLIP: Regime changed against position direction
    - STRUCTURAL_FAILURE: Alpha decay / thesis broken
    - TIME_STOP: Position exceeded max holding period
    - SEATBELT: Stop-loss catastrophe protection (last resort)
    """
    symbol: str
    position_side: str  # "LONG" / "SHORT"
    qty: float
    exit_reason: str  # DoctrineExitReason.value
    urgency: str  # ExitUrgency.value
    trigger_price: float
    entry_price: Optional[float] = None
    entry_regime: Optional[str] = None
    current_regime: Optional[str] = None
    bars_held: int = 0
    alpha_survival: Optional[float] = None
    rationale: str = ""
    strategy: str = "vol_target"


# ---------------------------------------------------------------------------
# Doctrine Exit Scanning
# ---------------------------------------------------------------------------

def _load_sentinel_x_state() -> Dict[str, Any]:
    """Load Sentinel-X state from logs/state/sentinel_x.json."""
    try:
        with open("logs/state/sentinel_x.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_alpha_diagnostics() -> Dict[str, Any]:
    """Load alpha diagnostics from logs/state/alpha_diagnostics.json."""
    try:
        with open("logs/state/alpha_diagnostics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_position_metadata(symbol: str, side: str) -> Dict[str, Any]:
    """
    Load position metadata for a symbol/side.
    Returns entry_regime, entry_time, etc. if available.
    """
    try:
        from execution.position_tp_sl_registry import get_position_tp_sl
        entry = get_position_tp_sl(symbol, side)
        if entry:
            return {
                "entry_regime": entry.get("entry_regime"),
                "entry_time": entry.get("entry_time"),
                "entry_regime_confidence": entry.get("entry_regime_confidence"),
                "entry_head": entry.get("entry_head"),
            }
    except Exception:
        pass
    return {}


def _calculate_bars_held(entry_time: Optional[float], bar_interval_seconds: int = 900) -> int:
    """
    Calculate how many bars a position has been held.
    Default bar interval is 15 minutes (900 seconds).
    """
    if entry_time is None:
        return 0
    try:
        elapsed_seconds = time.time() - entry_time
        return int(elapsed_seconds / bar_interval_seconds)
    except Exception:
        return 0


def scan_doctrine_exits(
    positions: List[Dict[str, Any]],
    price_map: Dict[str, float],
) -> List[DoctrineExitCandidate]:
    """
    Scan positions for doctrine-based exits.
    
    Doctrine exit precedence (strict order):
      1. CRISIS_OVERRIDE - Sentinel-X crisis detection (immediate)
      2. REGIME_FLIP - Regime changed against position (immediate)
      3. STRUCTURAL_FAILURE - Alpha decay / trend broken (stepped)
      4. TIME_STOP - Exceeded max holding period (patient)
      5. SEATBELT - Stop-loss catastrophe protection (last resort)
    
    Note: TP (take-profit) is NOT in doctrine exits. Positions die when
    the thesis dies, not when they hit arbitrary profit targets.
    
    Args:
        positions: List of position dicts from exchange or ledger
        price_map: Dict mapping symbol to current price
        
    Returns:
        List of DoctrineExitCandidate for positions that should exit
    """
    if not _DOCTRINE_AVAILABLE:
        LOG.debug("[exit_scanner] doctrine unavailable - skipping doctrine exits")
        return []
    
    results: List[DoctrineExitCandidate] = []
    
    # Load regime state once for all positions
    sentinel_state = _load_sentinel_x_state()
    if not sentinel_state:
        LOG.debug("[exit_scanner] sentinel state unavailable - skipping doctrine exits")
        return []
    
    try:
        regime_snapshot = build_regime_snapshot_from_state(sentinel_state)
    except Exception as exc:
        LOG.warning("[exit_scanner] failed to build regime snapshot: %s", exc)
        return []
    
    # Load alpha diagnostics for structural failure detection
    alpha_diag = _load_alpha_diagnostics()
    
    for pos in positions:
        symbol = pos.get("symbol")
        if not symbol:
            continue
        
        side = _get_side(pos)
        if not side:
            continue
        
        qty = abs(float(pos.get("qty") or pos.get("positionAmt") or 0))
        if qty <= 0:
            continue
        
        last_price = _get_last_price(symbol, price_map)
        if last_price is None:
            continue
        
        # Get position metadata (entry_regime, entry_time, etc.)
        pos_meta = _load_position_metadata(symbol, side)
        entry_time = pos_meta.get("entry_time")
        entry_regime = pos_meta.get("entry_regime")
        entry_price_raw = pos.get("entryPrice") or pos.get("entry_price")
        entry_price = float(entry_price_raw) if entry_price_raw else None
        
        bars_held = _calculate_bars_held(entry_time)
        
        # Get alpha survival for this symbol if available
        symbol_alpha = alpha_diag.get("symbols", {}).get(symbol, {})
        alpha_survival = symbol_alpha.get("alpha_survival")
        
        # Calculate unrealized PnL % for position snapshot
        effective_entry = entry_price or last_price
        if side == "LONG":
            unrealized_pnl_pct = (last_price - effective_entry) / effective_entry if effective_entry else 0.0
        else:
            unrealized_pnl_pct = (effective_entry - last_price) / effective_entry if effective_entry else 0.0
        
        # Get entry metadata with defaults
        entry_regime_confidence = pos_meta.get("entry_regime_confidence", 0.5)
        entry_ts = pos_meta.get("entry_time") or time.time()
        if isinstance(entry_ts, str):
            try:
                entry_ts = float(entry_ts)
            except (ValueError, TypeError):
                entry_ts = time.time()
        
        # Build position snapshot for doctrine
        position_snapshot = PositionSnapshot(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=effective_entry,
            current_price=last_price,
            entry_regime=entry_regime,
            entry_regime_confidence=entry_regime_confidence,
            entry_ts=entry_ts,
            bars_held=bars_held,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )
        
        # Build alpha health snapshot
        alpha_health = AlphaHealthSnapshot(
            survival_probability=alpha_survival if alpha_survival is not None else 1.0,
            trend_strength=symbol_alpha.get("trend_strength", 0.5),
        )
        
        # Build execution snapshot with defaults (execution quality not tracked per-position)
        execution_snapshot = ExecutionSnapshot()
        
        # Get doctrine exit verdict
        exit_decision: ExitDecision = doctrine_exit_verdict(
            regime=regime_snapshot,
            position=position_snapshot,
            execution=execution_snapshot,
            alpha_health=alpha_health,
        )
        
        # If doctrine says HOLD, no exit
        if exit_decision.reason == DoctrineExitReason.HOLD:
            continue
        
        # Build candidate
        candidate = DoctrineExitCandidate(
            symbol=symbol,
            position_side=side,
            qty=qty,
            exit_reason=exit_decision.reason.value,
            urgency=exit_decision.urgency.value,
            trigger_price=last_price,
            entry_price=entry_price,
            entry_regime=entry_regime,
            current_regime=regime_snapshot.primary_regime if regime_snapshot else None,
            bars_held=bars_held,
            alpha_survival=alpha_survival,
            rationale=exit_decision.explanation,
        )
        
        LOG.info(
            "[exit_scanner] DOCTRINE EXIT %s %s reason=%s urgency=%s explanation=%s",
            symbol,
            side,
            exit_decision.reason.value,
            exit_decision.urgency.value,
            exit_decision.explanation,
        )
        
        # Log doctrine event
        try:
            log_doctrine_event(
                event_type="doctrine_exit",
                symbol=symbol,
                details={
                    "side": side,
                    "reason": exit_decision.reason.value,
                    "urgency": exit_decision.urgency.value,
                    "explanation": exit_decision.explanation,
                    "bars_held": bars_held,
                    "alpha_survival": alpha_survival,
                    "entry_regime": entry_regime,
                    "current_regime": regime_snapshot.primary_regime if regime_snapshot else None,
                },
            )
        except Exception:
            pass
        
        try:
            record_exit_trigger()
        except Exception:
            pass
        
        # Record to cycle statistics (non-invasive)
        try:
            if _CYCLE_STATS_AVAILABLE:
                _record_exit_stat(
                    symbol=symbol,
                    exit_reason=exit_decision.reason.value,
                    regime=regime_snapshot.primary_regime if regime_snapshot else "UNKNOWN",
                    urgency=exit_decision.urgency.value,
                )
        except Exception:
            pass
        
        results.append(candidate)
    
    return results


def build_doctrine_exit_intent(candidate: DoctrineExitCandidate) -> Dict[str, Any]:
    """
    Build a reduceOnly exit intent from a DoctrineExitCandidate.
    
    Args:
        candidate: DoctrineExitCandidate from scan_doctrine_exits
        
    Returns:
        Intent dict ready for executor
    """
    from datetime import datetime, timezone
    
    # For closing a LONG position, we SELL. For SHORT, we BUY.
    close_side = "SELL" if candidate.position_side == "LONG" else "BUY"
    
    exit_block = {
        "reason": candidate.exit_reason,
        "urgency": candidate.urgency,
        "trigger_price": candidate.trigger_price,
        "entry_price": candidate.entry_price,
        "entry_regime": candidate.entry_regime,
        "current_regime": candidate.current_regime,
        "bars_held": candidate.bars_held,
        "alpha_survival": candidate.alpha_survival,
        "rationale": candidate.rationale,
        "source": "doctrine_kernel",
    }
    
    metadata = {
        "strategy": "doctrine_exit",
        "exit_reason": candidate.exit_reason,
        "exit_urgency": candidate.urgency,
        "trigger_price": candidate.trigger_price,
        "entry_price": candidate.entry_price,
        "rationale": candidate.rationale,
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
        "doctrine_exit": True,  # Flag for executor to prioritize
    }


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
    SEATBELT ONLY: Scan positions for TP/SL level crossings as emergency protection.
    
    v7.X_DOCTRINE: This function is now a SEATBELT (catastrophe protection) only.
    Primary exits should come from scan_doctrine_exits() which respects:
      CRISIS → REGIME_FLIP → STRUCTURAL_FAILURE → TIME_STOP → SEATBELT
    
    TP (take-profit) hits are logged but NOT recommended as primary exit mechanism.
    Positions should die when the THESIS dies, not when arbitrary profit targets hit.
    
    SL (stop-loss) hits are still actioned as catastrophe protection.
    
    V7.4_C3: Uses build_position_ledger() as canonical source for positions + TP/SL.
    Falls back to legacy registry if ledger unavailable.
    
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
        from execution.position_ledger import (
            LedgerReconciliationReport,
            build_position_ledger,
            load_positions_state,
            load_tp_sl_registry,
            reconcile_ledger_and_registry,
        )

        positions_state = load_positions_state()
        ledger = build_position_ledger()
        registry = load_tp_sl_registry()
        recon: LedgerReconciliationReport = reconcile_ledger_and_registry(
            positions_state=positions_state,
            ledger=ledger,
            tp_sl_registry=registry,
        )

        open_positions = list(ledger.values())
        open_positions_count = len(recon.position_keys)

        tp_sl_registered_count = sum(
            1 for entry in open_positions
            if entry.tp_sl.tp is not None or entry.tp_sl.sl is not None
        )
        tp_sl_missing_count = len(recon.missing_tp_sl_entries) + len(recon.missing_ledger_positions)
        underwater_without_tp_sl_count = 0

        # Count underwater positions that are missing TP/SL registrations
        for key in recon.missing_tp_sl_entries:
            entry = ledger.get(key)
            if not entry:
                continue
            last_price = _get_last_price(entry.symbol, price_map)
            if last_price is not None:
                pnl_pct_val = _pnl_pct(float(entry.entry_price), float(last_price), entry.side)
                if pnl_pct_val is not None and pnl_pct_val <= underwater_threshold_pct:
                    underwater_without_tp_sl_count += 1

        coverage_pct = (tp_sl_registered_count / open_positions_count) if open_positions_count > 0 else 0.0
        ledger_registry_mismatch = recon.has_mismatch

        update_exit_pipeline_status(
            open_positions_count=open_positions_count,
            tp_sl_registered_count=tp_sl_registered_count,
            tp_sl_missing_count=tp_sl_missing_count,
            underwater_without_tp_sl_count=underwater_without_tp_sl_count,
            tp_sl_coverage_pct=coverage_pct,
            ledger_registry_mismatch=ledger_registry_mismatch,
            mismatch_breakdown=recon.breakdown_counts(),
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


# ---------------------------------------------------------------------------
# Combined Doctrine + Seatbelt Exit Scanning
# ---------------------------------------------------------------------------

def scan_all_exits(
    positions: List[Dict[str, Any]],
    price_map: Dict[str, float],
) -> Tuple[List[DoctrineExitCandidate], List[ExitCandidate]]:
    """
    Combined exit scanning: doctrine exits first, then seatbelt (TP/SL).
    
    v7.X_DOCTRINE: This is the primary exit scanning function that respects
    the doctrine exit precedence:
      1. CRISIS_OVERRIDE (immediate)
      2. REGIME_FLIP (immediate)
      3. STRUCTURAL_FAILURE (stepped)
      4. TIME_STOP (patient)
      5. SEATBELT - Stop-loss catastrophe protection (last resort)
    
    v7.X_DOCTRINE: TP (take-profit) is NOT a seatbelt exit. Positions die when
    the THESIS dies, not when arbitrary profit targets hit. Only SL (stop-loss)
    qualifies as seatbelt (catastrophe protection).
    
    Returns doctrine exits and seatbelt exits separately so executor can
    prioritize appropriately. Doctrine exits should be processed BEFORE
    seatbelt exits.
    
    Args:
        positions: List of position dicts from exchange
        price_map: Dict mapping symbol to current price
        
    Returns:
        Tuple of (doctrine_exits, seatbelt_exits)
    """
    # Doctrine exits first (thesis-based)
    doctrine_exits: List[DoctrineExitCandidate] = []
    if _DOCTRINE_AVAILABLE:
        try:
            doctrine_exits = scan_doctrine_exits(positions, price_map)
        except Exception as exc:
            LOG.warning("[exit_scanner] doctrine exit scan failed: %s", exc)
    
    # Get symbols already marked for doctrine exit - exclude from seatbelt scan
    doctrine_symbols = {
        (c.symbol, c.position_side) for c in doctrine_exits
    }
    
    # Seatbelt exits (SL catastrophe protection ONLY - not TP)
    all_seatbelt_exits = scan_tp_sl_exits(positions, price_map)
    
    # v7.X_DOCTRINE: Filter to STOP_LOSS only (TP is not seatbelt)
    # Also filter out positions already covered by doctrine exits
    seatbelt_exits = [
        c for c in all_seatbelt_exits
        if (c.symbol, c.position_side) not in doctrine_symbols
        and c.exit_reason == ExitReason.STOP_LOSS  # Only SL is seatbelt
    ]
    
    # Log TP hits that are being IGNORED (not seatbelt-worthy)
    tp_hits_ignored = [
        c for c in all_seatbelt_exits
        if (c.symbol, c.position_side) not in doctrine_symbols
        and c.exit_reason == ExitReason.TAKE_PROFIT
    ]
    for tp_hit in tp_hits_ignored:
        LOG.info(
            "[exit_scanner] TP hit IGNORED (not seatbelt): %s %s at %.4f - "
            "positions die when THESIS dies, not on TP",
            tp_hit.symbol,
            tp_hit.position_side,
            tp_hit.trigger_price,
        )
    
    # Log summary
    if doctrine_exits or seatbelt_exits:
        LOG.info(
            "[exit_scanner] scan_all_exits: %d doctrine exits, %d seatbelt exits (SL only)",
            len(doctrine_exits),
            len(seatbelt_exits),
        )
    
    return doctrine_exits, seatbelt_exits


def build_combined_exit_intents(
    doctrine_exits: List[DoctrineExitCandidate],
    seatbelt_exits: List[ExitCandidate],
) -> List[Dict[str, Any]]:
    """
    Build exit intents from combined doctrine + seatbelt exits.
    
    Doctrine exits are prioritized and marked with doctrine_exit=True.
    
    Args:
        doctrine_exits: List from scan_doctrine_exits
        seatbelt_exits: List from scan_tp_sl_exits
        
    Returns:
        List of exit intents ready for executor
    """
    intents = []
    
    # Doctrine exits first (higher priority)
    for candidate in doctrine_exits:
        intent = build_doctrine_exit_intent(candidate)
        intents.append(intent)
    
    # Seatbelt exits (lower priority, catastrophe protection)
    for candidate in seatbelt_exits:
        intent = build_exit_intent(candidate)
        intent["seatbelt_exit"] = True  # Mark as seatbelt
        intents.append(intent)
    
    return intents
