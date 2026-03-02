"""
Doctrine Kernel — v7.X (CONSTITUTIONAL LAW)

The Doctrine Kernel is the SUPREME AUTHORITY over all trading decisions.
It encodes hard laws that cannot be bypassed, disabled, or configured away.

This module:
1. Decides WHETHER the system is allowed to trade (entry permission)
2. Decides WHETHER a position must exit (exit permission)
3. Produces multipliers for position sizing
4. Never consults configs for "enabled" flags — it IS the law

Architecture:
    Regime (Sentinel-X) ──┐
    Intent (Signal)     ──┼──▶ doctrine_entry_verdict() ──▶ ALLOW / VETO
    Execution (Minotaur)──┤
    Portfolio (Hydra)   ──┘

    Regime    ──┐
    Position  ──┼──▶ doctrine_exit_verdict() ──▶ EXIT / HOLD
    Alpha     ──┘

Single Source of Truth: This module defines the trading constitution.
No other module may authorize trades. All roads lead through here.

Safety Axiom — Kill Switch Exit Exemption (v7.9-KS):
    KILL_SWITCH may only block risk-increasing orders (new entries).
    It MUST NEVER block reduceOnly exits issued under doctrine authority.
    Positions whose thesis has died (REGIME_FLIP, CRISIS_OVERRIDE, etc.)
    require unconditional exit capability — blocking them converts a
    risk-limiting mechanism into a risk-amplifying one.
    Enforced in executor_live._send_order via the two-flag guard:
        _is_doctrine_exit = bool(intent["doctrine_exit"]) and bool(intent["reduceOnly"])
    Both flags must be True; spoofing doctrine_exit on entries has no effect.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constitutional Constants (HARD-CODED, NOT CONFIGURABLE)
# ---------------------------------------------------------------------------

# Regime must be stable for at least N cycles before entry is permitted
REGIME_STABILITY_CYCLES = 2

# Minimum regime confidence to permit any entry
REGIME_CONFIDENCE_FLOOR = 0.45

# Regimes that permit directional trades
REGIME_DIRECTION_MAP = {
    "TREND_UP": ["BUY", "LONG"],
    "TREND_DOWN": ["SELL", "SHORT"],
    "MEAN_REVERT": ["BUY", "SELL", "LONG", "SHORT"],  # Both directions via z-score
    "BREAKOUT": ["BUY", "SELL", "LONG", "SHORT"],  # After confirmation
    "CHOPPY": [],  # NO TRADES (micro-size only, effectively zero)
    "CRISIS": [],  # NO TRADES (forced contraction)
}

# Execution regimes that block trading
EXECUTION_BLOCKED_REGIMES = ["CRUNCH", "CRISIS", "HALT"]

# Alpha survival floor — below this, strategy is dying
ALPHA_SURVIVAL_FLOOR = 0.20

# Time stop — position has not progressed after T bars
TIME_STOP_BARS = 96  # ~24 hours on 15m

# Regime staleness — if no update in N seconds, refuse trades
REGIME_STALENESS_SECONDS = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DoctrineVerdict(str, Enum):
    """Outcome of doctrine entry evaluation."""
    
    ALLOW = "ALLOW"
    VETO_NO_REGIME = "VETO_NO_REGIME"
    VETO_REGIME_STALE = "VETO_REGIME_STALE"
    VETO_REGIME_UNSTABLE = "VETO_REGIME_UNSTABLE"
    VETO_REGIME_CONFIDENCE = "VETO_REGIME_CONFIDENCE"
    VETO_DIRECTION_MISMATCH = "VETO_DIRECTION_MISMATCH"
    VETO_CRISIS = "VETO_CRISIS"
    VETO_EXECUTION_CRUNCH = "VETO_EXECUTION_CRUNCH"
    VETO_NO_HEAD_BUDGET = "VETO_NO_HEAD_BUDGET"
    VETO_ALPHA_ROUTER_FLOOR = "VETO_ALPHA_ROUTER_FLOOR"
    VETO_ALPHA_SURVIVAL = "VETO_ALPHA_SURVIVAL"


class ExitReason(str, Enum):
    """Why a position should exit."""
    
    HOLD = "HOLD"  # No exit required
    
    # Priority 1: Crisis
    CRISIS_OVERRIDE = "CRISIS_OVERRIDE"
    
    # Priority 2: Regime invalidation
    REGIME_FLIP = "REGIME_FLIP"
    REGIME_CONFIDENCE_COLLAPSE = "REGIME_CONFIDENCE_COLLAPSE"
    
    # Priority 3: Structural failure
    TREND_DECAY = "TREND_DECAY"
    CARRY_DISAPPEARED = "CARRY_DISAPPEARED"
    CROSSFIRE_RESOLVED = "CROSSFIRE_RESOLVED"
    EXECUTION_ALPHA_DRAG = "EXECUTION_ALPHA_DRAG"
    
    # Priority 4: Time stop
    TIME_STOP = "TIME_STOP"
    
    # Priority 5: Seatbelt (emergency only)
    STOP_LOSS_SEATBELT = "STOP_LOSS_SEATBELT"
    

class ExitUrgency(str, Enum):
    """How urgently to exit."""
    
    IMMEDIATE = "IMMEDIATE"  # Market order now
    STEPPED = "STEPPED"  # TWAP over N minutes
    PATIENT = "PATIENT"  # Limit order, can wait


# ---------------------------------------------------------------------------
# Snapshots (Immutable Inputs to Doctrine)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSnapshot:
    """Immutable snapshot of Sentinel-X regime state."""
    
    primary_regime: str
    confidence: float
    secondary_regime: Optional[str] = None
    secondary_confidence: float = 0.0
    cycles_stable: int = 0
    crisis_flag: bool = False
    crisis_reason: str = ""
    updated_ts: float = 0.0
    
    def is_stale(self, max_age_s: float = REGIME_STALENESS_SECONDS) -> bool:
        """Check if regime data is stale."""
        if self.updated_ts <= 0:
            return True
        return (time.time() - self.updated_ts) > max_age_s
    
    def permits_direction(self, direction: str) -> bool:
        """Check if regime permits this direction."""
        allowed = REGIME_DIRECTION_MAP.get(self.primary_regime, [])
        return direction.upper() in allowed


@dataclass(frozen=True)
class IntentSnapshot:
    """Immutable snapshot of a trade intent."""
    
    symbol: str
    direction: str  # BUY/SELL/LONG/SHORT
    head: str  # Which Hydra head generated this
    raw_size_usd: float = 0.0
    alpha_router_allocation: float = 1.0
    conviction: float = 0.5


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Immutable snapshot of Minotaur execution state."""
    
    regime: str = "NORMAL"  # NORMAL, THIN, WIDE_SPREAD, SPIKE, CRUNCH
    quality_score: float = 0.8
    avg_slippage_bps: float = 0.0
    throttling_active: bool = False


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Immutable snapshot of portfolio/Hydra state."""
    
    head_budget_remaining: Dict[str, float] = field(default_factory=dict)
    total_exposure_pct: float = 0.0
    drawdown_pct: float = 0.0
    risk_mode: str = "OK"


@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable snapshot of an open position."""
    
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    current_price: float
    qty: float
    entry_regime: str
    entry_regime_confidence: float
    entry_ts: float
    bars_held: int
    unrealized_pnl_pct: float
    entry_head: str = ""
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


@dataclass(frozen=True)
class AlphaHealthSnapshot:
    """Immutable snapshot of alpha/strategy health."""
    
    survival_probability: float = 1.0
    trend_strength: float = 0.5
    carry_edge: float = 0.0
    execution_drag_bps: float = 0.0
    crossfire_spread_remaining: float = 1.0


# ---------------------------------------------------------------------------
# Decision Outputs
# ---------------------------------------------------------------------------


@dataclass
class DoctrineDecision:
    """Result of doctrine_entry_verdict."""
    
    verdict: DoctrineVerdict
    allowed: bool
    reason: str
    
    # Multipliers (only meaningful if allowed=True)
    regime_multiplier: float = 1.0
    execution_multiplier: float = 1.0
    alpha_multiplier: float = 1.0
    
    # Final composite multiplier
    @property
    def composite_multiplier(self) -> float:
        if not self.allowed:
            return 0.0
        return self.regime_multiplier * self.execution_multiplier * self.alpha_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "regime_multiplier": round(self.regime_multiplier, 4),
            "execution_multiplier": round(self.execution_multiplier, 4),
            "alpha_multiplier": round(self.alpha_multiplier, 4),
            "composite_multiplier": round(self.composite_multiplier, 4),
        }


@dataclass
class ExitDecision:
    """Result of doctrine_exit_verdict."""
    
    should_exit: bool
    reason: ExitReason
    urgency: ExitUrgency
    exit_pct: float = 1.0  # Partial exit support
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_exit": self.should_exit,
            "reason": self.reason.value,
            "urgency": self.urgency.value,
            "exit_pct": round(self.exit_pct, 4),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# DOCTRINE ENTRY VERDICT (THE LAW)
# ---------------------------------------------------------------------------


def doctrine_entry_verdict(
    regime: RegimeSnapshot,
    intent: IntentSnapshot,
    execution: ExecutionSnapshot,
    portfolio: PortfolioSnapshot,
    alpha_health: Optional[AlphaHealthSnapshot] = None,
) -> DoctrineDecision:
    """
    THE SUPREME ENTRY GATE.
    
    Returns ALLOW only if ALL conditions are met:
    1. Regime exists and is not stale
    2. Regime has been stable for >= N cycles
    3. Regime confidence >= floor
    4. Regime permits the intended direction
    5. Not in CRISIS
    6. Execution regime != CRUNCH
    7. Head has remaining budget > 0
    8. Alpha Router allocation > 0
    9. Alpha survival > floor (if available)
    
    This function NEVER reads config files.
    This function NEVER checks "enabled" flags.
    This function IS the law.
    """
    
    # 1. Regime exists and is not stale
    if regime.primary_regime is None or regime.primary_regime == "":
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_NO_REGIME,
            allowed=False,
            reason="No regime classification available",
        )
    
    if regime.is_stale():
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_REGIME_STALE,
            allowed=False,
            reason=f"Regime data stale (updated {regime.updated_ts})",
        )
    
    # 2. Regime stability (must be stable for N cycles)
    if regime.cycles_stable < REGIME_STABILITY_CYCLES:
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_REGIME_UNSTABLE,
            allowed=False,
            reason=f"Regime only stable for {regime.cycles_stable} cycles (need {REGIME_STABILITY_CYCLES})",
        )
    
    # 3. Regime confidence floor
    if regime.confidence < REGIME_CONFIDENCE_FLOOR:
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_REGIME_CONFIDENCE,
            allowed=False,
            reason=f"Regime confidence {regime.confidence:.2f} < floor {REGIME_CONFIDENCE_FLOOR}",
        )
    
    # 4. Crisis check (hard override)
    if regime.crisis_flag or regime.primary_regime == "CRISIS":
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_CRISIS,
            allowed=False,
            reason=f"CRISIS active: {regime.crisis_reason or 'regime=CRISIS'}",
        )
    
    # 5. Direction alignment
    if not regime.permits_direction(intent.direction):
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_DIRECTION_MISMATCH,
            allowed=False,
            reason=f"Regime {regime.primary_regime} does not permit {intent.direction}",
        )
    
    # 6. Execution regime check
    if execution.regime.upper() in EXECUTION_BLOCKED_REGIMES:
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_EXECUTION_CRUNCH,
            allowed=False,
            reason=f"Execution regime {execution.regime} blocks trading",
        )
    
    # 7. Head budget check
    head_budget = portfolio.head_budget_remaining.get(intent.head, 1.0)
    if head_budget <= 0:
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_NO_HEAD_BUDGET,
            allowed=False,
            reason=f"Head {intent.head} has no remaining budget",
        )
    
    # 8. Alpha Router allocation check
    if intent.alpha_router_allocation <= 0:
        return DoctrineDecision(
            verdict=DoctrineVerdict.VETO_ALPHA_ROUTER_FLOOR,
            allowed=False,
            reason="Alpha Router allocation is zero",
        )
    
    # 9. Alpha survival check (if available)
    if alpha_health is not None:
        if alpha_health.survival_probability < ALPHA_SURVIVAL_FLOOR:
            return DoctrineDecision(
                verdict=DoctrineVerdict.VETO_ALPHA_SURVIVAL,
                allowed=False,
                reason=f"Alpha survival {alpha_health.survival_probability:.2f} < floor {ALPHA_SURVIVAL_FLOOR}",
            )
    
    # ALL CHECKS PASSED — Calculate multipliers
    
    # Regime multiplier: scale down for lower confidence
    regime_mult = min(1.0, regime.confidence / 0.7)  # Full size at 70%+ confidence
    
    # Execution multiplier: scale down for poor execution quality
    exec_mult = min(1.0, execution.quality_score)
    
    # Alpha multiplier: scale down for lower survival
    alpha_mult = 1.0
    if alpha_health is not None:
        alpha_mult = min(1.0, alpha_health.survival_probability / 0.5)
    
    return DoctrineDecision(
        verdict=DoctrineVerdict.ALLOW,
        allowed=True,
        reason=f"Entry permitted: regime={regime.primary_regime} conf={regime.confidence:.2f}",
        regime_multiplier=regime_mult,
        execution_multiplier=exec_mult,
        alpha_multiplier=alpha_mult,
    )


# ---------------------------------------------------------------------------
# DOCTRINE EXIT VERDICT (THE LAW)
# ---------------------------------------------------------------------------


def doctrine_exit_verdict(
    regime: RegimeSnapshot,
    position: PositionSnapshot,
    execution: ExecutionSnapshot,
    alpha_health: Optional[AlphaHealthSnapshot] = None,
) -> ExitDecision:
    """
    THE SUPREME EXIT GATE.
    
    Exit precedence (strict order):
    1. CRISIS OVERRIDE — Immediate exit
    2. REGIME INVALIDATION — Regime flipped against position
    3. STRUCTURAL FAILURE — Trend/carry/crossfire edge disappeared
    4. TIME STOP — Position not progressing
    5. SEATBELT — Stop-loss hit (emergency only)
    
    TP/SL are NOT primary exit mechanisms.
    Positions die when the THESIS dies.
    """
    
    # PRIORITY 1: Crisis override
    if regime.crisis_flag or regime.primary_regime == "CRISIS":
        return ExitDecision(
            should_exit=True,
            reason=ExitReason.CRISIS_OVERRIDE,
            urgency=ExitUrgency.IMMEDIATE,
            exit_pct=1.0,
            explanation=f"CRISIS active: {regime.crisis_reason or 'regime=CRISIS'}",
        )
    
    # PRIORITY 2: Regime invalidation
    # Check if regime flipped against position direction
    position_dir = "BUY" if position.side.upper() == "LONG" else "SELL"
    if not regime.permits_direction(position_dir):
        # Regime no longer permits our direction
        if regime.primary_regime != position.entry_regime:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.REGIME_FLIP,
                urgency=ExitUrgency.STEPPED,
                exit_pct=1.0,
                explanation=f"Regime flipped from {position.entry_regime} to {regime.primary_regime}",
            )
    
    # Check for confidence collapse
    if regime.confidence < REGIME_CONFIDENCE_FLOOR * 0.7:  # 30% below floor
        return ExitDecision(
            should_exit=True,
            reason=ExitReason.REGIME_CONFIDENCE_COLLAPSE,
            urgency=ExitUrgency.STEPPED,
            exit_pct=1.0,
            explanation=f"Regime confidence collapsed to {regime.confidence:.2f}",
        )
    
    # PRIORITY 3: Structural failure (if alpha_health available)
    if alpha_health is not None:
        # Trend decay
        if alpha_health.trend_strength < 0.2:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TREND_DECAY,
                urgency=ExitUrgency.PATIENT,
                exit_pct=1.0,
                explanation=f"Trend strength decayed to {alpha_health.trend_strength:.2f}",
            )
        
        # Carry disappeared
        if abs(alpha_health.carry_edge) < 0.001:  # Near zero
            # Only exit if we entered for carry
            pass  # Would need entry thesis to check
        
        # Crossfire resolved
        if alpha_health.crossfire_spread_remaining < 0.1:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.CROSSFIRE_RESOLVED,
                urgency=ExitUrgency.PATIENT,
                exit_pct=1.0,
                explanation="Crossfire spread has resolved",
            )
        
        # Execution alpha drag
        if alpha_health.execution_drag_bps > 20:  # Losing 20+ bps to execution
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.EXECUTION_ALPHA_DRAG,
                urgency=ExitUrgency.PATIENT,
                exit_pct=0.5,  # Partial exit
                explanation=f"Execution drag {alpha_health.execution_drag_bps:.1f} bps",
            )
    
    # PRIORITY 4: Time stop
    if position.bars_held >= TIME_STOP_BARS:
        # Only exit if position hasn't progressed
        if position.unrealized_pnl_pct < 0.005:  # Less than 0.5% profit
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                urgency=ExitUrgency.PATIENT,
                exit_pct=1.0,
                explanation=f"Position held {position.bars_held} bars without progress",
            )
    
    # PRIORITY 5: Stop-loss seatbelt (emergency only)
    if position.sl_price is not None:
        if position.side.upper() == "LONG" and position.current_price <= position.sl_price:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.STOP_LOSS_SEATBELT,
                urgency=ExitUrgency.IMMEDIATE,
                exit_pct=1.0,
                explanation=f"Stop-loss seatbelt triggered at {position.sl_price}",
            )
        elif position.side.upper() == "SHORT" and position.current_price >= position.sl_price:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.STOP_LOSS_SEATBELT,
                urgency=ExitUrgency.IMMEDIATE,
                exit_pct=1.0,
                explanation=f"Stop-loss seatbelt triggered at {position.sl_price}",
            )
    
    # NO EXIT REQUIRED
    return ExitDecision(
        should_exit=False,
        reason=ExitReason.HOLD,
        urgency=ExitUrgency.PATIENT,
        exit_pct=0.0,
        explanation="Position thesis remains valid",
    )


# ---------------------------------------------------------------------------
# Convenience Builders
# ---------------------------------------------------------------------------


def build_regime_snapshot_from_state(state: Dict[str, Any]) -> RegimeSnapshot:
    """Build RegimeSnapshot from sentinel_x.json state."""
    # Parse updated_ts to epoch
    updated_ts = 0.0
    ts_str = state.get("updated_ts", "")
    if ts_str:
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            updated_ts = dt.timestamp()
        except Exception:
            pass
    
    # Calculate cycles stable from history_meta
    history = state.get("history_meta", {})
    cycles_stable = history.get("consecutive_count", 0)
    
    # Get confidence from smoothed_probs or regime_probs
    probs = state.get("smoothed_probs", state.get("regime_probs", {}))
    primary = state.get("primary_regime", "CHOPPY")
    confidence = probs.get(primary, 0.5)
    
    secondary = state.get("secondary_regime")
    secondary_conf = probs.get(secondary, 0.0) if secondary else 0.0
    
    return RegimeSnapshot(
        primary_regime=primary,
        confidence=confidence,
        secondary_regime=secondary,
        secondary_confidence=secondary_conf,
        cycles_stable=cycles_stable,
        crisis_flag=state.get("crisis_flag", False),
        crisis_reason=state.get("crisis_reason", ""),
        updated_ts=updated_ts,
    )


def build_execution_snapshot_from_state(state: Dict[str, Any]) -> ExecutionSnapshot:
    """Build ExecutionSnapshot from execution_quality.json or minotaur state."""
    meta = state.get("meta", {})
    return ExecutionSnapshot(
        regime=_infer_execution_regime(state),
        quality_score=meta.get("quality_score", 0.8),
        avg_slippage_bps=meta.get("avg_slippage_bps", 0.0),
        throttling_active=meta.get("throttling_active", False),
    )


def _infer_execution_regime(state: Dict[str, Any]) -> str:
    """Infer execution regime from state."""
    meta = state.get("meta", {})
    
    if meta.get("throttling_active"):
        return "CRUNCH"
    
    thin_symbols = meta.get("thin_liquidity_symbols", [])
    crunch_symbols = meta.get("crunch_symbols", [])
    
    if len(crunch_symbols) > 3:
        return "CRUNCH"
    elif len(thin_symbols) > 5:
        return "THIN"
    
    return "NORMAL"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_doctrine_event(
    event_type: str,
    symbol: str,
    verdict: DoctrineVerdict | ExitReason,
    details: Dict[str, Any],
) -> None:
    """
    Log doctrine events to dedicated channel.
    
    v7.X_DOCTRINE: Refusal is first-class. Every veto is logged with full
    context so we can audit decision quality. "Declining a trade should be
    visible and auditable, not silent."
    
    Event types:
    - ENTRY_VETO: Doctrine vetoed a new position
    - ENTRY_ALLOW: Doctrine allowed a new position
    - EXIT_DOCTRINE: Doctrine-based exit triggered
    - EXIT_SEATBELT: Stop-loss seatbelt triggered
    """
    import json
    from pathlib import Path
    
    log_path = Path("logs/doctrine_events.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "symbol": symbol,
        "verdict": verdict.value if hasattr(verdict, "value") else str(verdict),
        **details,
    }
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        _LOG.warning("Failed to log doctrine event: %s", e)


def get_recent_refusals(hours: float = 24.0, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent doctrine refusals for audit/analysis.
    
    v7.X_DOCTRINE: Refusal is strength, not weakness. This function provides
    visibility into what trades the doctrine prevented.
    
    Args:
        hours: How many hours back to look
        limit: Maximum number of events to return
        
    Returns:
        List of refusal events, most recent first
    """
    import json
    from pathlib import Path
    
    log_path = Path("logs/doctrine_events.jsonl")
    if not log_path.exists():
        return []
    
    cutoff_ts = datetime.now(timezone.utc) - timedelta(hours=hours)
    refusals = []
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    # Only include vetoes
                    if event.get("type") != "ENTRY_VETO":
                        continue
                    # Check timestamp
                    ts_str = event.get("ts", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts < cutoff_ts:
                            continue
                    except Exception:
                        continue
                    refusals.append(event)
                except Exception:
                    continue
    except Exception as e:
        _LOG.warning("Failed to read doctrine events: %s", e)
    
    # Sort by ts descending, limit
    refusals.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return refusals[:limit]


def get_refusal_summary(hours: float = 24.0) -> Dict[str, Any]:
    """
    Get summary of doctrine refusals.
    
    Returns counts by verdict type and recent examples.
    """
    refusals = get_recent_refusals(hours=hours, limit=1000)
    
    by_verdict: Dict[str, int] = {}
    by_symbol: Dict[str, int] = {}
    
    for r in refusals:
        verdict = r.get("verdict", "unknown")
        symbol = r.get("symbol", "unknown")
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
    
    return {
        "period_hours": hours,
        "total_refusals": len(refusals),
        "by_verdict": by_verdict,
        "by_symbol": by_symbol,
        "recent_examples": refusals[:5],
    }


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "REGIME_STABILITY_CYCLES",
    "REGIME_CONFIDENCE_FLOOR",
    "REGIME_DIRECTION_MAP",
    "EXECUTION_BLOCKED_REGIMES",
    "ALPHA_SURVIVAL_FLOOR",
    "TIME_STOP_BARS",
    # Enums
    "DoctrineVerdict",
    "ExitReason",
    "ExitUrgency",
    # Snapshots
    "RegimeSnapshot",
    "IntentSnapshot",
    "ExecutionSnapshot",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "AlphaHealthSnapshot",
    # Decisions
    "DoctrineDecision",
    "ExitDecision",
    # Core functions
    "doctrine_entry_verdict",
    "doctrine_exit_verdict",
    # Builders
    "build_regime_snapshot_from_state",
    "build_execution_snapshot_from_state",
    # Logging & Audit
    "log_doctrine_event",
    "get_recent_refusals",
    "get_refusal_summary",
]
