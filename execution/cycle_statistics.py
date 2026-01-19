"""
Cycle Statistics Collector — Multi-Cycle Metrics Without Doctrine Changes

Collects statistics for falsification criteria evaluation across regime cycles.
Append-only, non-invasive, observational only.

Reference: docs/DOCTRINE_FALSIFICATION_CRITERIA.md

Usage:
    from execution.cycle_statistics import record_cycle_event, get_cycle_summary

Events are logged to: logs/cycle_statistics.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("cycle_stats")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATS_DIR = Path("logs/cycle_statistics")
EVENTS_FILE = STATS_DIR / "events.jsonl"
SUMMARY_FILE = STATS_DIR / "summary.json"


# ---------------------------------------------------------------------------
# Event Types (aligned with falsification criteria)
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Regime events
    REGIME_CHANGE = "regime_change"
    REGIME_STABLE = "regime_stable"
    
    # Entry events
    ENTRY_ALLOWED = "entry_allowed"
    ENTRY_VETOED = "entry_vetoed"
    
    # Exit events
    EXIT_REGIME_FLIP = "exit_regime_flip"
    EXIT_STRUCTURAL = "exit_structural"
    EXIT_TIME_STOP = "exit_time_stop"
    EXIT_CRISIS = "exit_crisis"
    EXIT_SEATBELT = "exit_seatbelt"
    
    # State events
    FLAT_ENTERED = "flat_entered"
    FLAT_MAINTAINED = "flat_maintained"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Flag events (for tracking)
    GREEN_FLAG = "green_flag"
    YELLOW_FLAG = "yellow_flag"
    RED_FLAG = "red_flag"


# ---------------------------------------------------------------------------
# Statistics Tracking
# ---------------------------------------------------------------------------

@dataclass
class CycleStats:
    """Aggregated statistics for falsification analysis."""
    
    # Counters
    total_regime_changes: int = 0
    trend_up_cycles: int = 0
    trend_down_cycles: int = 0
    choppy_cycles: int = 0
    crisis_cycles: int = 0
    
    # Entry metrics
    entries_allowed: int = 0
    entries_vetoed: int = 0
    entries_during_trend: int = 0
    vetoes_during_trend: int = 0  # RF-1 indicator
    
    # Exit metrics (RF-2 detection)
    exits_regime_flip: int = 0
    exits_structural: int = 0
    exits_time_stop: int = 0
    exits_crisis: int = 0
    exits_seatbelt: int = 0
    
    # Flat behavior (RF-3, GF-1)
    flat_periods: int = 0
    entries_during_choppy: int = 0  # RF-3 indicator
    
    # Time tracking
    total_flat_time_s: float = 0.0
    total_trend_time_s: float = 0.0
    
    # Flag counts
    green_flags: int = 0
    yellow_flags: int = 0
    red_flags: int = 0
    
    # Timestamps
    first_event_ts: Optional[float] = None
    last_event_ts: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def seatbelt_ratio(self) -> float:
        """RF-2: Ratio of seatbelt exits to total exits."""
        total = (self.exits_regime_flip + self.exits_structural + 
                 self.exits_time_stop + self.exits_crisis + self.exits_seatbelt)
        if total == 0:
            return 0.0
        return self.exits_seatbelt / total
    
    @property
    def trend_participation_ratio(self) -> float:
        """RF-1: Ratio of entries to opportunities during trend."""
        opportunities = self.entries_during_trend + self.vetoes_during_trend
        if opportunities == 0:
            return 1.0  # No opportunities = no failure
        return self.entries_during_trend / opportunities
    
    @property
    def choppy_discipline_ratio(self) -> float:
        """RF-3: Inverse of churn during choppy (1.0 = perfect discipline)."""
        if self.choppy_cycles == 0:
            return 1.0
        # Ideally zero entries during choppy
        if self.entries_during_choppy == 0:
            return 1.0
        return max(0.0, 1.0 - (self.entries_during_choppy / self.choppy_cycles))


# ---------------------------------------------------------------------------
# Event Recording
# ---------------------------------------------------------------------------

def _ensure_stats_dir():
    """Create stats directory if needed."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)


def record_cycle_event(
    event_type: EventType,
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a single cycle event to the statistics log.
    
    Append-only, non-blocking, failure-tolerant.
    """
    try:
        _ensure_stats_dir()
        
        event = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "event": event_type.value if isinstance(event_type, EventType) else event_type,
            "symbol": symbol,
            "regime": regime,
            "details": details or {},
        }
        
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
            
    except Exception as exc:
        LOG.debug("[cycle_stats] record failed: %s", exc)


def record_exit_event(
    symbol: str,
    exit_reason: str,
    regime: str,
    urgency: Optional[str] = None,
    pnl: Optional[float] = None,
) -> None:
    """Convenience wrapper for exit events."""
    reason_map = {
        "REGIME_FLIP": EventType.EXIT_REGIME_FLIP,
        "STRUCTURAL_FAILURE": EventType.EXIT_STRUCTURAL,
        "TIME_STOP": EventType.EXIT_TIME_STOP,
        "CRISIS": EventType.EXIT_CRISIS,
        "CRISIS_OVERRIDE": EventType.EXIT_CRISIS,
        "SEATBELT": EventType.EXIT_SEATBELT,
        "STOP_LOSS": EventType.EXIT_SEATBELT,
    }
    
    event_type = reason_map.get(exit_reason.upper(), EventType.EXIT_SEATBELT)
    
    record_cycle_event(
        event_type=event_type,
        symbol=symbol,
        regime=regime,
        details={
            "exit_reason": exit_reason,
            "urgency": urgency,
            "pnl": pnl,
        }
    )


def record_entry_event(
    symbol: str,
    allowed: bool,
    regime: str,
    veto_reason: Optional[str] = None,
) -> None:
    """Convenience wrapper for entry events."""
    is_trend = regime in ("TREND_UP", "TREND_DOWN")
    is_choppy = regime == "CHOPPY"
    
    event_type = EventType.ENTRY_ALLOWED if allowed else EventType.ENTRY_VETOED
    
    record_cycle_event(
        event_type=event_type,
        symbol=symbol,
        regime=regime,
        details={
            "allowed": allowed,
            "veto_reason": veto_reason,
            "during_trend": is_trend,
            "during_choppy": is_choppy,
        }
    )


def record_regime_change(
    old_regime: Optional[str],
    new_regime: str,
    confidence: Optional[float] = None,
    cycles_stable: int = 0,
) -> None:
    """Record regime transition."""
    record_cycle_event(
        event_type=EventType.REGIME_CHANGE,
        regime=new_regime,
        details={
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
            "cycles_stable": cycles_stable,
        }
    )


def record_flag(
    flag_type: str,  # "green", "yellow", "red"
    flag_id: str,    # e.g., "GF-1", "RF-2"
    description: str,
) -> None:
    """Record a falsification flag observation."""
    type_map = {
        "green": EventType.GREEN_FLAG,
        "yellow": EventType.YELLOW_FLAG,
        "red": EventType.RED_FLAG,
    }
    
    record_cycle_event(
        event_type=type_map.get(flag_type.lower(), EventType.GREEN_FLAG),
        details={
            "flag_id": flag_id,
            "description": description,
        }
    )


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------

def compute_statistics() -> CycleStats:
    """
    Compute aggregate statistics from all recorded events.
    
    Returns CycleStats with all metrics computed.
    """
    stats = CycleStats()
    
    if not EVENTS_FILE.exists():
        return stats
    
    try:
        with open(EVENTS_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    _update_stats(stats, event)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        LOG.warning("[cycle_stats] compute failed: %s", exc)
    
    return stats


def _update_stats(stats: CycleStats, event: Dict[str, Any]) -> None:
    """Update stats from a single event."""
    ts = event.get("ts", 0)
    event_type = event.get("event", "")
    details = event.get("details", {})
    regime = event.get("regime")
    
    # Track time bounds
    if stats.first_event_ts is None:
        stats.first_event_ts = ts
    stats.last_event_ts = ts
    
    # Regime events
    if event_type == "regime_change":
        stats.total_regime_changes += 1
        new_regime = details.get("new_regime", regime)
        if new_regime == "TREND_UP":
            stats.trend_up_cycles += 1
        elif new_regime == "TREND_DOWN":
            stats.trend_down_cycles += 1
        elif new_regime == "CHOPPY":
            stats.choppy_cycles += 1
        elif new_regime == "CRISIS":
            stats.crisis_cycles += 1
    
    # Entry events
    elif event_type == "entry_allowed":
        stats.entries_allowed += 1
        if details.get("during_trend"):
            stats.entries_during_trend += 1
        if details.get("during_choppy"):
            stats.entries_during_choppy += 1  # RF-3 signal
    
    elif event_type == "entry_vetoed":
        stats.entries_vetoed += 1
        if details.get("during_trend"):
            stats.vetoes_during_trend += 1  # RF-1 signal
    
    # Exit events (RF-2 tracking)
    elif event_type == "exit_regime_flip":
        stats.exits_regime_flip += 1
    elif event_type == "exit_structural":
        stats.exits_structural += 1
    elif event_type == "exit_time_stop":
        stats.exits_time_stop += 1
    elif event_type == "exit_crisis":
        stats.exits_crisis += 1
    elif event_type == "exit_seatbelt":
        stats.exits_seatbelt += 1
    
    # Flat events
    elif event_type == "flat_entered":
        stats.flat_periods += 1
    
    # Flag events
    elif event_type == "green_flag":
        stats.green_flags += 1
    elif event_type == "yellow_flag":
        stats.yellow_flags += 1
    elif event_type == "red_flag":
        stats.red_flags += 1


def get_cycle_summary() -> Dict[str, Any]:
    """
    Get a summary of cycle statistics with falsification indicators.
    
    Returns dict suitable for logging or dashboard display.
    """
    stats = compute_statistics()
    
    total_exits = (stats.exits_regime_flip + stats.exits_structural + 
                   stats.exits_time_stop + stats.exits_crisis + stats.exits_seatbelt)
    
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "observation_period": {
            "first_event": stats.first_event_ts,
            "last_event": stats.last_event_ts,
            "duration_hours": ((stats.last_event_ts or 0) - (stats.first_event_ts or 0)) / 3600 if stats.first_event_ts else 0,
        },
        "regime_distribution": {
            "total_changes": stats.total_regime_changes,
            "trend_up": stats.trend_up_cycles,
            "trend_down": stats.trend_down_cycles,
            "choppy": stats.choppy_cycles,
            "crisis": stats.crisis_cycles,
        },
        "entry_metrics": {
            "allowed": stats.entries_allowed,
            "vetoed": stats.entries_vetoed,
            "during_trend": stats.entries_during_trend,
            "vetoed_during_trend": stats.vetoes_during_trend,
        },
        "exit_distribution": {
            "total": total_exits,
            "regime_flip": stats.exits_regime_flip,
            "structural": stats.exits_structural,
            "time_stop": stats.exits_time_stop,
            "crisis": stats.exits_crisis,
            "seatbelt": stats.exits_seatbelt,
        },
        "falsification_indicators": {
            "RF1_trend_participation": round(stats.trend_participation_ratio, 3),
            "RF2_seatbelt_ratio": round(stats.seatbelt_ratio, 3),
            "RF3_choppy_discipline": round(stats.choppy_discipline_ratio, 3),
            "entries_during_choppy": stats.entries_during_choppy,
        },
        "flag_counts": {
            "green": stats.green_flags,
            "yellow": stats.yellow_flags,
            "red": stats.red_flags,
        },
        "health_assessment": _assess_health(stats),
    }


def _assess_health(stats: CycleStats) -> Dict[str, Any]:
    """Assess system health based on falsification criteria."""
    issues = []
    status = "HEALTHY"
    
    # RF-1: Trend participation too low
    if stats.vetoes_during_trend > 3 and stats.trend_participation_ratio < 0.5:
        issues.append("RF-1: Low participation during trend regimes")
        status = "WARNING"
    
    # RF-2: Seatbelt dominated
    if stats.seatbelt_ratio > 0.5 and (stats.exits_seatbelt > 2):
        issues.append("RF-2: Seatbelt exits dominating")
        status = "CRITICAL"
    
    # RF-3: Churn during choppy
    if stats.entries_during_choppy > 0:
        issues.append(f"RF-3: {stats.entries_during_choppy} entries during CHOPPY")
        status = "CRITICAL"
    
    # Red flags observed
    if stats.red_flags > 0:
        issues.append(f"Red flags observed: {stats.red_flags}")
        status = "CRITICAL"
    
    return {
        "status": status,
        "issues": issues,
        "green_flags_observed": stats.green_flags,
    }


def write_summary() -> None:
    """Write current summary to file."""
    try:
        _ensure_stats_dir()
        summary = get_cycle_summary()
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        LOG.warning("[cycle_stats] summary write failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI for manual inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary = get_cycle_summary()
        print(json.dumps(summary, indent=2))
    else:
        print("Usage: python -m execution.cycle_statistics summary")
