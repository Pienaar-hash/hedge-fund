"""
Regime Pressure Dashboard — v7.8

Exposes how hard the market is trying to force a mistake — without changing
a single decision.

WARNING:
Regime pressure metrics are OBSERVATIONAL ONLY.
They must never be used to gate entries, size trades,
alter exits, or modify regime logic.

Any future use of regime pressure metrics for execution requires a doctrine amendment.

Architectural Placement:
    Owner: Sentinel-X (called after regime decision)
    Output: Read-only state surface
    Write path: logs/state/regime_pressure.json
    Read-only consumers: dashboards, post-mortems, shadow allocators

Hard rule:
    Nothing is allowed to read this file inside:
    - screener
    - doctrine
    - hydra
    - cerberus  
    - exits
    - risk
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESSURE_STATE_PATH = Path("logs/state/regime_pressure.json")
EVENTS_FILE = Path("logs/cycle_statistics/events.jsonl")

# Near-flip detection band
NEAR_FLIP_BAND_LOW = 0.45
NEAR_FLIP_BAND_HIGH = 0.55

# Time windows
HOURS_24 = 24
HOURS_7D = 168

# EMA smoothing for velocity
VELOCITY_ALPHA = 0.3


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RegimePressureState:
    """
    Regime pressure metrics — purely observational.
    
    These metrics reveal market pressure without affecting any decisions.
    """
    
    updated_ts: str = ""
    
    # Current regime snapshot
    current_regime: str = "CHOPPY"
    current_confidence: float = 0.5
    dwell_time_hours: float = 0.0
    stability_distance: int = 2  # cycles until stability threshold
    
    # Pressure metrics
    confidence_velocity: float = 0.0  # Δ confidence per cycle (EMA-smoothed)
    near_flip_count_24h: int = 0  # times confidence was in [0.45, 0.55]
    near_flip_count_7d: int = 0
    
    # Churn metrics
    regime_changes_24h: int = 0
    regime_changes_7d: int = 0
    avg_dwell_time_hours_7d: float = 0.0
    
    # Configuration
    near_flip_band: List[float] = field(default_factory=lambda: [NEAR_FLIP_BAND_LOW, NEAR_FLIP_BAND_HIGH])
    
    # Meta
    cycles_observed: int = 0
    calculation_window_hours: int = HOURS_7D
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "current": {
                "regime": self.current_regime,
                "confidence": round(self.current_confidence, 4),
                "dwell_time_hours": round(self.dwell_time_hours, 2),
                "stability_distance": self.stability_distance,
            },
            "pressure": {
                "confidence_velocity": round(self.confidence_velocity, 4),
                "near_flip_count_24h": self.near_flip_count_24h,
                "near_flip_count_7d": self.near_flip_count_7d,
            },
            "churn": {
                "regime_changes_24h": self.regime_changes_24h,
                "regime_changes_7d": self.regime_changes_7d,
                "avg_dwell_time_hours_7d": round(self.avg_dwell_time_hours_7d, 2),
            },
            "context": {
                "market_hostility": self._compute_hostility_label(),
            },
            "bands": {
                "near_flip_band": self.near_flip_band,
            },
            "meta": {
                "cycles_observed": self.cycles_observed,
                "calculation_window_hours": self.calculation_window_hours,
            },
        }
    
    def _compute_hostility_label(self) -> str:
        """
        Derive semantic hostility label from churn metrics.
        
        This is for HUMAN CONTEXT ONLY — never for decision logic.
        
        Labels:
            CALM     — < 2 changes/day, avg dwell > 12h
            MODERATE — 2-4 changes/day or avg dwell 6-12h
            HOSTILE  — 4-5 changes/day or avg dwell 4-6h
            EXTREME  — > 5 changes/day or avg dwell < 4h
        """
        changes_per_day = self.regime_changes_7d / 7.0 if self.regime_changes_7d > 0 else 0
        avg_dwell = self.avg_dwell_time_hours_7d
        
        # EXTREME: very high churn or very short dwell
        if changes_per_day > 5 or (avg_dwell > 0 and avg_dwell < 4):
            return "EXTREME"
        
        # HOSTILE: high churn or short dwell
        if changes_per_day > 4 or (avg_dwell > 0 and avg_dwell < 6):
            return "HOSTILE"
        
        # CALM: low churn and long dwell
        if changes_per_day < 2 and avg_dwell > 12:
            return "CALM"
        
        # MODERATE: normal market churn
        return "MODERATE"


# ---------------------------------------------------------------------------
# Pressure History (In-Memory Ring Buffer)
# ---------------------------------------------------------------------------

# Keep last N confidence readings for velocity calculation
_CONFIDENCE_HISTORY: List[float] = []
_MAX_HISTORY = 50

# Track last regime change timestamp
_LAST_REGIME_CHANGE_TS: float = 0.0
_LAST_REGIME: Optional[str] = None


def _update_confidence_history(confidence: float) -> None:
    """Add confidence reading to history ring buffer."""
    global _CONFIDENCE_HISTORY
    _CONFIDENCE_HISTORY.append(confidence)
    if len(_CONFIDENCE_HISTORY) > _MAX_HISTORY:
        _CONFIDENCE_HISTORY = _CONFIDENCE_HISTORY[-_MAX_HISTORY:]


def _compute_confidence_velocity() -> float:
    """
    Compute EMA-smoothed confidence velocity.
    
    Positive = conviction forming
    Negative = regime eroding
    """
    if len(_CONFIDENCE_HISTORY) < 2:
        return 0.0
    
    # Compute raw velocity (Δ confidence per reading)
    velocities = []
    for i in range(1, len(_CONFIDENCE_HISTORY)):
        velocities.append(_CONFIDENCE_HISTORY[i] - _CONFIDENCE_HISTORY[i - 1])
    
    if not velocities:
        return 0.0
    
    # EMA smoothing
    ema = velocities[0]
    for v in velocities[1:]:
        ema = VELOCITY_ALPHA * v + (1 - VELOCITY_ALPHA) * ema
    
    return ema


# ---------------------------------------------------------------------------
# Event-Based Metrics
# ---------------------------------------------------------------------------

def _load_regime_change_events() -> List[Dict[str, Any]]:
    """Load regime_change events from cycle statistics."""
    if not EVENTS_FILE.exists():
        return []
    
    events = []
    try:
        with open(EVENTS_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") == "regime_change":
                        events.append(event)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        _LOG.debug("[regime_pressure] Failed to load events: %s", exc)
    
    return events


def _count_regime_changes_in_window(events: List[Dict[str, Any]], hours: float) -> int:
    """Count regime changes within the last N hours."""
    now = time.time()
    cutoff = now - (hours * 3600)
    
    count = 0
    for event in events:
        ts = event.get("ts", 0)
        if ts >= cutoff:
            count += 1
    
    return count


def _compute_avg_dwell_time(events: List[Dict[str, Any]], hours: float) -> float:
    """Compute average regime dwell time over window."""
    now = time.time()
    cutoff = now - (hours * 3600)
    
    # Filter events in window
    recent = [e for e in events if e.get("ts", 0) >= cutoff]
    
    if len(recent) < 2:
        # Not enough data — use time since last change
        if recent:
            last_ts = recent[-1].get("ts", now)
            return (now - last_ts) / 3600
        return 0.0
    
    # Compute dwell times between consecutive changes
    dwell_times = []
    for i in range(1, len(recent)):
        prev_ts = recent[i - 1].get("ts", 0)
        curr_ts = recent[i].get("ts", 0)
        dwell_hours = (curr_ts - prev_ts) / 3600
        dwell_times.append(dwell_hours)
    
    if not dwell_times:
        return 0.0
    
    return sum(dwell_times) / len(dwell_times)


def _get_last_regime_change_ts(events: List[Dict[str, Any]]) -> float:
    """Get timestamp of most recent regime change."""
    if not events:
        return 0.0
    return events[-1].get("ts", 0.0)


# ---------------------------------------------------------------------------
# Near-Flip Detection
# ---------------------------------------------------------------------------

# Track near-flip events in memory (ts -> True)
_NEAR_FLIP_EVENTS: List[float] = []


def _record_near_flip_if_applicable(confidence: float) -> None:
    """Record if confidence is in near-flip band."""
    global _NEAR_FLIP_EVENTS
    
    if NEAR_FLIP_BAND_LOW <= confidence <= NEAR_FLIP_BAND_HIGH:
        _NEAR_FLIP_EVENTS.append(time.time())
        
        # Prune old events (keep 7 days)
        cutoff = time.time() - (HOURS_7D * 3600)
        _NEAR_FLIP_EVENTS = [ts for ts in _NEAR_FLIP_EVENTS if ts >= cutoff]


def _count_near_flips(hours: float) -> int:
    """Count near-flip events in last N hours."""
    cutoff = time.time() - (hours * 3600)
    return sum(1 for ts in _NEAR_FLIP_EVENTS if ts >= cutoff)


# ---------------------------------------------------------------------------
# Main Computation
# ---------------------------------------------------------------------------

def compute_regime_pressure(
    current_regime: str,
    current_confidence: float,
    cycles_stable: int = 0,
    stability_threshold: int = 2,
) -> RegimePressureState:
    """
    Compute regime pressure metrics.
    
    This function is called after Sentinel-X makes its regime decision.
    It only observes — it never affects the decision.
    
    Args:
        current_regime: Current regime label from Sentinel-X
        current_confidence: Confidence in current regime [0, 1]
        cycles_stable: How many cycles regime has been stable
        stability_threshold: Required cycles for doctrine stability
        
    Returns:
        RegimePressureState with all metrics computed
    """
    global _LAST_REGIME_CHANGE_TS, _LAST_REGIME
    
    now = time.time()
    
    # Update confidence history
    _update_confidence_history(current_confidence)
    
    # Record near-flip if applicable
    _record_near_flip_if_applicable(current_confidence)
    
    # Track regime changes
    if _LAST_REGIME is not None and _LAST_REGIME != current_regime:
        _LAST_REGIME_CHANGE_TS = now
    _LAST_REGIME = current_regime
    
    # Load historical events
    events = _load_regime_change_events()
    
    # If we have events, use the last one's timestamp for dwell time
    if events:
        last_change_ts = _get_last_regime_change_ts(events)
    else:
        last_change_ts = _LAST_REGIME_CHANGE_TS or now
    
    # Compute dwell time
    dwell_time_hours = (now - last_change_ts) / 3600 if last_change_ts > 0 else 0.0
    
    # Compute stability distance
    stability_distance = max(0, stability_threshold - cycles_stable)
    
    # Build state
    state = RegimePressureState(
        updated_ts=datetime.now(timezone.utc).isoformat(),
        current_regime=current_regime,
        current_confidence=current_confidence,
        dwell_time_hours=dwell_time_hours,
        stability_distance=stability_distance,
        confidence_velocity=_compute_confidence_velocity(),
        near_flip_count_24h=_count_near_flips(HOURS_24),
        near_flip_count_7d=_count_near_flips(HOURS_7D),
        regime_changes_24h=_count_regime_changes_in_window(events, HOURS_24),
        regime_changes_7d=_count_regime_changes_in_window(events, HOURS_7D),
        avg_dwell_time_hours_7d=_compute_avg_dwell_time(events, HOURS_7D),
        cycles_observed=len(_CONFIDENCE_HISTORY),
    )
    
    return state


def save_regime_pressure_state(state: RegimePressureState) -> bool:
    """
    Save regime pressure state to file.
    
    Returns True if successful.
    """
    try:
        PRESSURE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        tmp = PRESSURE_STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2))
        tmp.replace(PRESSURE_STATE_PATH)
        
        return True
    except Exception as exc:
        _LOG.error("[regime_pressure] Save failed: %s", exc)
        return False


def load_regime_pressure_state() -> Optional[RegimePressureState]:
    """Load regime pressure state from file."""
    if not PRESSURE_STATE_PATH.exists():
        return None
    
    try:
        data = json.loads(PRESSURE_STATE_PATH.read_text())
        current = data.get("current", {})
        pressure = data.get("pressure", {})
        churn = data.get("churn", {})
        meta = data.get("meta", {})
        bands = data.get("bands", {})
        
        return RegimePressureState(
            updated_ts=data.get("updated_ts", ""),
            current_regime=current.get("regime", "CHOPPY"),
            current_confidence=current.get("confidence", 0.5),
            dwell_time_hours=current.get("dwell_time_hours", 0.0),
            stability_distance=current.get("stability_distance", 2),
            confidence_velocity=pressure.get("confidence_velocity", 0.0),
            near_flip_count_24h=pressure.get("near_flip_count_24h", 0),
            near_flip_count_7d=pressure.get("near_flip_count_7d", 0),
            regime_changes_24h=churn.get("regime_changes_24h", 0),
            regime_changes_7d=churn.get("regime_changes_7d", 0),
            avg_dwell_time_hours_7d=churn.get("avg_dwell_time_hours_7d", 0.0),
            near_flip_band=bands.get("near_flip_band", [NEAR_FLIP_BAND_LOW, NEAR_FLIP_BAND_HIGH]),
            cycles_observed=meta.get("cycles_observed", 0),
            calculation_window_hours=meta.get("calculation_window_hours", HOURS_7D),
        )
    except Exception as exc:
        _LOG.warning("[regime_pressure] Load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Dashboard Summary (for human-readable output)
# ---------------------------------------------------------------------------

def get_pressure_summary(state: Optional[RegimePressureState] = None) -> str:
    """
    Get human-readable pressure summary.
    
    Example output:
    
    REGIME PRESSURE
    ---------------------------
    Regime: MEAN_REVERT (0.54)
    Dwell: 12.4h
    Stability Distance: 1 cycle
    
    Pressure:
      Confidence velocity:  -0.018 ↓
      Near-flips (24h):     7 ⚠️
      Near-flips (7d):      31
    
    Churn:
      Changes (24h):        5
      Changes (7d):         36
      Avg dwell (7d):       6.9h
    
    Context: EXTREME
    """
    if state is None:
        state = load_regime_pressure_state()
    
    if state is None:
        return "REGIME PRESSURE: No data available"
    
    # Velocity arrow
    if state.confidence_velocity > 0.005:
        vel_arrow = "↑"
    elif state.confidence_velocity < -0.005:
        vel_arrow = "↓"
    else:
        vel_arrow = "→"
    
    # Near-flip warning
    nf_warn = " ⚠️" if state.near_flip_count_24h >= 5 else ""
    
    # Hostility label
    hostility = state._compute_hostility_label()
    
    return f"""REGIME PRESSURE
---------------------------
Regime: {state.current_regime} ({state.current_confidence:.2f})
Dwell: {state.dwell_time_hours:.1f}h
Stability Distance: {state.stability_distance} cycle(s)

Pressure:
  Confidence velocity:  {state.confidence_velocity:+.3f} {vel_arrow}
  Near-flips (24h):     {state.near_flip_count_24h}{nf_warn}
  Near-flips (7d):      {state.near_flip_count_7d}

Churn:
  Changes (24h):        {state.regime_changes_24h}
  Changes (7d):         {state.regime_changes_7d}
  Avg dwell (7d):       {state.avg_dwell_time_hours_7d:.1f}h

Context: {hostility}"""
