"""
Binary Lab S1 — signal extraction from the existing futures pipeline.

Reads published state surfaces (sentinel_x.json, symbol_scores_v6.json)
to produce directional signals and eligibility decisions.  No new indicators.

Signal source: ``entry_gate.signal_source == "futures_pipeline"``

Direction rule:
    TREND_UP   → UP
    TREND_DOWN → DOWN
    MEAN_REVERT / BREAKOUT → derived from Sentinel-X trend_slope sign
    CHOPPY / CRISIS → blocked (no direction)

Conviction proxy:
    Primary regime probability → mapped through the standard band thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State file paths (same as executor_live reads)
# ---------------------------------------------------------------------------
SENTINEL_X_STATE_PATH = Path("logs/state/sentinel_x.json")
SYMBOL_SCORES_STATE_PATH = Path("logs/state/symbol_scores_v6.json")

# ---------------------------------------------------------------------------
# Regime → direction mapping (deterministic, no new logic)
# ---------------------------------------------------------------------------
_REGIME_DIRECTION: Dict[str, Optional[str]] = {
    "TREND_UP": "UP",
    "TREND_DOWN": "DOWN",
    "MEAN_REVERT": None,   # derived from trend_slope
    "BREAKOUT": None,      # derived from trend_slope
    "CHOPPY": None,        # blocked
    "CRISIS": None,        # blocked
}

# ---------------------------------------------------------------------------
# Conviction band thresholds — mirrors conviction_engine.ConvictionConfig
# ---------------------------------------------------------------------------
_BAND_THRESHOLDS: Dict[str, float] = {
    "very_high": 0.92,
    "high": 0.80,
    "medium": 0.60,
    "low": 0.40,
    "very_low": 0.20,
}

_BAND_ORDER = ["very_high", "high", "medium", "low", "very_low"]

# Minimum band values for gating comparisons
_BAND_RANK: Dict[str, int] = {
    "very_low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinaryLabSignal:
    """Immutable signal snapshot from the futures pipeline."""
    symbol: str
    regime: str
    regime_confidence: float
    direction: Optional[str]      # "UP" | "DOWN" | None
    conviction_score: float
    conviction_band: str
    trend_slope: Optional[float]
    vol_regime_z: Optional[float]
    volume_z: Optional[float]
    hybrid_score: Optional[float]
    ts: str                       # ISO 8601 from sentinel_x updated_ts


@dataclass(frozen=True)
class EligibilityResult:
    """Gate outcome for a single round."""
    eligible: bool
    deny_reason: Optional[str]
    signal: Optional[BinaryLabSignal]


# ---------------------------------------------------------------------------
# State file readers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("binary_lab_signals: failed to read %s: %s", path, exc)
        return None


def read_sentinel_state(
    path: Path = SENTINEL_X_STATE_PATH,
) -> Optional[Dict[str, Any]]:
    """Read current regime state from Sentinel-X."""
    return _read_json(path)


def read_symbol_scores(
    path: Path = SYMBOL_SCORES_STATE_PATH,
) -> Optional[Dict[str, Any]]:
    """Read published symbol scores (v6 surface)."""
    return _read_json(path)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def _regime_to_direction(regime: str, trend_slope: Optional[float]) -> Optional[str]:
    """
    Map regime to trade direction.

    For TREND_UP/DOWN the mapping is direct.
    For MEAN_REVERT/BREAKOUT, fall back to the sign of Sentinel-X trend_slope.
    CHOPPY/CRISIS yield None (blocked).
    """
    direct = _REGIME_DIRECTION.get(regime)
    if direct is not None:
        return direct

    if regime in ("MEAN_REVERT", "BREAKOUT"):
        if trend_slope is not None and trend_slope > 0:
            return "UP"
        if trend_slope is not None and trend_slope < 0:
            return "DOWN"
    return None


def _score_to_band(score: float) -> str:
    """Map a 0-1 conviction proxy score to a band label."""
    for band in _BAND_ORDER:
        if score >= _BAND_THRESHOLDS[band]:
            return band
    return "very_low"


def _find_symbol_score(scores_data: Optional[Dict[str, Any]], symbol: str) -> Optional[float]:
    """Extract hybrid_score for *symbol* from the v6 scores surface."""
    if scores_data is None:
        return None
    symbols_list = scores_data.get("symbols")
    if isinstance(symbols_list, list):
        for entry in symbols_list:
            if isinstance(entry, dict) and entry.get("symbol") == symbol:
                return entry.get("hybrid_score") or entry.get("score")
    return None


def extract_signal(
    symbol: str = "BTCUSDT",
    *,
    sentinel_path: Path = SENTINEL_X_STATE_PATH,
    scores_path: Path = SYMBOL_SCORES_STATE_PATH,
) -> Optional[BinaryLabSignal]:
    """
    Build a :class:`BinaryLabSignal` from published state surfaces.

    Returns ``None`` when upstream data is missing or stale.
    """
    sentinel = read_sentinel_state(sentinel_path)
    if sentinel is None:
        logger.debug("binary_lab_signals: sentinel state unavailable")
        return None

    regime = str(sentinel.get("primary_regime") or "").upper()
    if not regime:
        return None

    # Regime confidence: use the primary regime probability as proxy.
    regime_probs = sentinel.get("smoothed_probs") or sentinel.get("regime_probs") or {}
    regime_conf = float(regime_probs.get(regime, 0.0))

    features = sentinel.get("features") or {}
    trend_slope = features.get("trend_slope")
    if trend_slope is not None:
        trend_slope = float(trend_slope)

    vol_regime_z_raw = features.get("vol_regime_z")
    vol_regime_z = float(vol_regime_z_raw) if vol_regime_z_raw is not None else None
    volume_z_raw = features.get("volume_z")
    volume_z = float(volume_z_raw) if volume_z_raw is not None else None

    direction = _regime_to_direction(regime, trend_slope)

    # Conviction proxy: use regime probability mapped through the standard bands.
    scores_data = read_symbol_scores(scores_path)
    hybrid = _find_symbol_score(scores_data, symbol)

    # Conviction score = max(regime_prob, hybrid_score) capped at 1.0
    # This gives the stronger of regime certainty or factor scoring.
    conviction_score = regime_conf
    if hybrid is not None and hybrid > conviction_score:
        conviction_score = min(float(hybrid), 1.0)
    conviction_band = _score_to_band(conviction_score)

    ts = str(sentinel.get("updated_ts") or "")

    return BinaryLabSignal(
        symbol=symbol,
        regime=regime,
        regime_confidence=regime_conf,
        direction=direction,
        conviction_score=round(conviction_score, 6),
        conviction_band=conviction_band,
        trend_slope=trend_slope,
        vol_regime_z=vol_regime_z,
        volume_z=volume_z,
        hybrid_score=hybrid,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Eligibility gate — deterministic, from S1 spec §2
# ---------------------------------------------------------------------------
def check_eligibility(
    signal: BinaryLabSignal,
    limits: Mapping[str, Any],
    *,
    current_nav_usd: float,
    open_positions: int,
    freeze_intact: bool,
) -> EligibilityResult:
    """
    Full eligibility gate per S1 §2 Step A.

    Conditions (all must be true):
        1. regime NOT IN blocked_regimes
        2. regime confidence >= fallback_min
        3. conviction_band >= min_conviction_band
        4. open_positions < max_concurrent
        5. direction is not None (pipeline has a signal)
        6. freeze_intact == true
        7. NAV > kill_nav_usd AND kill distance > 0
    """
    entry_gate = limits.get("entry_gate") or {}
    position_rules = limits.get("position_rules") or {}
    kill_cfg = limits.get("kill_conditions") or {}

    blocked = entry_gate.get("blocked_regimes") or []
    if signal.regime in blocked:
        return EligibilityResult(False, f"regime_blocked:{signal.regime}", signal)

    conf_min = float(entry_gate.get("regime_confidence_fallback_min", 0.60))
    if signal.regime_confidence < conf_min:
        return EligibilityResult(
            False,
            f"confidence_below_min:{signal.regime_confidence:.3f}<{conf_min}",
            signal,
        )

    min_band = str(entry_gate.get("min_conviction_band", "medium"))
    if _BAND_RANK.get(signal.conviction_band, 0) < _BAND_RANK.get(min_band, 0):
        return EligibilityResult(
            False,
            f"conviction_band_below_min:{signal.conviction_band}<{min_band}",
            signal,
        )

    max_concurrent = int(position_rules.get("max_concurrent", 3))
    if open_positions >= max_concurrent:
        return EligibilityResult(
            False,
            f"concurrent_cap:{open_positions}>={max_concurrent}",
            signal,
        )

    if signal.direction is None:
        return EligibilityResult(False, "no_direction_signal", signal)

    if not freeze_intact:
        return EligibilityResult(False, "freeze_broken", signal)

    kill_nav = float(kill_cfg.get("kill_nav_usd", 0))
    if kill_nav > 0 and current_nav_usd <= kill_nav:
        return EligibilityResult(False, "kill_line_breached", signal)
    if kill_nav > 0 and (current_nav_usd - kill_nav) <= 0:
        return EligibilityResult(False, "kill_distance_zero", signal)

    return EligibilityResult(True, None, signal)
