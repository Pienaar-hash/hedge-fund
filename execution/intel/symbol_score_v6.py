"""Symbol scoring helpers for v6.0 telemetry (analysis-only)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import execution.router_metrics as router_metrics
import execution.rv_momentum as rv_momentum
from execution.utils.vol import get_hybrid_weight_modifiers

DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
DEFAULT_EXPECTANCY_PATH = DEFAULT_STATE_DIR / "expectancy_v6.json"
DEFAULT_ROUTER_HEALTH_PATH = DEFAULT_STATE_DIR / "router_health.json"
DEFAULT_STRATEGY_CONFIG_PATH = Path("config/strategy_config.json")
DEFAULT_FACTOR_DIAG_PATH = DEFAULT_STATE_DIR / "factor_diagnostics.json"

# Default hybrid scoring weights (fallback if config missing)
DEFAULT_W_TREND = 0.7
DEFAULT_W_CARRY = 0.3
DEFAULT_MIN_HYBRID_SCORE_LONG = 0.15
DEFAULT_MIN_HYBRID_SCORE_SHORT = 0.15

# Carry scoring normalization constants
FUNDING_CLAMP_MIN = -0.01  # -1% per 8h
FUNDING_CLAMP_MAX = 0.01   # +1% per 8h
FUNDING_SCALE_MAX = 0.7    # Funding contributes up to ±0.7 to carry_score

BASIS_CLAMP_MIN = -0.20    # -20% annualized
BASIS_CLAMP_MAX = 0.20     # +20% annualized
BASIS_SCALE_MAX = 0.3      # Basis contributes up to ±0.3 to carry_score


@dataclass
class CarryInputs:
    """Inputs for carry scoring."""
    funding_rate_8h: Optional[float] = None  # raw funding rate per 8h (e.g., 0.0005 = 0.05%)
    basis_bps: Optional[float] = None        # annualized basis in bps (e.g., 200 = 2%)


@dataclass
class SymbolScore:
    """Complete symbol score including trend, carry, and hybrid components."""
    symbol: str
    score: float  # Legacy composite score (expectancy + router)
    trend_score: float = 0.0
    carry_score: float = 0.0
    hybrid_score: float = 0.0
    vol_regime: str = "normal"  # "low" | "normal" | "high" | "crisis"
    vol_short: float = 0.0
    vol_long: float = 0.0
    vol_ratio: float = 1.0
    components: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorVector:
    """
    v7.5_C2: Per-symbol factor vector for diagnostics and attribution.
    
    Contains all factor components used in hybrid scoring, plus the final score.
    Used by factor_diagnostics module for normalization and covariance analysis.
    """
    symbol: str
    factors: Dict[str, float] = field(default_factory=dict)  # e.g. {"trend": 0.8, "carry": 0.2, ...}
    hybrid_score: float = 0.0
    direction: str = "LONG"
    regime: str = "normal"


def build_factor_vector(
    symbol: str,
    components: Dict[str, float],
    hybrid_score: float,
    direction: str = "LONG",
    regime: str = "normal",
) -> FactorVector:
    """
    Build a FactorVector from hybrid score components.
    
    Args:
        symbol: Trading pair
        components: Dict mapping factor names to their values
        hybrid_score: Final blended hybrid score
        direction: LONG or SHORT
        regime: Volatility regime
        
    Returns:
        FactorVector with all factor components
    """
    return FactorVector(
        symbol=symbol.upper(),
        factors=dict(components),
        hybrid_score=hybrid_score,
        direction=direction.upper(),
        regime=regime,
    )


@dataclass
class HybridScoringConfig:
    """Configuration for hybrid scoring weights and thresholds."""
    w_trend: float = DEFAULT_W_TREND
    w_carry: float = DEFAULT_W_CARRY
    min_hybrid_score_long: float = DEFAULT_MIN_HYBRID_SCORE_LONG
    min_hybrid_score_short: float = DEFAULT_MIN_HYBRID_SCORE_SHORT


def load_expectancy_snapshot(path: Path | str = DEFAULT_EXPECTANCY_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_router_health_snapshot(path: Path | str = DEFAULT_ROUTER_HEALTH_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _get_symbol_stats(expectancy_snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(expectancy_snapshot, Mapping):
        return {}
    data = expectancy_snapshot.get("symbols")
    if isinstance(data, Mapping):
        return {str(k).upper(): v for k, v in data.items() if isinstance(v, Mapping)}
    by_symbol = expectancy_snapshot.get("by_symbol")
    if isinstance(by_symbol, Mapping):
        return {str(k).upper(): v for k, v in by_symbol.items() if isinstance(v, Mapping)}
    return {}


def _extract_router_health_entry(router_health: Mapping[str, Any], symbol: str) -> Mapping[str, Any]:
    sym = symbol.upper()
    if not isinstance(router_health, Mapping):
        return {}
    by_symbol = router_health.get("by_symbol")
    if isinstance(by_symbol, Mapping):
        entry = by_symbol.get(sym) or by_symbol.get(symbol)
        if isinstance(entry, Mapping):
            return entry
    rh_block = router_health.get("router_health")
    if isinstance(rh_block, Mapping):
        per_symbol = rh_block.get("per_symbol")
        if isinstance(per_symbol, Mapping):
            entry = per_symbol.get(sym) or per_symbol.get(symbol)
            if isinstance(entry, Mapping):
                return entry
        elif isinstance(per_symbol, list):
            for item in per_symbol:
                if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == sym:
                    return item
    # legacy layouts
    symbols = router_health.get("symbols") if isinstance(router_health, Mapping) else None
    if isinstance(symbols, list):
        for entry in symbols:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("symbol", "")).upper() == sym:
                return entry
    per_symbol_map = router_health.get("per_symbol") if isinstance(router_health, Mapping) else None
    if isinstance(per_symbol_map, Mapping):
        maybe = per_symbol_map.get(sym) or per_symbol_map.get(symbol)
        if isinstance(maybe, Mapping):
            return maybe
    return {}


def _unpack_router_health(router_health: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(router_health, Mapping):
        return {}
    by_symbol = router_health.get("by_symbol")
    if isinstance(by_symbol, Mapping):
        return {str(k).upper(): v for k, v in by_symbol.items() if isinstance(v, Mapping)}
    rh_block = router_health.get("router_health") if isinstance(router_health, Mapping) else None
    if isinstance(rh_block, Mapping):
        per_symbol = rh_block.get("per_symbol")
        if isinstance(per_symbol, Mapping):
            return {str(k).upper(): v for k, v in per_symbol.items() if isinstance(v, Mapping)}
        if isinstance(per_symbol, list):
            return {
                str(entry.get("symbol")).upper(): entry
                for entry in per_symbol
                if isinstance(entry, Mapping) and entry.get("symbol")
            }
    symbols = router_health.get("symbols") if isinstance(router_health, Mapping) else None
    if isinstance(symbols, list):
        return {str(entry.get("symbol")).upper(): entry for entry in symbols if entry.get("symbol")}
    per_symbol_map = router_health.get("per_symbol") if isinstance(router_health, Mapping) else None
    if isinstance(per_symbol_map, Mapping):
        return {str(k).upper(): v for k, v in per_symbol_map.items() if isinstance(v, Mapping)}
    return {}


def _get_symbol_entry(snapshot: Mapping[str, Any], symbol: str) -> Any:
    """
    Fetch a per-symbol entry from either symbols/by_symbol mappings or lists.
    Returns {} if not found.
    """
    if not isinstance(snapshot, Mapping):
        return {}
    sym = symbol.upper()
    for key in ("symbols", "by_symbol"):
        block = snapshot.get(key)
        if isinstance(block, Mapping):
            entry = block.get(sym) or block.get(symbol)
            if entry is not None:
                return entry
    symbols_block = snapshot.get("symbols")
    if isinstance(symbols_block, list):
        for entry in symbols_block:
            if isinstance(entry, Mapping) and str(entry.get("symbol", "")).upper() == sym:
                return entry
    return {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_weight_map(weights: Mapping[str, float]) -> Dict[str, float]:
    """Normalize a weight map so it sums to 1.0 (or return zeros if empty)."""
    cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {k: 0.0 for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def _scale_expectancy(value: float) -> float:
    return 0.5 + 0.5 * math.tanh(value / 10.0)


def _scale_router_quality(router: Mapping[str, Any]) -> float:
    if not isinstance(router, Mapping):
        return 0.5
    quality_score = router.get("quality_score") or router.get("router_health_score")
    if quality_score is not None:
        try:
            return _clamp01(float(quality_score))
        except Exception:
            pass
    maker = float(router.get("maker_fill_rate") or 0.0)
    fallback = float(router.get("fallback_rate") or 0.0)
    raw = maker - fallback
    return 0.5 + 0.5 * math.tanh(raw)


def _slippage_penalty(router: Mapping[str, Any]) -> float:
    slip = float(router.get("slippage_p50") or 0.0)
    if slip <= 0:
        return 0.0
    return min(0.4, slip / 20.0)


def _fee_drag_penalty(router: Mapping[str, Any]) -> float:
    fees = router.get("fees_total")
    pnl = router.get("realized_pnl") or router.get("cum_pnl")
    if fees is None or pnl in (None, 0):
        return 0.0
    try:
        ratio = abs(float(fees)) / max(1e-6, abs(float(pnl)))
    except Exception:
        return 0.0
    return min(0.4, ratio * 0.2)


def _volatility_penalty(router: Mapping[str, Any]) -> float:
    scale = router.get("volatility_scale")
    if scale is None:
        return 0.0
    try:
        delta = abs(float(scale) - 1.0)
    except Exception:
        delta = 0.0
    return min(0.5, delta)


def _router_quality_score_from_health(
    router_health_snapshot: Mapping[str, Any],
    symbol: str,
    default: float = 0.0,
) -> float:
    entry = _extract_router_health_entry(router_health_snapshot, symbol)
    if entry:
        for key in ("quality_score", "router_health_score", "score"):
            if key in entry:
                try:
                    return float(entry.get(key))
                except Exception:
                    continue
    try:
        return _scale_router_quality(entry)
    except Exception:
        return default


def score_symbol(symbol: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    expect = metrics.get("expectancy") if isinstance(metrics, Mapping) else None
    router = metrics.get("router") if isinstance(metrics, Mapping) else None
    expect = expect if isinstance(expect, Mapping) else {}
    router = router if isinstance(router, Mapping) else {}
    expectancy_score = _scale_expectancy(float(expect.get("expectancy") or 0.0))
    hit_rate = expect.get("hit_rate")
    if isinstance(hit_rate, (int, float)):
        expectancy_score = 0.7 * expectancy_score + 0.3 * _clamp01(float(hit_rate))
    router_score = _scale_router_quality(router)
    slippage_pen = _slippage_penalty(router)
    fee_pen = _fee_drag_penalty(router)
    vol_pen = _volatility_penalty(router)
    raw = expectancy_score * 0.55 + router_score * 0.35 - slippage_pen - fee_pen - vol_pen * 0.3
    score = _clamp01(raw)
    return {
        "symbol": symbol,
        "score": score,
        "components": {
            "expectancy": expectancy_score,
            "router": router_score,
            "slippage_penalty": slippage_pen,
            "fee_drag_penalty": fee_pen,
            "volatility_penalty": vol_pen,
        },
        "inputs": {
            "expectancy": expect,
            "router": router,
        },
    }


def score_universe(expectancy_snapshot: Mapping[str, Any], router_health_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    exp_data = _get_symbol_stats(expectancy_snapshot)
    router_map = _unpack_router_health(router_health_snapshot)
    rq_cfg = router_metrics.load_router_quality_config()
    symbols = sorted({sym for sym in exp_data.keys() | router_map.keys() if sym})
    rows = []
    for symbol in symbols:
        entry = score_symbol(
            symbol,
            {
                "expectancy": exp_data.get(symbol, {}),
                "router": router_map.get(symbol, {}),
            },
        )
        rq_score = _router_quality_score_from_health(router_health_snapshot, symbol, default=entry["components"]["router"])
        entry["router_quality_score"] = rq_score
        entry["hybrid"] = {
            "score": entry["score"],
            "router_quality_score": rq_score,
            "passes_emission": entry["score"] >= rq_cfg.min_for_emission,
            "min_for_emission": rq_cfg.min_for_emission,
        }
        entry["hybrid_score"] = entry["hybrid"]["score"]
        rows.append(entry)
    rows.sort(key=lambda item: item["score"], reverse=True)
    return {"updated_ts": time.time(), "symbols": rows}


def build_symbol_scores(expectancy_snapshot: Mapping[str, Any], router_health_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Alias used by executor_live for publishing intel."""
    return score_universe(expectancy_snapshot, router_health_snapshot)


# ---------------------------------------------------------------------------
# Carry Scoring (Funding Rate + Basis)
# ---------------------------------------------------------------------------

DEFAULT_FUNDING_SNAPSHOT_PATH = Path("logs/state/funding_snapshot.json")
DEFAULT_BASIS_SNAPSHOT_PATH = Path("logs/state/basis_snapshot.json")


def load_funding_rate_snapshot(path: Path | str = DEFAULT_FUNDING_SNAPSHOT_PATH) -> Dict[str, Any]:
    """Load funding rate snapshot from state file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_basis_snapshot(path: Path | str = DEFAULT_BASIS_SNAPSHOT_PATH) -> Dict[str, Any]:
    """Load basis (spot-perp spread) snapshot from state file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _scale_funding_rate(funding_rate: float, direction: str) -> float:
    """
    Scale funding rate contribution to carry score.
    
    Positive funding when SHORT = favorable (shorts get paid)
    Negative funding when LONG = favorable (longs get paid)
    
    Returns 0.0-1.0 where higher = better carry for given direction.
    """
    # funding_rate is typically in range -0.001 to +0.001 (8h rate)
    # Annualized: multiply by 3 * 365 = 1095
    annual_rate = funding_rate * 1095
    
    if direction.upper() == "LONG":
        # Negative funding favors longs (they receive payment)
        raw = -annual_rate
    else:
        # Positive funding favors shorts (they receive payment)
        raw = annual_rate
    
    # Map to 0-1 using tanh, centered at 0 = 0.5
    # ±20% annualized maps to ~0.2-0.8
    return 0.5 + 0.5 * math.tanh(raw / 0.20)


def _scale_basis(basis_pct: float, direction: str) -> float:
    """
    Scale basis (perp premium/discount) contribution to carry score.
    
    Positive basis (perp > spot) = bearish signal (favors short)
    Negative basis (perp < spot) = bullish signal (favors long)
    
    Returns 0.0-1.0 where higher = better for given direction.
    """
    if direction.upper() == "LONG":
        # Negative basis (discount) favors longs
        raw = -basis_pct
    else:
        # Positive basis (premium) favors shorts
        raw = basis_pct
    
    # Map to 0-1: ±2% basis maps to ~0.2-0.8
    return 0.5 + 0.5 * math.tanh(raw / 0.02)


def carry_score(
    symbol: str,
    direction: str,
    funding_snapshot: Mapping[str, Any],
    basis_snapshot: Mapping[str, Any],
    funding_weight: float = 0.6,
    basis_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    Compute carry score for a symbol-direction pair.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        direction: 'LONG' or 'SHORT'
        funding_snapshot: Funding rate data keyed by symbol
        basis_snapshot: Basis data keyed by symbol
        funding_weight: Weight for funding rate component
        basis_weight: Weight for basis component
    
    Returns:
        Dict with score, components, and inputs
    """
    symbol = symbol.upper()
    direction = direction.upper()
    
    # Extract funding rate
    funding_data = _get_symbol_entry(funding_snapshot, symbol)
    if isinstance(funding_data, (int, float)):
        funding_rate = float(funding_data)
    elif isinstance(funding_data, Mapping):
        funding_rate = float(
            funding_data.get("rate")
            or funding_data.get("funding_rate")
            or funding_data.get("fundingRate")
            or 0.0
        )
    else:
        funding_rate = 0.0
    
    # Extract basis
    basis_data = _get_symbol_entry(basis_snapshot, symbol)
    if isinstance(basis_data, (int, float)):
        basis_pct = float(basis_data)
    elif isinstance(basis_data, Mapping):
        basis_pct = float(basis_data.get("basis_pct") or basis_data.get("basis") or 0.0)
    else:
        basis_pct = 0.0
    
    funding_score = _scale_funding_rate(funding_rate, direction)
    basis_score = _scale_basis(basis_pct, direction)
    
    # Weighted blend
    total_weight = funding_weight + basis_weight
    if total_weight > 0:
        score = (funding_score * funding_weight + basis_score * basis_weight) / total_weight
    else:
        score = 0.5
    
    return {
        "symbol": symbol,
        "direction": direction,
        "score": _clamp01(score),
        "components": {
            "funding_score": funding_score,
            "basis_score": basis_score,
        },
        "inputs": {
            "funding_rate": funding_rate,
            "basis_pct": basis_pct,
            "funding_weight": funding_weight,
            "basis_weight": basis_weight,
        },
    }


# ---------------------------------------------------------------------------
# Alpha Decay Engine (v7.5_A1)
# ---------------------------------------------------------------------------

DEFAULT_SIGNAL_TIMESTAMPS_PATH = DEFAULT_STATE_DIR / "signal_timestamps.json"
DEFAULT_HYBRID_SCORES_PATH = DEFAULT_STATE_DIR / "hybrid_scores.json"


@dataclass
class AlphaDecayConfig:
    """Configuration for alpha signal decay."""
    enabled: bool = True
    half_life_minutes: float = 45.0
    min_decay_multiplier: float = 0.35


def load_alpha_decay_config(strategy_config: Mapping[str, Any] | None = None) -> AlphaDecayConfig:
    """Load alpha decay configuration from strategy config."""
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return AlphaDecayConfig()
    
    decay_cfg = strategy_config.get("alpha_decay", {})
    
    return AlphaDecayConfig(
        enabled=bool(decay_cfg.get("enabled", True)),
        half_life_minutes=float(decay_cfg.get("half_life_minutes", 45.0)),
        min_decay_multiplier=float(decay_cfg.get("min_decay_multiplier", 0.35)),
    )


def load_signal_timestamps(path: Path | str = DEFAULT_SIGNAL_TIMESTAMPS_PATH) -> Dict[str, float]:
    """
    Load last signal update timestamps per symbol.
    
    Returns dict mapping "SYMBOL:DIRECTION" -> timestamp (Unix seconds)
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def save_signal_timestamps(
    timestamps: Dict[str, float],
    path: Path | str = DEFAULT_SIGNAL_TIMESTAMPS_PATH,
) -> None:
    """Save signal timestamps to state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(timestamps, f)
        tmp.replace(path)
    except Exception:
        pass


def update_signal_timestamp(
    symbol: str,
    direction: str,
    timestamp: float | None = None,
    path: Path | str = DEFAULT_SIGNAL_TIMESTAMPS_PATH,
) -> None:
    """Update the signal timestamp for a symbol-direction pair."""
    if timestamp is None:
        timestamp = time.time()
    
    key = f"{symbol.upper()}:{direction.upper()}"
    timestamps = load_signal_timestamps(path)
    timestamps[key] = timestamp
    save_signal_timestamps(timestamps, path)


def get_signal_age_minutes(
    symbol: str,
    direction: str,
    timestamps: Dict[str, float] | None = None,
    now: float | None = None,
) -> float:
    """
    Get the age of a signal in minutes.
    
    Args:
        symbol: Trading pair
        direction: LONG or SHORT
        timestamps: Preloaded timestamps (loads from file if None)
        now: Current timestamp (uses time.time() if None)
        
    Returns:
        Age in minutes, or 0.0 if no timestamp found (fresh signal)
    """
    if timestamps is None:
        timestamps = load_signal_timestamps()
    if now is None:
        now = time.time()
    
    key = f"{symbol.upper()}:{direction.upper()}"
    last_ts = timestamps.get(key)
    
    if last_ts is None:
        return 0.0  # No recorded timestamp = fresh signal
    
    age_seconds = max(0.0, now - last_ts)
    return age_seconds / 60.0


def compute_alpha_decay(
    age_minutes: float,
    config: AlphaDecayConfig,
) -> float:
    """
    Compute decay multiplier for a signal based on its age.
    
    Uses exponential decay: decay = exp(-age / half_life)
    Clamped to minimum multiplier.
    
    Args:
        age_minutes: Age of signal in minutes
        config: Decay configuration
        
    Returns:
        Decay multiplier in range [min_decay_multiplier, 1.0]
    """
    if not config.enabled or config.half_life_minutes <= 0:
        return 1.0
    
    # Exponential decay formula
    decay = math.exp(-age_minutes / config.half_life_minutes * math.log(2))
    
    # Clamp to minimum
    return max(config.min_decay_multiplier, min(1.0, decay))


def apply_alpha_decay(
    hybrid_score: float,
    symbol: str,
    direction: str,
    config: AlphaDecayConfig | None = None,
    timestamps: Dict[str, float] | None = None,
    now: float | None = None,
) -> Dict[str, Any]:
    """
    Apply alpha decay to a hybrid score.
    
    Args:
        hybrid_score: The base hybrid score (0.0-1.0)
        symbol: Trading pair
        direction: LONG or SHORT
        config: Decay configuration (loads from file if None)
        timestamps: Preloaded timestamps (loads from file if None)
        now: Current timestamp (uses time.time() if None)
        
    Returns:
        Dict with decayed score and decay metadata
    """
    if config is None:
        config = load_alpha_decay_config()
    
    if not config.enabled:
        return {
            "decayed_score": hybrid_score,
            "decay_multiplier": 1.0,
            "age_minutes": 0.0,
            "decay_enabled": False,
        }
    
    age_minutes = get_signal_age_minutes(symbol, direction, timestamps, now)
    decay_multiplier = compute_alpha_decay(age_minutes, config)
    decayed_score = hybrid_score * decay_multiplier
    
    return {
        "decayed_score": decayed_score,
        "decay_multiplier": decay_multiplier,
        "age_minutes": age_minutes,
        "half_life_minutes": config.half_life_minutes,
        "min_decay_multiplier": config.min_decay_multiplier,
        "decay_enabled": True,
        "at_minimum": decay_multiplier <= config.min_decay_multiplier + 0.01,
    }


def build_alpha_decay_snapshot(
    symbols: list[str],
    directions: list[str] | None = None,
    config: AlphaDecayConfig | None = None,
) -> Dict[str, Any]:
    """
    Build alpha decay state for all symbols for publishing.
    
    Args:
        symbols: List of trading pairs
        directions: List of directions (defaults to ["LONG", "SHORT"] for each)
        config: Decay configuration
        
    Returns:
        Dict suitable for JSON serialization
    """
    if config is None:
        config = load_alpha_decay_config()
    
    if directions is None:
        directions = ["LONG", "SHORT"]
    
    timestamps = load_signal_timestamps()
    now = time.time()
    
    result: Dict[str, Any] = {
        "updated_ts": now,
        "config": {
            "enabled": config.enabled,
            "half_life_minutes": config.half_life_minutes,
            "min_decay_multiplier": config.min_decay_multiplier,
        },
        "symbols": {},
    }
    
    for symbol in symbols:
        symbol = symbol.upper()
        symbol_data: Dict[str, Any] = {}
        
        for direction in directions:
            direction = direction.upper()
            age_minutes = get_signal_age_minutes(symbol, direction, timestamps, now)
            decay_multiplier = compute_alpha_decay(age_minutes, config)
            
            symbol_data[direction.lower()] = {
                "decay_multiplier": decay_multiplier,
                "age_minutes": age_minutes,
                "at_minimum": decay_multiplier <= config.min_decay_multiplier + 0.01,
            }
        
        result["symbols"][symbol] = symbol_data
    
    return result


# ---------------------------------------------------------------------------
# Hybrid Score (Blends Trend, Carry, Expectancy, Router)
# ---------------------------------------------------------------------------

@dataclass
class HybridScoreConfig:
    """Configuration for hybrid score blending weights and thresholds (v7.4 B1)."""
    trend_weight: float = 0.40
    carry_weight: float = 0.25
    expectancy_weight: float = 0.20
    router_weight: float = 0.15
    min_hybrid_score_long: float = DEFAULT_MIN_HYBRID_SCORE_LONG
    min_hybrid_score_short: float = DEFAULT_MIN_HYBRID_SCORE_SHORT


DEFAULT_HYBRID_WEIGHTS = HybridScoreConfig()


def load_hybrid_config(strategy_config: Mapping[str, Any] | None = None) -> HybridScoreConfig:
    """Load hybrid scoring config from strategy config."""
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return HybridScoreConfig()
    
    hybrid_block = strategy_config.get("hybrid_scoring", {})
    if not hybrid_block.get("enabled", True):
        return HybridScoreConfig()
    
    return HybridScoreConfig(
        trend_weight=float(hybrid_block.get("trend_weight", 0.40)),
        carry_weight=float(hybrid_block.get("carry_weight", 0.25)),
        expectancy_weight=float(hybrid_block.get("expectancy_weight", 0.20)),
        router_weight=float(hybrid_block.get("router_weight", 0.15)),
        min_hybrid_score_long=float(hybrid_block.get("intent_ranking", {}).get("min_hybrid_score_long", 0.55)),
        min_hybrid_score_short=float(hybrid_block.get("intent_ranking", {}).get("min_hybrid_score_short", 0.50)),
    )


def _get_regime_multiplier(regime: str | None, config: Mapping[str, Any] | None = None) -> float:
    """Get regime-based weight multiplier for carry component."""
    if config is None:
        config = {}
    
    modulation = config.get("hybrid_scoring", {}).get("regime_modulation", {})
    regime_key = (regime or "normal").lower()
    
    # Default multipliers by regime
    defaults = {
        "crisis": 0.5,
        "high": 0.8,
        "low": 1.2,
        "normal": 1.0,
    }
    
    return float(modulation.get(regime_key, defaults.get(regime_key, 1.0)))


_VOL_REGIME_FACTOR_MAP = {
    "low": 0.25,
    "normal": 0.0,
    "high": -0.35,
    "crisis": -0.75,
}


def _vol_regime_value(vol_regime_label: str | None) -> float:
    """
    Map volatility regime label to a signed factor value used in hybrid scoring.
    Negative values dampen scores in stressed regimes; positive boosts in calm regimes.
    """
    key = str(vol_regime_label or "normal").lower()
    return float(_VOL_REGIME_FACTOR_MAP.get(key, 0.0))


def hybrid_score(
    symbol: str,
    direction: str,
    trend_score: float,
    expectancy_snapshot: Mapping[str, Any],
    router_health_snapshot: Mapping[str, Any],
    funding_snapshot: Mapping[str, Any],
    basis_snapshot: Mapping[str, Any],
    regime: str | None = None,
    config: HybridScoreConfig | None = None,
    strategy_config: Mapping[str, Any] | None = None,
    factor_weights: Dict[str, float] | None = None,
    orthogonalized_factors: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    results = hybrid_score_universe(
        intents=[
            {
                "symbol": symbol,
                "direction": direction,
                "trend_score": trend_score,
            }
        ],
        expectancy_snapshot=expectancy_snapshot,
        router_health_snapshot=router_health_snapshot,
        funding_snapshot=funding_snapshot,
        basis_snapshot=basis_snapshot,
        regime=regime,
        config=config,
        strategy_config=strategy_config,
        factor_weights=factor_weights,
        orthogonalized_factors=orthogonalized_factors,
    )
    return results[0] if results else {
        "symbol": symbol.upper(),
        "direction": direction.upper(),
        "hybrid_score": 0.0,
        "passes_threshold": False,
        "min_threshold": router_metrics.load_router_quality_config(strategy_config).min_for_emission,
    }


def hybrid_score_universe(
    intents: list[Mapping[str, Any]],
    expectancy_snapshot: Mapping[str, Any],
    router_health_snapshot: Mapping[str, Any],
    funding_snapshot: Mapping[str, Any],
    basis_snapshot: Mapping[str, Any],
    regime: str | None = None,
    config: HybridScoreConfig | None = None,
    strategy_config: Mapping[str, Any] | None = None,
    factor_weights: Mapping[str, float] | None = None,
    orthogonalized_factors: Dict[str, Dict[str, float]] | None = None,
) -> list[Dict[str, Any]]:
    """
    Compute hybrid scores for all intents and return sorted by score.
    """
    rq_cfg = router_metrics.load_router_quality_config(strategy_config)
    rv_cfg = rv_momentum.load_rv_config(strategy_config)
    alpha_cfg = load_alpha_decay_config(strategy_config)
    cfg = config or load_hybrid_config(strategy_config)

    router_map = _unpack_router_health(router_health_snapshot)
    expect_map = _get_symbol_stats(expectancy_snapshot)

    regime_label = (regime or "normal") if regime else "normal"
    weight_mods = get_hybrid_weight_modifiers(regime_label, strategy_config)

    results: list[Dict[str, Any]] = []
    side_mode = str(((strategy_config or {}).get("vol_target") or {}).get("side_mode", "trend")).lower()

    for intent in intents:
        symbol = str(intent.get("symbol", "")).upper()
        if not symbol:
            continue
        direction = str(intent.get("direction", intent.get("side", "LONG"))).upper()
        trend_input = float(intent.get("trend_score", intent.get("signal_strength", 0.5)))
        trend_sc = _clamp01(trend_input)

        sym_score = score_symbol(
            symbol,
            {
                "expectancy": expect_map.get(symbol, {}),
                "router": router_map.get(symbol, {}),
            },
        )
        carry_result = carry_score(symbol, direction, funding_snapshot, basis_snapshot)
        expectancy_score = sym_score["components"]["expectancy"]
        router_score = sym_score["components"]["router"]

        rq_score = _router_quality_score_from_health(router_health_snapshot, symbol, default=router_score)
        if rq_cfg.enabled:
            try:
                rq_score = router_metrics.get_router_quality_score(symbol, cfg=rq_cfg)
            except Exception:
                rq_score = _router_quality_score_from_health(router_health_snapshot, symbol, default=router_score)

        rv_score_val = 0.0
        rv_enabled = bool(rv_cfg and rv_cfg.enabled)
        if rv_enabled:
            try:
                rv_score_val = float(rv_momentum.get_rv_score(symbol, cfg=rv_cfg))
            except Exception:
                rv_score_val = 0.0
        rv_weight = float(getattr(rv_cfg, "hybrid_weight", 0.0) if rv_enabled else 0.0)
        rv_weight = max(0.0, min(1.0, rv_weight))

        vol_regime_value = _vol_regime_value(regime_label)
        raw_factor_vector: Dict[str, float] = {
            "trend": trend_sc,
            "carry": carry_result["score"],
            "expectancy": expectancy_score,
            "router": router_score,
            "router_quality": rq_score,
            "rv_momentum": rv_score_val,
            "vol_regime": vol_regime_value,
        }

        factors_for_score = dict(raw_factor_vector)
        using_orthogonalized = False
        if orthogonalized_factors and symbol in orthogonalized_factors:
            for key, value in orthogonalized_factors[symbol].items():
                factors_for_score[key] = float(value)
            using_orthogonalized = True

        using_factor_weights = bool(factor_weights)
        base_weights: Dict[str, float] = {}
        if factor_weights:
            base_weights = {str(k): float(v) for k, v in factor_weights.items()}
        else:
            base_weights = {
                "trend": cfg.trend_weight,
                "carry": cfg.carry_weight,
                "expectancy": cfg.expectancy_weight,
                "router": cfg.router_weight,
            }

        # Apply vol regime weight modifiers and normalize
        base_weights["carry"] = base_weights.get("carry", 0.0) * weight_mods.carry
        base_weights["expectancy"] = base_weights.get("expectancy", 0.0) * weight_mods.expectancy
        base_weights["router"] = base_weights.get("router", 0.0) * weight_mods.router
        weights_used = _normalize_weight_map(base_weights)

        base_score = sum(
            weights_used.get(name, 0.0) * factors_for_score.get(name, 0.0)
            for name in weights_used
        )

        if rv_weight > 0:
            base_score *= max(0.0, 1.0 - rv_weight)
        hybrid_raw = base_score + rv_weight * rv_score_val

        rq_multiplier = 1.0
        if rq_cfg.enabled:
            if rq_score <= rq_cfg.low_quality_threshold:
                rq_multiplier = rq_cfg.low_quality_hybrid_multiplier
            elif rq_score >= rq_cfg.high_quality_threshold:
                rq_multiplier = rq_cfg.high_quality_hybrid_multiplier
        hybrid_after_rq = hybrid_raw * rq_multiplier

        alpha_decay_meta = {
            "decayed_score": hybrid_after_rq,
            "decay_multiplier": 1.0,
            "age_minutes": 0.0,
            "decay_enabled": False,
        }
        if alpha_cfg.enabled:
            try:
                alpha_decay_meta = apply_alpha_decay(
                    hybrid_after_rq,
                    symbol,
                    direction,
                    config=alpha_cfg,
                )
            except Exception:
                alpha_decay_meta = {
                    "decayed_score": hybrid_after_rq,
                    "decay_multiplier": 1.0,
                    "age_minutes": 0.0,
                    "decay_enabled": False,
                }

        hybrid_final = alpha_decay_meta.get("decayed_score", hybrid_after_rq)
        hybrid_final = max(-1.0, min(1.0, hybrid_final))

        min_threshold = rq_cfg.min_for_emission
        passes_threshold = hybrid_final >= min_threshold

        classification = "NEUTRAL"
        if passes_threshold and side_mode != "neutral":
            classification = direction

        reported_weights = {str(k): float(v) for k, v in factor_weights.items()} if using_factor_weights and factor_weights else {}

        factor_weighting_block = {
            "factor_weights_used": reported_weights,
            "using_factor_weights": using_factor_weights,
            "using_orthogonalized": using_orthogonalized,
        }

        result = {
            "symbol": symbol,
            "direction": direction,
            "regime": regime_label,
            "hybrid_score": hybrid_final,
            "components": {
                "trend": trend_sc,
                "carry": carry_result["score"],
                "expectancy": expectancy_score,
                "router": router_score,
            },
            "carry_details": carry_result,
            "rv_momentum": {
                "score": rv_score_val,
                "enabled": rv_enabled,
                "weight": rv_weight,
            },
            "router_quality": {
                "enabled": rq_cfg.enabled,
                "score": rq_score,
                "multiplier": rq_multiplier,
            },
            "factor_vector": {
                **raw_factor_vector,
                **({} if not using_orthogonalized else {k: factors_for_score.get(k, raw_factor_vector.get(k, 0.0)) for k in factors_for_score}),
            },
            "factor_weighting": factor_weighting_block,
            "weight_modifiers": {
                "carry": weight_mods.carry,
                "expectancy": weight_mods.expectancy,
                "router": weight_mods.router,
            },
            "alpha_decay": alpha_decay_meta,
            "passes_threshold": passes_threshold,
            "min_threshold": min_threshold,
            "intent": dict(intent),
            "classification": classification,
            "vol_regime_modifier": weight_mods.router,  # keep compatibility field
        }

        results.append(result)

    results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return results


def rank_intents_by_hybrid_score(
    intents: list[Mapping[str, Any]],
    expectancy_snapshot: Mapping[str, Any] | None = None,
    router_health_snapshot: Mapping[str, Any] | None = None,
    funding_snapshot: Mapping[str, Any] | None = None,
    basis_snapshot: Mapping[str, Any] | None = None,
    regime: str | None = None,
    strategy_config: Mapping[str, Any] | None = None,
    filter_below_threshold: bool = True,
) -> list[Dict[str, Any]]:
    """
    Rank intents by hybrid score and optionally filter below threshold.
    
    Convenience function that loads snapshots if not provided.
    
    Args:
        intents: List of intent dicts
        *_snapshot: Data snapshots (loaded from defaults if None)
        regime: Volatility regime
        strategy_config: Strategy config
        filter_below_threshold: If True, exclude intents below min threshold
    
    Returns:
        Ranked (and optionally filtered) list of hybrid score results
    """
    # Load snapshots if not provided
    if expectancy_snapshot is None:
        expectancy_snapshot = load_expectancy_snapshot()
    if router_health_snapshot is None:
        router_health_snapshot = load_router_health_snapshot()
    if funding_snapshot is None:
        funding_snapshot = load_funding_rate_snapshot()
    if basis_snapshot is None:
        basis_snapshot = load_basis_snapshot()
    
    results = hybrid_score_universe(
        intents=intents,
        expectancy_snapshot=expectancy_snapshot,
        router_health_snapshot=router_health_snapshot,
        funding_snapshot=funding_snapshot,
        basis_snapshot=basis_snapshot,
        regime=regime,
        strategy_config=strategy_config,
    )
    
    if filter_below_threshold:
        results = [r for r in results if r.get("passes_threshold", False)]
    
    return results


__all__ = [
    "load_expectancy_snapshot",
    "load_router_health_snapshot",
    "load_funding_rate_snapshot",
    "load_basis_snapshot",
    "build_symbol_scores",
    "score_symbol",
    "score_universe",
    "carry_score",
    "hybrid_score",
    "hybrid_score_universe",
    "rank_intents_by_hybrid_score",
    "load_hybrid_config",
    "HybridScoreConfig",
    # v7.5_A1 Alpha Decay exports
    "AlphaDecayConfig",
    "load_alpha_decay_config",
    "compute_alpha_decay",
    "apply_alpha_decay",
    "get_signal_age_minutes",
    "load_signal_timestamps",
    "save_signal_timestamps",
    # v7.5_C2 Factor Vector exports
    "FactorVector",
    "build_factor_vector",
]
