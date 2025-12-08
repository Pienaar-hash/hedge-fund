"""
Volatility helpers for execution sizing.

Backed by pnl_tracker or other utilities that compute realized volatility.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from execution import pnl_tracker


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# EWMA Volatility Regime Model (v7.4 B2)
# ---------------------------------------------------------------------------

VolRegimeLabel = Literal["low", "normal", "high", "crisis"]


@dataclass
class VolRegimeConfig:
    """Configuration for EWMA volatility regime classification."""
    short_window_bars: int = 168       # ~7 days at hourly
    long_window_bars: int = 720        # ~30 days at hourly
    short_halflife_bars: int = 84      # halflife for short EWMA
    long_halflife_bars: int = 360      # halflife for long EWMA
    ratio_low: float = 0.6             # below this = "low" regime
    ratio_normal: float = 1.2          # below this = "normal" regime
    ratio_high: float = 1.8            # below this = "high", above = "crisis"


@dataclass
class VolRegime:
    """Result of volatility regime classification."""
    label: VolRegimeLabel
    vol_short: float
    vol_long: float
    ratio: float


@dataclass
class VolRegimeSizingConfig:
    """Tier-specific sizing multipliers by volatility regime."""
    low: float = 1.0
    normal: float = 1.0
    high: float = 1.0
    crisis: float = 1.0


@dataclass
class HybridWeightModifiers:
    """Regime-specific modifiers for hybrid score components."""
    carry: float = 1.0
    expectancy: float = 1.0
    router: float = 1.0


DEFAULT_VOL_REGIME_CONFIG = VolRegimeConfig()


def load_vol_regime_config(strategy_config: Mapping[str, Any] | None = None) -> VolRegimeConfig:
    """Load volatility regime config from strategy_config."""
    if strategy_config is None:
        return VolRegimeConfig()
    
    vol_cfg = strategy_config.get("vol_regimes", {})
    if not vol_cfg.get("enabled", True):
        return VolRegimeConfig()
    
    defaults = vol_cfg.get("defaults", {})
    thresholds = defaults.get("ratio_thresholds", {})
    
    return VolRegimeConfig(
        short_window_bars=int(defaults.get("short_window_bars", 168)),
        long_window_bars=int(defaults.get("long_window_bars", 720)),
        short_halflife_bars=int(defaults.get("short_halflife_bars", 84)),
        long_halflife_bars=int(defaults.get("long_halflife_bars", 360)),
        ratio_low=float(thresholds.get("low", 0.6)),
        ratio_normal=float(thresholds.get("normal", 1.2)),
        ratio_high=float(thresholds.get("high", 1.8)),
    )


def compute_log_returns(prices: Sequence[float]) -> list[float]:
    """
    Compute log returns from a price series.
    
    Args:
        prices: Sequence of prices (oldest to newest)
    
    Returns:
        List of log returns (length = len(prices) - 1)
    """
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i - 1]))
        else:
            returns.append(0.0)
    
    return returns


def compute_ewma_vol(
    returns: Sequence[float],
    halflife_bars: int,
    window_bars: int | None = None,
) -> float:
    """
    Compute EWMA volatility (standard deviation) of returns.
    
    Uses exponential weighting with decay lambda = 0.5 ** (1 / halflife_bars).
    
    Args:
        returns: Sequence of log returns (oldest to newest)
        halflife_bars: Halflife for EWMA decay
        window_bars: Optional max window to consider (from end)
    
    Returns:
        EWMA standard deviation of returns
    """
    if not returns or halflife_bars <= 0:
        return 0.0
    
    # Use only the last window_bars if specified
    if window_bars is not None and len(returns) > window_bars:
        returns = returns[-window_bars:]
    
    if len(returns) < 2:
        return 0.0
    
    # Decay factor: lambda = 0.5^(1/halflife)
    decay = 0.5 ** (1.0 / halflife_bars)
    
    # Compute EWMA variance
    weights = []
    weight = 1.0
    for _ in range(len(returns)):
        weights.append(weight)
        weight *= decay
    
    # Reverse weights so most recent has highest weight
    weights = weights[::-1]
    
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    
    # Weighted mean
    mean = sum(r * w for r, w in zip(returns, weights)) / total_weight
    
    # Weighted variance
    variance = sum(w * (r - mean) ** 2 for r, w in zip(returns, weights)) / total_weight
    
    return math.sqrt(variance)


def classify_vol_regime(
    vol_short: float,
    vol_long: float,
    config: VolRegimeConfig | None = None,
) -> VolRegime:
    """
    Classify volatility into a regime based on short/long vol ratio.
    
    Args:
        vol_short: Short-term EWMA volatility
        vol_long: Long-term EWMA volatility
        config: Volatility regime configuration
    
    Returns:
        VolRegime with label, vol values, and ratio
    """
    if config is None:
        config = DEFAULT_VOL_REGIME_CONFIG
    
    # Handle edge cases
    if vol_long <= 0 or vol_short < 0:
        return VolRegime(
            label="normal",
            vol_short=vol_short if vol_short >= 0 else 0.0,
            vol_long=vol_long if vol_long >= 0 else 0.0,
            ratio=1.0,
        )
    
    ratio = vol_short / vol_long
    
    if ratio < config.ratio_low:
        label: VolRegimeLabel = "low"
    elif ratio < config.ratio_normal:
        label = "normal"
    elif ratio < config.ratio_high:
        label = "high"
    else:
        label = "crisis"
    
    return VolRegime(
        label=label,
        vol_short=vol_short,
        vol_long=vol_long,
        ratio=ratio,
    )


def compute_vol_regime_from_prices(
    prices: Sequence[float],
    config: VolRegimeConfig | None = None,
) -> VolRegime:
    """
    Compute volatility regime from a price series.
    
    Convenience function that computes returns and classifies regime.
    
    Args:
        prices: Sequence of prices (oldest to newest)
        config: Volatility regime configuration
    
    Returns:
        VolRegime with classification
    """
    if config is None:
        config = DEFAULT_VOL_REGIME_CONFIG
    
    returns = compute_log_returns(prices)
    
    if len(returns) < config.short_window_bars:
        # Insufficient data - return normal with zeros
        return VolRegime(
            label="normal",
            vol_short=0.0,
            vol_long=0.0,
            ratio=1.0,
        )
    
    vol_short = compute_ewma_vol(
        returns,
        halflife_bars=config.short_halflife_bars,
        window_bars=config.short_window_bars,
    )
    
    vol_long = compute_ewma_vol(
        returns,
        halflife_bars=config.long_halflife_bars,
        window_bars=config.long_window_bars,
    )
    
    return classify_vol_regime(vol_short, vol_long, config)


def get_sizing_multiplier(
    tier: str,
    regime_label: VolRegimeLabel,
    strategy_config: Mapping[str, Any] | None = None,
) -> float:
    """
    Get the sizing multiplier for a tier and volatility regime.
    
    Args:
        tier: Strategy tier (CORE, SATELLITE, TACTICAL, ALT-EXT)
        regime_label: Volatility regime label
        strategy_config: Full strategy config
    
    Returns:
        Sizing multiplier (default 1.0)
    """
    if strategy_config is None:
        return 1.0
    
    vol_cfg = strategy_config.get("vol_regimes", {})
    if not vol_cfg.get("enabled", True):
        return 1.0
    
    sizing_mults = vol_cfg.get("sizing_multipliers", {})
    
    # Try exact tier, then fallback to CORE, then default
    tier_mults = sizing_mults.get(tier.upper())
    if tier_mults is None:
        tier_mults = sizing_mults.get("CORE", {})
    
    return float(tier_mults.get(regime_label, 1.0))


def get_hybrid_weight_modifiers(
    regime_label: VolRegimeLabel,
    strategy_config: Mapping[str, Any] | None = None,
) -> HybridWeightModifiers:
    """
    Get hybrid score weight modifiers for a volatility regime.
    
    Args:
        regime_label: Volatility regime label
        strategy_config: Full strategy config
    
    Returns:
        HybridWeightModifiers with multipliers for carry/expectancy/router
    """
    if strategy_config is None:
        return HybridWeightModifiers()
    
    vol_cfg = strategy_config.get("vol_regimes", {})
    if not vol_cfg.get("enabled", True):
        return HybridWeightModifiers()
    
    hybrid_mods = vol_cfg.get("hybrid_weight_modifiers", {})
    default_mods = hybrid_mods.get("default", {})
    regime_mods = default_mods.get(regime_label, {})
    
    return HybridWeightModifiers(
        carry=float(regime_mods.get("carry", 1.0)),
        expectancy=float(regime_mods.get("expectancy", 1.0)),
        router=float(regime_mods.get("router", 1.0)),
    )


def build_vol_regime_snapshot(
    symbol: str,
    prices: Sequence[float] | None = None,
    config: VolRegimeConfig | None = None,
) -> dict:
    """
    Build a volatility regime snapshot for a symbol.
    
    Args:
        symbol: Trading symbol
        prices: Optional price series (if None, returns default)
        config: Volatility regime configuration
    
    Returns:
        Dict with regime info for state publishing
    """
    if prices is None or len(prices) < 2:
        return {
            "symbol": symbol.upper(),
            "vol_regime": "normal",
            "vol": {
                "short": 0.0,
                "long": 0.0,
                "ratio": 1.0,
            },
            "updated_ts": time.time(),
        }
    
    regime = compute_vol_regime_from_prices(prices, config)
    
    return {
        "symbol": symbol.upper(),
        "vol_regime": regime.label,
        "vol": {
            "short": round(regime.vol_short, 6),
            "long": round(regime.vol_long, 6),
            "ratio": round(regime.ratio, 4),
        },
        "updated_ts": time.time(),
    }


def build_vol_regime_summary(regimes: Sequence[VolRegime | str]) -> dict:
    """
    Build a summary of regime distribution across symbols.
    
    Args:
        regimes: Sequence of VolRegime objects or label strings
    
    Returns:
        Dict with counts per regime: {"low": 3, "normal": 7, ...}
    """
    summary = {"low": 0, "normal": 0, "high": 0, "crisis": 0}
    
    for r in regimes:
        label = r.label if isinstance(r, VolRegime) else str(r)
        if label in summary:
            summary[label] += 1
    
    return summary


def rolling_sigma(symbol: str, lookback: int = 50) -> float:
    """
    Realized std of per-trade or per-bar returns for `symbol` over `lookback`.
    Delegates to pnl_tracker; normalize as needed downstream.
    """
    if pnl_tracker is None:
        return 0.0
    stats: Mapping[str, Any] | None = None
    try:
        stats = pnl_tracker.get_symbol_stats(symbol, window_trades=lookback)
    except TypeError:
        try:
            stats = pnl_tracker.get_symbol_stats(symbol, lookback)  # type: ignore[misc]
        except Exception:
            stats = None
    except Exception:
        stats = None
    if not isinstance(stats, Mapping):
        return 0.0
    return float(stats.get("std", 0.0) or 0.0)


def atr_pct(symbol: str, lookback_bars: int = 50, *, median_only: bool = False) -> float:
    """
    Percent ATR over `lookback_bars`. Falls back to medians when requested.
    """
    kwargs = {"lookback_bars": lookback_bars}
    if median_only:
        kwargs["median_only"] = True
    payload: Mapping[str, Any] | None = None
    try:
        payload = pnl_tracker.get_symbol_atr(symbol, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Legacy signatures may not accept kwargs.
        try:
            payload = pnl_tracker.get_symbol_atr(symbol, lookback_bars)  # type: ignore[misc]
        except Exception:
            payload = None
    except Exception:
        payload = None
    if not isinstance(payload, Mapping):
        return 0.0
    key = "atr_pct_median" if median_only else "atr_pct"
    value = payload.get(key)
    if value is None and median_only:
        value = payload.get("atr_pct")
    return _as_float(value)


# ─────────────────────────────────────────────────────────────────────────────
# ATR Regime Classification (v7)
# ─────────────────────────────────────────────────────────────────────────────

# ATR regime bucket boundaries (percentile-based)
# 0 = Low       (ATR <= p20)
# 1 = Normal    (p20 < ATR <= p50)
# 2 = Elevated  (p50 < ATR <= p80)
# 3 = Extreme   (ATR > p80)

ATR_REGIME_LOW = 0
ATR_REGIME_NORMAL = 1
ATR_REGIME_ELEVATED = 2
ATR_REGIME_EXTREME = 3

ATR_REGIME_NAMES = {
    ATR_REGIME_LOW: "low",
    ATR_REGIME_NORMAL: "normal",
    ATR_REGIME_ELEVATED: "elevated",
    ATR_REGIME_EXTREME: "extreme",
}

# Default percentile thresholds (can be overridden)
DEFAULT_ATR_PERCENTILES = {
    "p20": 0.15,  # 0.15% ATR
    "p50": 0.25,  # 0.25% ATR
    "p80": 0.40,  # 0.40% ATR
}


def classify_atr_regime(
    atr_value: float,
    percentiles: Mapping[str, float] | None = None,
) -> int:
    """
    Classify an ATR value into a regime bucket.

    ATR Regime Buckets:
        0 = Low       (ATR <= p20)
        1 = Normal    (p20 < ATR <= p50)
        2 = Elevated  (p50 < ATR <= p80)
        3 = Extreme   (ATR > p80)

    Args:
        atr_value: The ATR percentage value to classify
        percentiles: Optional dict with p20, p50, p80 thresholds.
                     Defaults to DEFAULT_ATR_PERCENTILES.

    Returns:
        Integer regime bucket (0-3)
    """
    if percentiles is None:
        percentiles = DEFAULT_ATR_PERCENTILES

    p20 = _as_float(percentiles.get("p20", DEFAULT_ATR_PERCENTILES["p20"]))
    p50 = _as_float(percentiles.get("p50", DEFAULT_ATR_PERCENTILES["p50"]))
    p80 = _as_float(percentiles.get("p80", DEFAULT_ATR_PERCENTILES["p80"]))

    atr = _as_float(atr_value)

    if atr <= p20:
        return ATR_REGIME_LOW
    elif atr <= p50:
        return ATR_REGIME_NORMAL
    elif atr <= p80:
        return ATR_REGIME_ELEVATED
    else:
        return ATR_REGIME_EXTREME


def compute_atr_regime(
    symbol: str,
    atr_value: float | None = None,
    percentiles: Mapping[str, float] | None = None,
) -> dict:
    """
    Compute ATR regime for a symbol.

    Args:
        symbol: The trading symbol
        atr_value: Optional ATR value. If None, fetches from atr_pct().
        percentiles: Optional percentile thresholds for regime boundaries.

    Returns:
        Dict with regime info:
        {
            "symbol": str,
            "atr_value": float,
            "atr_regime": int,
            "atr_regime_name": str,
            "percentiles": dict,
        }
    """
    if atr_value is None:
        atr_value = atr_pct(symbol)

    atr_val = _as_float(atr_value)
    pcts = dict(percentiles) if percentiles else dict(DEFAULT_ATR_PERCENTILES)
    regime = classify_atr_regime(atr_val, pcts)

    return {
        "symbol": symbol,
        "atr_value": atr_val,
        "atr_regime": regime,
        "atr_regime_name": ATR_REGIME_NAMES.get(regime, "unknown"),
        "percentiles": pcts,
    }


def get_atr_regime_name(regime: int) -> str:
    """Get the name for an ATR regime bucket."""
    return ATR_REGIME_NAMES.get(regime, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown Regime Classification (v7)
# ─────────────────────────────────────────────────────────────────────────────

# Drawdown regime buckets
# 0 = dd_frac < 0.05
# 1 = 0.05 <= dd_frac < 0.15
# 2 = 0.15 <= dd_frac < 0.30
# 3 = dd_frac >= 0.30

DD_REGIME_LOW = 0
DD_REGIME_MODERATE = 1
DD_REGIME_HIGH = 2
DD_REGIME_CRITICAL = 3

DD_REGIME_NAMES = {
    DD_REGIME_LOW: "low",
    DD_REGIME_MODERATE: "moderate",
    DD_REGIME_HIGH: "high",
    DD_REGIME_CRITICAL: "critical",
}

# Drawdown regime thresholds (fractional)
DD_THRESHOLD_MODERATE = 0.05
DD_THRESHOLD_HIGH = 0.15
DD_THRESHOLD_CRITICAL = 0.30


def classify_dd_regime(dd_frac: float) -> int:
    """
    Classify a drawdown fraction into a regime bucket.

    Drawdown Regime Buckets:
        0 = Low       (dd_frac < 0.05)
        1 = Moderate  (0.05 <= dd_frac < 0.15)
        2 = High      (0.15 <= dd_frac < 0.30)
        3 = Critical  (dd_frac >= 0.30)

    Args:
        dd_frac: Drawdown as a fraction (0-1)

    Returns:
        Integer regime bucket (0-3)
    """
    dd = _as_float(dd_frac)

    if dd < DD_THRESHOLD_MODERATE:
        return DD_REGIME_LOW
    elif dd < DD_THRESHOLD_HIGH:
        return DD_REGIME_MODERATE
    elif dd < DD_THRESHOLD_CRITICAL:
        return DD_REGIME_HIGH
    else:
        return DD_REGIME_CRITICAL


def get_dd_regime_name(regime: int) -> str:
    """Get the name for a drawdown regime bucket."""
    return DD_REGIME_NAMES.get(regime, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Regime Matrix (v7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_matrix(
    atr_regime: int,
    dd_regime: int,
) -> list:
    """
    Compute a 4x4 regime matrix showing current position.

    The matrix is [dd_regime][atr_regime] indexed:
    - Rows: dd_regime (0=low to 3=critical)
    - Cols: atr_regime (0=low to 3=extreme)

    Each cell contains 1 if it's the current regime, 0 otherwise.

    Args:
        atr_regime: Current ATR regime (0-3)
        dd_regime: Current DD regime (0-3)

    Returns:
        4x4 list of lists with 0s and a single 1 at [dd_regime][atr_regime]
    """
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    
    # Clamp to valid range
    dd_idx = max(0, min(3, int(dd_regime)))
    atr_idx = max(0, min(3, int(atr_regime)))
    
    matrix[dd_idx][atr_idx] = 1
    return matrix


def build_regime_snapshot(
    atr_value: float | None = None,
    dd_frac: float | None = None,
    atr_percentiles: Mapping[str, float] | None = None,
) -> dict:
    """
    Build a complete regime snapshot for telemetry.

    Args:
        atr_value: ATR percentage value
        dd_frac: Drawdown fraction (0-1)
        atr_percentiles: Optional ATR percentile thresholds

    Returns:
        Dict suitable for regimes.json:
        {
            "atr_regime": int,
            "atr_regime_name": str,
            "atr_value": float,
            "dd_regime": int,
            "dd_regime_name": str,
            "dd_frac": float,
            "regime_matrix": [[...], [...]],
            "updated_ts": float,
        }
    """
    import time

    atr_val = _as_float(atr_value) if atr_value is not None else 0.0
    dd_val = _as_float(dd_frac) if dd_frac is not None else 0.0

    atr_regime = classify_atr_regime(atr_val, atr_percentiles)
    dd_regime = classify_dd_regime(dd_val)
    matrix = compute_regime_matrix(atr_regime, dd_regime)

    return {
        "atr_regime": atr_regime,
        "atr_regime_name": get_atr_regime_name(atr_regime),
        "atr_value": atr_val,
        "atr_percentiles": dict(atr_percentiles) if atr_percentiles else dict(DEFAULT_ATR_PERCENTILES),
        "dd_regime": dd_regime,
        "dd_regime_name": get_dd_regime_name(dd_regime),
        "dd_frac": dd_val,
        "regime_matrix": matrix,
        "updated_ts": time.time(),
    }

