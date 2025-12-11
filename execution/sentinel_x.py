"""
Sentinel-X: Hybrid ML Market Regime Classifier — v7.8_P6

A hybrid regime engine that:
1. Extracts rich market features from price/volume/volatility data
2. Uses ML-style scoring to produce regime probabilities
3. Applies rule-based thresholds and crisis overrides
4. Produces hard regime labels with stickiness for stability

Canonical Regimes:
    TREND_UP    — Strong positive directional bias
    TREND_DOWN  — Strong negative directional bias
    MEAN_REVERT — Mean-reverting / range-bound conditions
    BREAKOUT    — Volatility expansion with directional intent
    CHOPPY      — High noise, low signal, uncertain direction
    CRISIS      — Extreme stress (forced by hard rules)

Architecture:
    Price/Volume Data → Feature Extraction → ML-Style Scoring → Rule-Based Labels
                                                                      ↓
                                            logs/state/sentinel_x.json

Single writer rule: Only executor/intel pipeline may write sentinel_x.json.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/sentinel_x.json")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")

_LOG = logging.getLogger(__name__)

# Canonical regime labels
REGIMES = [
    "TREND_UP",
    "TREND_DOWN",
    "MEAN_REVERT",
    "BREAKOUT",
    "CHOPPY",
    "CRISIS",
]

# Safety constants
MIN_BARS_FOR_FEATURES = 20
SAFE_STD_FLOOR = 1e-10
MAX_FEATURE_VALUE = 1e6


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SentinelXConfig:
    """Configuration for Sentinel-X regime classifier."""

    enabled: bool = False
    lookback_bars: int = 240
    feature_agg_bars: int = 48
    min_history_days: int = 30
    regimes: List[str] = field(default_factory=lambda: REGIMES.copy())

    # Probability thresholds for regime assignment
    prob_threshold_primary: float = 0.55
    prob_threshold_secondary: float = 0.25

    # Crisis hard rules (override ML predictions)
    crisis_dd_threshold: float = 0.12  # 12% drawdown triggers crisis
    crisis_vol_spike_multiple: float = 3.0  # 3x normal vol triggers crisis

    # Model parameters (conceptual - used in scoring)
    model_type: str = "gradient_boosting"
    model_max_depth: int = 4
    model_n_estimators: int = 100

    # Smoothing and stability
    prob_alpha: float = 0.3  # EMA alpha for probability smoothing
    label_stickiness: int = 3  # Require N consecutive hints before flipping

    # Run interval
    run_interval_cycles: int = 5

    def __post_init__(self) -> None:
        """Validate and normalize config."""
        if self.lookback_bars < MIN_BARS_FOR_FEATURES:
            self.lookback_bars = MIN_BARS_FOR_FEATURES
        if self.feature_agg_bars < 5:
            self.feature_agg_bars = 48
        if self.prob_threshold_primary < 0.3 or self.prob_threshold_primary > 0.9:
            self.prob_threshold_primary = 0.55
        if self.prob_threshold_secondary < 0.1 or self.prob_threshold_secondary > 0.5:
            self.prob_threshold_secondary = 0.25
        if self.prob_alpha < 0.05 or self.prob_alpha > 1.0:
            self.prob_alpha = 0.3
        if self.label_stickiness < 1:
            self.label_stickiness = 3
        # Normalize regimes to uppercase
        self.regimes = [r.upper() for r in self.regimes]


@dataclass
class RegimeFeatures:
    """
    Extracted market features for regime classification.
    
    All features are computed over the feature_agg_bars window.
    """

    # Return distribution features
    returns_mean: float = 0.0  # Mean return (directional bias)
    returns_std: float = 0.0  # Return volatility
    returns_skew: float = 0.0  # Return skewness (asymmetry)
    returns_kurtosis: float = 0.0  # Return kurtosis (tail heaviness)

    # Volatility features
    atr_norm: float = 0.0  # ATR / price (normalized volatility)
    vol_regime_z: float = 0.0  # Z-score of current vol vs historical

    # Trend features
    trend_slope: float = 0.0  # Linear regression slope on log price
    trend_r2: float = 0.0  # R² of trend fit (trend strength)
    trend_acceleration: float = 0.0  # 2nd derivative of trend

    # Breakout/range features
    breakout_score: float = 0.0  # Distance from rolling max/min
    range_position: float = 0.5  # Position in recent range [0, 1]
    
    # Mean reversion features
    mean_reversion_score: float = 0.0  # Measure of reversal frequency
    hl_spread: float = 0.0  # High-low spread normalized

    # Volume features
    volume_z: float = 0.0  # Z-score of volume vs rolling mean
    volume_trend: float = 0.0  # Volume trend (rising/falling)

    # Correlation features
    realized_corr: float = 0.0  # Correlation vs reference (e.g., BTC)

    # Microstructure (if available)
    spread_vol: float = 0.0  # Volatility of bid-ask spread proxy

    # Data quality
    data_quality: float = 1.0  # Completeness factor [0, 1]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "returns_mean": round(self.returns_mean, 6),
            "returns_std": round(self.returns_std, 6),
            "returns_skew": round(self.returns_skew, 4),
            "returns_kurtosis": round(self.returns_kurtosis, 4),
            "atr_norm": round(self.atr_norm, 6),
            "vol_regime_z": round(self.vol_regime_z, 4),
            "trend_slope": round(self.trend_slope, 6),
            "trend_r2": round(self.trend_r2, 4),
            "trend_acceleration": round(self.trend_acceleration, 6),
            "breakout_score": round(self.breakout_score, 4),
            "range_position": round(self.range_position, 4),
            "mean_reversion_score": round(self.mean_reversion_score, 4),
            "hl_spread": round(self.hl_spread, 6),
            "volume_z": round(self.volume_z, 4),
            "volume_trend": round(self.volume_trend, 4),
            "realized_corr": round(self.realized_corr, 4),
            "spread_vol": round(self.spread_vol, 6),
            "data_quality": round(self.data_quality, 2),
        }


@dataclass
class RegimeProbabilities:
    """Probability distribution over regimes."""

    probs: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure all regimes have a probability."""
        for regime in REGIMES:
            if regime not in self.probs:
                self.probs[regime] = 0.0

    def normalize(self) -> None:
        """Normalize probabilities to sum to 1."""
        total = sum(self.probs.values())
        if total > 0:
            for k in self.probs:
                self.probs[k] /= total
        else:
            # Uniform distribution if no signal
            n = len(self.probs)
            for k in self.probs:
                self.probs[k] = 1.0 / n

    def get_primary(self) -> Tuple[str, float]:
        """Get primary (highest probability) regime."""
        if not self.probs:
            return ("CHOPPY", 0.0)
        primary = max(self.probs.items(), key=lambda x: x[1])
        return primary

    def get_secondary(self) -> Optional[Tuple[str, float]]:
        """Get secondary (second highest) regime."""
        if len(self.probs) < 2:
            return None
        sorted_probs = sorted(self.probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[1] if len(sorted_probs) > 1 else None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {k: round(v, 4) for k, v in self.probs.items()}


@dataclass
class HistoryMeta:
    """Tracking metadata for label stability."""

    last_n_labels: List[str] = field(default_factory=list)
    consecutive_count: int = 0
    pending_regime: Optional[str] = None
    last_primary: str = "CHOPPY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "last_n_labels": self.last_n_labels[-10:],  # Keep last 10
            "consecutive_count": self.consecutive_count,
            "pending_regime": self.pending_regime,
            "last_primary": self.last_primary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HistoryMeta":
        """Create from dictionary."""
        return cls(
            last_n_labels=d.get("last_n_labels", []),
            consecutive_count=d.get("consecutive_count", 0),
            pending_regime=d.get("pending_regime"),
            last_primary=d.get("last_primary", "CHOPPY"),
        )


@dataclass
class SentinelXState:
    """
    State persisted to logs/state/sentinel_x.json.
    """

    updated_ts: str = ""
    cycle_count: int = 0

    # Primary regime classification
    primary_regime: str = "CHOPPY"
    secondary_regime: Optional[str] = None

    # Regime probabilities
    regime_probs: Dict[str, float] = field(default_factory=dict)
    smoothed_probs: Dict[str, float] = field(default_factory=dict)

    # Feature snapshot
    features: Dict[str, float] = field(default_factory=dict)

    # Crisis override
    crisis_flag: bool = False
    crisis_reason: str = ""

    # History tracking
    history_meta: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "cycle_count": self.cycle_count,
            "primary_regime": self.primary_regime,
            "secondary_regime": self.secondary_regime,
            "regime_probs": {k: round(v, 4) for k, v in self.regime_probs.items()},
            "smoothed_probs": {k: round(v, 4) for k, v in self.smoothed_probs.items()},
            "features": self.features,
            "crisis_flag": self.crisis_flag,
            "crisis_reason": self.crisis_reason,
            "history_meta": self.history_meta,
            "meta": self.meta,
            "notes": self.notes,
            "errors": self.errors[-10:],
        }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_sentinel_x_config(
    strategy_cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> SentinelXConfig:
    """
    Load Sentinel-X config from strategy_config.json.

    Args:
        strategy_cfg: Optional pre-loaded strategy config dict
        config_path: Optional path override

    Returns:
        SentinelXConfig instance
    """
    if strategy_cfg is None:
        path = config_path or DEFAULT_CONFIG_PATH
        try:
            with open(path, "r") as f:
                strategy_cfg = json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as exc:
            _LOG.warning("sentinel_x_config_load_failed: %s", exc)
            return SentinelXConfig(enabled=False)

    section = strategy_cfg.get("sentinel_x", {})
    if not section:
        return SentinelXConfig(enabled=False)

    # Parse nested structures
    prob_thresholds = section.get("prob_thresholds", {})
    crisis_rules = section.get("crisis_hard_rules", {})
    model_params = section.get("model", {})
    smoothing = section.get("smoothing", {})

    return SentinelXConfig(
        enabled=bool(section.get("enabled", False)),
        lookback_bars=int(section.get("lookback_bars", 240)),
        feature_agg_bars=int(section.get("feature_agg_bars", 48)),
        min_history_days=int(section.get("min_history_days", 30)),
        regimes=section.get("regimes", REGIMES.copy()),
        prob_threshold_primary=float(prob_thresholds.get("primary", 0.55)),
        prob_threshold_secondary=float(prob_thresholds.get("secondary", 0.25)),
        crisis_dd_threshold=float(crisis_rules.get("dd_threshold", 0.12)),
        crisis_vol_spike_multiple=float(crisis_rules.get("vol_spike_multiple", 3.0)),
        model_type=str(model_params.get("type", "gradient_boosting")),
        model_max_depth=int(model_params.get("max_depth", 4)),
        model_n_estimators=int(model_params.get("n_estimators", 100)),
        prob_alpha=float(smoothing.get("prob_alpha", 0.3)),
        label_stickiness=int(smoothing.get("label_stickiness", 3)),
        run_interval_cycles=int(section.get("run_interval_cycles", 5)),
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_sentinel_x_state(
    state_path: Optional[Path] = None,
) -> SentinelXState:
    """
    Load Sentinel-X state from file.

    Returns empty state if file doesn't exist.
    """
    path = state_path or DEFAULT_STATE_PATH
    if not path.exists():
        return SentinelXState()

    try:
        data = json.loads(path.read_text())
        return SentinelXState(
            updated_ts=data.get("updated_ts", ""),
            cycle_count=data.get("cycle_count", 0),
            primary_regime=data.get("primary_regime", "CHOPPY"),
            secondary_regime=data.get("secondary_regime"),
            regime_probs=data.get("regime_probs", {}),
            smoothed_probs=data.get("smoothed_probs", {}),
            features=data.get("features", {}),
            crisis_flag=data.get("crisis_flag", False),
            crisis_reason=data.get("crisis_reason", ""),
            history_meta=data.get("history_meta", {}),
            meta=data.get("meta", {}),
            notes=data.get("notes", ""),
            errors=data.get("errors", []),
        )
    except (json.JSONDecodeError, IOError, OSError) as exc:
        _LOG.warning("sentinel_x_state_load_failed: %s", exc)
        return SentinelXState()


def save_sentinel_x_state(
    state: SentinelXState,
    state_path: Optional[Path] = None,
) -> bool:
    """
    Save Sentinel-X state to file.

    Returns True if successful.
    """
    path = state_path or DEFAULT_STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2))
        return True
    except (IOError, OSError) as exc:
        _LOG.error("sentinel_x_state_save_failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def compute_returns(prices: List[float]) -> List[float]:
    """Compute log returns from price series."""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i - 1]))
        else:
            returns.append(0.0)
    return returns


def compute_mean(series: List[float]) -> float:
    """Compute mean of series."""
    if not series:
        return 0.0
    return sum(series) / len(series)


def compute_std(series: List[float], mean: Optional[float] = None) -> float:
    """Compute standard deviation of series."""
    if len(series) < 2:
        return SAFE_STD_FLOOR
    if mean is None:
        mean = compute_mean(series)
    variance = sum((x - mean) ** 2 for x in series) / len(series)
    return max(math.sqrt(variance), SAFE_STD_FLOOR)


def compute_skewness(series: List[float], mean: Optional[float] = None, std: Optional[float] = None) -> float:
    """Compute skewness of series."""
    if len(series) < 3:
        return 0.0
    if mean is None:
        mean = compute_mean(series)
    if std is None:
        std = compute_std(series, mean)
    if std < SAFE_STD_FLOOR:
        return 0.0
    n = len(series)
    m3 = sum((x - mean) ** 3 for x in series) / n
    return m3 / (std ** 3)


def compute_kurtosis(series: List[float], mean: Optional[float] = None, std: Optional[float] = None) -> float:
    """Compute excess kurtosis of series."""
    if len(series) < 4:
        return 0.0
    if mean is None:
        mean = compute_mean(series)
    if std is None:
        std = compute_std(series, mean)
    if std < SAFE_STD_FLOOR:
        return 0.0
    n = len(series)
    m4 = sum((x - mean) ** 4 for x in series) / n
    return (m4 / (std ** 4)) - 3.0  # Excess kurtosis


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Compute Average True Range."""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return 0.0
    
    tr_values = []
    for i in range(1, min(len(highs), len(lows), len(closes))):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return compute_mean(tr_values) if tr_values else 0.0
    
    return compute_mean(tr_values[-period:])


def compute_linear_regression(y: List[float]) -> Tuple[float, float, float]:
    """
    Compute linear regression on series.
    
    Returns (slope, intercept, r_squared).
    """
    n = len(y)
    if n < 3:
        return (0.0, 0.0, 0.0)
    
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    # Compute slope and intercept
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if abs(denominator) < SAFE_STD_FLOOR:
        return (0.0, y_mean, 0.0)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Compute R²
    y_pred = [slope * x[i] + intercept for i in range(n)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > SAFE_STD_FLOOR else 0.0
    r_squared = max(0.0, min(1.0, r_squared))
    
    return (slope, intercept, r_squared)


def compute_z_score(value: float, series: List[float]) -> float:
    """Compute z-score of value relative to series."""
    if len(series) < 2:
        return 0.0
    mean = compute_mean(series)
    std = compute_std(series, mean)
    if std < SAFE_STD_FLOOR:
        return 0.0
    return (value - mean) / std


def compute_range_position(price: float, rolling_high: float, rolling_low: float) -> float:
    """Compute position within range [0, 1]."""
    range_size = rolling_high - rolling_low
    if range_size < SAFE_STD_FLOOR:
        return 0.5
    return (price - rolling_low) / range_size


def compute_reversal_frequency(returns: List[float]) -> float:
    """
    Compute reversal frequency (sign changes in returns).
    
    Higher value = more mean-reverting.
    """
    if len(returns) < 3:
        return 0.0
    
    sign_changes = 0
    for i in range(1, len(returns)):
        if returns[i] * returns[i - 1] < 0:
            sign_changes += 1
    
    return sign_changes / (len(returns) - 1)


def extract_regime_features(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
    reference_prices: Optional[List[float]] = None,
    cfg: Optional[SentinelXConfig] = None,
) -> RegimeFeatures:
    """
    Extract regime classification features from market data.
    
    Args:
        prices: Close price series (newest last)
        volumes: Volume series (optional)
        highs: High price series (optional)
        lows: Low price series (optional)
        reference_prices: Reference asset prices for correlation (optional)
        cfg: Configuration
        
    Returns:
        RegimeFeatures with all computed metrics
    """
    cfg = cfg or SentinelXConfig()
    agg_bars = cfg.feature_agg_bars
    
    # Ensure minimum data
    if len(prices) < MIN_BARS_FOR_FEATURES:
        return RegimeFeatures(data_quality=0.0)
    
    # Use recent window
    recent_prices = prices[-agg_bars:] if len(prices) >= agg_bars else prices
    recent_n = len(recent_prices)
    
    # Compute returns
    returns = compute_returns(recent_prices)
    if not returns:
        return RegimeFeatures(data_quality=0.0)
    
    # --- Return Distribution ---
    returns_mean = compute_mean(returns)
    returns_std = compute_std(returns, returns_mean)
    returns_skew = compute_skewness(returns, returns_mean, returns_std)
    returns_kurtosis = compute_kurtosis(returns, returns_mean, returns_std)
    
    # --- Volatility ---
    # ATR normalized by price
    if highs and lows and len(highs) >= agg_bars and len(lows) >= agg_bars:
        atr = compute_atr(highs[-agg_bars:], lows[-agg_bars:], recent_prices)
        atr_norm = atr / recent_prices[-1] if recent_prices[-1] > 0 else 0.0
    else:
        # Fallback: use return std as proxy
        atr_norm = returns_std
    
    # Vol regime z-score (current vol vs longer history)
    if len(prices) >= cfg.lookback_bars:
        long_returns = compute_returns(prices[-cfg.lookback_bars:])
        long_std = compute_std(long_returns)
        vol_regime_z = (returns_std - long_std) / max(long_std, SAFE_STD_FLOOR)
    else:
        vol_regime_z = 0.0
    
    # --- Trend ---
    log_prices = [math.log(p) if p > 0 else 0 for p in recent_prices]
    trend_slope, _, trend_r2 = compute_linear_regression(log_prices)
    
    # Trend acceleration (change in slope)
    if len(prices) >= agg_bars * 2:
        earlier_prices = prices[-(agg_bars * 2):-agg_bars]
        earlier_log = [math.log(p) if p > 0 else 0 for p in earlier_prices]
        earlier_slope, _, _ = compute_linear_regression(earlier_log)
        trend_acceleration = trend_slope - earlier_slope
    else:
        trend_acceleration = 0.0
    
    # --- Breakout / Range ---
    if len(prices) >= agg_bars:
        lookback_prices = prices[-agg_bars:]
        rolling_high = max(lookback_prices[:-1]) if len(lookback_prices) > 1 else lookback_prices[-1]
        rolling_low = min(lookback_prices[:-1]) if len(lookback_prices) > 1 else lookback_prices[-1]
        current_price = lookback_prices[-1]
        
        # Breakout score: how far above/below recent range
        if current_price > rolling_high:
            breakout_score = (current_price - rolling_high) / max(rolling_high, SAFE_STD_FLOOR)
        elif current_price < rolling_low:
            breakout_score = (rolling_low - current_price) / max(rolling_low, SAFE_STD_FLOOR)
        else:
            breakout_score = 0.0
        
        range_position = compute_range_position(current_price, rolling_high, rolling_low)
    else:
        breakout_score = 0.0
        range_position = 0.5
    
    # --- Mean Reversion ---
    mean_reversion_score = compute_reversal_frequency(returns)
    
    # High-low spread normalized
    if highs and lows and len(highs) >= agg_bars and len(lows) >= agg_bars:
        hl_spreads = [
            (highs[i] - lows[i]) / max(prices[i], SAFE_STD_FLOOR) 
            for i in range(-agg_bars, 0)
        ]
        hl_spread = compute_mean(hl_spreads)
    else:
        hl_spread = atr_norm
    
    # --- Volume ---
    if volumes and len(volumes) >= agg_bars:
        recent_volumes = volumes[-agg_bars:]
        current_vol = recent_volumes[-1]
        volume_z = compute_z_score(current_vol, recent_volumes)
        
        # Volume trend
        vol_slope, _, _ = compute_linear_regression(recent_volumes)
        volume_trend = vol_slope / max(compute_mean(recent_volumes), SAFE_STD_FLOOR)
    else:
        volume_z = 0.0
        volume_trend = 0.0
    
    # --- Correlation ---
    if reference_prices and len(reference_prices) >= agg_bars:
        ref_returns = compute_returns(reference_prices[-agg_bars:])
        if len(ref_returns) == len(returns):
            # Pearson correlation
            r_mean = compute_mean(ref_returns)
            r_std = compute_std(ref_returns, r_mean)
            if returns_std > SAFE_STD_FLOOR and r_std > SAFE_STD_FLOOR:
                cov = sum((returns[i] - returns_mean) * (ref_returns[i] - r_mean) 
                         for i in range(len(returns))) / len(returns)
                realized_corr = cov / (returns_std * r_std)
            else:
                realized_corr = 0.0
        else:
            realized_corr = 0.0
    else:
        realized_corr = 0.0
    
    # --- Microstructure (spread volatility proxy) ---
    # Use high-low as proxy for spread volatility if no actual spread data
    spread_vol = compute_std([hl_spread]) if hl_spread > 0 else 0.0
    
    # --- Data Quality ---
    data_quality = min(1.0, recent_n / cfg.feature_agg_bars)
    
    return RegimeFeatures(
        returns_mean=returns_mean,
        returns_std=returns_std,
        returns_skew=returns_skew,
        returns_kurtosis=returns_kurtosis,
        atr_norm=atr_norm,
        vol_regime_z=vol_regime_z,
        trend_slope=trend_slope,
        trend_r2=trend_r2,
        trend_acceleration=trend_acceleration,
        breakout_score=breakout_score,
        range_position=range_position,
        mean_reversion_score=mean_reversion_score,
        hl_spread=hl_spread,
        volume_z=volume_z,
        volume_trend=volume_trend,
        realized_corr=realized_corr,
        spread_vol=spread_vol,
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# ML-Style Regime Scoring (Hybrid Model)
# ---------------------------------------------------------------------------


class SimpleRegimeModel:
    """
    Hybrid ML-style regime classifier.
    
    This implements ML-inspired scoring using handcrafted feature-to-regime
    mappings. The "model" is deterministic and pure-Python, making it
    unit-testable without external dependencies.
    
    The scoring mimics what a gradient boosting classifier would learn:
    - Strong positive trend + high R² → TREND_UP
    - Strong negative trend + high R² → TREND_DOWN
    - High reversal frequency + neutral slope → MEAN_REVERT
    - High vol spike + breakout → BREAKOUT
    - High noise + low R² + mixed signals → CHOPPY
    - Extreme conditions → CRISIS (handled by rules)
    """

    def __init__(self, cfg: SentinelXConfig):
        """Initialize model with config."""
        self.cfg = cfg
        
        # Scoring parameters (conceptual "learned" weights)
        # In a real ML model, these would be fitted from data
        self.trend_slope_threshold = 0.001  # Slope threshold for trend
        self.trend_r2_threshold = 0.3  # R² threshold for trend confidence
        self.reversal_threshold = 0.45  # Reversal frequency for mean-revert
        self.vol_spike_threshold = 1.5  # Z-score for vol spike
        self.breakout_threshold = 0.02  # Breakout score threshold

    def predict_proba(self, features: RegimeFeatures) -> RegimeProbabilities:
        """
        Predict regime probabilities from features.
        
        Uses ML-inspired scoring:
        - Each regime gets a raw score based on feature combinations
        - Scores are normalized to probabilities
        
        Args:
            features: Extracted market features
            
        Returns:
            RegimeProbabilities with normalized probabilities
        """
        if features.data_quality < 0.3:
            # Insufficient data → uniform
            probs = RegimeProbabilities()
            probs.normalize()
            return probs
        
        scores = {regime: 0.0 for regime in REGIMES}
        
        # --- TREND_UP Scoring ---
        # Strong positive slope + high R² + positive returns
        trend_up_score = 0.0
        if features.trend_slope > self.trend_slope_threshold:
            slope_signal = min(1.0, features.trend_slope / (self.trend_slope_threshold * 3))
            r2_boost = features.trend_r2 ** 0.5  # Sqrt to moderate effect
            returns_boost = 1.0 if features.returns_mean > 0 else 0.5
            trend_up_score = slope_signal * r2_boost * returns_boost
            
            # Bonus for acceleration
            if features.trend_acceleration > 0:
                trend_up_score *= 1.1
        
        scores["TREND_UP"] = trend_up_score
        
        # --- TREND_DOWN Scoring ---
        # Strong negative slope + high R² + negative returns
        trend_down_score = 0.0
        if features.trend_slope < -self.trend_slope_threshold:
            slope_signal = min(1.0, abs(features.trend_slope) / (self.trend_slope_threshold * 3))
            r2_boost = features.trend_r2 ** 0.5
            returns_boost = 1.0 if features.returns_mean < 0 else 0.5
            trend_down_score = slope_signal * r2_boost * returns_boost
            
            # Bonus for acceleration (more negative)
            if features.trend_acceleration < 0:
                trend_down_score *= 1.1
        
        scores["TREND_DOWN"] = trend_down_score
        
        # --- MEAN_REVERT Scoring ---
        # High reversal frequency + low trend R² + moderate volatility
        mean_revert_score = 0.0
        if features.mean_reversion_score > self.reversal_threshold:
            reversal_signal = min(1.0, features.mean_reversion_score / 0.6)
            anti_trend = 1.0 - features.trend_r2  # Higher when less trendy
            vol_factor = 1.0 if features.vol_regime_z < 1.0 else 0.7  # Prefer normal vol
            mean_revert_score = reversal_signal * anti_trend * vol_factor
            
            # Bonus for range-bound position
            if 0.3 < features.range_position < 0.7:
                mean_revert_score *= 1.15
        
        scores["MEAN_REVERT"] = mean_revert_score
        
        # --- BREAKOUT Scoring ---
        # High breakout score + vol spike + directional move
        breakout_score_val = 0.0
        if features.breakout_score > self.breakout_threshold:
            breakout_signal = min(1.0, features.breakout_score / (self.breakout_threshold * 2))
            vol_spike = max(0.5, min(1.5, 0.5 + features.vol_regime_z * 0.25))
            volume_confirm = 1.0 if features.volume_z > 0.5 else 0.7
            breakout_score_val = breakout_signal * vol_spike * volume_confirm
            
            # Trend acceleration adds confidence
            if abs(features.trend_acceleration) > 0:
                breakout_score_val *= 1.1
        
        scores["BREAKOUT"] = breakout_score_val
        
        # --- CHOPPY Scoring ---
        # Low R² + high volatility + mixed signals
        choppy_score = 0.0
        low_r2 = max(0, 1.0 - features.trend_r2 * 2)
        mixed_signal = 0.0
        if abs(features.trend_slope) < self.trend_slope_threshold * 0.5:
            mixed_signal = 0.8
        noise_factor = min(1.0, features.returns_kurtosis / 3.0) if features.returns_kurtosis > 0 else 0.3
        choppy_score = low_r2 * 0.5 + mixed_signal * 0.3 + noise_factor * 0.2
        
        # High skewness without clear direction adds to chop
        if abs(features.returns_skew) > 1.0 and abs(features.trend_slope) < self.trend_slope_threshold:
            choppy_score *= 1.15
        
        scores["CHOPPY"] = choppy_score
        
        # --- CRISIS Scoring ---
        # Extreme conditions (will often be overridden by rules anyway)
        crisis_score = 0.0
        # Very high volatility spike
        if features.vol_regime_z > 2.5:
            crisis_score += 0.4
        # Extreme kurtosis (fat tails)
        if features.returns_kurtosis > 5.0:
            crisis_score += 0.3
        # Large negative returns with acceleration
        if features.returns_mean < -0.02 and features.trend_acceleration < -0.001:
            crisis_score += 0.3
        
        scores["CRISIS"] = min(1.0, crisis_score)
        
        # Apply data quality discount
        for regime in scores:
            scores[regime] *= features.data_quality
        
        # Create probabilities and normalize
        probs = RegimeProbabilities(probs=scores)
        probs.normalize()
        
        return probs


# ---------------------------------------------------------------------------
# Rule-Based Classification
# ---------------------------------------------------------------------------


def check_crisis_override(
    features: RegimeFeatures,
    cfg: SentinelXConfig,
    current_dd: float = 0.0,
) -> Tuple[bool, str]:
    """
    Check if crisis conditions should override ML predictions.
    
    Args:
        features: Extracted features
        cfg: Configuration
        current_dd: Current portfolio drawdown (0.12 = 12%)
        
    Returns:
        (crisis_triggered, reason)
    """
    reasons = []
    
    # Drawdown override
    if current_dd >= cfg.crisis_dd_threshold:
        reasons.append(f"DD={current_dd:.1%} >= {cfg.crisis_dd_threshold:.1%}")
    
    # Volatility spike override
    if features.vol_regime_z >= cfg.crisis_vol_spike_multiple:
        reasons.append(f"VolZ={features.vol_regime_z:.2f} >= {cfg.crisis_vol_spike_multiple}")
    
    if reasons:
        return (True, "; ".join(reasons))
    
    return (False, "")


def apply_label_stickiness(
    new_primary: str,
    prob: float,
    history: HistoryMeta,
    cfg: SentinelXConfig,
) -> Tuple[str, HistoryMeta]:
    """
    Apply label stickiness to prevent regime flicker.
    
    Requires N consecutive predictions before changing label.
    
    Args:
        new_primary: Newly predicted primary regime
        prob: Probability of new primary
        history: History tracking metadata
        cfg: Configuration
        
    Returns:
        (final_label, updated_history)
    """
    # If same as current, keep it
    if new_primary == history.last_primary:
        history.consecutive_count = 0
        history.pending_regime = None
        return (new_primary, history)
    
    # If same as pending, increment counter
    if new_primary == history.pending_regime:
        history.consecutive_count += 1
    else:
        # New pending regime
        history.pending_regime = new_primary
        history.consecutive_count = 1
    
    # Check if we've hit stickiness threshold
    if history.consecutive_count >= cfg.label_stickiness:
        # Also require sufficient probability
        if prob >= cfg.prob_threshold_primary:
            history.last_primary = new_primary
            history.consecutive_count = 0
            history.pending_regime = None
            return (new_primary, history)
    
    # Not enough consecutive hints, keep old label
    return (history.last_primary, history)


def smooth_probabilities(
    current_probs: Dict[str, float],
    prev_probs: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """
    Apply EMA smoothing to probabilities.
    
    Args:
        current_probs: Current raw probabilities
        prev_probs: Previous smoothed probabilities
        alpha: EMA alpha (higher = more responsive)
        
    Returns:
        Smoothed probabilities
    """
    smoothed = {}
    for regime in REGIMES:
        current = current_probs.get(regime, 0.0)
        prev = prev_probs.get(regime, 1.0 / len(REGIMES))
        smoothed[regime] = alpha * current + (1 - alpha) * prev
    
    # Renormalize
    total = sum(smoothed.values())
    if total > 0:
        for k in smoothed:
            smoothed[k] /= total
    
    return smoothed


def classify_regime(
    probs: RegimeProbabilities,
    features: RegimeFeatures,
    cfg: SentinelXConfig,
    prev_state: Optional[SentinelXState] = None,
    current_dd: float = 0.0,
) -> SentinelXState:
    """
    Classify regime from probabilities using rule-based thresholds.
    
    Steps:
    1. Check crisis override (hard rules)
    2. If not crisis, use ML probabilities
    3. Apply stickiness to prevent flicker
    4. Smooth probabilities
    
    Args:
        probs: Regime probabilities from model
        features: Extracted features
        cfg: Configuration
        prev_state: Previous state (for smoothing and stickiness)
        current_dd: Current portfolio drawdown
        
    Returns:
        New SentinelXState
    """
    now = datetime.now(timezone.utc).isoformat()
    
    # Load history from previous state
    if prev_state and prev_state.history_meta:
        history = HistoryMeta.from_dict(prev_state.history_meta)
    else:
        history = HistoryMeta()
    
    cycle_count = (prev_state.cycle_count + 1) if prev_state else 1
    
    # Step 1: Crisis override check
    crisis_flag, crisis_reason = check_crisis_override(features, cfg, current_dd)
    
    if crisis_flag:
        # Crisis overrides everything
        primary_regime = "CRISIS"
        secondary_regime = None
        history.last_primary = "CRISIS"
        history.last_n_labels.append("CRISIS")
        
        # Still smooth probabilities for continuity
        prev_probs = prev_state.smoothed_probs if prev_state else {}
        smoothed_probs = smooth_probabilities(probs.probs, prev_probs, cfg.prob_alpha)
        
        return SentinelXState(
            updated_ts=now,
            cycle_count=cycle_count,
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            regime_probs=probs.to_dict(),
            smoothed_probs=smoothed_probs,
            features=features.to_dict(),
            crisis_flag=True,
            crisis_reason=crisis_reason,
            history_meta=history.to_dict(),
        )
    
    # Step 2: Get primary and secondary from probabilities
    raw_primary, raw_primary_prob = probs.get_primary()
    raw_secondary = probs.get_secondary()
    
    # Step 3: Apply stickiness
    primary_regime, history = apply_label_stickiness(
        raw_primary, raw_primary_prob, history, cfg
    )
    
    # Secondary: only if probability meets threshold
    secondary_regime = None
    if raw_secondary and raw_secondary[1] >= cfg.prob_threshold_secondary:
        secondary_regime = raw_secondary[0]
    
    # Update history
    history.last_n_labels.append(primary_regime)
    if len(history.last_n_labels) > 20:
        history.last_n_labels = history.last_n_labels[-20:]
    
    # Step 4: Smooth probabilities
    prev_probs = prev_state.smoothed_probs if prev_state else {}
    smoothed_probs = smooth_probabilities(probs.probs, prev_probs, cfg.prob_alpha)
    
    return SentinelXState(
        updated_ts=now,
        cycle_count=cycle_count,
        primary_regime=primary_regime,
        secondary_regime=secondary_regime,
        regime_probs=probs.to_dict(),
        smoothed_probs=smoothed_probs,
        features=features.to_dict(),
        crisis_flag=False,
        crisis_reason="",
        history_meta=history.to_dict(),
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

_SENTINEL_X_CYCLE_COUNT = 0  # Module-level counter


def should_run_sentinel_x(
    cycle_count: int,
    cfg: SentinelXConfig,
) -> bool:
    """
    Check if Sentinel-X should run this cycle.
    
    Args:
        cycle_count: Current intel cycle
        cfg: Configuration
        
    Returns:
        True if should run
    """
    if not cfg.enabled:
        return False
    if cfg.run_interval_cycles <= 0:
        return True
    if cycle_count % cfg.run_interval_cycles != 0:
        return False
    return True


def run_sentinel_x_step(
    prices: List[float],
    cfg: Optional[SentinelXConfig] = None,
    strategy_cfg: Optional[Dict[str, Any]] = None,
    volumes: Optional[List[float]] = None,
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
    reference_prices: Optional[List[float]] = None,
    current_dd: float = 0.0,
    state_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[SentinelXState]:
    """
    Run a single Sentinel-X classification step.
    
    Args:
        prices: Close price series (e.g., BTC index)
        cfg: Optional config override
        strategy_cfg: Optional strategy config dict
        volumes: Volume series (optional)
        highs: High price series (optional)
        lows: Low price series (optional)
        reference_prices: Reference prices for correlation (optional)
        current_dd: Current portfolio drawdown
        state_path: Optional state file path
        dry_run: If True, don't save state
        
    Returns:
        SentinelXState or None if disabled
    """
    global _SENTINEL_X_CYCLE_COUNT
    _SENTINEL_X_CYCLE_COUNT += 1
    
    cfg = cfg or load_sentinel_x_config(strategy_cfg)
    
    if not cfg.enabled:
        return None
    
    # Load previous state
    prev_state = load_sentinel_x_state(state_path)
    
    # Extract features
    features = extract_regime_features(
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        reference_prices=reference_prices,
        cfg=cfg,
    )
    
    # Initialize model and predict
    model = SimpleRegimeModel(cfg)
    probs = model.predict_proba(features)
    
    # Classify with rules
    state = classify_regime(
        probs=probs,
        features=features,
        cfg=cfg,
        prev_state=prev_state,
        current_dd=current_dd,
    )
    
    # Add metadata
    state.meta = {
        "model_type": cfg.model_type,
        "lookback_bars": cfg.lookback_bars,
        "feature_agg_bars": cfg.feature_agg_bars,
        "data_points": len(prices),
    }
    
    # Save state
    if not dry_run:
        save_sentinel_x_state(state, state_path)
    
    return state


# ---------------------------------------------------------------------------
# View Functions (for Dashboard and EdgeInsights)
# ---------------------------------------------------------------------------


def get_sentinel_x_summary(
    state: Optional[SentinelXState] = None,
) -> Dict[str, Any]:
    """
    Get summary for dashboard display.
    
    Args:
        state: Optional state (loads from file if not provided)
        
    Returns:
        Summary dict
    """
    if state is None:
        state = load_sentinel_x_state()
    
    return {
        "primary_regime": state.primary_regime,
        "secondary_regime": state.secondary_regime,
        "regime_probs": state.smoothed_probs,
        "crisis_flag": state.crisis_flag,
        "crisis_reason": state.crisis_reason,
        "updated_ts": state.updated_ts,
        "cycle_count": state.cycle_count,
    }


def get_sentinel_x_for_insights(
    state: Optional[SentinelXState] = None,
) -> Dict[str, Any]:
    """
    Get Sentinel-X data for EdgeInsights.
    
    Args:
        state: Optional state
        
    Returns:
        Dict suitable for EdgeInsights.sentinel_x field
    """
    if state is None:
        state = load_sentinel_x_state()
    
    return {
        "primary_regime": state.primary_regime,
        "secondary_regime": state.secondary_regime,
        "regime_probs": state.smoothed_probs,
        "crisis_flag": state.crisis_flag,
        "features": state.features,
    }


def get_regime_conviction_weight(
    regime: str,
    weights_config: Optional[Dict[str, float]] = None,
) -> float:
    """
    Get conviction weight multiplier for a regime.
    
    Default weights:
    - TREND_UP: 1.10 (boost)
    - TREND_DOWN: 0.90 (reduce)
    - MEAN_REVERT: 0.95 (slight reduce)
    - BREAKOUT: 1.05 (slight boost)
    - CHOPPY: 0.85 (reduce)
    - CRISIS: 0.50 (strong reduce)
    
    Args:
        regime: Regime label
        weights_config: Optional custom weights
        
    Returns:
        Multiplier in [0.5, 1.5]
    """
    default_weights = {
        "TREND_UP": 1.10,
        "TREND_DOWN": 0.90,
        "MEAN_REVERT": 0.95,
        "BREAKOUT": 1.05,
        "CHOPPY": 0.85,
        "CRISIS": 0.50,
    }
    
    weights = weights_config or default_weights
    weight = weights.get(regime, 1.0)
    
    # Clamp to safe range
    return max(0.5, min(1.5, weight))


def get_factor_regime_weights(
    regime: str,
    factor_weights_config: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Get factor weight adjustments for a regime.
    
    Args:
        regime: Regime label
        factor_weights_config: Optional custom weights
        
    Returns:
        Dict of factor_name → multiplier
    """
    default_weights = {
        "TREND_UP": {"trend": 1.15, "mean_revert": 0.90, "momentum": 1.10, "value": 0.95},
        "TREND_DOWN": {"trend": 1.10, "mean_revert": 0.85, "momentum": 0.90, "value": 1.05},
        "MEAN_REVERT": {"trend": 0.90, "mean_revert": 1.15, "momentum": 0.95, "value": 1.05},
        "BREAKOUT": {"trend": 1.10, "mean_revert": 0.85, "momentum": 1.15, "value": 0.90},
        "CHOPPY": {"trend": 0.85, "mean_revert": 1.00, "momentum": 0.85, "value": 1.00},
        "CRISIS": {"trend": 0.70, "mean_revert": 0.80, "momentum": 0.70, "value": 1.10},
    }
    
    weights = factor_weights_config or default_weights
    return weights.get(regime, {})


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def get_regime_allocation_factor(
    regime: str,
    base_factor: float = 1.0,
) -> float:
    """
    Get allocation factor for Alpha Router based on regime.
    
    Args:
        regime: Current regime
        base_factor: Base factor to adjust
        
    Returns:
        Adjusted factor
    """
    adjustments = {
        "TREND_UP": 1.05,
        "TREND_DOWN": 0.90,
        "MEAN_REVERT": 0.95,
        "BREAKOUT": 1.00,
        "CHOPPY": 0.85,
        "CRISIS": 0.60,
    }
    
    adjustment = adjustments.get(regime, 1.0)
    return base_factor * adjustment


def get_regime_universe_shrink(
    regime: str,
) -> float:
    """
    Get universe shrink factor for Universe Optimizer based on regime.
    
    Higher value = more aggressive shrinkage.
    
    Args:
        regime: Current regime
        
    Returns:
        Shrink factor [0, 1]
    """
    shrink_factors = {
        "TREND_UP": 0.0,  # No extra shrink
        "TREND_DOWN": 0.15,  # Slight shrink
        "MEAN_REVERT": 0.10,
        "BREAKOUT": 0.05,
        "CHOPPY": 0.25,  # More shrink in chop
        "CRISIS": 0.40,  # Aggressive shrink
    }
    
    return shrink_factors.get(regime, 0.0)
