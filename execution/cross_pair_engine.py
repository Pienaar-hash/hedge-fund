"""
Cross-Pair Statistical Arbitrage Engine (Crossfire) — v7.8_P5

Detects statistical mispricings between correlated symbol pairs and produces
pair-level edge scores for research and intel surfaces.

Architecture:
    Price Series → OLS Hedge Ratio → Spread Computation → Z-Score → Edge Scoring
                                                                ↓
                                        logs/state/cross_pair_edges.json

Features:
    - Hedge ratio estimation via OLS regression
    - Spread z-score calculation with mean reversion signals
    - Correlation and half-life quality metrics
    - Residual momentum for trend detection
    - EMA smoothing for edge score stability

This module is RESEARCH-ONLY and does NOT place trades.
It reads price data and produces intel surfaces for dashboards and research.

Single writer rule: Only executor/intel pipeline may write cross_pair_edges.json.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/cross_pair_edges.json")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")

_LOG = logging.getLogger(__name__)

# Safety constants
MIN_BARS_FOR_STATS = 30  # Minimum bars for reliable statistics
SAFE_STD_FLOOR = 1e-10  # Floor for standard deviation to avoid division by zero
MAX_HALF_LIFE = 500  # Cap half-life estimate


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CrossPairConfig:
    """Configuration for the Cross-Pair Engine."""

    enabled: bool = False
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    lookback_bars: int = 240
    half_life_bars: int = 60
    min_corr: float = 0.75
    max_spread_z: float = 3.0
    entry_z: float = 2.0
    exit_z: float = 0.5
    min_liquidity_usd: float = 50_000.0
    smoothing_alpha: float = 0.20
    run_interval_cycles: int = 10  # Run every N cycles

    def __post_init__(self) -> None:
        """Validate and normalize config values."""
        if self.lookback_bars < MIN_BARS_FOR_STATS:
            self.lookback_bars = MIN_BARS_FOR_STATS
        if self.half_life_bars < 1:
            self.half_life_bars = 60
        if self.min_corr < 0 or self.min_corr > 1:
            self.min_corr = 0.75
        if self.entry_z <= 0:
            self.entry_z = 2.0
        if self.exit_z < 0:
            self.exit_z = 0.5
        if self.smoothing_alpha < 0.01 or self.smoothing_alpha > 1.0:
            self.smoothing_alpha = 0.20
        # Normalize pairs to tuples
        normalized_pairs = []
        for pair in self.pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                normalized_pairs.append((str(pair[0]).upper(), str(pair[1]).upper()))
        self.pairs = normalized_pairs


@dataclass
class PairStats:
    """
    Statistical metrics for a symbol pair.
    """

    base: str  # First symbol (e.g., BTCUSDT)
    quote: str  # Second symbol (e.g., ETHUSDT)
    hedge_ratio: float  # OLS beta coefficient
    spread_mean: float  # Mean of spread series
    spread_std: float  # Std dev of spread series
    spread_z: float  # Current z-score of spread
    spread_last: float  # Last spread value
    corr: float  # Correlation between price series
    half_life_est: float  # Estimated mean-reversion half-life (bars)
    residual_momo: float  # Residual momentum (recent spread change)
    liquidity_ok: bool  # Both legs have sufficient liquidity
    eligible: bool  # Pair passes all eligibility criteria
    data_quality: float = 1.0  # Data completeness [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "base": self.base,
            "quote": self.quote,
            "hedge_ratio": round(self.hedge_ratio, 6),
            "spread_mean": round(self.spread_mean, 6),
            "spread_std": round(self.spread_std, 6),
            "spread_z": round(self.spread_z, 4),
            "spread_last": round(self.spread_last, 6),
            "corr": round(self.corr, 4),
            "half_life_est": round(self.half_life_est, 2),
            "residual_momo": round(self.residual_momo, 4),
            "liquidity_ok": self.liquidity_ok,
            "eligible": self.eligible,
            "data_quality": round(self.data_quality, 2),
        }


@dataclass
class PairEdge:
    """
    Edge assessment for a symbol pair.
    """

    pair: Tuple[str, str]  # (base, quote)
    edge_score: float  # Composite edge score [0, 1]
    ema_score: float  # EMA-smoothed edge score
    long_leg: Optional[str]  # Symbol to go long (underpriced)
    short_leg: Optional[str]  # Symbol to go short (overpriced)
    signal: str  # "ENTER", "EXIT", "NONE"
    reason: str  # Human-readable explanation
    stats: PairStats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "pair": list(self.pair),
            "edge_score": round(self.edge_score, 4),
            "ema_score": round(self.ema_score, 4),
            "long_leg": self.long_leg,
            "short_leg": self.short_leg,
            "signal": self.signal,
            "reason": self.reason,
            "stats": self.stats.to_dict(),
        }

    @property
    def pair_key(self) -> str:
        """Get standardized pair key."""
        return f"{self.pair[0]}-{self.pair[1]}"


@dataclass
class CrossPairState:
    """
    State persisted to logs/state/cross_pair_edges.json.
    """

    updated_ts: float = 0.0
    cycle_count: int = 0
    pairs_analyzed: int = 0
    pairs_eligible: int = 0
    pair_edges: Dict[str, PairEdge] = field(default_factory=dict)
    ema_scores: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "cycle_count": self.cycle_count,
            "pairs_analyzed": self.pairs_analyzed,
            "pairs_eligible": self.pairs_eligible,
            "pair_edges": {k: v.to_dict() for k, v in self.pair_edges.items()},
            "ema_scores": {k: round(v, 4) for k, v in self.ema_scores.items()},
            "meta": self.meta,
            "notes": self.notes,
            "errors": self.errors[-10:],  # Keep last 10 errors
        }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_cross_pair_config(
    strategy_cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> CrossPairConfig:
    """
    Load cross-pair engine config from strategy_config.json.

    Args:
        strategy_cfg: Optional pre-loaded strategy config dict
        config_path: Optional path override

    Returns:
        CrossPairConfig instance
    """
    if strategy_cfg is None:
        path = config_path or DEFAULT_CONFIG_PATH
        try:
            with open(path, "r") as f:
                strategy_cfg = json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as exc:
            _LOG.warning("cross_pair_config_load_failed: %s", exc)
            return CrossPairConfig(enabled=False)

    section = strategy_cfg.get("cross_pair_engine", {})
    if not section:
        return CrossPairConfig(enabled=False)

    pairs = section.get("pairs", [])

    return CrossPairConfig(
        enabled=bool(section.get("enabled", False)),
        pairs=pairs,
        lookback_bars=int(section.get("lookback_bars", 240)),
        half_life_bars=int(section.get("half_life_bars", 60)),
        min_corr=float(section.get("min_corr", 0.75)),
        max_spread_z=float(section.get("max_spread_z", 3.0)),
        entry_z=float(section.get("entry_z", 2.0)),
        exit_z=float(section.get("exit_z", 0.5)),
        min_liquidity_usd=float(section.get("min_liquidity_usd", 50_000.0)),
        smoothing_alpha=float(section.get("smoothing_alpha", 0.20)),
        run_interval_cycles=int(section.get("run_interval_cycles", 10)),
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_cross_pair_state(
    state_path: Optional[Path] = None,
) -> CrossPairState:
    """
    Load cross-pair state from JSON file.

    Args:
        state_path: Optional path override

    Returns:
        CrossPairState instance
    """
    path = state_path or DEFAULT_STATE_PATH
    try:
        with open(path, "r") as f:
            data = json.load(f)

        pair_edges = {}
        for key, edge_data in data.get("pair_edges", {}).items():
            stats_data = edge_data.get("stats", {})
            stats = PairStats(
                base=stats_data.get("base", ""),
                quote=stats_data.get("quote", ""),
                hedge_ratio=float(stats_data.get("hedge_ratio", 1.0)),
                spread_mean=float(stats_data.get("spread_mean", 0.0)),
                spread_std=float(stats_data.get("spread_std", 1.0)),
                spread_z=float(stats_data.get("spread_z", 0.0)),
                spread_last=float(stats_data.get("spread_last", 0.0)),
                corr=float(stats_data.get("corr", 0.0)),
                half_life_est=float(stats_data.get("half_life_est", 100.0)),
                residual_momo=float(stats_data.get("residual_momo", 0.0)),
                liquidity_ok=bool(stats_data.get("liquidity_ok", False)),
                eligible=bool(stats_data.get("eligible", False)),
                data_quality=float(stats_data.get("data_quality", 1.0)),
            )
            pair_tuple = tuple(edge_data.get("pair", []))
            if len(pair_tuple) != 2:
                continue
            edge = PairEdge(
                pair=pair_tuple,  # type: ignore
                edge_score=float(edge_data.get("edge_score", 0.0)),
                ema_score=float(edge_data.get("ema_score", 0.0)),
                long_leg=edge_data.get("long_leg"),
                short_leg=edge_data.get("short_leg"),
                signal=str(edge_data.get("signal", "NONE")),
                reason=str(edge_data.get("reason", "")),
                stats=stats,
            )
            pair_edges[key] = edge

        return CrossPairState(
            updated_ts=float(data.get("updated_ts", 0.0)),
            cycle_count=int(data.get("cycle_count", 0)),
            pairs_analyzed=int(data.get("pairs_analyzed", 0)),
            pairs_eligible=int(data.get("pairs_eligible", 0)),
            pair_edges=pair_edges,
            ema_scores=data.get("ema_scores", {}),
            meta=data.get("meta", {}),
            notes=str(data.get("notes", "")),
            errors=data.get("errors", []),
        )
    except (json.JSONDecodeError, IOError, OSError, FileNotFoundError):
        return CrossPairState()


def save_cross_pair_state(
    state: CrossPairState,
    state_path: Optional[Path] = None,
) -> bool:
    """
    Save cross-pair state to JSON file.

    Args:
        state: State to save
        state_path: Optional path override

    Returns:
        True if successful
    """
    path = state_path or DEFAULT_STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return True
    except (IOError, OSError) as exc:
        _LOG.error("cross_pair_state_save_failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Price Data Fetching
# ---------------------------------------------------------------------------


def get_price_series(
    symbol: str,
    lookback_bars: int,
    interval: str = "4h",
) -> Optional[List[float]]:
    """
    Get price series (closes) for a symbol.

    Args:
        symbol: Trading pair
        lookback_bars: Number of bars to fetch
        interval: Kline interval

    Returns:
        List of close prices or None if unavailable
    """
    try:
        from execution.exchange_utils import get_klines

        klines = get_klines(symbol, interval, lookback_bars)
        if not klines or len(klines) < MIN_BARS_FOR_STATS:
            return None
        closes = [float(row[4]) for row in klines if len(row) > 4]
        return closes if len(closes) >= MIN_BARS_FOR_STATS else None
    except Exception as exc:
        _LOG.debug("get_price_series_failed symbol=%s: %s", symbol, exc)
        return None


def get_symbol_liquidity(symbol: str) -> float:
    """
    Get approximate 24h USD volume for a symbol.

    Args:
        symbol: Trading pair

    Returns:
        24h volume in USD, or 0 if unavailable
    """
    try:
        from execution.exchange_utils import get_klines

        # Get last 6 4h bars = 24h
        klines = get_klines(symbol, "4h", 6)
        if not klines:
            return 0.0
        # Sum volume and multiply by last close for USD estimate
        total_vol = sum(float(row[5]) for row in klines if len(row) > 5)
        last_close = float(klines[-1][4]) if len(klines[-1]) > 4 else 0.0
        return total_vol * last_close
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Statistical Computations
# ---------------------------------------------------------------------------


def compute_ols_hedge_ratio(
    prices_a: List[float],
    prices_b: List[float],
) -> Tuple[float, float]:
    """
    Compute OLS hedge ratio: B = alpha + beta * A.

    Args:
        prices_a: Price series for first symbol (X)
        prices_b: Price series for second symbol (Y)

    Returns:
        Tuple of (beta/hedge_ratio, alpha/intercept)
    """
    n = len(prices_a)
    if n != len(prices_b) or n < 2:
        return 1.0, 0.0

    # Simple OLS: beta = cov(A,B) / var(A)
    mean_a = sum(prices_a) / n
    mean_b = sum(prices_b) / n

    cov_ab = sum((a - mean_a) * (b - mean_b) for a, b in zip(prices_a, prices_b)) / n
    var_a = sum((a - mean_a) ** 2 for a in prices_a) / n

    if var_a < SAFE_STD_FLOOR:
        return 1.0, 0.0

    beta = cov_ab / var_a
    alpha = mean_b - beta * mean_a

    return beta, alpha


def compute_spread_series(
    prices_a: List[float],
    prices_b: List[float],
    hedge_ratio: float,
) -> List[float]:
    """
    Compute spread series: spread_t = B_t - beta * A_t.

    Args:
        prices_a: Price series for first symbol
        prices_b: Price series for second symbol
        hedge_ratio: Beta coefficient

    Returns:
        Spread series
    """
    return [b - hedge_ratio * a for a, b in zip(prices_a, prices_b)]


def compute_correlation(
    series_a: List[float],
    series_b: List[float],
) -> float:
    """
    Compute Pearson correlation between two series.

    Args:
        series_a: First series
        series_b: Second series

    Returns:
        Correlation coefficient [-1, 1]
    """
    n = len(series_a)
    if n != len(series_b) or n < 2:
        return 0.0

    mean_a = sum(series_a) / n
    mean_b = sum(series_b) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b)) / n
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in series_a) / n)
    std_b = math.sqrt(sum((b - mean_b) ** 2 for b in series_b) / n)

    if std_a < SAFE_STD_FLOOR or std_b < SAFE_STD_FLOOR:
        return 0.0

    return cov / (std_a * std_b)


def compute_half_life(spread: List[float]) -> float:
    """
    Estimate mean-reversion half-life from AR(1) model.

    spread_t = phi * spread_{t-1} + epsilon
    half_life = -log(2) / log(phi)

    Args:
        spread: Spread series

    Returns:
        Estimated half-life in bars (capped at MAX_HALF_LIFE)
    """
    if len(spread) < 10:
        return MAX_HALF_LIFE

    # AR(1) regression: spread_t = phi * spread_{t-1}
    lag = spread[:-1]
    current = spread[1:]
    n = len(lag)

    mean_lag = sum(lag) / n
    mean_curr = sum(current) / n

    cov = sum((lv - mean_lag) * (c - mean_curr) for lv, c in zip(lag, current)) / n
    var_lag = sum((lv - mean_lag) ** 2 for lv in lag) / n

    if var_lag < SAFE_STD_FLOOR:
        return MAX_HALF_LIFE

    phi = cov / var_lag

    # Clamp phi to valid range for log
    if phi <= 0 or phi >= 1:
        return MAX_HALF_LIFE

    half_life = -math.log(2) / math.log(phi)
    return min(MAX_HALF_LIFE, max(1.0, half_life))


def compute_residual_momentum(
    spread: List[float],
    window: int = 10,
) -> float:
    """
    Compute residual momentum as recent spread change.

    Args:
        spread: Spread series
        window: Lookback window for momentum

    Returns:
        Normalized momentum [-1, 1]
    """
    if len(spread) < window + 1:
        return 0.0

    recent = spread[-1]
    older = spread[-(window + 1)]

    if abs(older) < SAFE_STD_FLOOR:
        return 0.0

    momo = (recent - older) / abs(older)
    # Clamp to reasonable range
    return max(-1.0, min(1.0, momo))


def compute_pair_stats(
    base: str,
    quote: str,
    prices_base: List[float],
    prices_quote: List[float],
    config: CrossPairConfig,
    base_liquidity: float,
    quote_liquidity: float,
) -> PairStats:
    """
    Compute all statistical metrics for a pair.

    Args:
        base: Base symbol
        quote: Quote symbol
        prices_base: Price series for base
        prices_quote: Price series for quote
        config: Engine configuration
        base_liquidity: Base symbol 24h volume
        quote_liquidity: Quote symbol 24h volume

    Returns:
        PairStats with all computed metrics
    """
    # Align series to same length
    min_len = min(len(prices_base), len(prices_quote))
    prices_a = prices_base[-min_len:]
    prices_b = prices_quote[-min_len:]

    # OLS hedge ratio
    hedge_ratio, _ = compute_ols_hedge_ratio(prices_a, prices_b)

    # Spread series
    spread = compute_spread_series(prices_a, prices_b, hedge_ratio)

    # Spread statistics
    spread_mean = sum(spread) / len(spread) if spread else 0.0
    spread_variance = sum((s - spread_mean) ** 2 for s in spread) / len(spread) if spread else 0.0
    spread_std = math.sqrt(spread_variance) if spread_variance > 0 else SAFE_STD_FLOOR

    # Current z-score
    spread_last = spread[-1] if spread else 0.0
    spread_z = (spread_last - spread_mean) / max(spread_std, SAFE_STD_FLOOR)

    # Correlation
    corr = compute_correlation(prices_a, prices_b)

    # Half-life
    half_life_est = compute_half_life(spread)

    # Residual momentum
    residual_momo = compute_residual_momentum(spread)

    # Liquidity check
    liquidity_ok = (
        base_liquidity >= config.min_liquidity_usd
        and quote_liquidity >= config.min_liquidity_usd
    )

    # Data quality
    data_quality = min(1.0, min_len / config.lookback_bars)

    # Eligibility
    eligible = (
        corr >= config.min_corr
        and half_life_est <= config.half_life_bars
        and liquidity_ok
        and data_quality >= 0.5
    )

    return PairStats(
        base=base,
        quote=quote,
        hedge_ratio=hedge_ratio,
        spread_mean=spread_mean,
        spread_std=spread_std,
        spread_z=spread_z,
        spread_last=spread_last,
        corr=corr,
        half_life_est=half_life_est,
        residual_momo=residual_momo,
        liquidity_ok=liquidity_ok,
        eligible=eligible,
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# Edge Scoring
# ---------------------------------------------------------------------------


def compute_pair_edge(
    stats: PairStats,
    config: CrossPairConfig,
    prev_ema: float = 0.0,
) -> PairEdge:
    """
    Compute edge score and signal for a pair.

    Edge components:
    - |spread_z| relative to entry_z / max_spread_z
    - Sign of residual_momo (mean-reversion favorable vs runaway)
    - Correlation strength
    - Half-life quality

    Args:
        stats: Pair statistics
        config: Engine configuration
        prev_ema: Previous EMA score for smoothing

    Returns:
        PairEdge with score and signal
    """
    pair = (stats.base, stats.quote)

    # If not eligible, return zero edge
    if not stats.eligible:
        return PairEdge(
            pair=pair,
            edge_score=0.0,
            ema_score=prev_ema * (1 - config.smoothing_alpha),
            long_leg=None,
            short_leg=None,
            signal="NONE",
            reason="not eligible",
            stats=stats,
        )

    # --- Edge Components ---

    # 1. Z-score magnitude component (higher |z| within bounds = higher edge)
    abs_z = abs(stats.spread_z)
    if abs_z > config.max_spread_z:
        # Beyond max z = too risky
        z_component = 0.0
    elif abs_z >= config.entry_z:
        # In entry zone = good setup
        z_component = min(1.0, abs_z / config.max_spread_z)
    elif abs_z >= config.exit_z:
        # Between exit and entry = moderate setup
        z_component = 0.5 * (abs_z - config.exit_z) / max(0.1, config.entry_z - config.exit_z)
    else:
        # Below exit z = no setup
        z_component = 0.0

    # 2. Momentum component (mean-reversion favorable = spread moving back)
    # If z > 0 (spread overextended high), we want momo < 0 (coming back)
    # If z < 0 (spread underextended), we want momo > 0 (coming back)
    momo_favorable = (stats.spread_z > 0 and stats.residual_momo < 0) or \
                     (stats.spread_z < 0 and stats.residual_momo > 0)
    momo_component = 0.6 if momo_favorable else 0.3

    # 3. Correlation component (higher corr = better pair)
    corr_component = (stats.corr - config.min_corr) / max(0.1, 1.0 - config.min_corr)
    corr_component = max(0.0, min(1.0, corr_component))

    # 4. Half-life component (faster mean reversion = better)
    hl_component = 1.0 - (stats.half_life_est / config.half_life_bars)
    hl_component = max(0.0, min(1.0, hl_component))

    # Weighted combination
    edge_score = (
        0.40 * z_component
        + 0.25 * momo_component
        + 0.20 * corr_component
        + 0.15 * hl_component
    )

    # Apply data quality discount
    edge_score *= stats.data_quality

    # Clamp to [0, 1]
    edge_score = max(0.0, min(1.0, edge_score))

    # EMA smoothing
    if prev_ema > 0:
        ema_score = config.smoothing_alpha * edge_score + (1 - config.smoothing_alpha) * prev_ema
    else:
        ema_score = edge_score

    # --- Signal Determination ---
    signal = "NONE"
    long_leg: Optional[str] = None
    short_leg: Optional[str] = None
    reason = ""

    if abs_z >= config.entry_z and abs_z <= config.max_spread_z:
        signal = "ENTER"
        if stats.spread_z > 0:
            # Spread too high: B is overpriced relative to A
            # Long A, Short B
            long_leg = stats.base
            short_leg = stats.quote
            reason = f"spread z={stats.spread_z:.2f} > entry; long {stats.base}, short {stats.quote}"
        else:
            # Spread too low: A is overpriced relative to B
            # Long B, Short A
            long_leg = stats.quote
            short_leg = stats.base
            reason = f"spread z={stats.spread_z:.2f} < -entry; long {stats.quote}, short {stats.base}"
    elif abs_z < config.exit_z:
        signal = "EXIT"
        reason = f"spread z={stats.spread_z:.2f} within exit zone"
    else:
        # Between exit_z and entry_z
        signal = "NONE"
        reason = f"spread z={stats.spread_z:.2f} between zones"

    return PairEdge(
        pair=pair,
        edge_score=edge_score,
        ema_score=ema_score,
        long_leg=long_leg,
        short_leg=short_leg,
        signal=signal,
        reason=reason,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

_CROSS_PAIR_CYCLE_COUNT = 0  # Module-level counter


def should_run_cross_pair(
    cycle_count: int,
    config: CrossPairConfig,
) -> bool:
    """
    Check if cross-pair engine should run this cycle.

    Args:
        cycle_count: Current intel cycle
        config: Engine configuration

    Returns:
        True if should run
    """
    if not config.enabled:
        return False
    if cycle_count % config.run_interval_cycles != 0:
        return False
    return True


def run_cross_pair_scan(
    config: Optional[CrossPairConfig] = None,
    strategy_cfg: Optional[Dict[str, Any]] = None,
    state_path: Optional[Path] = None,
    dry_run: bool = False,
) -> CrossPairState:
    """
    Run a single cross-pair scan.

    This scans all configured pairs, computes statistics, and produces edges.

    Args:
        config: Optional config override
        strategy_cfg: Optional strategy config dict
        state_path: Optional state path override
        dry_run: If True, don't save state

    Returns:
        CrossPairState with scan results
    """
    global _CROSS_PAIR_CYCLE_COUNT
    _CROSS_PAIR_CYCLE_COUNT += 1

    cfg = config or load_cross_pair_config(strategy_cfg)
    if not cfg.enabled:
        _LOG.debug("cross_pair_engine: disabled, skipping")
        return CrossPairState(notes="disabled")

    if not cfg.pairs:
        _LOG.debug("cross_pair_engine: no pairs configured")
        return CrossPairState(notes="no_pairs")

    start_ts = time.time()
    errors: List[str] = []

    # Load previous state for EMA
    prev_state = load_cross_pair_state(state_path)

    pair_edges: Dict[str, PairEdge] = {}
    ema_scores: Dict[str, float] = {}
    pairs_eligible = 0

    for base, quote in cfg.pairs:
        pair_key = f"{base}-{quote}"
        try:
            # Fetch price series
            prices_base = get_price_series(base, cfg.lookback_bars)
            prices_quote = get_price_series(quote, cfg.lookback_bars)

            if prices_base is None or prices_quote is None:
                _LOG.debug("cross_pair: insufficient data for %s", pair_key)
                errors.append(f"{pair_key}: insufficient data")
                continue

            # Get liquidity
            base_liq = get_symbol_liquidity(base)
            quote_liq = get_symbol_liquidity(quote)

            # Compute stats
            stats = compute_pair_stats(
                base=base,
                quote=quote,
                prices_base=prices_base,
                prices_quote=prices_quote,
                config=cfg,
                base_liquidity=base_liq,
                quote_liquidity=quote_liq,
            )

            # Get previous EMA
            prev_ema = prev_state.ema_scores.get(pair_key, 0.0)

            # Compute edge
            edge = compute_pair_edge(stats, cfg, prev_ema)
            pair_edges[pair_key] = edge
            ema_scores[pair_key] = edge.ema_score

            if stats.eligible:
                pairs_eligible += 1

        except Exception as exc:
            _LOG.warning("cross_pair_scan_failed pair=%s: %s", pair_key, exc)
            errors.append(f"{pair_key}: {exc}")

    elapsed = time.time() - start_ts

    # Build state
    state = CrossPairState(
        updated_ts=time.time(),
        cycle_count=prev_state.cycle_count + 1,
        pairs_analyzed=len(cfg.pairs),
        pairs_eligible=pairs_eligible,
        pair_edges=pair_edges,
        ema_scores=ema_scores,
        meta={
            "lookback_bars": cfg.lookback_bars,
            "entry_z": cfg.entry_z,
            "exit_z": cfg.exit_z,
            "min_corr": cfg.min_corr,
        },
        notes=f"scanned {len(cfg.pairs)} pairs in {elapsed:.2f}s",
        errors=errors,
    )

    # Save state
    if not dry_run:
        save_cross_pair_state(state, state_path)

    _LOG.info(
        "cross_pair: analyzed %d pairs, %d eligible, in %.2fs",
        len(cfg.pairs),
        pairs_eligible,
        elapsed,
    )

    return state


# ---------------------------------------------------------------------------
# View Functions (for dashboard and EdgeInsights)
# ---------------------------------------------------------------------------


def get_top_pair_edges(
    state: Optional[CrossPairState] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get top pair edges sorted by EMA score.

    Args:
        state: Optional state (loads from file if not provided)
        top_k: Number of top pairs to return

    Returns:
        List of pair edge dicts
    """
    if state is None:
        state = load_cross_pair_state()

    edges = list(state.pair_edges.values())
    edges.sort(key=lambda e: e.ema_score, reverse=True)
    return [e.to_dict() for e in edges[:top_k]]


def get_cross_pair_summary(
    state: Optional[CrossPairState] = None,
) -> Dict[str, Any]:
    """
    Get summary stats for dashboard.

    Args:
        state: Optional state

    Returns:
        Summary dict
    """
    if state is None:
        state = load_cross_pair_state()

    now = time.time()
    age_s = now - state.updated_ts if state.updated_ts > 0 else 0

    # Count signals
    enter_count = sum(1 for e in state.pair_edges.values() if e.signal == "ENTER")
    exit_count = sum(1 for e in state.pair_edges.values() if e.signal == "EXIT")

    top_edge = None
    if state.pair_edges:
        best = max(state.pair_edges.values(), key=lambda e: e.ema_score)
        top_edge = {
            "pair": best.pair_key,
            "score": round(best.ema_score, 4),
            "signal": best.signal,
        }

    return {
        "updated_ts": state.updated_ts,
        "age_s": round(age_s, 1),
        "cycle_count": state.cycle_count,
        "pairs_analyzed": state.pairs_analyzed,
        "pairs_eligible": state.pairs_eligible,
        "enter_signals": enter_count,
        "exit_signals": exit_count,
        "top_edge": top_edge,
        "notes": state.notes,
    }


def get_pair_edges_for_insights(
    state: Optional[CrossPairState] = None,
) -> Dict[str, Any]:
    """
    Get pair edges in format suitable for EdgeInsights.

    Args:
        state: Optional state

    Returns:
        Dict with pair edges summary
    """
    if state is None:
        state = load_cross_pair_state()

    if not state.pair_edges:
        return {}

    edges_list = []
    for key, edge in state.pair_edges.items():
        edges_list.append({
            "pair": key,
            "edge_score": round(edge.ema_score, 4),
            "spread_z": round(edge.stats.spread_z, 4),
            "corr": round(edge.stats.corr, 4),
            "half_life": round(edge.stats.half_life_est, 2),
            "signal": edge.signal,
            "eligible": edge.stats.eligible,
        })

    # Sort by edge score
    edges_list.sort(key=lambda x: x["edge_score"], reverse=True)

    return {
        "top_pairs": edges_list[:5],
        "pairs_analyzed": state.pairs_analyzed,
        "pairs_eligible": state.pairs_eligible,
        "updated_ts": state.updated_ts,
    }


# ---------------------------------------------------------------------------
# Universe Optimizer Integration
# ---------------------------------------------------------------------------


def get_pair_leg_boost(
    symbol: str,
    state: Optional[CrossPairState] = None,
    min_edge_threshold: float = 0.5,
    max_boost: float = 0.10,
) -> float:
    """
    Get optional score boost for symbols that are legs of strong pair edges.

    This is a soft bias for Universe Optimizer integration.

    Args:
        symbol: Symbol to check
        state: Optional cross-pair state
        min_edge_threshold: Minimum edge score to trigger boost
        max_boost: Maximum boost amount

    Returns:
        Boost amount [0, max_boost]
    """
    if state is None:
        state = load_cross_pair_state()

    if not state.pair_edges:
        return 0.0

    max_edge = 0.0
    for edge in state.pair_edges.values():
        if not edge.stats.eligible:
            continue
        if symbol == edge.stats.base or symbol == edge.stats.quote:
            max_edge = max(max_edge, edge.ema_score)

    if max_edge < min_edge_threshold:
        return 0.0

    # Scale boost linearly with edge score
    boost = max_boost * (max_edge - min_edge_threshold) / (1.0 - min_edge_threshold)
    return max(0.0, min(max_boost, boost))


def get_cerberus_crossfire_multiplier(
    cerberus_state: Optional[Dict[str, Any]],
) -> float:
    """
    Get Cerberus RELATIVE_VALUE head multiplier for Crossfire edge scaling (v7.8_P8).
    
    Args:
        cerberus_state: Cerberus state dict (or None)
        
    Returns:
        Multiplier typically in [0.5, 2.0]
    """
    if cerberus_state is None:
        return 1.0
    
    head_state = cerberus_state.get("head_state", {})
    if not head_state:
        return 1.0
    
    heads = head_state.get("heads", {})
    rv_head = heads.get("RELATIVE_VALUE", {})
    
    if isinstance(rv_head, dict):
        multiplier = rv_head.get("multiplier", 1.0)
        return max(0.5, min(2.0, multiplier))
    
    return 1.0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "CrossPairConfig",
    "load_cross_pair_config",
    # Stats
    "PairStats",
    "compute_pair_stats",
    "compute_ols_hedge_ratio",
    "compute_spread_series",
    "compute_correlation",
    "compute_half_life",
    "compute_residual_momentum",
    # Edge
    "PairEdge",
    "compute_pair_edge",
    # State
    "CrossPairState",
    "load_cross_pair_state",
    "save_cross_pair_state",
    # Runner
    "should_run_cross_pair",
    "run_cross_pair_scan",
    # Views
    "get_top_pair_edges",
    "get_cross_pair_summary",
    "get_pair_edges_for_insights",
    # Integration
    "get_pair_leg_boost",
    "get_cerberus_crossfire_multiplier",  # v7.8_P8
    # Helpers
    "get_price_series",
    "get_symbol_liquidity",
]
