"""
Alpha Miner (Prospector) — v7.8_P4

Autonomous discovery of new tradable symbols from the entire exchange.
Scans all listed futures, extracts alpha features, scores candidates,
and surfaces recommendations for universe expansion.

The miner does NOT modify the universe — it only produces candidates
for review by the Universe Optimizer (P3) or manual inspection.

Architecture:
    Exchange symbols → Feature extraction → Scoring → EMA smoothing → Candidate selection
                                                           ↓
                                                logs/state/alpha_miner.json

Features:
    - short_momo: 7-period price momentum (%)
    - long_momo: 30-period price momentum (%)
    - volatility: Annualized volatility from 4h closes
    - trend_consistency: % of periods with positive returns
    - liquidity_score: Derived from 24h volume / market cap proxy
    - spread_quality: Inverse of typical bid-ask spread
    - router_score: Expected fill quality from slippage model

Scoring Formula:
    score = (w1 * short_momo + w2 * long_momo + w3 * (1/vol) +
             w4 * trend_consistency + w5 * router_score + w6 * liq_quality)

State:
    Written to logs/state/alpha_miner.json
    Dashboard reads for candidate display

Integration:
    Called from edge_scanner.py after universe_optimizer step
    Runs every N cycles (expensive operation, default 50 cycles)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/alpha_miner.json")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")
DEFAULT_CATEGORIES_PATH = Path("config/symbol_categories.json")
DEFAULT_UNIVERSE_PATH = Path("config/pairs_universe.json")

_LOG = logging.getLogger(__name__)

# Minimum data requirements
MIN_KLINE_BARS = 50  # Need at least 50 bars for feature extraction
SAFE_VOL_FLOOR = 0.001  # Floor for volatility to avoid division by zero

# Scoring weights (default, can be overridden via config)
DEFAULT_WEIGHTS = {
    "short_momo": 0.15,
    "long_momo": 0.20,
    "inv_vol": 0.15,
    "trend_consistency": 0.20,
    "router_score": 0.10,
    "liq_quality": 0.20,
}

# Category inference heuristics (suffix patterns)
CATEGORY_PATTERNS = {
    "BTCUSDT": "L1_MAJOR",
    "ETHUSDT": "L1_MAJOR",
    "BTC": "L1_MAJOR",
    "ETH": "L1_MAJOR",
    "SOL": "L1_ALT",
    "DOGE": "MEME",
    "SHIB": "MEME",
    "PEPE": "MEME",
    "BONK": "MEME",
    "WIF": "MEME",
    "FLOKI": "MEME",
    "LINK": "DEFI",
    "UNI": "DEFI",
    "AAVE": "DEFI",
    "AVAX": "L1_ALT",
    "MATIC": "L1_ALT",
    "ADA": "L1_ALT",
    "DOT": "L1_ALT",
    "ATOM": "L1_ALT",
    "NEAR": "L1_ALT",
    "APT": "L1_ALT",
    "SUI": "L1_ALT",
    "ARB": "L2",
    "OP": "L2",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class AlphaMinerConfig:
    """Configuration for the Alpha Miner."""

    enabled: bool = False  # Must be explicitly enabled
    min_liquidity_usd: float = 500_000.0  # Minimum 24h volume in USD
    max_spread_pct: float = 0.30  # Maximum spread as % of mid-price
    min_age_days: int = 30  # Minimum listing age (days)
    score_threshold: float = 0.50  # Minimum score to be a candidate
    top_k: int = 20  # Maximum candidates to surface
    smoothing_alpha: float = 0.15  # EMA smoothing for scores
    lookback_bars: int = 100  # Bars for feature extraction
    interval: str = "4h"  # Kline interval for features
    run_interval_cycles: int = 50  # Run every N edge_scanner cycles
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["BUSD", "TUSD", "USDC", "DAI", "USDP"]
    )

    def __post_init__(self) -> None:
        """Validate config values."""
        if self.smoothing_alpha < 0.01 or self.smoothing_alpha > 1.0:
            self.smoothing_alpha = 0.15
        if self.top_k < 1:
            self.top_k = 20
        if self.score_threshold < 0 or self.score_threshold > 1.0:
            self.score_threshold = 0.50


@dataclass
class SymbolAlphaFeatures:
    """
    Alpha features extracted for a single symbol.
    """

    symbol: str
    short_momo: float = 0.0  # 7-period momentum (%)
    long_momo: float = 0.0  # 30-period momentum (%)
    volatility: float = 0.0  # Annualized volatility
    trend_consistency: float = 0.0  # Fraction of periods with positive returns
    liquidity_score: float = 0.0  # Normalized liquidity score [0, 1]
    spread_quality: float = 0.0  # Inverse spread score [0, 1]
    router_score: float = 0.5  # Expected fill quality [0, 1]
    category_hint: str = "OTHER"  # Inferred category
    volume_24h: float = 0.0  # Raw 24h volume in USD
    price: float = 0.0  # Current price
    data_quality: float = 1.0  # Data completeness [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class AlphaMinerCandidate:
    """
    A candidate symbol for universe expansion.
    """

    symbol: str
    score: float  # Composite alpha score [0, 1]
    ema_score: float  # EMA-smoothed score
    features: SymbolAlphaFeatures
    reason: str  # Why this is a candidate
    in_universe: bool = False  # Already in active universe
    first_seen_ts: float = 0.0  # First seen timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "ema_score": round(self.ema_score, 4),
            "features": self.features.to_dict(),
            "reason": self.reason,
            "in_universe": self.in_universe,
            "first_seen_ts": self.first_seen_ts,
        }


@dataclass
class AlphaMinerState:
    """
    State persisted to logs/state/alpha_miner.json.
    """

    updated_ts: float = 0.0
    cycle_count: int = 0
    symbols_scanned: int = 0
    symbols_passed_filter: int = 0
    candidates: List[AlphaMinerCandidate] = field(default_factory=list)
    ema_scores: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "cycle_count": self.cycle_count,
            "symbols_scanned": self.symbols_scanned,
            "symbols_passed_filter": self.symbols_passed_filter,
            "candidates": [c.to_dict() for c in self.candidates],
            "ema_scores": {k: round(v, 4) for k, v in self.ema_scores.items()},
            "notes": self.notes,
            "errors": self.errors[-10:],  # Keep last 10 errors
        }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_alpha_miner_config(
    config_path: Optional[Path] = None,
) -> AlphaMinerConfig:
    """
    Load alpha miner config from strategy_config.json.

    Args:
        config_path: Optional path override

    Returns:
        AlphaMinerConfig instance
    """
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        with open(path, "r") as f:
            data = json.load(f)
        section = data.get("alpha_miner", {})
        if not section:
            return AlphaMinerConfig(enabled=False)

        weights = section.get("weights", DEFAULT_WEIGHTS.copy())
        exclude = section.get("exclude_patterns", ["BUSD", "TUSD", "USDC", "DAI", "USDP"])

        return AlphaMinerConfig(
            enabled=bool(section.get("enabled", False)),
            min_liquidity_usd=float(section.get("min_liquidity_usd", 500_000.0)),
            max_spread_pct=float(section.get("max_spread_pct", 0.30)),
            min_age_days=int(section.get("min_age_days", 30)),
            score_threshold=float(section.get("score_threshold", 0.50)),
            top_k=int(section.get("top_k", 20)),
            smoothing_alpha=float(section.get("smoothing_alpha", 0.15)),
            lookback_bars=int(section.get("lookback_bars", 100)),
            interval=str(section.get("interval", "4h")),
            run_interval_cycles=int(section.get("run_interval_cycles", 50)),
            weights=weights,
            exclude_patterns=exclude,
        )
    except (json.JSONDecodeError, IOError, OSError) as exc:
        _LOG.warning("alpha_miner_config_load_failed: %s", exc)
        return AlphaMinerConfig(enabled=False)


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_alpha_miner_state(
    state_path: Optional[Path] = None,
) -> AlphaMinerState:
    """
    Load alpha miner state from JSON file.

    Args:
        state_path: Optional path override

    Returns:
        AlphaMinerState instance
    """
    path = state_path or DEFAULT_STATE_PATH
    try:
        with open(path, "r") as f:
            data = json.load(f)

        candidates = []
        for c in data.get("candidates", []):
            feat_data = c.get("features", {})
            feat = SymbolAlphaFeatures(
                symbol=feat_data.get("symbol", c.get("symbol", "")),
                short_momo=float(feat_data.get("short_momo", 0.0)),
                long_momo=float(feat_data.get("long_momo", 0.0)),
                volatility=float(feat_data.get("volatility", 0.0)),
                trend_consistency=float(feat_data.get("trend_consistency", 0.0)),
                liquidity_score=float(feat_data.get("liquidity_score", 0.0)),
                spread_quality=float(feat_data.get("spread_quality", 0.0)),
                router_score=float(feat_data.get("router_score", 0.5)),
                category_hint=str(feat_data.get("category_hint", "OTHER")),
                volume_24h=float(feat_data.get("volume_24h", 0.0)),
                price=float(feat_data.get("price", 0.0)),
                data_quality=float(feat_data.get("data_quality", 1.0)),
            )
            cand = AlphaMinerCandidate(
                symbol=c.get("symbol", ""),
                score=float(c.get("score", 0.0)),
                ema_score=float(c.get("ema_score", 0.0)),
                features=feat,
                reason=str(c.get("reason", "")),
                in_universe=bool(c.get("in_universe", False)),
                first_seen_ts=float(c.get("first_seen_ts", 0.0)),
            )
            candidates.append(cand)

        return AlphaMinerState(
            updated_ts=float(data.get("updated_ts", 0.0)),
            cycle_count=int(data.get("cycle_count", 0)),
            symbols_scanned=int(data.get("symbols_scanned", 0)),
            symbols_passed_filter=int(data.get("symbols_passed_filter", 0)),
            candidates=candidates,
            ema_scores=data.get("ema_scores", {}),
            notes=str(data.get("notes", "")),
            errors=data.get("errors", []),
        )
    except (json.JSONDecodeError, IOError, OSError, FileNotFoundError):
        return AlphaMinerState()


def save_alpha_miner_state(
    state: AlphaMinerState,
    state_path: Optional[Path] = None,
) -> bool:
    """
    Save alpha miner state to JSON file.

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
        _LOG.error("alpha_miner_state_save_failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Exchange Symbol Discovery
# ---------------------------------------------------------------------------


def get_all_exchange_symbols() -> List[str]:
    """
    Get all USDT-margined futures symbols from the exchange.

    Returns:
        List of symbol names (e.g., ["BTCUSDT", "ETHUSDT", ...])
    """
    try:
        # Import here to avoid circular imports
        from execution.exchange_utils import _load_exchange_filters

        filters = _load_exchange_filters()
        # Filter for USDT perpetuals
        symbols = [
            sym
            for sym in filters.keys()
            if sym.endswith("USDT") and not sym.endswith("_PERP")
        ]
        return sorted(symbols)
    except Exception as exc:
        _LOG.warning("alpha_miner_symbols_fetch_failed: %s", exc)
        return []


def load_current_universe(
    universe_path: Optional[Path] = None,
) -> List[str]:
    """
    Load the current trading universe.

    Args:
        universe_path: Optional path override

    Returns:
        List of symbols in the current universe
    """
    path = universe_path or DEFAULT_UNIVERSE_PATH
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [str(k).upper() for k in data.keys()]
        return []
    except (json.JSONDecodeError, IOError, OSError):
        return []


def filter_excluded_symbols(
    symbols: List[str],
    exclude_patterns: List[str],
    current_universe: List[str],
) -> List[str]:
    """
    Filter symbols based on exclusion patterns and current universe.

    Args:
        symbols: All symbols to filter
        exclude_patterns: Patterns to exclude (e.g., stablecoins)
        current_universe: Symbols already in universe (skip scanning)

    Returns:
        Filtered list of symbols to scan
    """
    result = []
    for sym in symbols:
        # Skip if already in universe
        if sym in current_universe:
            continue
        # Skip if matches exclusion pattern
        skip = False
        for pattern in exclude_patterns:
            if pattern.upper() in sym.upper():
                skip = True
                break
        if not skip:
            result.append(sym)
    return result


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def infer_category(symbol: str) -> str:
    """
    Infer category from symbol name using heuristics.

    Args:
        symbol: Symbol name (e.g., "PEPEUSDT")

    Returns:
        Inferred category name
    """
    sym_upper = symbol.upper()
    # Direct match
    if sym_upper in CATEGORY_PATTERNS:
        return CATEGORY_PATTERNS[sym_upper]
    # Remove USDT suffix and check
    base = sym_upper.replace("USDT", "")
    if base in CATEGORY_PATTERNS:
        return CATEGORY_PATTERNS[base]
    # Default to OTHER
    return "OTHER"


def compute_momentum(closes: List[float], period: int) -> float:
    """
    Compute price momentum over a period.

    Args:
        closes: List of close prices (oldest first)
        period: Lookback period

    Returns:
        Momentum as percentage (-1 to +1 range, clamped)
    """
    if len(closes) < period + 1:
        return 0.0
    recent = closes[-1]
    older = closes[-(period + 1)]
    if older <= 0:
        return 0.0
    momo = (recent - older) / older
    # Clamp to reasonable range
    return max(-1.0, min(1.0, momo))


def compute_volatility(closes: List[float]) -> float:
    """
    Compute annualized volatility from closes.

    Args:
        closes: List of close prices (oldest first)

    Returns:
        Annualized volatility (assuming 4h bars)
    """
    if len(closes) < 10:
        return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            returns.append(ret)
    if len(returns) < 5:
        return 0.0
    # Standard deviation of returns
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance) if variance > 0 else SAFE_VOL_FLOOR
    # Annualize assuming 4h bars: sqrt(6 * 365)
    ann_factor = math.sqrt(6 * 365)
    return std * ann_factor


def compute_trend_consistency(closes: List[float], period: int = 30) -> float:
    """
    Compute fraction of periods with positive returns.

    Args:
        closes: List of close prices (oldest first)
        period: Lookback period

    Returns:
        Fraction [0, 1] of positive return periods
    """
    if len(closes) < period + 1:
        return 0.5
    start_idx = max(0, len(closes) - period - 1)
    positive_count = 0
    total_count = 0
    for i in range(start_idx + 1, len(closes)):
        if closes[i - 1] > 0:
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            if ret > 0:
                positive_count += 1
            total_count += 1
    if total_count == 0:
        return 0.5
    return positive_count / total_count


def compute_liquidity_score(volume_24h: float, min_volume: float) -> float:
    """
    Normalize volume into a [0, 1] liquidity score.

    Args:
        volume_24h: 24h volume in USD
        min_volume: Minimum volume threshold

    Returns:
        Liquidity score [0, 1]
    """
    if volume_24h <= 0:
        return 0.0
    if volume_24h < min_volume:
        return volume_24h / min_volume
    # Log-scale above threshold
    # At 10x min_volume → score ~0.7
    # At 100x min_volume → score ~0.85
    ratio = volume_24h / min_volume
    score = 0.5 + 0.5 * (1 - 1 / (1 + math.log10(ratio)))
    return min(1.0, score)


def compute_spread_quality(
    spread_pct: float,
    max_spread_pct: float,
) -> float:
    """
    Convert spread percentage to quality score.

    Args:
        spread_pct: Spread as percentage of mid-price
        max_spread_pct: Maximum acceptable spread

    Returns:
        Spread quality score [0, 1] (higher = tighter spread)
    """
    if spread_pct <= 0:
        return 1.0  # Perfect spread
    if spread_pct >= max_spread_pct:
        return 0.0
    # Linear interpolation: 0% spread = 1.0, max_spread = 0.0
    return 1.0 - (spread_pct / max_spread_pct)


def extract_symbol_features(
    symbol: str,
    config: AlphaMinerConfig,
    klines: Optional[List[List[float]]] = None,
    orderbook: Optional[Dict[str, Any]] = None,
    volume_24h: float = 0.0,
    price: float = 0.0,
) -> Optional[SymbolAlphaFeatures]:
    """
    Extract alpha features for a single symbol.

    Args:
        symbol: Symbol name
        config: Miner configuration
        klines: Optional klines data [[ts, o, h, l, c, v], ...]
        orderbook: Optional orderbook data {"bids": [...], "asks": [...]}
        volume_24h: 24h volume in USD
        price: Current price

    Returns:
        SymbolAlphaFeatures or None if insufficient data
    """
    if klines is None:
        try:
            from execution.exchange_utils import get_klines

            klines = get_klines(symbol, config.interval, config.lookback_bars)
        except Exception as exc:
            _LOG.debug("alpha_miner_klines_failed symbol=%s: %s", symbol, exc)
            return None

    if not klines or len(klines) < MIN_KLINE_BARS:
        return None

    # Extract closes (index 4 in kline row)
    closes = [float(row[4]) for row in klines if len(row) > 4]
    if len(closes) < MIN_KLINE_BARS:
        return None

    # Compute features
    short_momo = compute_momentum(closes, 7)
    long_momo = compute_momentum(closes, 30)
    volatility = compute_volatility(closes)
    trend_consistency = compute_trend_consistency(closes, 30)
    liq_score = compute_liquidity_score(volume_24h, config.min_liquidity_usd)

    # Spread quality from orderbook
    spread_pct = 0.0
    if orderbook:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            if best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2
                spread_pct = (best_ask - best_bid) / mid * 100
    spread_quality = compute_spread_quality(spread_pct, config.max_spread_pct)

    # Router score (placeholder - use slippage model if available)
    router_score = 0.5
    try:
        from execution.slippage_model import estimate_expected_slippage_bps

        # Estimate slippage for a small test order
        test_qty = 1000 / max(price, 1.0)  # ~$1000 order
        depth = orderbook if orderbook else {"bids": [], "asks": []}
        slippage_bps = estimate_expected_slippage_bps("BUY", test_qty, depth, price)
        # Convert slippage to score: 0 bps = 1.0, 50 bps = 0.5, 100+ bps = 0.0
        router_score = max(0.0, 1.0 - slippage_bps / 100)
    except Exception:
        pass  # Use default

    # Infer category
    category = infer_category(symbol)

    # Data quality based on bar count
    data_quality = min(1.0, len(closes) / config.lookback_bars)

    return SymbolAlphaFeatures(
        symbol=symbol,
        short_momo=short_momo,
        long_momo=long_momo,
        volatility=max(SAFE_VOL_FLOOR, volatility),
        trend_consistency=trend_consistency,
        liquidity_score=liq_score,
        spread_quality=spread_quality,
        router_score=router_score,
        category_hint=category,
        volume_24h=volume_24h,
        price=price,
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_alpha_score(
    features: SymbolAlphaFeatures,
    weights: Dict[str, float],
) -> float:
    """
    Compute composite alpha score from features.

    Scoring formula:
        score = sum(weight_i * normalized_feature_i)

    All features are normalized to [0, 1] range before weighting.

    Args:
        features: Extracted features
        weights: Feature weights

    Returns:
        Composite score [0, 1]
    """
    # Normalize momentum to [0, 1] from [-1, 1]
    norm_short_momo = (features.short_momo + 1) / 2
    norm_long_momo = (features.long_momo + 1) / 2

    # Inverse volatility (lower vol = higher score)
    # Typical crypto vol is 50-200% annualized
    # At 50% → inv_vol = 1.0, at 200% → inv_vol = 0.25
    inv_vol = 1.0 / max(0.5, features.volatility)
    norm_inv_vol = min(1.0, inv_vol)

    # Other features already [0, 1]
    norm_trend = features.trend_consistency
    norm_router = features.router_score
    norm_liq = features.liquidity_score

    # Weighted sum
    w = weights
    score = (
        w.get("short_momo", 0.15) * norm_short_momo
        + w.get("long_momo", 0.20) * norm_long_momo
        + w.get("inv_vol", 0.15) * norm_inv_vol
        + w.get("trend_consistency", 0.20) * norm_trend
        + w.get("router_score", 0.10) * norm_router
        + w.get("liq_quality", 0.20) * norm_liq
    )

    # Apply data quality discount
    score *= features.data_quality

    return max(0.0, min(1.0, score))


def apply_ema_smoothing(
    current_scores: Dict[str, float],
    previous_ema: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """
    Apply EMA smoothing to scores.

    Args:
        current_scores: New raw scores {symbol: score}
        previous_ema: Previous EMA scores {symbol: ema}
        alpha: Smoothing factor (0 = full smoothing, 1 = no smoothing)

    Returns:
        Updated EMA scores
    """
    result = {}
    for symbol, score in current_scores.items():
        prev = previous_ema.get(symbol, score)
        ema = alpha * score + (1 - alpha) * prev
        result[symbol] = ema
    return result


# ---------------------------------------------------------------------------
# Candidate Selection
# ---------------------------------------------------------------------------


def generate_candidate_reason(features: SymbolAlphaFeatures) -> str:
    """
    Generate a human-readable reason for candidacy.

    Args:
        features: Symbol features

    Returns:
        Reason string
    """
    reasons = []

    # Momentum
    if features.long_momo > 0.15:
        reasons.append(f"strong momentum (+{features.long_momo*100:.1f}%)")
    elif features.long_momo > 0.05:
        reasons.append(f"positive momentum (+{features.long_momo*100:.1f}%)")

    # Trend
    if features.trend_consistency > 0.6:
        reasons.append(f"consistent trend ({features.trend_consistency*100:.0f}%)")

    # Liquidity
    if features.liquidity_score > 0.7:
        reasons.append("good liquidity")

    # Category
    if features.category_hint in ("L1_MAJOR", "L1_ALT"):
        reasons.append(f"category: {features.category_hint}")

    if not reasons:
        reasons.append("composite score above threshold")

    return "; ".join(reasons)


def select_candidates(
    scores: Dict[str, float],
    ema_scores: Dict[str, float],
    features_map: Dict[str, SymbolAlphaFeatures],
    config: AlphaMinerConfig,
    current_universe: List[str],
    previous_candidates: List[AlphaMinerCandidate],
) -> List[AlphaMinerCandidate]:
    """
    Select top candidates based on EMA scores.

    Args:
        scores: Raw scores {symbol: score}
        ema_scores: EMA-smoothed scores {symbol: ema}
        features_map: Features by symbol
        config: Configuration
        current_universe: Symbols already in universe
        previous_candidates: Previous candidate list (for first_seen tracking)

    Returns:
        List of top candidates
    """
    # Build first_seen map
    first_seen_map = {}
    for c in previous_candidates:
        first_seen_map[c.symbol] = c.first_seen_ts

    now = time.time()

    # Filter by threshold and sort by EMA score
    eligible = []
    for symbol, ema in ema_scores.items():
        if ema < config.score_threshold:
            continue
        if symbol not in features_map:
            continue
        eligible.append((symbol, ema))

    # Sort by EMA score descending
    eligible.sort(key=lambda x: x[1], reverse=True)

    # Take top_k
    candidates = []
    for symbol, ema in eligible[: config.top_k]:
        feat = features_map[symbol]
        raw_score = scores.get(symbol, 0.0)
        reason = generate_candidate_reason(feat)
        first_seen = first_seen_map.get(symbol, now)

        cand = AlphaMinerCandidate(
            symbol=symbol,
            score=raw_score,
            ema_score=ema,
            features=feat,
            reason=reason,
            in_universe=symbol in current_universe,
            first_seen_ts=first_seen,
        )
        candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def should_run_miner(
    cycle_count: int,
    config: AlphaMinerConfig,
    previous_state: AlphaMinerState,
) -> bool:
    """
    Check if miner should run this cycle.

    Args:
        cycle_count: Current edge_scanner cycle
        config: Miner configuration
        previous_state: Previous state

    Returns:
        True if should run
    """
    if not config.enabled:
        return False
    # Run every N cycles
    if cycle_count % config.run_interval_cycles != 0:
        return False
    return True


def run_alpha_miner_step(
    config: Optional[AlphaMinerConfig] = None,
    state_path: Optional[Path] = None,
    fetch_orderbook: bool = False,
    dry_run: bool = False,
) -> AlphaMinerState:
    """
    Run a single alpha miner scan.

    This is an expensive operation that:
    1. Fetches all exchange symbols
    2. Filters by exclusion patterns and current universe
    3. Extracts features for each candidate (requires API calls)
    4. Computes scores and applies EMA smoothing
    5. Selects top candidates
    6. Saves state

    Args:
        config: Optional config override
        state_path: Optional state path override
        fetch_orderbook: Whether to fetch orderbook for spread/slippage
        dry_run: If True, don't save state

    Returns:
        Updated AlphaMinerState
    """
    cfg = config or load_alpha_miner_config()
    if not cfg.enabled:
        _LOG.debug("alpha_miner: disabled, skipping")
        return AlphaMinerState(notes="disabled")

    start_ts = time.time()
    errors: List[str] = []

    # Load previous state
    prev_state = load_alpha_miner_state(state_path)

    # Get all symbols
    all_symbols = get_all_exchange_symbols()
    if not all_symbols:
        _LOG.warning("alpha_miner: no symbols from exchange")
        return AlphaMinerState(
            updated_ts=time.time(),
            cycle_count=prev_state.cycle_count + 1,
            notes="no_symbols",
            errors=["no_symbols_from_exchange"],
        )

    # Load current universe
    current_universe = load_current_universe()

    # Filter symbols
    symbols_to_scan = filter_excluded_symbols(
        all_symbols,
        cfg.exclude_patterns,
        current_universe,
    )

    _LOG.info(
        "alpha_miner: scanning %d symbols (filtered from %d, universe=%d)",
        len(symbols_to_scan),
        len(all_symbols),
        len(current_universe),
    )

    # Extract features
    features_map: Dict[str, SymbolAlphaFeatures] = {}
    scores: Dict[str, float] = {}

    for symbol in symbols_to_scan:
        try:
            # Get orderbook if requested
            orderbook = None
            if fetch_orderbook:
                try:
                    from execution.exchange_utils import get_orderbook

                    orderbook = get_orderbook(symbol, limit=5)
                except Exception:
                    pass

            # Get 24h volume (approximate from klines)
            volume_24h = 0.0
            price = 0.0
            try:
                from execution.exchange_utils import get_klines

                klines = get_klines(symbol, cfg.interval, cfg.lookback_bars)
                if klines and len(klines) > 0:
                    # Sum volume from last 6 bars (assuming 4h = 24h)
                    vol_sum = sum(float(row[5]) for row in klines[-6:] if len(row) > 5)
                    price = float(klines[-1][4]) if len(klines[-1]) > 4 else 0.0
                    volume_24h = vol_sum * price  # Approximate USD volume
                else:
                    klines = []
            except Exception as exc:
                _LOG.debug("alpha_miner_klines_failed symbol=%s: %s", symbol, exc)
                klines = []

            # Extract features
            feat = extract_symbol_features(
                symbol=symbol,
                config=cfg,
                klines=klines if klines else None,
                orderbook=orderbook,
                volume_24h=volume_24h,
                price=price,
            )

            if feat is None:
                continue

            # Filter by minimum liquidity
            if feat.volume_24h < cfg.min_liquidity_usd:
                continue

            features_map[symbol] = feat

            # Compute score
            score = compute_alpha_score(feat, cfg.weights)
            scores[symbol] = score

        except Exception as exc:
            _LOG.debug("alpha_miner_feature_failed symbol=%s: %s", symbol, exc)
            errors.append(f"{symbol}: {exc}")

    # Apply EMA smoothing
    ema_scores = apply_ema_smoothing(scores, prev_state.ema_scores, cfg.smoothing_alpha)

    # Select candidates
    candidates = select_candidates(
        scores=scores,
        ema_scores=ema_scores,
        features_map=features_map,
        config=cfg,
        current_universe=current_universe,
        previous_candidates=prev_state.candidates,
    )

    elapsed = time.time() - start_ts
    notes = f"scanned {len(symbols_to_scan)} in {elapsed:.1f}s"

    # Build state
    state = AlphaMinerState(
        updated_ts=time.time(),
        cycle_count=prev_state.cycle_count + 1,
        symbols_scanned=len(symbols_to_scan),
        symbols_passed_filter=len(features_map),
        candidates=candidates,
        ema_scores=ema_scores,
        notes=notes,
        errors=errors,
    )

    # Save state
    if not dry_run:
        save_alpha_miner_state(state, state_path)

    _LOG.info(
        "alpha_miner: found %d candidates from %d passed (%d scanned) in %.1fs",
        len(candidates),
        len(features_map),
        len(symbols_to_scan),
        elapsed,
    )

    return state


# ---------------------------------------------------------------------------
# View Functions (for dashboard)
# ---------------------------------------------------------------------------


def get_top_candidates(
    state: Optional[AlphaMinerState] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top candidates for dashboard display.

    Args:
        state: Optional state (loads from file if not provided)
        top_k: Number of candidates to return

    Returns:
        List of candidate dicts
    """
    if state is None:
        state = load_alpha_miner_state()
    return [c.to_dict() for c in state.candidates[:top_k]]


def get_miner_summary(
    state: Optional[AlphaMinerState] = None,
) -> Dict[str, Any]:
    """
    Get summary stats for dashboard.

    Args:
        state: Optional state

    Returns:
        Summary dict
    """
    if state is None:
        state = load_alpha_miner_state()

    now = time.time()
    age_s = now - state.updated_ts if state.updated_ts > 0 else 0

    return {
        "updated_ts": state.updated_ts,
        "age_s": round(age_s, 1),
        "cycle_count": state.cycle_count,
        "symbols_scanned": state.symbols_scanned,
        "symbols_passed_filter": state.symbols_passed_filter,
        "num_candidates": len(state.candidates),
        "top_candidate": state.candidates[0].symbol if state.candidates else None,
        "top_score": round(state.candidates[0].ema_score, 4) if state.candidates else 0,
        "notes": state.notes,
    }


def get_candidates_by_category(
    state: Optional[AlphaMinerState] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group candidates by inferred category.

    Args:
        state: Optional state

    Returns:
        Dict of category → candidate list
    """
    if state is None:
        state = load_alpha_miner_state()

    result: Dict[str, List[Dict[str, Any]]] = {}
    for cand in state.candidates:
        cat = cand.features.category_hint
        if cat not in result:
            result[cat] = []
        result[cat].append(cand.to_dict())

    return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "AlphaMinerConfig",
    "load_alpha_miner_config",
    # Features
    "SymbolAlphaFeatures",
    "extract_symbol_features",
    # Scoring
    "compute_alpha_score",
    "apply_ema_smoothing",
    # Candidates
    "AlphaMinerCandidate",
    "select_candidates",
    # State
    "AlphaMinerState",
    "load_alpha_miner_state",
    "save_alpha_miner_state",
    # Runner
    "should_run_miner",
    "run_alpha_miner_step",
    # Views
    "get_top_candidates",
    "get_miner_summary",
    "get_candidates_by_category",
    # Helpers
    "get_all_exchange_symbols",
    "load_current_universe",
    "filter_excluded_symbols",
    "infer_category",
    "compute_momentum",
    "compute_volatility",
    "compute_trend_consistency",
    "compute_liquidity_score",
    "compute_spread_quality",
    "generate_candidate_reason",
]
