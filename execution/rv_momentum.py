"""
v7.5_C1 — Cross-Asset Relative Momentum (RV-MOMO) Factor.

Provides relative momentum scoring across asset groups:
- BTC vs ETH pair momentum
- L1 vs ALT basket momentum
- Meme vs Rest basket momentum
- Per-symbol relative strength

The rv_score is integrated into hybrid scoring to prefer
strong coins vs weak coins, not just good setups in isolation.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RvConfig:
    """Configuration for relative momentum factor."""

    enabled: bool = True
    lookback_bars: int = 48
    half_life_bars: int = 24
    normalize_mode: str = "zscore"  # "zscore" or "rank"
    btc_vs_eth_weight: float = 0.35
    l1_vs_alt_weight: float = 0.35
    meme_vs_rest_weight: float = 0.20
    per_symbol_weight: float = 0.10
    max_abs_score: float = 1.0
    # Hybrid integration
    hybrid_weight: float = 0.10  # Weight in hybrid score composition


@dataclass
class RvSymbolScore:
    """Per-symbol relative momentum score."""

    symbol: str
    score: float  # Final normalized score in [-max_abs, max_abs]
    raw_score: float = 0.0  # Pre-normalization score
    baskets: List[str] = field(default_factory=list)  # Which baskets symbol belongs to


@dataclass
class RvSnapshot:
    """Snapshot of relative momentum state."""

    per_symbol: Dict[str, RvSymbolScore] = field(default_factory=dict)
    btc_vs_eth_spread: float = 0.0
    l1_vs_alt_spread: float = 0.0
    meme_vs_rest_spread: float = 0.0
    updated_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state publishing."""
        return {
            "updated_ts": self.updated_ts,
            "per_symbol": {
                sym: {
                    "score": round(entry.score, 4),
                    "raw_score": round(entry.raw_score, 4),
                    "baskets": entry.baskets,
                }
                for sym, entry in self.per_symbol.items()
            },
            "spreads": {
                "btc_vs_eth": round(self.btc_vs_eth_spread, 4),
                "l1_vs_alt": round(self.l1_vs_alt_spread, 4),
                "meme_vs_rest": round(self.meme_vs_rest_spread, 4),
            },
        }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

_BASKETS_PATH = Path("config/rv_momo_baskets.json")


def load_rv_config(strategy_config: Mapping[str, Any] | None = None) -> RvConfig:
    """
    Load RV momentum config from strategy_config.

    Args:
        strategy_config: Strategy config dict, or None to load from file.

    Returns:
        RvConfig with values from config or defaults.
    """
    if strategy_config is None:
        try:
            cfg_path = Path("config/strategy_config.json")
            if cfg_path.exists():
                strategy_config = json.loads(cfg_path.read_text())
            else:
                strategy_config = {}
        except Exception:
            strategy_config = {}

    rv_cfg = strategy_config.get("rv_momentum", {})

    return RvConfig(
        enabled=rv_cfg.get("enabled", True),
        lookback_bars=rv_cfg.get("lookback_bars", 48),
        half_life_bars=rv_cfg.get("half_life_bars", 24),
        normalize_mode=rv_cfg.get("normalize_mode", "zscore"),
        btc_vs_eth_weight=rv_cfg.get("btc_vs_eth_weight", 0.35),
        l1_vs_alt_weight=rv_cfg.get("l1_vs_alt_weight", 0.35),
        meme_vs_rest_weight=rv_cfg.get("meme_vs_rest_weight", 0.20),
        per_symbol_weight=rv_cfg.get("per_symbol_weight", 0.10),
        max_abs_score=rv_cfg.get("max_abs_score", 1.0),
        hybrid_weight=rv_cfg.get("hybrid_weight", 0.10),
    )


def load_baskets_config(path: Path | None = None) -> Dict[str, Any]:
    """
    Load basket definitions from config file.

    Args:
        path: Path to baskets config, or None to use default.

    Returns:
        Dict with 'pairs' and 'baskets' keys.
    """
    if path is None:
        path = _BASKETS_PATH

    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass

    # Return default structure
    return {
        "pairs": {
            "btc_vs_eth": {
                "long": "BTCUSDT",
                "short": "ETHUSDT",
            }
        },
        "baskets": {
            "l1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "alts": ["LTCUSDT", "LINKUSDT", "SUIUSDT"],
            "meme": ["DOGEUSDT", "WIFUSDT"],
        },
    }


def get_symbol_baskets(symbol: str, baskets_cfg: Dict[str, Any]) -> List[str]:
    """
    Get list of basket names a symbol belongs to.

    Args:
        symbol: Symbol to check.
        baskets_cfg: Baskets config dict.

    Returns:
        List of basket names (e.g., ["l1", "meme"]).
    """
    symbol_upper = symbol.upper()
    baskets = baskets_cfg.get("baskets", {})
    result = []

    for basket_name, symbols in baskets.items():
        if symbol_upper in [s.upper() for s in symbols]:
            result.append(basket_name)

    return result


# ---------------------------------------------------------------------------
# Returns Loading
# ---------------------------------------------------------------------------


def load_returns_for_symbols(
    symbols: List[str],
    lookback_bars: int,
    half_life_bars: int = 24,
) -> Dict[str, np.ndarray]:
    """
    Load returns data for symbols.

    This is a simplified implementation that attempts to load from
    the OHLCV cache. In production, this would integrate with the
    existing data pipeline.

    Args:
        symbols: List of symbols to load.
        lookback_bars: Number of bars to load.
        half_life_bars: Half-life for EWMA smoothing.

    Returns:
        Dict mapping symbol -> np.array of returns.
    """
    returns: Dict[str, np.ndarray] = {}

    for symbol in symbols:
        try:
            # Try to load from OHLCV cache
            ohlcv_path = Path(f"data/ohlcv/{symbol}_1h.json")
            if ohlcv_path.exists():
                data = json.loads(ohlcv_path.read_text())
                closes = [float(bar.get("close", bar.get("c", 0))) for bar in data[-lookback_bars:]]
                if len(closes) >= 2:
                    # Compute log returns
                    prices = np.array(closes)
                    log_returns = np.diff(np.log(prices + 1e-10))

                    # Apply EWMA smoothing
                    alpha = 1 - math.exp(-math.log(2) / half_life_bars)
                    ewma_returns = _ewma(log_returns, alpha)
                    returns[symbol.upper()] = ewma_returns
                    continue
        except Exception:
            pass

        # Fallback: return zeros if no data
        returns[symbol.upper()] = np.zeros(max(1, lookback_bars - 1))

    return returns


def _ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute exponentially weighted moving average.

    Args:
        data: Input array.
        alpha: Smoothing factor (0 < alpha <= 1).

    Returns:
        EWMA of input data.
    """
    if len(data) == 0:
        return data

    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


# ---------------------------------------------------------------------------
# Pair Relative Momentum
# ---------------------------------------------------------------------------


def compute_pair_relative_momentum(
    long_symbol: str,
    short_symbol: str,
    returns: Dict[str, np.ndarray],
) -> float:
    """
    Compute relative momentum of long vs short symbol.

    Args:
        long_symbol: Symbol expected to outperform.
        short_symbol: Symbol expected to underperform.
        returns: Dict of symbol -> returns array.

    Returns:
        Relative momentum (positive = long outperforming short).
    """
    long_returns = returns.get(long_symbol.upper(), np.zeros(1))
    short_returns = returns.get(short_symbol.upper(), np.zeros(1))

    # Align lengths
    min_len = min(len(long_returns), len(short_returns))
    if min_len == 0:
        return 0.0

    long_ret = long_returns[-min_len:]
    short_ret = short_returns[-min_len:]

    # Compute mean relative return
    relative_return = np.mean(long_ret - short_ret)

    return float(relative_return)


# ---------------------------------------------------------------------------
# Basket Relative Momentum
# ---------------------------------------------------------------------------


def compute_basket_relative_momentum(
    group_a: List[str],
    group_b: List[str],
    returns: Dict[str, np.ndarray],
) -> float:
    """
    Compute relative momentum of group A vs group B.

    Uses equal-weight average of available symbols in each group.

    Args:
        group_a: Symbols in first group (expected to outperform).
        group_b: Symbols in second group (expected to underperform).
        returns: Dict of symbol -> returns array.

    Returns:
        Relative momentum (positive = group A outperforming group B).
    """
    # Get returns for each group
    a_returns = [returns.get(s.upper(), np.zeros(1)) for s in group_a]
    b_returns = [returns.get(s.upper(), np.zeros(1)) for s in group_b]

    # Filter out empty arrays
    a_returns = [r for r in a_returns if len(r) > 0 and np.any(r != 0)]
    b_returns = [r for r in b_returns if len(r) > 0 and np.any(r != 0)]

    if not a_returns or not b_returns:
        return 0.0

    # Compute average return for each group
    # First, find minimum common length
    min_len_a = min(len(r) for r in a_returns)
    min_len_b = min(len(r) for r in b_returns)
    min_len = min(min_len_a, min_len_b)

    if min_len == 0:
        return 0.0

    # Truncate and average
    a_avg = np.mean([r[-min_len:] for r in a_returns], axis=0)
    b_avg = np.mean([r[-min_len:] for r in b_returns], axis=0)

    # Relative momentum is mean of difference
    relative_return = np.mean(a_avg - b_avg)

    return float(relative_return)


# ---------------------------------------------------------------------------
# Score Normalization
# ---------------------------------------------------------------------------


def normalize_scores(
    raw_scores: Dict[str, float],
    mode: str = "zscore",
    max_abs: float = 1.0,
) -> Dict[str, float]:
    """
    Normalize raw scores to [-max_abs, max_abs].

    Args:
        raw_scores: Dict of symbol -> raw score.
        mode: "zscore" or "rank".
        max_abs: Maximum absolute value for output.

    Returns:
        Dict of symbol -> normalized score.
    """
    if not raw_scores:
        return {}

    values = list(raw_scores.values())

    if mode == "rank":
        # Rank-based normalization
        sorted_items = sorted(raw_scores.items(), key=lambda x: x[1])
        n = len(sorted_items)
        if n == 1:
            return {sorted_items[0][0]: 0.0}

        normalized = {}
        for i, (sym, _) in enumerate(sorted_items):
            # Map rank to [-max_abs, max_abs]
            normalized[sym] = -max_abs + (2 * max_abs * i / (n - 1))
        return normalized

    else:  # zscore mode
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val < 1e-10:
            # All values are the same
            return {sym: 0.0 for sym in raw_scores}

        normalized = {}
        for sym, raw in raw_scores.items():
            z = (raw - mean_val) / std_val
            # Rescale using tanh to bound to [-max_abs, max_abs]
            normalized[sym] = max_abs * np.tanh(z / 2)

        return normalized


# ---------------------------------------------------------------------------
# Build RV Snapshot
# ---------------------------------------------------------------------------


def build_rv_snapshot(
    cfg: RvConfig,
    returns: Dict[str, np.ndarray] | None = None,
    baskets_cfg: Dict[str, Any] | None = None,
    symbols: List[str] | None = None,
) -> RvSnapshot:
    """
    Build complete RV momentum snapshot.

    Computes:
    - BTC vs ETH relative momentum
    - L1 vs ALT basket momentum
    - Meme vs Rest basket momentum
    - Per-symbol final rv_score

    Args:
        cfg: RV config.
        returns: Pre-loaded returns dict, or None to load.
        baskets_cfg: Basket definitions, or None to load.
        symbols: List of symbols to score, or None to use all in returns.

    Returns:
        RvSnapshot with per-symbol scores and spreads.
    """
    if baskets_cfg is None:
        baskets_cfg = load_baskets_config()

    # Collect all symbols we need
    all_basket_symbols = set()
    for basket_symbols in baskets_cfg.get("baskets", {}).values():
        all_basket_symbols.update(s.upper() for s in basket_symbols)

    for pair_cfg in baskets_cfg.get("pairs", {}).values():
        if "long" in pair_cfg:
            all_basket_symbols.add(pair_cfg["long"].upper())
        if "short" in pair_cfg:
            all_basket_symbols.add(pair_cfg["short"].upper())

    if symbols:
        all_basket_symbols.update(s.upper() for s in symbols)

    # Load returns if not provided
    if returns is None:
        returns = load_returns_for_symbols(
            list(all_basket_symbols),
            cfg.lookback_bars,
            cfg.half_life_bars,
        )

    # Get baskets
    baskets = baskets_cfg.get("baskets", {})
    l1_basket = [s.upper() for s in baskets.get("l1", [])]
    alts_basket = [s.upper() for s in baskets.get("alts", [])]
    meme_basket = [s.upper() for s in baskets.get("meme", [])]

    # Compute pair relative momentum (BTC vs ETH)
    pairs = baskets_cfg.get("pairs", {})
    btc_eth_pair = pairs.get("btc_vs_eth", {})
    btc_vs_eth_spread = compute_pair_relative_momentum(
        btc_eth_pair.get("long", "BTCUSDT"),
        btc_eth_pair.get("short", "ETHUSDT"),
        returns,
    )

    # Compute basket relative momentum
    l1_vs_alt_spread = compute_basket_relative_momentum(l1_basket, alts_basket, returns)

    # Meme vs Rest (rest = all symbols not in meme)
    meme_set = set(meme_basket)
    rest_symbols = [s for s in returns.keys() if s not in meme_set]
    meme_vs_rest_spread = compute_basket_relative_momentum(meme_basket, rest_symbols, returns)

    # Compute per-symbol raw scores
    raw_scores: Dict[str, float] = {}
    symbol_baskets: Dict[str, List[str]] = {}

    for symbol in returns.keys():
        symbol_upper = symbol.upper()
        raw_score = 0.0
        sym_baskets = get_symbol_baskets(symbol_upper, baskets_cfg)
        symbol_baskets[symbol_upper] = sym_baskets

        # BTC vs ETH contribution
        if symbol_upper == btc_eth_pair.get("long", "BTCUSDT").upper():
            raw_score += cfg.btc_vs_eth_weight * btc_vs_eth_spread
        elif symbol_upper == btc_eth_pair.get("short", "ETHUSDT").upper():
            raw_score -= cfg.btc_vs_eth_weight * btc_vs_eth_spread

        # L1 vs ALT contribution
        if symbol_upper in l1_basket:
            raw_score += cfg.l1_vs_alt_weight * l1_vs_alt_spread
        elif symbol_upper in alts_basket:
            raw_score -= cfg.l1_vs_alt_weight * l1_vs_alt_spread

        # Meme vs Rest contribution
        if symbol_upper in meme_basket:
            raw_score += cfg.meme_vs_rest_weight * meme_vs_rest_spread
        else:
            raw_score -= cfg.meme_vs_rest_weight * meme_vs_rest_spread

        # Per-symbol own momentum contribution
        sym_returns = returns.get(symbol_upper, np.zeros(1))
        if len(sym_returns) > 0:
            own_momentum = np.sum(sym_returns)  # Cumulative return
            raw_score += cfg.per_symbol_weight * own_momentum * 100  # Scale up

        raw_scores[symbol_upper] = raw_score

    # Normalize scores
    normalized_scores = normalize_scores(
        raw_scores,
        mode=cfg.normalize_mode,
        max_abs=cfg.max_abs_score,
    )

    # Build per-symbol entries
    per_symbol: Dict[str, RvSymbolScore] = {}
    for symbol, score in normalized_scores.items():
        per_symbol[symbol] = RvSymbolScore(
            symbol=symbol,
            score=score,
            raw_score=raw_scores.get(symbol, 0.0),
            baskets=symbol_baskets.get(symbol, []),
        )

    return RvSnapshot(
        per_symbol=per_symbol,
        btc_vs_eth_spread=btc_vs_eth_spread,
        l1_vs_alt_spread=l1_vs_alt_spread,
        meme_vs_rest_spread=meme_vs_rest_spread,
        updated_ts=time.time(),
    )


def get_rv_score(
    symbol: str,
    rv_snapshot: RvSnapshot | None = None,
    cfg: RvConfig | None = None,
) -> float:
    """
    Get RV momentum score for a symbol.

    Args:
        symbol: Symbol to get score for.
        rv_snapshot: Pre-computed snapshot, or None to compute fresh.
        cfg: RV config, or None to load.

    Returns:
        RV score in [-max_abs, max_abs], or 0.0 if not available.
    """
    if rv_snapshot is None:
        if cfg is None:
            cfg = load_rv_config()
        if not cfg.enabled:
            return 0.0
        rv_snapshot = build_rv_snapshot(cfg, symbols=[symbol])

    entry = rv_snapshot.per_symbol.get(symbol.upper())
    if entry:
        return entry.score
    return 0.0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "RvConfig",
    "RvSymbolScore",
    "RvSnapshot",
    "load_rv_config",
    "load_baskets_config",
    "get_symbol_baskets",
    "load_returns_for_symbols",
    "compute_pair_relative_momentum",
    "compute_basket_relative_momentum",
    "normalize_scores",
    "build_rv_snapshot",
    "get_rv_score",
]
