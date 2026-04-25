"""
P6 Candidate 1 — Regime-Permissioned Simple Rules (v7.9-P6A)

Four candidate_ids evaluated simultaneously in shadow:
  C1_TREND_NORMAL   — EMA crossover + TREND_UP→LONG, TREND_DOWN→SHORT
  C1_TREND_INVERTED — EMA crossover + TREND_UP→SHORT, TREND_DOWN→LONG
  C1_MR_NORMAL      — z-score extremes + range_position + MEAN_REVERT→reverting side
  C1_MR_INVERTED    — same triggers, opposite polarity

Polarity is a shadow-evaluation dimension, NOT a runtime tuning knob.

Signal suppression: max ONE canonical signal per symbol per loop.
Priority: regime-matching trend → regime-matching MR → none.
Suppressed alternatives are logged for audit.

Thresholds are frozen before evaluation and must not be modified during
the evaluation window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("p6_simple_rules")


# ── Frozen thresholds (set once, do not touch during evaluation) ─────────

@dataclass(frozen=True)
class C1Config:
    """Frozen configuration for Candidate 1 simple rules."""

    # Trend rule thresholds
    ema_fast_period: int = 15
    ema_slow_period: int = 50
    trend_r2_min: float = 0.5

    # Mean-reversion rule thresholds
    zscore_lookback: int = 48
    zscore_entry: float = 1.5
    range_pos_long_max: float = 0.2   # range_position < this → long trigger
    range_pos_short_min: float = 0.8  # range_position > this → short trigger

    # Conviction proxy coefficients (frozen, logged)
    # Trend: conv = clip(0.5 + k1*(r2 - 0.5) + k2*|ema_spread|/price, 0, 1)
    conv_k1: float = 0.30   # weight on trend_r2 deviation from 0.5
    conv_k2: float = 0.20   # weight on normalized EMA spread
    # MR: conv = clip(0.5 + k3*(|z|/z_cap) + k4*|range_pos - 0.5|, 0, 1)
    conv_k3: float = 0.25   # weight on normalized z-score magnitude
    conv_k4: float = 0.20   # weight on range_position extremity
    conv_z_cap: float = 4.0 # z-score cap for normalization

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = C1Config()


# ── P6Signal — shared signal type ────────────────────────────────────────

@dataclass
class P6Signal:
    """A candidate entry signal for shadow evaluation."""

    candidate_id: str          # e.g. "C1_TREND_NORMAL"
    candidate_family: str      # "C1" or "C2"
    rule_name: str             # e.g. "trend_ema_crossover", "mr_zscore_range"
    symbol: str
    side: str                  # "LONG" or "SHORT"
    polarity: str              # "normal" or "inverted"
    regime: str                # Sentinel-X regime at signal time
    feature_snapshot: Dict[str, Any]  # Input features for replay
    conviction: float = 0.5    # Proxy conviction for bridge lookup
    ts: float = 0.0           # Unix timestamp
    selected_for_eval: bool = True
    region: str = ""           # For C2 only

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Technical indicators ─────────────────────────────────────────────────

def compute_ema(prices: Sequence[float], period: int) -> List[float]:
    """Exponential moving average. Returns list same length as prices.

    First (period-1) values use expanding EMA.
    """
    if len(prices) == 0:
        return []
    alpha = 2.0 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(ema[-1] + alpha * (p - ema[-1]))
    return ema


def compute_zscore(prices: Sequence[float], lookback: int) -> float:
    """Z-score of the last price relative to the lookback window mean/std."""
    if len(prices) < 2:
        return 0.0
    window = list(prices[-lookback:]) if len(prices) >= lookback else list(prices)
    mean = sum(window) / len(window)
    var = sum((x - mean) ** 2 for x in window) / len(window)
    std = var ** 0.5
    if std < 1e-12:
        return 0.0
    return (prices[-1] - mean) / std


# ── Conviction proxy ─────────────────────────────────────────────────────

def compute_trend_conviction(
    trend_r2: float,
    ema_fast: float,
    ema_slow: float,
    price: float,
    config: C1Config,
) -> float:
    """Deterministic conviction proxy for trend signals.

    conv = clip(0.5 + k1*(r2 - 0.5) + k2*|ema_spread|/price, 0, 1)
    """
    if price < 1e-12:
        return 0.5
    r2_term = config.conv_k1 * (trend_r2 - 0.5)
    spread_term = config.conv_k2 * abs(ema_fast - ema_slow) / price
    return max(0.0, min(1.0, 0.5 + r2_term + spread_term))


def compute_mr_conviction(
    zscore: float,
    range_pos: float,
    config: C1Config,
) -> float:
    """Deterministic conviction proxy for mean-reversion signals.

    conv = clip(0.5 + k3*(|z|/z_cap) + k4*|range_pos - 0.5|, 0, 1)
    """
    z_norm = min(abs(zscore), config.conv_z_cap) / config.conv_z_cap
    z_term = config.conv_k3 * z_norm
    rp_term = config.conv_k4 * abs(range_pos - 0.5)
    return max(0.0, min(1.0, 0.5 + z_term + rp_term))


# ── Rule generators ──────────────────────────────────────────────────────

def _trend_signals(
    symbol: str,
    closes: Sequence[float],
    regime: str,
    features: Dict[str, float],
    config: C1Config,
    ts: float,
) -> List[P6Signal]:
    """Generate trend EMA-crossover signals for both polarities."""
    if len(closes) < config.ema_slow_period + 1:
        return []

    ema_fast = compute_ema(closes, config.ema_fast_period)
    ema_slow = compute_ema(closes, config.ema_slow_period)

    fast_last = ema_fast[-1]
    slow_last = ema_slow[-1]
    trend_r2 = features.get("trend_r2", 0.0)

    # No crossover or weak trend → no signal
    if abs(fast_last - slow_last) < 1e-12:
        return []
    if trend_r2 < config.trend_r2_min:
        return []

    crossover_bullish = fast_last > slow_last

    # Compute proxy conviction
    price = closes[-1] if closes else 1.0
    conv = compute_trend_conviction(trend_r2, fast_last, slow_last, price, config)

    snap = {
        "ema_fast": round(fast_last, 6),
        "ema_slow": round(slow_last, 6),
        "trend_r2": round(trend_r2, 4),
        "trend_slope": round(features.get("trend_slope", 0.0), 8),
        "regime": regime,
        "conviction_proxy": round(conv, 6),
    }

    signals: List[P6Signal] = []

    # ── Normal polarity ──
    # TREND_UP + bullish crossover → LONG; TREND_DOWN + bearish → SHORT
    if regime == "TREND_UP" and crossover_bullish:
        signals.append(P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol=symbol,
            side="LONG", polarity="normal", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))
    elif regime == "TREND_DOWN" and not crossover_bullish:
        signals.append(P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol=symbol,
            side="SHORT", polarity="normal", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))

    # ── Inverted polarity ──
    # TREND_UP + bullish crossover → SHORT; TREND_DOWN + bearish → LONG
    if regime == "TREND_UP" and crossover_bullish:
        signals.append(P6Signal(
            candidate_id="C1_TREND_INVERTED", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol=symbol,
            side="SHORT", polarity="inverted", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))
    elif regime == "TREND_DOWN" and not crossover_bullish:
        signals.append(P6Signal(
            candidate_id="C1_TREND_INVERTED", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol=symbol,
            side="LONG", polarity="inverted", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))

    return signals


def _mr_signals(
    symbol: str,
    closes: Sequence[float],
    regime: str,
    features: Dict[str, float],
    config: C1Config,
    ts: float,
) -> List[P6Signal]:
    """Generate mean-reversion z-score + range_position signals for both polarities."""
    if regime != "MEAN_REVERT":
        return []

    zscore = compute_zscore(closes, config.zscore_lookback)
    range_pos = features.get("range_position", 0.5)

    # Compute proxy conviction
    conv = compute_mr_conviction(zscore, range_pos, config)

    snap = {
        "zscore": round(zscore, 4),
        "range_position": round(range_pos, 4),
        "mean_reversion_score": round(features.get("mean_reversion_score", 0.0), 4),
        "regime": regime,
        "conviction_proxy": round(conv, 6),
    }

    signals: List[P6Signal] = []

    # Check for extreme z-score conditions
    long_trigger = zscore < -config.zscore_entry and range_pos < config.range_pos_long_max
    short_trigger = zscore > config.zscore_entry and range_pos > config.range_pos_short_min

    if long_trigger:
        # Normal: low z-score + low range → LONG (mean reverts up)
        signals.append(P6Signal(
            candidate_id="C1_MR_NORMAL", candidate_family="C1",
            rule_name="mr_zscore_range", symbol=symbol,
            side="LONG", polarity="normal", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))
        # Inverted: same trigger → SHORT
        signals.append(P6Signal(
            candidate_id="C1_MR_INVERTED", candidate_family="C1",
            rule_name="mr_zscore_range", symbol=symbol,
            side="SHORT", polarity="inverted", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))
    elif short_trigger:
        # Normal: high z-score + high range → SHORT (mean reverts down)
        signals.append(P6Signal(
            candidate_id="C1_MR_NORMAL", candidate_family="C1",
            rule_name="mr_zscore_range", symbol=symbol,
            side="SHORT", polarity="normal", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))
        # Inverted: same trigger → LONG
        signals.append(P6Signal(
            candidate_id="C1_MR_INVERTED", candidate_family="C1",
            rule_name="mr_zscore_range", symbol=symbol,
            side="LONG", polarity="inverted", regime=regime,
            feature_snapshot=snap, conviction=conv, ts=ts,
        ))

    return signals


# ── Signal suppression ───────────────────────────────────────────────────

def _suppress_per_symbol(
    signals: List[P6Signal],
) -> tuple[List[P6Signal], List[P6Signal]]:
    """Apply signal suppression: max ONE selected signal per (symbol, candidate_id).

    Priority: trend rule > MR rule.
    Returns (selected_signals, suppressed_signals).
    """
    RULE_PRIORITY = {"trend_ema_crossover": 0, "mr_zscore_range": 1}

    by_key: Dict[str, List[P6Signal]] = {}
    for sig in signals:
        key = f"{sig.symbol}:{sig.candidate_id}"
        by_key.setdefault(key, []).append(sig)

    selected: List[P6Signal] = []
    suppressed: List[P6Signal] = []

    for key, sigs in by_key.items():
        sigs.sort(key=lambda s: RULE_PRIORITY.get(s.rule_name, 99))
        winner = sigs[0]
        winner.selected_for_eval = True
        selected.append(winner)
        for loser in sigs[1:]:
            loser.selected_for_eval = False
            suppressed.append(loser)

    return selected, suppressed


# ── Main entry point ─────────────────────────────────────────────────────

def generate_simple_rule_signals(
    closes: Sequence[float],
    sentinel_features: Dict[str, float],
    regime: str,
    symbol: str,
    config: Optional[C1Config] = None,
    ts: Optional[float] = None,
) -> tuple[List[P6Signal], List[P6Signal]]:
    """Generate all C1 candidate signals for a single symbol.

    Args:
        closes: Close prices (chronological, most recent last).
        sentinel_features: Dict of Sentinel-X features for this symbol.
        regime: Current primary regime from Sentinel-X.
        symbol: Trading pair (e.g. "BTCUSDT").
        config: Frozen thresholds. Uses DEFAULT_CONFIG if None.
        ts: Timestamp. Uses time.time() if None.

    Returns:
        (selected_signals, suppressed_signals)
        selected: at most one per (symbol, candidate_id)
        suppressed: alternatives that lost priority
    """
    if config is None:
        config = DEFAULT_CONFIG
    if ts is None:
        ts = time.time()

    all_signals: List[P6Signal] = []

    # Generate trend signals (fire on TREND_UP / TREND_DOWN)
    all_signals.extend(_trend_signals(
        symbol, closes, regime, sentinel_features, config, ts,
    ))

    # Generate MR signals (fire on MEAN_REVERT)
    all_signals.extend(_mr_signals(
        symbol, closes, regime, sentinel_features, config, ts,
    ))

    if not all_signals:
        return [], []

    return _suppress_per_symbol(all_signals)
