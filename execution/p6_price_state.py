"""
P6 Candidate 2 — Price-State / Region Surface (v7.9-P6A)

Classifies current price state into mutually-exclusive, exhaustive regions
using Sentinel-X features, then generates directional signals per region.

Regions (frozen if/elif precedence — do NOT reorder):
  1. low_range      — range_position < threshold_low
  2. high_range     — range_position > threshold_high
  3. vol_compressed — vol_regime_z < vol_compress_z
  4. vol_expanded   — vol_regime_z > vol_expand_z
  5. center         — everything else (catch-all, guarantees exhaustive)

Each region emits signals with BOTH polarities for shadow evaluation:
  C2_REGION_NORMAL   — directional hypothesis based on region meaning
  C2_REGION_INVERTED — opposite polarity for falsification test

Thresholds are frozen before evaluation and must not be modified during
the evaluation window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from execution.p6_simple_rules import P6Signal

LOG = logging.getLogger("p6_price_state")


# ── Frozen thresholds ────────────────────────────────────────────────────

@dataclass(frozen=True)
class C2Config:
    """Frozen configuration for Candidate 2 price-state classifier."""

    # Region boundaries (applied in if/elif order — do NOT reorder)
    range_pos_low: float = 0.20      # range_position < this → low_range
    range_pos_high: float = 0.80     # range_position > this → high_range
    vol_compress_z: float = -1.0     # vol_regime_z < this → vol_compressed
    vol_expand_z: float = 1.0        # vol_regime_z > this → vol_expanded

    # Minimum data quality to emit signal
    min_data_quality: float = 0.5

    # Conviction proxy: frozen region base scale
    conv_low_range: float = 0.58
    conv_high_range: float = 0.58
    conv_vol_compressed: float = 0.56
    conv_vol_expanded: float = 0.52
    conv_center: float = 0.50
    # Deterministic bump from |range_position - 0.5| (capped)
    conv_rp_bump_k: float = 0.10     # coefficient for range_position extremity bump
    conv_bump_cap: float = 0.08      # max bump magnitude

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = C2Config()

# Canonical region list — order matches classifier precedence
REGIONS = ("low_range", "high_range", "vol_compressed", "vol_expanded", "center")


# ── Conviction proxy ─────────────────────────────────────────────────────

def compute_region_conviction(
    region: str,
    features: Dict[str, float],
    config: C2Config,
) -> float:
    """Deterministic conviction proxy for price-state signals.

    Base conviction is a frozen per-region constant. A small deterministic
    bump is added from |range_position - 0.5| to allow the bridge to
    express band differences.
    """
    base_map = {
        "low_range": config.conv_low_range,
        "high_range": config.conv_high_range,
        "vol_compressed": config.conv_vol_compressed,
        "vol_expanded": config.conv_vol_expanded,
        "center": config.conv_center,
    }
    base = base_map.get(region, config.conv_center)
    rp = features.get("range_position", 0.5)
    bump = config.conv_rp_bump_k * abs(rp - 0.5)
    bump = min(bump, config.conv_bump_cap)
    return max(0.0, min(1.0, base + bump))


# ── Region classifier ────────────────────────────────────────────────────

def classify_region(
    features: Dict[str, float],
    config: C2Config,
) -> str:
    """Classify price state into a mutually-exclusive, exhaustive region.

    Precedence is frozen. Do NOT reorder.
    """
    range_pos = features.get("range_position", 0.5)
    vol_z = features.get("vol_regime_z", 0.0)

    # 1. low_range
    if range_pos < config.range_pos_low:
        return "low_range"
    # 2. high_range
    if range_pos > config.range_pos_high:
        return "high_range"
    # 3. vol_compressed
    if vol_z < config.vol_compress_z:
        return "vol_compressed"
    # 4. vol_expanded
    if vol_z > config.vol_expand_z:
        return "vol_expanded"
    # 5. center (catch-all — guarantees exhaustive)
    return "center"


# ── Directional hypothesis per region ────────────────────────────────────

# Normal polarity mapping: region → (side, rationale)
# low_range  → LONG  (mean-revert up from bottom)
# high_range → SHORT (mean-revert down from top)
# vol_compressed → LONG  (breakout anticipation)
# vol_expanded  → SHORT (volatility fade)
# center     → None  (no directional edge expected)

_NORMAL_DIRECTION: Dict[str, Optional[str]] = {
    "low_range": "LONG",
    "high_range": "SHORT",
    "vol_compressed": "LONG",
    "vol_expanded": "SHORT",
    "center": None,       # center has no directional hypothesis
}


# ── Signal generation ────────────────────────────────────────────────────

def generate_price_state_signals(
    sentinel_features: Dict[str, float],
    regime: str,
    symbol: str,
    config: Optional[C2Config] = None,
    ts: Optional[float] = None,
) -> List[P6Signal]:
    """Generate C2 candidate signals for a single symbol.

    Args:
        sentinel_features: Dict of Sentinel-X features.
        regime: Current primary regime.
        symbol: Trading pair.
        config: Frozen thresholds. Uses DEFAULT_CONFIG if None.
        ts: Timestamp. Uses time.time() if None.

    Returns:
        List of P6Signal (0 or 2 — normal + inverted, or empty if
        region is center or data quality insufficient).
    """
    if config is None:
        config = DEFAULT_CONFIG
    if ts is None:
        ts = time.time()

    # Data quality gate
    dq = sentinel_features.get("data_quality", 1.0)
    if dq < config.min_data_quality:
        LOG.debug("P6 C2 skip %s: data_quality=%.2f < %.2f", symbol, dq, config.min_data_quality)
        return []

    region = classify_region(sentinel_features, config)
    normal_side = _NORMAL_DIRECTION[region]

    if normal_side is None:
        # center region — no directional hypothesis, no signal
        return []

    inverted_side = "SHORT" if normal_side == "LONG" else "LONG"

    # Compute proxy conviction
    conv = compute_region_conviction(region, sentinel_features, config)

    # Pre-class continuous scores (logged even though classification is discrete)
    range_pos = sentinel_features.get("range_position", 0.5)
    vol_z = sentinel_features.get("vol_regime_z", 0.0)

    snap: Dict[str, Any] = {
        # Pre-class continuous measures
        "range_position": round(range_pos, 4),
        "vol_regime_z": round(vol_z, 4),
        "data_quality": round(dq, 4),
        # Region result
        "regime": regime,
        "region": region,
        "region_assigned": region,
        # Conviction provenance
        "conviction_proxy": round(conv, 6),
        # Region score vector (normalized distances to each boundary)
        "region_score_vector": {
            "low_range_dist": round(config.range_pos_low - range_pos, 4),
            "high_range_dist": round(range_pos - config.range_pos_high, 4),
            "vol_compress_dist": round(config.vol_compress_z - vol_z, 4),
            "vol_expand_dist": round(vol_z - config.vol_expand_z, 4),
        },
    }

    signals: List[P6Signal] = []

    # Normal polarity
    signals.append(P6Signal(
        candidate_id="C2_REGION_NORMAL",
        candidate_family="C2",
        rule_name=f"region_{region}",
        symbol=symbol,
        side=normal_side,
        polarity="normal",
        regime=regime,
        feature_snapshot=snap,
        conviction=conv,
        ts=ts,
        region=region,
    ))

    # Inverted polarity
    signals.append(P6Signal(
        candidate_id="C2_REGION_INVERTED",
        candidate_family="C2",
        rule_name=f"region_{region}",
        symbol=symbol,
        side=inverted_side,
        polarity="inverted",
        regime=regime,
        feature_snapshot=snap,
        conviction=conv,
        ts=ts,
        region=region,
    ))

    return signals
