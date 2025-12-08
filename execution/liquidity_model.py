"""
v7.5_B1 — Liquidity Model

Classifies symbols into liquidity buckets (A/B/C) with bucket-specific
rules for spread thresholds and maker/taker bias.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

LOG = logging.getLogger("liquidity_model")

DEFAULT_CONFIG_PATH = Path("config/liquidity_buckets.json")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LiquidityBucketConfig:
    """Configuration for a liquidity bucket."""
    name: str
    max_spread_bps: float
    default_maker_bias: float
    symbols: list = field(default_factory=list)


@dataclass
class LiquidityModel:
    """
    Maps symbols to their liquidity bucket configuration.
    Provides default bucket for unknown symbols.
    """
    symbol_to_bucket: Dict[str, LiquidityBucketConfig] = field(default_factory=dict)
    default_bucket: LiquidityBucketConfig = field(default_factory=lambda: LiquidityBucketConfig(
        name="GENERIC",
        max_spread_bps=15.0,
        default_maker_bias=0.5,
    ))
    buckets: Dict[str, LiquidityBucketConfig] = field(default_factory=dict)

    def get_bucket(self, symbol: str) -> LiquidityBucketConfig:
        """Get the bucket for a symbol, or default if unknown."""
        return self.symbol_to_bucket.get(symbol.upper(), self.default_bucket)

    def get_bucket_name(self, symbol: str) -> str:
        """Get the bucket name for a symbol."""
        return self.get_bucket(symbol).name


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_liquidity_model(
    path: str | Path | None = None,
) -> LiquidityModel:
    """
    Load liquidity bucket configuration from JSON file.
    
    Args:
        path: Path to config file (default: config/liquidity_buckets.json)
        
    Returns:
        LiquidityModel with symbol -> bucket mapping
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    
    path = Path(path)
    
    # Default bucket if file missing
    default_bucket = LiquidityBucketConfig(
        name="GENERIC",
        max_spread_bps=15.0,
        default_maker_bias=0.5,
    )
    
    if not path.exists():
        LOG.warning("liquidity_buckets config not found at %s, using defaults", path)
        return LiquidityModel(
            symbol_to_bucket={},
            default_bucket=default_bucket,
            buckets={"GENERIC": default_bucket},
        )
    
    try:
        with path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        LOG.error("failed to load liquidity_buckets config: %s", exc)
        return LiquidityModel(
            symbol_to_bucket={},
            default_bucket=default_bucket,
            buckets={"GENERIC": default_bucket},
        )
    
    # Parse defaults
    defaults = cfg.get("defaults", {})
    default_bucket = LiquidityBucketConfig(
        name="GENERIC",
        max_spread_bps=float(defaults.get("max_spread_bps", 15.0)),
        default_maker_bias=float(defaults.get("default_maker_bias", 0.5)),
    )
    
    # Parse buckets
    buckets_cfg = cfg.get("buckets", {})
    symbol_to_bucket: Dict[str, LiquidityBucketConfig] = {}
    buckets: Dict[str, LiquidityBucketConfig] = {"GENERIC": default_bucket}
    
    for bucket_name, bucket_data in buckets_cfg.items():
        bucket = LiquidityBucketConfig(
            name=bucket_name,
            max_spread_bps=float(bucket_data.get("max_spread_bps", default_bucket.max_spread_bps)),
            default_maker_bias=float(bucket_data.get("default_maker_bias", default_bucket.default_maker_bias)),
            symbols=bucket_data.get("symbols", []),
        )
        buckets[bucket_name] = bucket
        
        # Map each symbol to this bucket
        for symbol in bucket_data.get("symbols", []):
            symbol_upper = symbol.upper()
            if symbol_upper in symbol_to_bucket:
                LOG.debug(
                    "duplicate symbol %s in buckets, overwriting with %s",
                    symbol_upper, bucket_name
                )
            symbol_to_bucket[symbol_upper] = bucket
    
    LOG.info(
        "loaded liquidity model: %d buckets, %d symbols mapped",
        len(buckets), len(symbol_to_bucket)
    )
    
    return LiquidityModel(
        symbol_to_bucket=symbol_to_bucket,
        default_bucket=default_bucket,
        buckets=buckets,
    )


# ---------------------------------------------------------------------------
# Singleton / Cached Instance
# ---------------------------------------------------------------------------

_LIQUIDITY_MODEL: Optional[LiquidityModel] = None


def get_liquidity_model() -> LiquidityModel:
    """
    Get the cached liquidity model instance.
    Loads from config on first call.
    """
    global _LIQUIDITY_MODEL
    if _LIQUIDITY_MODEL is None:
        _LIQUIDITY_MODEL = load_liquidity_model()
    return _LIQUIDITY_MODEL


def reload_liquidity_model() -> LiquidityModel:
    """Force reload the liquidity model from config."""
    global _LIQUIDITY_MODEL
    _LIQUIDITY_MODEL = load_liquidity_model()
    return _LIQUIDITY_MODEL


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_bucket_for_symbol(symbol: str) -> LiquidityBucketConfig:
    """Get the liquidity bucket for a symbol."""
    return get_liquidity_model().get_bucket(symbol)


def get_max_spread_bps(symbol: str) -> float:
    """Get the maximum acceptable spread (in bps) for a symbol."""
    return get_bucket_for_symbol(symbol).max_spread_bps


def get_default_maker_bias(symbol: str) -> float:
    """Get the default maker bias (0-1) for a symbol."""
    return get_bucket_for_symbol(symbol).default_maker_bias


def is_high_liquidity(symbol: str) -> bool:
    """Check if symbol is in the high-liquidity bucket."""
    bucket = get_bucket_for_symbol(symbol)
    return bucket.name == "A_HIGH"


def is_low_liquidity(symbol: str) -> bool:
    """Check if symbol is in the low-liquidity bucket."""
    bucket = get_bucket_for_symbol(symbol)
    return bucket.name == "C_LOW"


def build_liquidity_snapshot() -> Dict[str, dict]:
    """
    Build a snapshot of liquidity bucket assignments for state publishing.
    
    Returns:
        Dict mapping symbol -> {bucket, max_spread_bps, default_maker_bias}
    """
    model = get_liquidity_model()
    snapshot = {}
    
    for symbol, bucket in model.symbol_to_bucket.items():
        snapshot[symbol] = {
            "bucket": bucket.name,
            "max_spread_bps": bucket.max_spread_bps,
            "default_maker_bias": bucket.default_maker_bias,
        }
    
    return snapshot
