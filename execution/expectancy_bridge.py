"""
Expectancy Bridge (v7.9-P4B)

Replaces the heuristic confidence-threshold model in true_edge.py with a
deterministic, empirical mapping from conviction/score bands to realized
net expectancy computed from historical episodes.

Contract:
    expected_edge_pct = E[net_edge_pct | confidence_band, symbol?, regime?]

The bridge is:
  - Empirical: derived from closed episodes with known PnL
  - Deterministic: same inputs → same output (no randomness)
  - Replayable: band table is serializable and versioned
  - Banded: conviction is bucketed into fixed-width bands
  - Shadow-only: does NOT gate execution until promoted (P4D)

Band table schema:
    {
        band_key: {
            "band_lo": float,      # inclusive lower bound
            "band_hi": float,      # exclusive upper bound
            "n_episodes": int,     # sample count
            "net_pnl_sum": float,  # total net PnL USD
            "notional_sum": float, # total entry notional USD
            "avg_edge_pct": float, # net_pnl_sum / notional_sum
            "win_rate": float,     # fraction with net_pnl > 0
            "sufficient": bool,    # n_episodes >= min_sample
        }
    }

Lookup priority (most specific → least):
    1. (confidence_band, symbol, regime) — if sufficient
    2. (confidence_band, symbol)         — if sufficient
    3. (confidence_band)                 — pooled across all
    4. global fallback                   — all episodes pooled

Edge source: "empirical_expectancy_bridge"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("expectancy_bridge")

# ── Configuration ─────────────────────────────────────────────────────────
DEFAULT_BAND_WIDTH: float = 0.05       # 5pp conviction bands
DEFAULT_MIN_SAMPLE: int = 10           # Minimum episodes per band
DEFAULT_MIN_CONVICTION: float = 0.20   # Ignore episodes below this
DEFAULT_MAX_CONVICTION: float = 1.00   # Upper bound
DEFAULT_STALE_HOURS: float = 24.0      # Rebuild if table older than this


@dataclass
class BridgeConfig:
    """Configuration for the expectancy bridge."""

    band_width: float = DEFAULT_BAND_WIDTH
    min_sample: int = DEFAULT_MIN_SAMPLE
    min_conviction: float = DEFAULT_MIN_CONVICTION
    max_conviction: float = DEFAULT_MAX_CONVICTION
    stale_hours: float = DEFAULT_STALE_HOURS
    # Path to the episode ledger (source of truth)
    episode_ledger_path: str = "logs/state/episode_ledger.json"
    # Path to persist the band table
    band_table_path: str = "logs/state/expectancy_bridge.json"


# ── Band Table ────────────────────────────────────────────────────────────

@dataclass
class BandEntry:
    """Aggregated statistics for a single conviction band."""

    band_lo: float
    band_hi: float
    n_episodes: int = 0
    net_pnl_sum: float = 0.0
    notional_sum: float = 0.0
    win_count: int = 0

    @property
    def avg_edge_pct(self) -> float:
        if self.notional_sum <= 0:
            return 0.0
        return self.net_pnl_sum / self.notional_sum

    @property
    def win_rate(self) -> float:
        if self.n_episodes <= 0:
            return 0.0
        return self.win_count / self.n_episodes

    @property
    def sufficient(self) -> bool:
        return self.n_episodes >= DEFAULT_MIN_SAMPLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_lo": round(self.band_lo, 4),
            "band_hi": round(self.band_hi, 4),
            "n_episodes": self.n_episodes,
            "net_pnl_sum": round(self.net_pnl_sum, 6),
            "notional_sum": round(self.notional_sum, 2),
            "avg_edge_pct": round(self.avg_edge_pct, 8),
            "win_rate": round(self.win_rate, 4),
            "sufficient": self.sufficient,
        }


@dataclass
class BandTable:
    """Complete band table with metadata."""

    bands: Dict[str, BandEntry] = field(default_factory=dict)
    global_entry: Optional[BandEntry] = None
    build_ts: float = 0.0
    n_episodes_total: int = 0
    n_episodes_scored: int = 0
    config: Optional[BridgeConfig] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "build_ts": self.build_ts,
            "build_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.build_ts))
            if self.build_ts
            else None,
            "n_episodes_total": self.n_episodes_total,
            "n_episodes_scored": self.n_episodes_scored,
            "config": asdict(self.config) if self.config else None,
            "global": self.global_entry.to_dict() if self.global_entry else None,
            "bands": {k: v.to_dict() for k, v in sorted(self.bands.items())},
        }


# ── Build ─────────────────────────────────────────────────────────────────

def _band_key(conviction: float, band_width: float) -> str:
    """Compute the band key for a given conviction score.

    Bands are [lo, hi) — lower-inclusive, upper-exclusive.
    Uses integer arithmetic to avoid float precision issues.
    """
    import math
    band_idx = math.floor(conviction / band_width + 1e-9)
    band_lo = round(band_idx * band_width, 4)
    band_hi = round(band_lo + band_width, 4)
    return f"{band_lo:.2f}-{band_hi:.2f}"


def _get_band_bounds(conviction: float, band_width: float) -> tuple[float, float]:
    """Return (lo, hi) for the band containing this conviction."""
    import math
    band_idx = math.floor(conviction / band_width + 1e-9)
    band_lo = round(band_idx * band_width, 4)
    return (band_lo, round(band_lo + band_width, 4))


def build_band_table(
    episodes: Sequence[Dict[str, Any]],
    config: Optional[BridgeConfig] = None,
) -> BandTable:
    """Build band table from episode records.

    Each episode must have at minimum:
      - conviction_score (float > 0)
      - net_pnl (float)
      - entry_notional (float > 0)

    Optional enrichment fields: symbol, regime_at_entry, side.

    Returns a BandTable with:
      - Per-band aggregates keyed by "lo-hi" string
      - Global aggregate across all scored episodes
    """
    if config is None:
        config = BridgeConfig()

    table = BandTable(
        build_ts=time.time(),
        n_episodes_total=len(episodes),
        config=config,
        version="1.0",
    )

    # Global accumulator
    global_entry = BandEntry(
        band_lo=config.min_conviction,
        band_hi=config.max_conviction,
    )

    scored_count = 0

    for ep in episodes:
        conviction = ep.get("conviction_score") or 0
        if conviction < config.min_conviction:
            continue

        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue

        notional = ep.get("entry_notional") or 0
        if notional <= 0:
            continue

        scored_count += 1
        net_pnl = float(net_pnl)
        notional = float(notional)
        is_win = net_pnl > 0

        # Per-band
        key = _band_key(conviction, config.band_width)
        if key not in table.bands:
            lo, hi = _get_band_bounds(conviction, config.band_width)
            table.bands[key] = BandEntry(band_lo=lo, band_hi=hi)

        band = table.bands[key]
        band.n_episodes += 1
        band.net_pnl_sum += net_pnl
        band.notional_sum += notional
        if is_win:
            band.win_count += 1

        # Global
        global_entry.n_episodes += 1
        global_entry.net_pnl_sum += net_pnl
        global_entry.notional_sum += notional
        if is_win:
            global_entry.win_count += 1

    table.n_episodes_scored = scored_count
    table.global_entry = global_entry

    return table


# ── Lookup ────────────────────────────────────────────────────────────────

@dataclass
class BridgeLookupResult:
    """Result of an expectancy bridge lookup."""

    expected_edge_pct: float
    band_key: str
    n_episodes: int
    sufficient: bool
    win_rate: float
    lookup_tier: str           # "band", "global", "cold_start"
    edge_contract: str = "empirical_expectancy_bridge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_expected_edge_pct": round(self.expected_edge_pct, 8),
            "bridge_band_key": self.band_key,
            "bridge_n_episodes": self.n_episodes,
            "bridge_sufficient": self.sufficient,
            "bridge_win_rate": round(self.win_rate, 4),
            "bridge_lookup_tier": self.lookup_tier,
            "bridge_edge_contract": self.edge_contract,
        }


def lookup_expected_edge(
    conviction: float,
    table: BandTable,
    config: Optional[BridgeConfig] = None,
) -> BridgeLookupResult:
    """Look up empirical expected edge for a conviction score.

    Lookup priority:
        1. Band-specific entry if sufficient samples
        2. Global (all scored episodes) if band insufficient
        3. Cold-start fallback (zero edge) if no data

    Returns BridgeLookupResult with the expected edge and provenance.
    """
    if config is None:
        config = table.config or BridgeConfig()

    band_width = config.band_width
    key = _band_key(conviction, band_width)

    # Tier 1: Band-specific
    band = table.bands.get(key)
    if band is not None and band.n_episodes >= config.min_sample:
        return BridgeLookupResult(
            expected_edge_pct=band.avg_edge_pct,
            band_key=key,
            n_episodes=band.n_episodes,
            sufficient=True,
            win_rate=band.win_rate,
            lookup_tier="band",
        )

    # Tier 2: Global pooled
    g = table.global_entry
    if g is not None and g.n_episodes >= config.min_sample:
        return BridgeLookupResult(
            expected_edge_pct=g.avg_edge_pct,
            band_key=key,
            n_episodes=g.n_episodes,
            sufficient=True,
            win_rate=g.win_rate,
            lookup_tier="global",
        )

    # Tier 3: Cold start — no data
    return BridgeLookupResult(
        expected_edge_pct=0.0,
        band_key=key,
        n_episodes=(band.n_episodes if band else 0),
        sufficient=False,
        win_rate=0.0,
        lookup_tier="cold_start",
    )


# ── Persistence ───────────────────────────────────────────────────────────

def save_band_table(table: BandTable, path: Optional[str] = None) -> str:
    """Persist band table to JSON."""
    if path is None:
        path = (table.config or BridgeConfig()).band_table_path
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(table.to_dict(), f, indent=2)
    LOG.info("[expectancy_bridge] saved band table to %s (%d bands, %d episodes)",
             path, len(table.bands), table.n_episodes_scored)
    return str(out)


def load_band_table(path: Optional[str] = None) -> Optional[BandTable]:
    """Load a previously saved band table. Returns None if not found."""
    if path is None:
        path = BridgeConfig().band_table_path
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)

        config = None
        if data.get("config"):
            config = BridgeConfig(**{
                k: v for k, v in data["config"].items()
                if k in BridgeConfig.__dataclass_fields__
            })

        table = BandTable(
            build_ts=data.get("build_ts", 0),
            n_episodes_total=data.get("n_episodes_total", 0),
            n_episodes_scored=data.get("n_episodes_scored", 0),
            config=config,
            version=data.get("version", "1.0"),
        )

        # Reconstruct bands
        for key, bd in (data.get("bands") or {}).items():
            table.bands[key] = BandEntry(
                band_lo=bd["band_lo"],
                band_hi=bd["band_hi"],
                n_episodes=bd["n_episodes"],
                net_pnl_sum=bd["net_pnl_sum"],
                notional_sum=bd["notional_sum"],
                win_count=int(round(bd["win_rate"] * bd["n_episodes"])),
            )

        # Reconstruct global
        g = data.get("global")
        if g:
            table.global_entry = BandEntry(
                band_lo=g.get("band_lo", 0),
                band_hi=g.get("band_hi", 1),
                n_episodes=g["n_episodes"],
                net_pnl_sum=g["net_pnl_sum"],
                notional_sum=g["notional_sum"],
                win_count=int(round(g["win_rate"] * g["n_episodes"])),
            )

        return table
    except Exception as exc:
        LOG.warning("[expectancy_bridge] failed to load band table from %s: %s", path, exc)
        return None


# ── Episode Loader ────────────────────────────────────────────────────────

def load_episodes(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load episodes from the episode ledger JSON."""
    if path is None:
        path = BridgeConfig().episode_ledger_path
    p = Path(path)
    if not p.exists():
        LOG.warning("[expectancy_bridge] episode ledger not found: %s", path)
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("episodes", [])
    if isinstance(data, list):
        return data
    return []


# ── Build + Save convenience ──────────────────────────────────────────────

def rebuild_band_table(config: Optional[BridgeConfig] = None) -> BandTable:
    """Load episodes, build band table, save to disk, return table."""
    if config is None:
        config = BridgeConfig()
    episodes = load_episodes(config.episode_ledger_path)
    table = build_band_table(episodes, config)
    save_band_table(table)
    return table


# ── Monotonicity diagnostic ──────────────────────────────────────────────

def check_monotonicity(table: BandTable) -> Dict[str, Any]:
    """Check if higher conviction bands have higher expected edge.

    Returns a diagnostic dict with:
      - monotonic: bool (True if edge increases with conviction)
      - inversions: list of (band_a, band_b) pairs where edge decreases
      - band_edges: ordered list of (band_key, avg_edge_pct, sufficient)
    """
    ordered = sorted(
        [(k, v) for k, v in table.bands.items()],
        key=lambda x: x[1].band_lo,
    )

    band_edges = [
        (k, v.avg_edge_pct, v.sufficient) for k, v in ordered
    ]

    inversions = []
    sufficient_only = [(k, v) for k, v in ordered if v.sufficient]
    for i in range(len(sufficient_only) - 1):
        k_a, v_a = sufficient_only[i]
        k_b, v_b = sufficient_only[i + 1]
        if v_b.avg_edge_pct < v_a.avg_edge_pct:
            inversions.append((k_a, k_b))

    return {
        "monotonic": len(inversions) == 0,
        "inversions": inversions,
        "band_edges": band_edges,
        "n_sufficient_bands": len(sufficient_only),
    }


# ── P5B: Regime-Conditional Expectancy Bridge ────────────────────────────

# Binary regime split: MEAN_REVERT vs everything else (OTHER)
REGIME_KEY_MR = "MEAN_REVERT"
REGIME_KEY_OTHER = "OTHER"

DEFAULT_REGIME_BRIDGE_PATH = "logs/state/expectancy_bridge_regime.json"


def _regime_bucket(regime: str) -> str:
    """Binary split: MEAN_REVERT vs OTHER."""
    if regime == REGIME_KEY_MR:
        return REGIME_KEY_MR
    return REGIME_KEY_OTHER


def build_regime_conditional_table(
    episodes: Sequence[Dict[str, Any]],
    config: Optional[BridgeConfig] = None,
) -> Dict[str, BandTable]:
    """Build per-regime band tables using binary split.

    Returns {"MEAN_REVERT": BandTable, "OTHER": BandTable}.

    Conservation invariant: sum of regime tables == pooled table
    (verified at call site, not internally).
    """
    if config is None:
        config = BridgeConfig()

    regime_eps: Dict[str, List[Dict[str, Any]]] = {
        REGIME_KEY_MR: [],
        REGIME_KEY_OTHER: [],
    }

    for ep in episodes:
        regime_raw = ep.get("regime_at_entry") or "unknown"
        bucket = _regime_bucket(regime_raw)
        regime_eps[bucket].append(ep)

    tables: Dict[str, BandTable] = {}
    for regime_key, eps in regime_eps.items():
        tables[regime_key] = build_band_table(eps, config)

    return tables


@dataclass
class RegimeBridgeLookupResult:
    """Result of a regime-conditional expectancy bridge lookup."""

    expected_edge_pct: float
    lookup_tier: str       # band_regime | band_pooled | global_regime | global_pooled | cold_start
    band_key: str
    regime_key: str
    sample_n: int
    is_sufficient: bool
    fallback_depth: int    # 0 = direct hit, 1-4 = cascade steps taken
    cold_start_reason: str = ""
    edge_contract: str = "empirical_expectancy_bridge_regime"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bridge_regime_expected_edge_pct": round(self.expected_edge_pct, 8),
            "bridge_regime_lookup_tier": self.lookup_tier,
            "bridge_regime_band_key": self.band_key,
            "bridge_regime_key": self.regime_key,
            "bridge_regime_sample_n": self.sample_n,
            "bridge_regime_is_sufficient": self.is_sufficient,
            "bridge_regime_fallback_depth": self.fallback_depth,
        }
        if self.cold_start_reason:
            d["bridge_regime_cold_start_reason"] = self.cold_start_reason
        return d


def lookup_expected_edge_conditional(
    conviction: float,
    regime: str,
    regime_tables: Dict[str, BandTable],
    pooled_table: Optional[BandTable] = None,
    config: Optional[BridgeConfig] = None,
) -> RegimeBridgeLookupResult:
    """Regime-conditional expectancy lookup with 4-level cascade.

    Cascade priority (most specific → least):
      1. (band, regime)   — band entry in regime-specific table, if sufficient
      2. (band, pooled)   — band entry in pooled table, if sufficient
      3. (global, regime)  — global entry in regime-specific table, if sufficient
      4. (global, pooled)  — global entry in pooled table, if sufficient
      5. cold_start        — no data

    Returns RegimeBridgeLookupResult with full provenance.
    """
    if config is None:
        config = BridgeConfig()

    band_width = config.band_width
    min_sample = config.min_sample
    key = _band_key(conviction, band_width)
    regime_bucket = _regime_bucket(regime)

    # Tier 1: (band, regime)
    rt = regime_tables.get(regime_bucket)
    if rt is not None:
        band = rt.bands.get(key)
        if band is not None and band.n_episodes >= min_sample:
            return RegimeBridgeLookupResult(
                expected_edge_pct=band.avg_edge_pct,
                lookup_tier="band_regime",
                band_key=key,
                regime_key=regime_bucket,
                sample_n=band.n_episodes,
                is_sufficient=True,
                fallback_depth=0,
            )

    # Tier 2: (band, pooled)
    if pooled_table is not None:
        band = pooled_table.bands.get(key)
        if band is not None and band.n_episodes >= min_sample:
            return RegimeBridgeLookupResult(
                expected_edge_pct=band.avg_edge_pct,
                lookup_tier="band_pooled",
                band_key=key,
                regime_key=regime_bucket,
                sample_n=band.n_episodes,
                is_sufficient=True,
                fallback_depth=1,
            )

    # Tier 3: (global, regime)
    if rt is not None:
        g = rt.global_entry
        if g is not None and g.n_episodes >= min_sample:
            return RegimeBridgeLookupResult(
                expected_edge_pct=g.avg_edge_pct,
                lookup_tier="global_regime",
                band_key=key,
                regime_key=regime_bucket,
                sample_n=g.n_episodes,
                is_sufficient=True,
                fallback_depth=2,
            )

    # Tier 4: (global, pooled)
    if pooled_table is not None:
        g = pooled_table.global_entry
        if g is not None and g.n_episodes >= min_sample:
            return RegimeBridgeLookupResult(
                expected_edge_pct=g.avg_edge_pct,
                lookup_tier="global_pooled",
                band_key=key,
                regime_key=regime_bucket,
                sample_n=g.n_episodes,
                is_sufficient=True,
                fallback_depth=3,
            )

    # Tier 5: cold start
    n = 0
    if rt is not None:
        band = rt.bands.get(key)
        if band is not None:
            n = band.n_episodes
    reason = f"no_sufficient_data_for_{regime_bucket}_band_{key}"
    return RegimeBridgeLookupResult(
        expected_edge_pct=0.0,
        lookup_tier="cold_start",
        band_key=key,
        regime_key=regime_bucket,
        sample_n=n,
        is_sufficient=False,
        fallback_depth=4,
        cold_start_reason=reason,
    )


# ── P5B: Regime Bridge Persistence ───────────────────────────────────────

def save_regime_bridge(
    regime_tables: Dict[str, BandTable],
    config: Optional[BridgeConfig] = None,
    path: Optional[str] = None,
) -> str:
    """Persist regime-conditional bridge tables to JSON."""
    if config is None:
        config = BridgeConfig()
    if path is None:
        path = DEFAULT_REGIME_BRIDGE_PATH

    out_data: Dict[str, Any] = {
        "version": "1.0",
        "build_ts": time.time(),
        "build_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime_split": "binary",
        "regime_keys": list(regime_tables.keys()),
        "sufficiency_threshold": config.min_sample,
        "config": asdict(config),
        "tables": {},
    }

    for regime_key, table in regime_tables.items():
        out_data["tables"][regime_key] = table.to_dict()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_data, f, indent=2)
    LOG.info("[expectancy_bridge] saved regime bridge to %s (%d regimes)",
             path, len(regime_tables))
    return str(out)


def load_regime_bridge(
    path: Optional[str] = None,
) -> Optional[Dict[str, BandTable]]:
    """Load regime-conditional bridge tables from JSON."""
    if path is None:
        path = DEFAULT_REGIME_BRIDGE_PATH
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)

        tables: Dict[str, BandTable] = {}
        config_data = data.get("config")
        config = None
        if config_data:
            config = BridgeConfig(**{
                k: v for k, v in config_data.items()
                if k in BridgeConfig.__dataclass_fields__
            })

        for regime_key, tdata in (data.get("tables") or {}).items():
            table = BandTable(
                build_ts=tdata.get("build_ts", 0),
                n_episodes_total=tdata.get("n_episodes_total", 0),
                n_episodes_scored=tdata.get("n_episodes_scored", 0),
                config=config,
                version=tdata.get("version", "1.0"),
            )
            for key, bd in (tdata.get("bands") or {}).items():
                table.bands[key] = BandEntry(
                    band_lo=bd["band_lo"],
                    band_hi=bd["band_hi"],
                    n_episodes=bd["n_episodes"],
                    net_pnl_sum=bd["net_pnl_sum"],
                    notional_sum=bd["notional_sum"],
                    win_count=int(round(bd["win_rate"] * bd["n_episodes"])),
                )
            g = tdata.get("global")
            if g:
                table.global_entry = BandEntry(
                    band_lo=g.get("band_lo", 0),
                    band_hi=g.get("band_hi", 1),
                    n_episodes=g["n_episodes"],
                    net_pnl_sum=g["net_pnl_sum"],
                    notional_sum=g["notional_sum"],
                    win_count=int(round(g["win_rate"] * g["n_episodes"])),
                )
            tables[regime_key] = table

        return tables
    except Exception as exc:
        LOG.warning("[expectancy_bridge] failed to load regime bridge: %s", exc)
        return None
