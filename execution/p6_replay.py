"""
P6B.5 Historical Replay — Deterministic episode-level replay (v7.9-P6B.5)

For each historical episode in the episode ledger:
  1. Look up klines around entry time
  2. Compute sentinel features via extract_regime_features()
  3. Generate C1 (simple rules) + C2 (price-state) signals
  4. Route each signal through the frozen expectancy bridge
  5. Log primary + control + realized PnL

The bridge tables, configs, and fee model are frozen at replay start.
All replay records are written to logs/execution/p6_replay_signals.jsonl.

This module is OBSERVATION-ONLY.  It never gates execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from execution.expectancy_bridge import (
    BandTable,
    BridgeConfig,
    load_band_table,
    load_regime_bridge,
)
from execution.log_utils import append_jsonl
from execution.p6_price_state import C2Config, generate_price_state_signals
from execution.p6_shadow_evaluator import (
    DEFAULT_FEE_THRESHOLD_PCT,
    evaluate_signal_against_bridge,
)
from execution.p6_simple_rules import (
    C1Config,
    P6Signal,
    generate_simple_rule_signals,
)
from execution.sentinel_x import extract_regime_features

LOG = logging.getLogger("p6_replay")

# ── Paths ────────────────────────────────────────────────────────────────

REPLAY_LOG_PATH = Path("logs/execution/p6_replay_signals.jsonl")
REPLAY_SUMMARY_PATH = Path("logs/state/p6_replay_summary.json")
REPLAY_TABLES_PATH = Path("logs/state/p6_replay_tables.csv")
REPLAY_RUN_PATH = Path("logs/state/p6_replay_run.json")

# ── Constants ────────────────────────────────────────────────────────────

BINANCE_FAPI_BASE = "https://fapi.binance.com"
INTERVAL_15M_MS = 900_000
DEFAULT_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WARMUP_BARS = 60  # >= max(ema_slow=50, zscore_lookback=48, BB) per user spec
KLINE_REQUEST_LIMIT = 1500
RATE_LIMIT_S = 0.12  # polite delay between API requests
KLINE_CACHE_PATH = Path("logs/state/p6_kline_cache.json")

ALL_CANDIDATE_IDS = [
    "C1_TREND_NORMAL",
    "C1_TREND_INVERTED",
    "C1_MR_NORMAL",
    "C1_MR_INVERTED",
    "C2_REGION_NORMAL",
    "C2_REGION_INVERTED",
]


# ── Kline Fetching ───────────────────────────────────────────────────────

def fetch_klines_range(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str = BINANCE_FAPI_BASE,
) -> List[Dict[str, Any]]:
    """Fetch all klines in [start_ms, end_ms] from Binance public FAPI.

    Uses pagination (1500 bars per request) and polite rate limiting.
    Returns list of dicts: {open_time, open, high, low, close, volume}.
    """
    all_klines: List[Dict[str, Any]] = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": KLINE_REQUEST_LIMIT,
        }
        resp = requests.get(
            f"{base_url}/fapi/v1/klines",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for row in data:
            all_klines.append({
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        current = int(data[-1][0]) + INTERVAL_15M_MS
        time.sleep(RATE_LIMIT_S)

    return all_klines


def build_kline_cache(
    symbols: List[str],
    start_ms: int,
    end_ms: int,
    interval: str = "15m",
    base_url: str = BINANCE_FAPI_BASE,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch and cache klines for all symbols in the date range."""
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for sym in symbols:
        LOG.info("Fetching klines for %s ...", sym)
        klines = fetch_klines_range(sym, interval, start_ms, end_ms, base_url)
        cache[sym] = klines
        LOG.info("  %s: %d bars fetched", sym, len(klines))
    return cache


def save_kline_cache(
    cache: Dict[str, List[Dict[str, Any]]],
    path: Optional[Path] = None,
) -> None:
    """Persist kline cache to JSON for reproducibility."""
    if path is None:
        path = KLINE_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)
    total = sum(len(v) for v in cache.values())
    LOG.info("Kline cache saved to %s (%d symbols, %d bars)", path, len(cache), total)


def load_kline_cache(path: Optional[Path] = None) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Load persisted kline cache. Returns None if not found."""
    if path is None:
        path = KLINE_CACHE_PATH
    if not path.exists():
        return None
    with open(path) as f:
        cache = json.load(f)
    total = sum(len(v) for v in cache.values())
    LOG.info("Kline cache loaded from %s (%d symbols, %d bars)", path, len(cache), total)
    return cache


# ── Episode helpers ──────────────────────────────────────────────────────

def _iso_to_ms(ts_iso: str) -> int:
    """Convert ISO-8601 timestamp string to Unix milliseconds."""
    dt = datetime.fromisoformat(ts_iso)
    return int(dt.timestamp() * 1000)


def _snap_to_15m_open(ts_ms: int) -> int:
    """Snap a timestamp to the start of its 15m bar."""
    return (ts_ms // INTERVAL_15M_MS) * INTERVAL_15M_MS


def _find_bar_index(klines: List[Dict[str, Any]], target_ms: int) -> Optional[int]:
    """Binary-search for the kline bar whose open_time is <= target_ms.

    The target_ms is first snapped to the 15m bar open for alignment.
    """
    if not klines:
        return None
    target_ms = _snap_to_15m_open(target_ms)
    lo, hi = 0, len(klines) - 1
    result: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if klines[mid]["open_time"] <= target_ms:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


# ── Feature extraction from kline window ─────────────────────────────────

def extract_features_at_bar(
    klines: List[Dict[str, Any]],
    bar_idx: int,
    lookback: int = WARMUP_BARS,
) -> Tuple[List[float], Dict[str, float]]:
    """Build close-price series and sentinel features for the window ending at bar_idx.

    Returns (closes, features_dict).
    """
    start = max(0, bar_idx - lookback + 1)
    window = klines[start : bar_idx + 1]

    closes = [k["close"] for k in window]
    highs = [k["high"] for k in window]
    lows = [k["low"] for k in window]
    volumes = [k["volume"] for k in window]

    features = extract_regime_features(
        prices=closes,
        volumes=volumes,
        highs=highs,
        lows=lows,
    )
    return closes, features.to_dict()


# ── Single-episode replay ───────────────────────────────────────────────

def replay_episode(
    episode: Dict[str, Any],
    klines: List[Dict[str, Any]],
    bar_idx: int,
    c1_config: C1Config,
    c2_config: C2Config,
    regime_tables: Optional[Dict[str, BandTable]],
    pooled_table: Optional[BandTable],
    bridge_config: Optional[BridgeConfig],
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
) -> List[Dict[str, Any]]:
    """Replay a single episode: signals → bridge → realized PnL.

    Returns a list of replay records (one per signal, or a single
    NO_SIGNAL / SKIPPED record if no signal fires).
    """
    symbol = episode["symbol"]
    regime = episode.get("regime_at_entry", "unknown")
    entry_ts = _iso_to_ms(episode["entry_ts"]) / 1000.0

    # ── Feature extraction ──
    closes, features_dict = extract_features_at_bar(klines, bar_idx)

    if len(closes) < WARMUP_BARS:
        return [_skip_record(episode, f"insufficient_warmup_{len(closes)}_bars")]

    # ── Signal generation ──
    c1_selected, c1_suppressed = generate_simple_rule_signals(
        closes=closes,
        sentinel_features=features_dict,
        regime=regime,
        symbol=symbol,
        config=c1_config,
        ts=entry_ts,
    )

    c2_signals = generate_price_state_signals(
        sentinel_features=features_dict,
        regime=regime,
        symbol=symbol,
        config=c2_config,
        ts=entry_ts,
    )

    all_signals = c1_selected + c2_signals

    # ── Realized PnL from episode ──
    entry_notional = float(episode.get("entry_notional") or 0)
    net_pnl = float(episode.get("net_pnl") or 0)
    gross_pnl = float(episode.get("gross_pnl") or 0)
    fees = float(episode.get("fees") or 0)
    realized_edge_pct = (net_pnl / entry_notional) if entry_notional > 0 else 0.0

    episode_meta = {
        "episode_id": episode.get("episode_id", ""),
        "episode_symbol": symbol,
        "episode_side": episode.get("side", ""),
        "episode_regime": regime,
        "signal_generated": True,
        "realized_gross_pnl": round(gross_pnl, 6),
        "realized_fees": round(fees, 6),
        "realized_net_pnl": round(net_pnl, 6),
        "realized_net_edge_pct": round(realized_edge_pct, 8),
        "entry_notional": round(entry_notional, 2),
        "exit_reason": episode.get("exit_reason", ""),
        "episode_conviction_score": float(episode.get("conviction_score") or 0),
    }

    if not all_signals:
        rec = _skip_record(episode, "no_signal")
        rec["signal_generated"] = False
        rec.update({
            "realized_gross_pnl": episode_meta["realized_gross_pnl"],
            "realized_fees": episode_meta["realized_fees"],
            "realized_net_pnl": episode_meta["realized_net_pnl"],
            "realized_net_edge_pct": episode_meta["realized_net_edge_pct"],
            "entry_notional": episode_meta["entry_notional"],
        })
        return [rec]

    # ── Bridge evaluation for each signal ──
    records: List[Dict[str, Any]] = []
    for sig in all_signals:
        rec = evaluate_signal_against_bridge(
            signal=sig,
            regime_tables=regime_tables,
            pooled_table=pooled_table,
            bridge_config=bridge_config,
            fee_threshold_pct=fee_threshold_pct,
        )
        rec.update(episode_meta)
        records.append(rec)

    return records


def _skip_record(episode: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Build a minimal skip/no-signal record for an episode."""
    return {
        "episode_id": episode.get("episode_id", ""),
        "symbol": episode.get("symbol", ""),
        "regime": episode.get("regime_at_entry", "unknown"),
        "candidate_id": reason.upper() if reason != "no_signal" else "NO_SIGNAL",
        "skip_reason": reason,
        "signal_generated": False,
    }


# ── Episode loader ───────────────────────────────────────────────────────

def load_episodes(path: str = "logs/state/episode_ledger.json") -> List[Dict[str, Any]]:
    """Load episodes from the episode ledger JSON."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("episodes", [])
    return data if isinstance(data, list) else []


# ── Main replay loop ────────────────────────────────────────────────────

def run_replay(
    episodes: Optional[List[Dict[str, Any]]] = None,
    c1_config: Optional[C1Config] = None,
    c2_config: Optional[C2Config] = None,
    bridge_config: Optional[BridgeConfig] = None,
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
    universe: Optional[List[str]] = None,
    kline_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    base_url: str = BINANCE_FAPI_BASE,
    log_path: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run the full P6B.5 historical replay.

    Args:
        episodes: Override episode list (default: load from ledger).
        c1_config: Frozen C1 thresholds.
        c2_config: Frozen C2 thresholds.
        bridge_config: Bridge lookup config.
        fee_threshold_pct: Roundtrip fee threshold (default 0.12%).
        universe: Symbols to include (default: BTC/ETH/SOL).
        kline_cache: Pre-built kline cache (skips fetching if provided).
        base_url: Binance API base URL.
        log_path: Output JSONL path.
        dry_run: If True, returns records without writing to disk.

    Returns:
        List of all replay records.
    """
    if episodes is None:
        episodes = load_episodes()
    if c1_config is None:
        c1_config = C1Config()
    if c2_config is None:
        c2_config = C2Config()
    if universe is None:
        universe = list(DEFAULT_UNIVERSE)
    if log_path is None:
        log_path = REPLAY_LOG_PATH

    # Filter to universe
    ep_universe = [e for e in episodes if e.get("symbol") in universe]
    ep_skipped = [e for e in episodes if e.get("symbol") not in universe]
    LOG.info(
        "Replay: %d episodes in universe, %d skipped (out-of-universe)",
        len(ep_universe),
        len(ep_skipped),
    )

    if not ep_universe:
        LOG.error("No episodes to replay")
        return []

    # Determine date range (pad warmup before earliest entry)
    entry_ms_list = [
        _iso_to_ms(e["entry_ts"])
        for e in ep_universe
        if e.get("entry_ts")
    ]
    start_ms = min(entry_ms_list) - (WARMUP_BARS * INTERVAL_15M_MS)
    end_ms = max(entry_ms_list) + INTERVAL_15M_MS

    # Build or reuse kline cache
    if kline_cache is None:
        kline_cache = build_kline_cache(
            universe, start_ms, end_ms, base_url=base_url,
        )
        # Persist for reproducibility
        if not dry_run:
            save_kline_cache(kline_cache)

    # Load frozen bridge tables
    regime_tables = load_regime_bridge() or {}
    pooled_table = load_band_table()

    LOG.info(
        "Bridge loaded: %d regime keys, pooled=%s",
        len(regime_tables),
        pooled_table is not None,
    )

    # Ensure output directory exists
    if not dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    for i, ep in enumerate(ep_universe):
        symbol = ep["symbol"]
        klines = kline_cache.get(symbol, [])

        if not klines:
            rec = _skip_record(ep, "no_klines")
            all_records.append(rec)
            if not dry_run:
                append_jsonl(log_path, rec)
            continue

        entry_ms = _iso_to_ms(ep["entry_ts"])
        bar_idx = _find_bar_index(klines, entry_ms)

        if bar_idx is None:
            rec = _skip_record(ep, "no_matching_bar")
            all_records.append(rec)
            if not dry_run:
                append_jsonl(log_path, rec)
            continue

        records = replay_episode(
            episode=ep,
            klines=klines,
            bar_idx=bar_idx,
            c1_config=c1_config,
            c2_config=c2_config,
            regime_tables=regime_tables,
            pooled_table=pooled_table,
            bridge_config=bridge_config,
            fee_threshold_pct=fee_threshold_pct,
        )

        all_records.extend(records)

        if not dry_run:
            for rec in records:
                try:
                    append_jsonl(log_path, rec)
                except Exception:
                    LOG.exception("Failed to write replay record for %s", symbol)

        if (i + 1) % 100 == 0:
            LOG.info("  Replay progress: %d / %d episodes", i + 1, len(ep_universe))

    # Log out-of-universe episodes as skipped
    for ep in ep_skipped:
        rec = _skip_record(ep, "out_of_universe")
        all_records.append(rec)
        if not dry_run:
            append_jsonl(log_path, rec)

    sig_count = sum(1 for r in all_records if r.get("signal_generated"))
    skip_count = sum(1 for r in all_records if r.get("skip_reason"))
    LOG.info(
        "Replay complete: %d records (%d signals, %d skipped) from %d episodes",
        len(all_records), sig_count, skip_count, len(episodes),
    )
    return all_records


# ── Run Manifest ─────────────────────────────────────────────────────────

def _file_sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_run_manifest(
    c1_config: Optional[C1Config] = None,
    c2_config: Optional[C2Config] = None,
    bridge_config: Optional[BridgeConfig] = None,
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
    universe: Optional[List[str]] = None,
    n_episodes: int = 0,
    n_records: int = 0,
) -> Dict[str, Any]:
    """Build a frozen run manifest for reproducibility.

    Contains all configs, file hashes, git commit, and run metadata.
    """
    if c1_config is None:
        c1_config = C1Config()
    if c2_config is None:
        c2_config = C2Config()
    if universe is None:
        universe = list(DEFAULT_UNIVERSE)

    ts_now = datetime.now(tz=timezone.utc)

    manifest: Dict[str, Any] = {
        "version": "p6b5_v1",
        "run_ts": ts_now.isoformat(),
        "run_ts_unix": time.time(),
        "git_commit": _git_commit_hash(),
        # Frozen configs
        "c1_config": asdict(c1_config),
        "c2_config": asdict(c2_config),
        "bridge_config": asdict(bridge_config) if bridge_config else asdict(BridgeConfig()),
        "fee_threshold_pct": fee_threshold_pct,
        "universe": universe,
        "warmup_bars": WARMUP_BARS,
        "interval": "15m",
        # File hashes for reproducibility
        "file_hashes": {
            "episode_ledger": _file_sha256("logs/state/episode_ledger.json"),
            "expectancy_bridge": _file_sha256("logs/state/expectancy_bridge.json"),
            "expectancy_bridge_regime": _file_sha256("logs/state/expectancy_bridge_regime.json"),
            "p6_simple_rules": _file_sha256("execution/p6_simple_rules.py"),
            "p6_price_state": _file_sha256("execution/p6_price_state.py"),
            "p6_shadow_evaluator": _file_sha256("execution/p6_shadow_evaluator.py"),
            "p6_replay": _file_sha256("execution/p6_replay.py"),
        },
        # Run statistics
        "n_episodes_total": n_episodes,
        "n_records_total": n_records,
    }
    return manifest


def save_run_manifest(manifest: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Save run manifest to JSON."""
    if path is None:
        path = REPLAY_RUN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    LOG.info("Run manifest saved to %s", path)


# ── Summary Statistics ───────────────────────────────────────────────────

def compute_replay_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-candidate summary statistics for fast-fail evaluation.

    Returns a dict with:
      - per_candidate: {candidate_id: {stats}}
      - global: overall stats
      - ts: ISO timestamp
    """
    ts_now = datetime.now(tz=timezone.utc).isoformat()

    # Filter to signal records only (exclude skips)
    signal_recs = [r for r in records if r.get("signal_generated")]
    skip_recs = [r for r in records if r.get("skip_reason")]
    no_signal_recs = [r for r in records if r.get("candidate_id") == "NO_SIGNAL"]

    if not signal_recs:
        return {
            "per_candidate": {},
            "global": {"n_episodes": len(records), "n_signals": 0, "n_skipped": len(skip_recs)},
            "ts": ts_now,
        }

    # Group by candidate_id
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for rec in signal_recs:
        cid = rec.get("candidate_id", "unknown")
        by_candidate.setdefault(cid, []).append(rec)

    per_candidate: Dict[str, Dict[str, Any]] = {}

    for cid, recs in by_candidate.items():
        per_candidate[cid] = _candidate_stats(cid, recs)

    return {
        "per_candidate": per_candidate,
        "global": {
            "n_episodes": len(records),
            "n_signal_records": len(signal_recs),
            "n_no_signal": len(no_signal_recs),
            "n_skipped": len(skip_recs),
            "candidate_ids": sorted(by_candidate.keys()),
        },
        "ts": ts_now,
    }


def _candidate_stats(cid: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute stats for a single candidate_id."""
    n = len(recs)

    # Bridge pass rates
    n_pass = sum(1 for r in recs if r.get("bridge_would_pass"))
    pass_rate = n_pass / n if n > 0 else 0.0

    # Control pass rates
    n_control_pass = sum(1 for r in recs if r.get("control_would_pass"))
    control_pass_rate = n_control_pass / n if n > 0 else 0.0

    # Expected edge stats
    edges = [r.get("bridge_expected_edge_pct", 0.0) for r in recs]
    control_edges = [r.get("control_expected_edge_pct", 0.0) for r in recs]
    mean_edge = sum(edges) / n if n > 0 else 0.0
    sorted_edges = sorted(edges)
    median_edge = sorted_edges[n // 2] if n > 0 else 0.0

    # Realized PnL stats (only for recs with realized data)
    realized = [r for r in recs if r.get("realized_net_edge_pct") is not None]
    realized_edges = [r["realized_net_edge_pct"] for r in realized]
    mean_realized = sum(realized_edges) / len(realized_edges) if realized_edges else 0.0

    # Conviction stats
    convictions = [r.get("conviction", 0.5) for r in recs]
    mean_conv = sum(convictions) / n if n > 0 else 0.0

    # Band distribution
    band_dist: Dict[str, int] = {}
    for r in recs:
        bk = r.get("bridge_band_key", "none")
        band_dist[bk] = band_dist.get(bk, 0) + 1

    # Regime distribution
    regime_dist: Dict[str, int] = {}
    for r in recs:
        rk = r.get("regime", r.get("episode_regime", "unknown"))
        regime_dist[rk] = regime_dist.get(rk, 0) + 1

    # Symbol distribution
    symbol_dist: Dict[str, int] = {}
    for r in recs:
        sym = r.get("symbol", r.get("episode_symbol", "unknown"))
        symbol_dist[sym] = symbol_dist.get(sym, 0) + 1

    # Lookup tier distribution
    tier_dist: Dict[str, int] = {}
    for r in recs:
        t = r.get("bridge_lookup_tier", "unknown")
        tier_dist[t] = tier_dist.get(t, 0) + 1

    # Concentration check (max share of single symbol)
    max_symbol_share = max(symbol_dist.values()) / n if n > 0 and symbol_dist else 0.0
    max_symbol = max(symbol_dist, key=symbol_dist.get) if symbol_dist else ""

    # Per-band edge stats
    band_edge_stats = _per_band_edge_stats(recs)

    # Best band (by avg realized edge, min n=10)
    best_band = _find_best_band(band_edge_stats)

    # Spearman rank correlation: conviction → realized_net_edge_pct
    spearman_rho = _spearman_rho(
        [r.get("conviction", 0.5) for r in realized],
        [r["realized_net_edge_pct"] for r in realized],
    )

    # Per-symbol realized edge (for fast-fail gate 4)
    per_symbol_edge: Dict[str, Dict[str, Any]] = {}
    for sym in symbol_dist:
        sym_recs = [r for r in recs if r.get("symbol", r.get("episode_symbol")) == sym]
        sym_realized = [r["realized_net_edge_pct"] for r in sym_recs if r.get("realized_net_edge_pct") is not None]
        sym_expected = [r.get("bridge_expected_edge_pct", 0.0) for r in sym_recs]
        sym_best = _find_best_band(_per_band_edge_stats(sym_recs)) if sym_recs else None
        per_symbol_edge[sym] = {
            "n": len(sym_recs),
            "mean_realized_edge_pct": round(sum(sym_realized) / len(sym_realized), 8) if sym_realized else 0.0,
            "mean_expected_edge_pct": round(sum(sym_expected) / len(sym_expected), 8) if sym_expected else 0.0,
            "best_band": sym_best,
        }

    # Median realized edge
    sorted_realized = sorted(realized_edges)
    median_realized = sorted_realized[len(sorted_realized) // 2] if sorted_realized else 0.0

    return {
        "candidate_id": cid,
        "n_signals": n,
        "n_pass": n_pass,
        "pass_rate": round(pass_rate, 4),
        "mean_expected_edge_pct": round(mean_edge, 8),
        "median_expected_edge_pct": round(median_edge, 8),
        "mean_realized_edge_pct": round(mean_realized, 8),
        "median_realized_edge_pct": round(median_realized, 8),
        "mean_conviction": round(mean_conv, 6),
        "spearman_rho": round(spearman_rho, 6) if spearman_rho is not None else None,
        "best_band": best_band,
        "band_distribution": band_dist,
        "band_edge_stats": band_edge_stats,
        "regime_distribution": regime_dist,
        "symbol_distribution": symbol_dist,
        "per_symbol_edge": per_symbol_edge,
        "tier_distribution": tier_dist,
        "max_symbol_share": round(max_symbol_share, 4),
        "max_symbol": max_symbol,
        "n_realized": len(realized),
        # Control comparison
        "control_pass_rate": round(control_pass_rate, 4),
        "control_mean_edge_pct": round(sum(control_edges) / n, 8) if n > 0 else 0.0,
        "pass_rate_delta": round(pass_rate - control_pass_rate, 4),
    }


def _per_band_edge_stats(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-band realized edge statistics."""
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        bk = r.get("bridge_band_key", "none")
        by_band.setdefault(bk, []).append(r)

    stats: Dict[str, Dict[str, Any]] = {}
    for bk, band_recs in sorted(by_band.items()):
        n = len(band_recs)
        realized = [r["realized_net_edge_pct"] for r in band_recs if r.get("realized_net_edge_pct") is not None]
        expected = [r.get("bridge_expected_edge_pct", 0.0) for r in band_recs]
        stats[bk] = {
            "n": n,
            "mean_expected_edge_pct": round(sum(expected) / n, 8) if n else 0.0,
            "mean_realized_edge_pct": round(sum(realized) / len(realized), 8) if realized else 0.0,
            "n_realized": len(realized),
            "n_pass": sum(1 for r in band_recs if r.get("bridge_would_pass")),
        }
    return stats


def _find_best_band(band_stats: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the band with highest avg realized edge (min n=10)."""
    candidates = [
        {"band_key": bk, **bs}
        for bk, bs in band_stats.items()
        if bs.get("n_realized", 0) >= 10
    ]
    if not candidates:
        # Fall back: any band with n >= 1
        candidates = [
            {"band_key": bk, **bs}
            for bk, bs in band_stats.items()
            if bs.get("n_realized", 0) >= 1
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.get("mean_realized_edge_pct", -999))


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Compute Spearman rank correlation coefficient.

    Returns None if fewer than 3 pairs or zero variance in ranks.
    """
    n = min(len(x), len(y))
    if n < 3:
        return None

    def _rank(vals: Sequence[float]) -> List[float]:
        indexed = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and vals[indexed[j]] == vals[indexed[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    rx = _rank(list(x[:n]))
    ry = _rank(list(y[:n]))

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = (var_x * var_y) ** 0.5
    if denom < 1e-12:
        return None
    return cov / denom


# ── Fast-Fail Gates ──────────────────────────────────────────────────────

def apply_fast_fail_gates(
    summary: Dict[str, Any],
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
    min_best_band_n: int = 10,
    max_single_symbol_share: float = 0.70,
) -> Dict[str, Dict[str, Any]]:
    """Apply fast-fail gates to each candidate.

    Gates (reject if ANY true):
      1. No band with realized edge > fee_threshold_pct
      2. Best band n < min_best_band_n
      3. Spearman ρ ≤ 0
      4. >70% single symbol with ≤ 0 edge
      5. Pass rate 0% or > 80%

    Returns {candidate_id: {passed: bool, fails: [...], ...}}.
    """
    per_candidate = summary.get("per_candidate", {})
    verdicts: Dict[str, Dict[str, Any]] = {}

    for cid, stats in per_candidate.items():
        fails: List[str] = []

        # Gate 1: any band with expected_edge > fee? (bridge-predicted)
        band_stats = stats.get("band_edge_stats", {})
        has_fee_clearing_band = any(
            bs.get("mean_expected_edge_pct", 0) > fee_threshold_pct
            and bs.get("n", 0) >= min_best_band_n
            for bs in band_stats.values()
        )
        if not has_fee_clearing_band:
            fails.append("no_fee_clearing_band")

        # Gate 2: best band sample size
        best_band = stats.get("best_band")
        if best_band is None or best_band.get("n_realized", 0) < min_best_band_n:
            fails.append(f"best_band_n_lt_{min_best_band_n}")

        # Gate 3: Spearman ρ ≤ 0
        rho = stats.get("spearman_rho")
        if rho is None or rho <= 0:
            fails.append(f"spearman_rho_lte_0 (rho={rho})")

        # Gate 4: >70% signals from one symbol AND that symbol's best band ≤ 0
        if stats.get("max_symbol_share", 0) > max_single_symbol_share:
            dominant_sym = stats.get("max_symbol", "")
            pse = stats.get("per_symbol_edge", {})
            sym_info = pse.get(dominant_sym, {})
            sym_best = sym_info.get("best_band")
            sym_best_edge = sym_best.get("mean_realized_edge_pct", 0) if sym_best else 0
            if sym_best_edge <= 0:
                fails.append(
                    f"dominant_symbol_{dominant_sym}_share="
                    f"{stats['max_symbol_share']:.2f}_best_band_edge={sym_best_edge:.6f}"
                )

        # Gate 5: pass rate 0% or > 80%
        pr = stats.get("pass_rate", 0)
        if pr == 0:
            fails.append("pass_rate_zero")
        elif pr > 0.80:
            fails.append(f"pass_rate_too_high ({pr:.2%})")

        verdicts[cid] = {
            "candidate_id": cid,
            "passed": len(fails) == 0,
            "n_fails": len(fails),
            "fails": fails,
            "n_signals": stats.get("n_signals", 0),
            "pass_rate": stats.get("pass_rate", 0),
            "spearman_rho": stats.get("spearman_rho"),
            "best_band": stats.get("best_band"),
        }

    return verdicts


# ── Promotion Pre-Gates ──────────────────────────────────────────────────

def apply_promotion_gates(
    summary: Dict[str, Any],
    fast_fail_verdicts: Dict[str, Dict[str, Any]],
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
    edge_buffer_pct: float = 0.05,
    min_best_band_n: int = 20,
    min_rho: float = 0.10,
    selectivity_range: Tuple[float, float] = (0.05, 0.80),
) -> Dict[str, Dict[str, Any]]:
    """Apply promotion pre-gates to candidates that passed fast-fail.

    Pre-gates (all must pass for promotion):
      1. Best band edge > friction + buffer (fee + 0.05%)
      2. Best band n ≥ 20 (or ≥30 total signals)
      3. Spearman ρ ≥ 0.10
      4. Selectivity (pass_rate) in [5%, 80%]
      5. No single-symbol dominance (>70%) unless that symbol's band
         is independently sufficient

    Returns {candidate_id: {promoted: bool, ...}}.
    """
    per_candidate = summary.get("per_candidate", {})
    results: Dict[str, Dict[str, Any]] = {}

    promotion_threshold = fee_threshold_pct + edge_buffer_pct

    for cid, stats in per_candidate.items():
        ff = fast_fail_verdicts.get(cid, {})
        if not ff.get("passed", False):
            results[cid] = {
                "candidate_id": cid,
                "promoted": False,
                "reason": "failed_fast_fail",
                "fast_fail_fails": ff.get("fails", []),
            }
            continue

        gates: List[str] = []

        # Gate 1: edge > friction + buffer
        best_band = stats.get("best_band")
        best_edge = best_band.get("mean_realized_edge_pct", 0) if best_band else 0
        if best_edge <= promotion_threshold:
            gates.append(
                f"best_band_edge={best_edge:.6f}_lt_threshold={promotion_threshold:.4f}"
            )

        # Gate 2: best band n ≥ 20 (or ≥30 total signals)
        best_n = best_band.get("n_realized", 0) if best_band else 0
        total_n = stats.get("n_signals", 0)
        if best_n < min_best_band_n and total_n < 30:
            gates.append(f"best_band_n={best_n}_lt_{min_best_band_n}_total_n={total_n}_lt_30")

        # Gate 3: Spearman ρ ≥ 0.10
        rho = stats.get("spearman_rho")
        if rho is None or rho < min_rho:
            gates.append(f"spearman_rho={rho}_lt_{min_rho}")

        # Gate 4: selectivity in range
        pr = stats.get("pass_rate", 0)
        if pr < selectivity_range[0] or pr > selectivity_range[1]:
            gates.append(f"selectivity={pr:.4f}_outside_{selectivity_range}")

        # Gate 5: no single-symbol dominance (>70%) unless independently sufficient
        max_share = stats.get("max_symbol_share", 0)
        if max_share > 0.70:
            dominant_sym = stats.get("max_symbol", "")
            pse = stats.get("per_symbol_edge", {})
            sym_info = pse.get(dominant_sym, {})
            sym_best = sym_info.get("best_band")
            sym_sufficient = (
                sym_best is not None
                and sym_best.get("n_realized", 0) >= 10
                and sym_best.get("mean_realized_edge_pct", 0) > fee_threshold_pct
            )
            if not sym_sufficient:
                gates.append(
                    f"symbol_dominance_{dominant_sym}_share={max_share:.2f}_not_independently_sufficient"
                )

        results[cid] = {
            "candidate_id": cid,
            "promoted": len(gates) == 0,
            "n_gate_fails": len(gates),
            "gate_fails": gates,
            "best_band_edge": round(best_edge, 8),
            "best_band_n": best_n,
            "spearman_rho": rho,
            "pass_rate": round(pr, 4),
        }

    return results


# ── CSV Export ───────────────────────────────────────────────────────────

def export_replay_tables_csv(
    summary: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Export per-candidate summary as CSV for external analysis."""
    if path is None:
        path = REPLAY_TABLES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    per_candidate = summary.get("per_candidate", {})
    if not per_candidate:
        path.write_text("candidate_id,n_signals,pass_rate,mean_edge,spearman_rho\n")
        return

    header = [
        "candidate_id", "n_signals", "pass_rate",
        "mean_expected_edge_pct", "median_expected_edge_pct",
        "mean_realized_edge_pct", "mean_conviction",
        "spearman_rho", "max_symbol_share", "max_symbol",
        "control_pass_rate", "control_mean_edge_pct", "pass_rate_delta",
    ]

    lines = [",".join(header)]
    for cid in sorted(per_candidate.keys()):
        s = per_candidate[cid]
        row = [
            cid,
            str(s.get("n_signals", 0)),
            f"{s.get('pass_rate', 0):.4f}",
            f"{s.get('mean_expected_edge_pct', 0):.8f}",
            f"{s.get('median_expected_edge_pct', 0):.8f}",
            f"{s.get('mean_realized_edge_pct', 0):.8f}",
            f"{s.get('mean_conviction', 0):.6f}",
            f"{s.get('spearman_rho', '')}" if s.get("spearman_rho") is not None else "",
            f"{s.get('max_symbol_share', 0):.4f}",
            s.get("max_symbol", ""),
            f"{s.get('control_pass_rate', 0):.4f}",
            f"{s.get('control_mean_edge_pct', 0):.8f}",
            f"{s.get('pass_rate_delta', 0):.4f}",
        ]
        lines.append(",".join(row))

    path.write_text("\n".join(lines) + "\n")
    LOG.info("Replay tables CSV saved to %s", path)


# ── Orchestrator ─────────────────────────────────────────────────────────

def run_full_replay_pipeline(
    dry_run: bool = False,
    kline_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    base_url: str = BINANCE_FAPI_BASE,
) -> Dict[str, Any]:
    """Full P6B.5 pipeline: manifest → replay → summary → gates → artifacts.

    Returns a result dict with {records, summary, fast_fail, promotion, manifest}.
    """
    c1_config = C1Config()
    c2_config = C2Config()
    fee = DEFAULT_FEE_THRESHOLD_PCT

    LOG.info("=" * 60)
    LOG.info("P6B.5 Historical Replay — starting")
    LOG.info("=" * 60)

    # 0. Build and save manifest FIRST (before any other artifact)
    manifest = build_run_manifest(
        c1_config=c1_config,
        c2_config=c2_config,
        fee_threshold_pct=fee,
        n_episodes=len(load_episodes()),
        n_records=0,  # updated after replay
    )
    if not dry_run:
        save_run_manifest(manifest)
        LOG.info("Manifest saved (pre-replay)")

    # 1. Run replay (kline_cache saved after fetch for reproducibility)
    records = run_replay(
        c1_config=c1_config,
        c2_config=c2_config,
        fee_threshold_pct=fee,
        kline_cache=kline_cache,
        base_url=base_url,
        dry_run=dry_run,
    )

    # 2. Compute summary
    summary = compute_replay_summary(records)

    # 3. Fast-fail gates
    ff_verdicts = apply_fast_fail_gates(summary, fee_threshold_pct=fee)

    # 4. Promotion gates (only for survivors)
    promo_verdicts = apply_promotion_gates(summary, ff_verdicts, fee_threshold_pct=fee)

    # 5. Determine outcome
    any_survived = any(v.get("passed") for v in ff_verdicts.values())
    any_promoted = any(v.get("promoted") for v in promo_verdicts.values())

    outcome = "BRANCH_A_CONTAINMENT"
    if any_promoted:
        outcome = "P6C_LIVE_SHADOW"
    elif any_survived:
        outcome = "P6C_LIVE_SHADOW_CONDITIONAL"

    # 6. Update manifest with final stats and save all artifacts
    manifest["n_records_total"] = len(records)
    manifest["outcome"] = outcome
    manifest["fast_fail_verdicts"] = ff_verdicts
    manifest["promotion_verdicts"] = promo_verdicts

    if not dry_run:
        save_run_manifest(manifest)  # overwrite with final stats

        summary_path = REPLAY_SUMMARY_PATH
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        LOG.info("Summary saved to %s", summary_path)

        export_replay_tables_csv(summary)

    LOG.info("=" * 60)
    LOG.info("P6B.5 Outcome: %s", outcome)
    LOG.info("  Fast-fail survivors: %s", [
        cid for cid, v in ff_verdicts.items() if v.get("passed")
    ])
    LOG.info("  Promotion candidates: %s", [
        cid for cid, v in promo_verdicts.items() if v.get("promoted")
    ])
    LOG.info("=" * 60)

    return {
        "records": records,
        "summary": summary,
        "fast_fail": ff_verdicts,
        "promotion": promo_verdicts,
        "manifest": manifest,
        "outcome": outcome,
    }
