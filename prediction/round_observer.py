"""Round Observer — Binary Sleeve Layer 3.

Observes each 15-minute BTC Up/Down round from start to resolution,
combining Layer 1 (RTDS oracle) and Layer 2 (CLOB market) data into
a complete round record.

What it captures
----------------
* ``oracle_start`` / ``oracle_end`` — Chainlink BTC/USD at boundaries
* ``best_bid_ask_start`` — top-of-book at round start
* ``spread_start`` / ``spread_min`` / ``spread_max`` — spread statistics
* **Terminal snapshots** at T−30s, T−10s, T−3s:
    best_bid, best_ask, spread, implied_probability
* ``direction`` — actual oracle direction (UP / DOWN / FLAT)
* ``bundle_cost`` — hypothetical cost to buy 1 unit of the winning outcome
* ``net_ev_after_fee`` — hypothetical net EV at assumed fee rate
* ``resolution_source`` — how outcome was determined

No trading.  No reducer integration.  Pure observation.

Log output
----------
* ``logs/prediction/binary_rounds.jsonl`` — one record per completed round

Design invariants
-----------------
* **Append-only** — never mutates round log.
* **Isolated** — no imports from ``execution/`` or ``dashboard/``.
* **Deterministic** — all timestamps are UTC ISO-8601 or epoch ms.
* **Resilient** — missing data produces partial records, never crashes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prediction.round_observer")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROUNDS_LOG = _PROJECT_ROOT / "logs" / "prediction" / "binary_rounds.jsonl"
ENV_EVENTS_LOG = _PROJECT_ROOT / "logs" / "execution" / "environment_events.jsonl"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
# Discovery polling interval for the observer
DISCOVERY_POLL_S: float = float(os.environ.get("OBSERVER_DISCOVERY_POLL_S", "30"))

# Pre-round subscription lead time: subscribe this many seconds before start
PRE_SUBSCRIBE_S: float = float(os.environ.get("OBSERVER_PRE_SUBSCRIBE_S", "60"))

# Terminal snapshot offsets (seconds before round end)
SNAPSHOT_OFFSETS_S: List[int] = [30, 10, 3]

# Intraround sampling interval (seconds)
INTRAROUND_INTERVAL_S: int = int(os.environ.get("OBSERVER_INTRAROUND_INTERVAL_S", "60"))

# Assumed latency for EV calculations (ms)
ASSUMED_LATENCY_MS: int = int(os.environ.get("OBSERVER_LATENCY_MS", "250"))

# Fee rate (bps) for net EV calculation — Polymarket taker fee
FEE_RATE_BPS: float = float(os.environ.get("OBSERVER_FEE_BPS", "0"))

# Polymarket taker fee rate (fractional).  Effective fee per contract
# is ``fee_rate * min(price, 1 - price)``.  Default 2 % matches the
# standard Polymarket CLOB taker schedule.
POLYMARKET_FEE_RATE: float = float(
    os.environ.get("OBSERVER_POLYMARKET_FEE_RATE", "0.02")
)

# Discovery timeframe
OBSERVER_TIMEFRAME: str = os.environ.get("OBSERVER_TIMEFRAME", "15m")

# Maximum rounds to observe before exiting (0 = unlimited)
MAX_ROUNDS: int = int(os.environ.get("OBSERVER_MAX_ROUNDS", "0"))

# Oracle stale threshold — if oracle tick is older than this (ms),
# mark the boundary as stale
ORACLE_STALE_MS: int = int(os.environ.get("OBSERVER_ORACLE_STALE_MS", "10000"))


# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


# Hard ceiling for any single tail read — prevents regression if a caller
# accidentally passes a huge max_bytes.  8 MB keeps peak allocation well
# within safe bounds on a 4 GB box.
_TAIL_MAX_BYTES: int = 8_000_000


def _tail_lines(path: Path, max_bytes: int = 2_000_000) -> List[str]:
    """Read the last *max_bytes* of a file and return as lines.

    This avoids loading multi-GB files into memory — only the tail is
    read.  The first (partial) line is discarded since the seek may
    land mid-line.  Returns an empty list on any I/O error.

    *max_bytes* is silently clamped to ``_TAIL_MAX_BYTES`` (8 MB) so
    no caller can accidentally regress into full-file reads.
    """
    max_bytes = min(max_bytes, _TAIL_MAX_BYTES)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(size - max_bytes, 0)
            f.seek(start)
            data = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    lines = data.splitlines()
    # Drop the first element — it is likely a partial line from mid-seek
    if start > 0 and lines:
        lines = lines[1:]
    return lines


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def polymarket_taker_fee(price: float) -> float:
    """Compute Polymarket taker fee for one contract at *price*.

    Fee schedule: ``fee_rate * min(price, 1 - price)``.
    Returns 0.0 for invalid / out-of-range prices.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return POLYMARKET_FEE_RATE * min(price, 1.0 - price)


# ---------------------------------------------------------------------------
# Snapshot data structure
# ---------------------------------------------------------------------------
@dataclass
class TerminalSnapshot:
    """Snapshot of market state at a specific time offset."""
    offset_label: str  # "t_minus_30", "t_minus_10", "t_minus_3"
    captured_at: str = ""  # ISO-8601
    captured_at_ms: int = 0
    best_bid_up: Optional[float] = None
    best_ask_up: Optional[float] = None
    best_bid_down: Optional[float] = None
    best_ask_down: Optional[float] = None
    spread_up: Optional[float] = None
    spread_down: Optional[float] = None
    mid_up: Optional[float] = None
    mid_down: Optional[float] = None
    implied_prob_up: Optional[float] = None  # mid_up as implied probability
    oracle_price: Optional[float] = None
    oracle_ts_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "captured_at_ms": self.captured_at_ms,
            "best_bid_up": self.best_bid_up,
            "best_ask_up": self.best_ask_up,
            "best_bid_down": self.best_bid_down,
            "best_ask_down": self.best_ask_down,
            "spread_up": self.spread_up,
            "spread_down": self.spread_down,
            "mid_up": self.mid_up,
            "mid_down": self.mid_down,
            "implied_prob_up": self.implied_prob_up,
            "oracle_price": self.oracle_price,
            "oracle_ts_ms": self.oracle_ts_ms,
        }


@dataclass
class RoundState:
    """Accumulates state for one 15-minute round observation."""
    slug: str
    question: str
    timeframe: str
    up_token: str
    down_token: str
    round_start_iso: str  # Scheduled start (from market metadata)
    round_end_iso: str    # Scheduled end
    round_start_ts: float  # Unix seconds
    round_end_ts: float    # Unix seconds
    condition_id: str = ""

    # Oracle boundaries
    oracle_start: Optional[float] = None
    oracle_start_ts_ms: Optional[int] = None
    oracle_end: Optional[float] = None
    oracle_end_ts_ms: Optional[int] = None

    # Market state at round start
    start_best_bid_up: Optional[float] = None
    start_best_ask_up: Optional[float] = None
    start_best_bid_down: Optional[float] = None
    start_best_ask_down: Optional[float] = None
    start_spread_up: Optional[float] = None
    start_spread_down: Optional[float] = None

    # Spread tracking during round
    spreads_up: List[float] = field(default_factory=list)
    spreads_down: List[float] = field(default_factory=list)

    # Terminal snapshots
    snapshots: Dict[str, TerminalSnapshot] = field(default_factory=dict)

    # Resolution
    resolved: bool = False
    resolved_outcome: Optional[str] = None  # "Up" or "Down"
    resolution_source: str = "oracle_direction"  # or "market_resolved"
    market_resolved_received: bool = False

    # Tick-size changes during round
    tick_size_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Intraround periodic samples (every 60s)
    intraround_samples: List[Dict[str, Any]] = field(default_factory=list)

    # Event counts during round
    event_count: int = 0

    def _compute_intraround_stats(self) -> Dict[str, Any]:
        """Compute aggregate dislocation statistics from intraround samples.

        Includes IDCP fields: fee-adjusted bundle stats, dislocation duration
        windows (consecutive sub-1.0 samples), and depth statistics.
        """
        if not self.intraround_samples:
            return {"sample_count": 0}

        bundle_costs = [
            s["bundle_cost"] for s in self.intraround_samples
            if s.get("bundle_cost") is not None
        ]
        if not bundle_costs:
            return {"sample_count": len(self.intraround_samples)}

        sub_one = [c for c in bundle_costs if c < 1.0]
        stats: Dict[str, Any] = {
            "sample_count": len(bundle_costs),
            "min_bundle_cost": round(min(bundle_costs), 6),
            "max_bundle_cost": round(max(bundle_costs), 6),
            "mean_bundle_cost": round(sum(bundle_costs) / len(bundle_costs), 6),
            "dislocation_count": len(sub_one),
            "min_dislocation": round(min(sub_one), 6) if sub_one else None,
        }

        # --- IDCP: fee-adjusted bundle statistics ---
        fee_adj = [
            s["fee_adjusted_bundle"] for s in self.intraround_samples
            if s.get("fee_adjusted_bundle") is not None
        ]
        if fee_adj:
            fee_adj_sub_one = [c for c in fee_adj if c < 1.0]
            stats["min_fee_adjusted_bundle"] = round(min(fee_adj), 6)
            stats["max_fee_adjusted_bundle"] = round(max(fee_adj), 6)
            stats["mean_fee_adjusted_bundle"] = round(
                sum(fee_adj) / len(fee_adj), 6
            )
            stats["dislocation_fee_adjusted_count"] = len(fee_adj_sub_one)
            stats["min_fee_adjusted_dislocation"] = (
                round(min(fee_adj_sub_one), 6) if fee_adj_sub_one else None
            )

        # --- IDCP: dislocation duration windows ---
        # Track consecutive fee-adjusted sub-1.0 samples to estimate how
        # long dislocations persist.  Each sample is ~INTRAROUND_INTERVAL_S
        # apart; we record window count, max window, and total duration.
        # NOTE: uses fee_adjusted_bundle (not raw bundle_cost) — raw
        # dislocation means nothing if fees eat the edge.
        windows: List[float] = []  # durations in seconds
        current_window_start_s: Optional[float] = None

        for s in self.intraround_samples:
            fab = s.get("fee_adjusted_bundle")
            elapsed = s.get("elapsed_s")
            if fab is not None and fab < 1.0 and elapsed is not None:
                if current_window_start_s is None:
                    current_window_start_s = elapsed
            else:
                if current_window_start_s is not None and elapsed is not None:
                    duration = elapsed - current_window_start_s
                    windows.append(max(duration, 0.0))
                    current_window_start_s = None

        # Close any open window at end of samples
        if current_window_start_s is not None:
            last_elapsed = None
            for s in reversed(self.intraround_samples):
                if s.get("elapsed_s") is not None:
                    last_elapsed = s["elapsed_s"]
                    break
            if last_elapsed is not None:
                windows.append(max(last_elapsed - current_window_start_s, 0.0))

        stats["dislocation_window_count"] = len(windows)
        if windows:
            stats["max_dislocation_window_s"] = round(max(windows), 1)
            stats["mean_dislocation_window_s"] = round(
                sum(windows) / len(windows), 1
            )
            stats["total_dislocation_s"] = round(sum(windows), 1)

        # --- IDCP: depth statistics ---
        depths_up = [
            s["depth_up"] for s in self.intraround_samples
            if s.get("depth_up") is not None
        ]
        depths_down = [
            s["depth_down"] for s in self.intraround_samples
            if s.get("depth_down") is not None
        ]
        if depths_up:
            stats["mean_depth_up"] = round(sum(depths_up) / len(depths_up), 2)
            stats["min_depth_up"] = round(min(depths_up), 2)
        if depths_down:
            stats["mean_depth_down"] = round(
                sum(depths_down) / len(depths_down), 2
            )
            stats["min_depth_down"] = round(min(depths_down), 2)

        # --- Sync audit: leg timestamp skew statistics ---
        skew_values = [
            s["skew_ms"] for s in self.intraround_samples
            if s.get("skew_ms") is not None
        ]
        if skew_values:
            stats["skew_ms_min"] = min(skew_values)
            stats["skew_ms_max"] = max(skew_values)
            stats["skew_ms_mean"] = round(sum(skew_values) / len(skew_values), 0)
            stats["skew_samples_gt_1s"] = sum(1 for v in skew_values if v > 1000)
            stats["skew_samples_gt_30s"] = sum(1 for v in skew_values if v > 30000)

        # --- Eligible dislocation decision rule ---
        # A sample is "eligible" (simultaneously tradable) only if ALL:
        #   1. fee_adjusted_bundle < 1.0
        #   2. abs(skew_ms) <= SKEW_STRICT (1000ms) or SKEW_LENIENT (2000ms)
        #   3. Both legs fresh: staleness_up_ms <= 2000 AND staleness_down_ms <= 2000
        SKEW_STRICT_MS = 1000
        SKEW_LENIENT_MS = 2000
        FRESHNESS_MS = 2000

        eligible_strict: list[dict] = []
        eligible_lenient: list[dict] = []

        for s in self.intraround_samples:
            fab = s.get("fee_adjusted_bundle")
            skew = s.get("skew_ms")
            stale_up = s.get("staleness_up_ms")
            stale_down = s.get("staleness_down_ms")

            if fab is None or fab >= 1.0:
                continue
            if skew is None or stale_up is None or stale_down is None:
                continue
            if stale_up > FRESHNESS_MS or stale_down > FRESHNESS_MS:
                continue

            # Passes freshness — check skew tiers
            if skew <= SKEW_LENIENT_MS:
                eligible_lenient.append(s)
            if skew <= SKEW_STRICT_MS:
                eligible_strict.append(s)

        stats["eligible_strict_count"] = len(eligible_strict)
        stats["eligible_lenient_count"] = len(eligible_lenient)

        if eligible_strict:
            strict_skews = [s["skew_ms"] for s in eligible_strict]
            strict_fabs = [s["fee_adjusted_bundle"] for s in eligible_strict]
            stats["eligible_strict_skew_mean"] = round(
                sum(strict_skews) / len(strict_skews), 0
            )
            stats["eligible_strict_skew_max"] = max(strict_skews)
            stats["eligible_strict_min_fab"] = round(min(strict_fabs), 6)
            # Depth during eligible dislocations
            strict_depths_up = [s["depth_up"] for s in eligible_strict if s.get("depth_up") is not None]
            strict_depths_down = [s["depth_down"] for s in eligible_strict if s.get("depth_down") is not None]
            if strict_depths_up:
                stats["eligible_strict_median_depth_up"] = round(
                    sorted(strict_depths_up)[len(strict_depths_up) // 2], 2
                )
            if strict_depths_down:
                stats["eligible_strict_median_depth_down"] = round(
                    sorted(strict_depths_down)[len(strict_depths_down) // 2], 2
                )

        if eligible_lenient:
            lenient_skews = [s["skew_ms"] for s in eligible_lenient]
            lenient_fabs = [s["fee_adjusted_bundle"] for s in eligible_lenient]
            stats["eligible_lenient_skew_mean"] = round(
                sum(lenient_skews) / len(lenient_skews), 0
            )
            stats["eligible_lenient_skew_max"] = max(lenient_skews)
            stats["eligible_lenient_min_fab"] = round(min(lenient_fabs), 6)

        # Eligible window duration (strict): consecutive eligible samples
        eligible_windows: list[float] = []
        ew_start: Optional[float] = None
        eligible_elapsed_set = {s.get("elapsed_s") for s in eligible_strict}

        for s in self.intraround_samples:
            elapsed = s.get("elapsed_s")
            if elapsed in eligible_elapsed_set and elapsed is not None:
                if ew_start is None:
                    ew_start = elapsed
            else:
                if ew_start is not None and elapsed is not None:
                    eligible_windows.append(max(elapsed - ew_start, 0.0))
                    ew_start = None

        # Close open window
        if ew_start is not None:
            for s in reversed(self.intraround_samples):
                if s.get("elapsed_s") is not None:
                    eligible_windows.append(max(s["elapsed_s"] - ew_start, 0.0))
                    break

        stats["eligible_strict_window_count"] = len(eligible_windows)
        if eligible_windows:
            stats["eligible_strict_max_window_s"] = round(max(eligible_windows), 1)

        return stats

    def direction_from_oracle(self) -> Optional[str]:
        """Determine direction from oracle start/end prices."""
        if self.oracle_start is None or self.oracle_end is None:
            return None
        if self.oracle_end > self.oracle_start:
            return "UP"
        elif self.oracle_end < self.oracle_start:
            return "DOWN"
        return "FLAT"

    def compute_bundle_cost(self) -> Optional[float]:
        """Compute hypothetical bundle cost for buying winning outcome.

        Bundle cost = price_up + price_down.  In a perfect market this
        equals 1.0; any excess is the market maker's fee embedded in
        prices.  We use the best-ask (worst fill) for both sides.
        """
        direction = self.direction_from_oracle()
        if not direction or direction == "FLAT":
            return None

        # Use the last terminal snapshot (T-3s) or T-10s for pricing
        snap = self.snapshots.get("t_minus_3") or self.snapshots.get("t_minus_10")
        if snap is None:
            return None

        ask_up = snap.best_ask_up
        ask_down = snap.best_ask_down
        if ask_up is not None and ask_down is not None:
            return round(ask_up + ask_down, 6)
        return None

    def compute_net_ev(self, fee_bps: float = FEE_RATE_BPS) -> Optional[float]:
        """Compute hypothetical net EV after fees.

        If direction is UP, you'd buy the UP token at best_ask_up.
        Winning payout = 1.0.  Net EV = 1.0 - cost - fees.
        """
        direction = self.direction_from_oracle()
        if not direction or direction == "FLAT":
            return None

        snap = self.snapshots.get("t_minus_3") or self.snapshots.get("t_minus_10")
        if snap is None:
            return None

        if direction == "UP":
            cost = snap.best_ask_up
        else:
            cost = snap.best_ask_down

        if cost is None:
            return None

        fee = cost * fee_bps / 10_000
        net_ev = round(1.0 - cost - fee, 6)
        return net_ev

    def to_record(self) -> Dict[str, Any]:
        """Serialize to the final round record for logging."""
        direction = self.direction_from_oracle()
        bundle = self.compute_bundle_cost()
        net_ev = self.compute_net_ev()

        spreads_up_sorted = sorted(self.spreads_up) if self.spreads_up else []
        spreads_down_sorted = sorted(self.spreads_down) if self.spreads_down else []

        record: Dict[str, Any] = {
            "slug": self.slug,
            "question": self.question,
            "timeframe": self.timeframe,
            "condition_id": self.condition_id,
            "round_start": self.round_start_iso,
            "round_end": self.round_end_iso,
            "oracle_start": self.oracle_start,
            "oracle_start_ts_ms": self.oracle_start_ts_ms,
            "oracle_end": self.oracle_end,
            "oracle_end_ts_ms": self.oracle_end_ts_ms,
            "direction": direction,
            "start_book": {
                "best_bid_up": self.start_best_bid_up,
                "best_ask_up": self.start_best_ask_up,
                "best_bid_down": self.start_best_bid_down,
                "best_ask_down": self.start_best_ask_down,
                "spread_up": self.start_spread_up,
                "spread_down": self.start_spread_down,
            },
            "spread_stats_up": {
                "min": round(spreads_up_sorted[0], 6) if spreads_up_sorted else None,
                "max": round(spreads_up_sorted[-1], 6) if spreads_up_sorted else None,
                "mean": round(sum(spreads_up_sorted) / len(spreads_up_sorted), 6) if spreads_up_sorted else None,
                "samples": len(spreads_up_sorted),
            },
            "spread_stats_down": {
                "min": round(spreads_down_sorted[0], 6) if spreads_down_sorted else None,
                "max": round(spreads_down_sorted[-1], 6) if spreads_down_sorted else None,
                "mean": round(sum(spreads_down_sorted) / len(spreads_down_sorted), 6) if spreads_down_sorted else None,
                "samples": len(spreads_down_sorted),
            },
            "snapshots": {
                k: v.to_dict() for k, v in self.snapshots.items()
            },
            "intraround_samples": self.intraround_samples,
            "intraround_stats": self._compute_intraround_stats(),
            "bundle_cost": bundle,
            "net_ev_after_fee": net_ev,
            "latency_assumed_ms": ASSUMED_LATENCY_MS,
            "fee_bps": FEE_RATE_BPS,
            "resolved_outcome": self.resolved_outcome,
            "resolution_source": self.resolution_source,
            "market_resolved_received": self.market_resolved_received,
            "tick_size_changes": self.tick_size_changes,
            "event_count": self.event_count,
            "logged_at": _now_iso(),
        }

        # Oracle boundary alignment
        if self.oracle_start_ts_ms is not None:
            start_misalign = abs(
                self.oracle_start_ts_ms - int(self.round_start_ts * 1000)
            )
            record["oracle_start_misalign_ms"] = start_misalign
        if self.oracle_end_ts_ms is not None:
            end_misalign = abs(
                self.oracle_end_ts_ms - int(self.round_end_ts * 1000)
            )
            record["oracle_end_misalign_ms"] = end_misalign

        return record


# ---------------------------------------------------------------------------
# Oracle reader — reads RTDS log for boundary ticks
# ---------------------------------------------------------------------------
def read_oracle_tick_near(
    target_ts: float,
    oracle_log: Path = _PROJECT_ROOT / "logs" / "prediction" / "rtds_oracle.jsonl",
    max_age_ms: int = ORACLE_STALE_MS,
    direction: str = "before",
) -> Optional[Dict[str, Any]]:
    """Find the oracle tick closest to *target_ts* (Unix seconds).

    Parameters
    ----------
    target_ts
        Target time in Unix seconds.
    oracle_log
        Path to the RTDS oracle JSONL log.
    max_age_ms
        Maximum acceptable distance (in ms) from target.
    direction
        ``"before"`` — last tick at or before target (for round start/end).
        ``"nearest"`` — closest tick in either direction.

    Returns
    -------
    Dict with ``price``, ``oracle_ts_ms``, ``latency_ms``, or None.
    """
    if not oracle_log.exists():
        return None

    target_ms = int(target_ts * 1000)
    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")

    # Read from the end for efficiency (recent ticks are more relevant)
    lines = _tail_lines(oracle_log)
    if not lines:
        return None

    # Scan in reverse for efficiency
    for line in reversed(lines[-5000:]):  # Only scan recent ticks
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        ots = rec.get("oracle_ts_ms")
        if ots is None:
            continue

        dist = abs(ots - target_ms)
        if direction == "before" and ots > target_ms:
            continue

        if dist < best_dist:
            best_dist = dist
            best = rec

        # If we've moved far past the target, stop scanning
        if ots < target_ms - max_age_ms * 5:
            break

    if best is not None and best_dist <= max_age_ms:
        return best
    return None


def read_latest_oracle_tick(
    oracle_log: Path = _PROJECT_ROOT / "logs" / "prediction" / "rtds_oracle.jsonl",
) -> Optional[Dict[str, Any]]:
    """Read the most recent oracle tick from the log."""
    if not oracle_log.exists():
        return None
    try:
        lines = _tail_lines(oracle_log, max_bytes=500_000)
        if not lines:
            return None
        # Last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# CLOB state reader — reads from live CLOB client or market log
# ---------------------------------------------------------------------------
def read_clob_state_for_asset(
    asset_id: str,
    market_log: Path = _PROJECT_ROOT / "logs" / "prediction" / "clob_market.jsonl",
) -> Optional[Dict[str, Any]]:
    """Read the latest best_bid/best_ask for an asset from the CLOB log.

    Scans recent log entries for best_bid_ask or price_change events
    with populated best_bid/best_ask for the given asset.
    """
    if not market_log.exists():
        return None

    lines = _tail_lines(market_log)

    # Scan in reverse for the latest state
    for line in reversed(lines[-5000:]):
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if rec.get("asset_id") != asset_id:
            continue
        bid = rec.get("best_bid")
        ask = rec.get("best_ask")
        if bid is not None and ask is not None:
            return {
                "best_bid": bid,
                "best_ask": ask,
                "spread": round(ask - bid, 6),
                "mid": round((bid + ask) / 2, 6),
                "ts_arrival_ms": rec.get("ts_arrival_ms"),
                "event_type": rec.get("event_type"),
            }
    return None


def read_clob_ask_depth(
    asset_id: str,
    market_log: Path = _PROJECT_ROOT / "logs" / "prediction" / "clob_market.jsonl",
) -> Optional[float]:
    """Return the most recent ask-side depth (size) for *asset_id*.

    Scans recent ``price_change`` events where ``side == "SELL"`` and the
    level ``price`` matches the current ``best_ask``.  Returns the ``size``
    field (number of contracts available at best ask), or ``None`` if no
    suitable record is found.
    """
    if not market_log.exists():
        return None

    lines = _tail_lines(market_log)

    for line in reversed(lines[-5000:]):
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if rec.get("asset_id") != asset_id:
            continue
        if rec.get("event_type") != "price_change":
            continue
        if rec.get("side") != "SELL":
            continue
        # Level price should equal best_ask (this is the top-of-book ask)
        price = rec.get("price")
        best_ask = rec.get("best_ask")
        if price is not None and best_ask is not None and price == best_ask:
            size_raw = rec.get("size")
            if size_raw is not None:
                try:
                    return float(size_raw)
                except (TypeError, ValueError):
                    pass
    return None


# ---------------------------------------------------------------------------
# Round Observer
# ---------------------------------------------------------------------------
class RoundObserver:
    """Observes 15-minute BTC Up/Down rounds and logs complete records.

    Lifecycle per round:
    1. Discovery poll finds current active market
    2. At round start: capture oracle price + opening book state
    3. During round: track spreads from CLOB log
    4. At T-30s / T-10s / T-3s: capture terminal snapshots
    5. At round end: capture oracle end price, determine direction
    6. Write complete round record to binary_rounds.jsonl
    """

    def __init__(
        self,
        rounds_log: Path = ROUNDS_LOG,
        env_events_log: Path = ENV_EVENTS_LOG,
        oracle_log: Optional[Path] = None,
        market_log: Optional[Path] = None,
        discovery_poll_s: float = DISCOVERY_POLL_S,
        timeframe: str = OBSERVER_TIMEFRAME,
        max_rounds: int = MAX_ROUNDS,
    ) -> None:
        self.rounds_log = rounds_log
        self.env_events_log = env_events_log
        self.oracle_log = oracle_log or (_PROJECT_ROOT / "logs" / "prediction" / "rtds_oracle.jsonl")
        self.market_log = market_log or (_PROJECT_ROOT / "logs" / "prediction" / "clob_market.jsonl")
        self.discovery_poll_s = discovery_poll_s
        self.timeframe = timeframe
        self.max_rounds = max_rounds

        self._running: bool = False
        self._current_round: Optional[RoundState] = None
        self._completed_rounds: int = 0

    # ----- snapshot capture -----

    def _capture_intraround_sample(self, rnd: RoundState) -> Dict[str, Any]:
        """Capture a periodic intraround sample for dislocation analysis.

        Returns a compact dict with ask_up, ask_down, bundle_cost, spreads,
        fee-adjusted bundle, ask-side depth, and per-leg timestamps for
        synchronization audit (IDCP fields).
        """
        up_state = read_clob_state_for_asset(rnd.up_token, self.market_log)
        down_state = read_clob_state_for_asset(rnd.down_token, self.market_log)

        ask_up = up_state["best_ask"] if up_state else None
        ask_down = down_state["best_ask"] if down_state else None
        spread_up = up_state["spread"] if up_state else None
        spread_down = down_state["spread"] if down_state else None
        mid_up = up_state["mid"] if up_state else None
        mid_down = down_state["mid"] if down_state else None

        # --- Leg timestamp extraction for synchronization audit ---
        sample_ts_ms = int(time.time() * 1000)
        ts_up_ms = up_state.get("ts_arrival_ms") if up_state else None
        ts_down_ms = down_state.get("ts_arrival_ms") if down_state else None
        skew_ms: Optional[int] = None
        staleness_up_ms: Optional[int] = None
        staleness_down_ms: Optional[int] = None
        if ts_up_ms is not None:
            staleness_up_ms = sample_ts_ms - int(ts_up_ms)
        if ts_down_ms is not None:
            staleness_down_ms = sample_ts_ms - int(ts_down_ms)
        if ts_up_ms is not None and ts_down_ms is not None:
            skew_ms = abs(int(ts_up_ms) - int(ts_down_ms))

        bundle_cost: Optional[float] = None
        if ask_up is not None and ask_down is not None:
            bundle_cost = round(ask_up + ask_down, 6)

        # --- IDCP fields (v7.9) ---
        # Fee-adjusted bundle: bundle + Polymarket taker fees on each leg
        fee_up: Optional[float] = None
        fee_down: Optional[float] = None
        fee_adjusted_bundle: Optional[float] = None
        bundle_sub_1: Optional[bool] = None
        fee_adjusted_sub_1: Optional[bool] = None

        if ask_up is not None:
            fee_up = round(polymarket_taker_fee(ask_up), 6)
        if ask_down is not None:
            fee_down = round(polymarket_taker_fee(ask_down), 6)
        if bundle_cost is not None and fee_up is not None and fee_down is not None:
            fee_adjusted_bundle = round(bundle_cost + fee_up + fee_down, 6)
            bundle_sub_1 = bundle_cost < 1.0
            fee_adjusted_sub_1 = fee_adjusted_bundle < 1.0

        # Ask-side depth (contracts available at best ask)
        depth_up = read_clob_ask_depth(rnd.up_token, self.market_log)
        depth_down = read_clob_ask_depth(rnd.down_token, self.market_log)

        return {
            "ts": _now_iso(),
            "elapsed_s": round(time.time() - rnd.round_start_ts, 1),
            "ask_up": ask_up,
            "ask_down": ask_down,
            "bundle_cost": bundle_cost,
            "spread_up": spread_up,
            "spread_down": spread_down,
            "mid_up": mid_up,
            "mid_down": mid_down,
            # IDCP fields
            "fee_up": fee_up,
            "fee_down": fee_down,
            "fee_adjusted_bundle": fee_adjusted_bundle,
            "bundle_sub_1": bundle_sub_1,
            "fee_adjusted_sub_1": fee_adjusted_sub_1,
            "depth_up": depth_up,
            "depth_down": depth_down,
            # Sync audit fields (v7.9-W6)
            "sample_ts_ms": sample_ts_ms,
            "ts_up_ms": ts_up_ms,
            "ts_down_ms": ts_down_ms,
            "skew_ms": skew_ms,
            "staleness_up_ms": staleness_up_ms,
            "staleness_down_ms": staleness_down_ms,
        }

    def _capture_book_state(self, up_token: str, down_token: str) -> Dict[str, Any]:
        """Capture current best_bid/ask for both UP and DOWN tokens."""
        up_state = read_clob_state_for_asset(up_token, self.market_log)
        down_state = read_clob_state_for_asset(down_token, self.market_log)
        return {
            "up": up_state,
            "down": down_state,
        }

    def _capture_terminal_snapshot(
        self,
        label: str,
        rnd: RoundState,
    ) -> TerminalSnapshot:
        """Capture a terminal snapshot at T-N seconds."""
        snap = TerminalSnapshot(offset_label=label)
        snap.captured_at = _now_iso()
        snap.captured_at_ms = _now_ms()

        # CLOB state
        up_state = read_clob_state_for_asset(rnd.up_token, self.market_log)
        down_state = read_clob_state_for_asset(rnd.down_token, self.market_log)

        if up_state:
            snap.best_bid_up = up_state["best_bid"]
            snap.best_ask_up = up_state["best_ask"]
            snap.spread_up = up_state["spread"]
            snap.mid_up = up_state["mid"]
            snap.implied_prob_up = up_state["mid"]  # Mid price ≈ implied probability

        if down_state:
            snap.best_bid_down = down_state["best_bid"]
            snap.best_ask_down = down_state["best_ask"]
            snap.spread_down = down_state["spread"]
            snap.mid_down = down_state["mid"]

        # Oracle state
        tick = read_latest_oracle_tick(self.oracle_log)
        if tick:
            snap.oracle_price = tick.get("price")
            snap.oracle_ts_ms = tick.get("oracle_ts_ms")

        return snap

    def _capture_start_state(self, rnd: RoundState) -> None:
        """Capture oracle + book state at round start."""
        # Oracle at start boundary
        tick = read_oracle_tick_near(
            rnd.round_start_ts,
            self.oracle_log,
            direction="before",
        )
        if tick:
            rnd.oracle_start = tick.get("price")
            rnd.oracle_start_ts_ms = tick.get("oracle_ts_ms")
        else:
            # Use latest available tick if no exact boundary match
            latest = read_latest_oracle_tick(self.oracle_log)
            if latest:
                rnd.oracle_start = latest.get("price")
                rnd.oracle_start_ts_ms = latest.get("oracle_ts_ms")

        # Book state at start
        book = self._capture_book_state(rnd.up_token, rnd.down_token)
        if book["up"]:
            rnd.start_best_bid_up = book["up"]["best_bid"]
            rnd.start_best_ask_up = book["up"]["best_ask"]
            rnd.start_spread_up = book["up"]["spread"]
        if book["down"]:
            rnd.start_best_bid_down = book["down"]["best_bid"]
            rnd.start_best_ask_down = book["down"]["best_ask"]
            rnd.start_spread_down = book["down"]["spread"]

    def _capture_end_state(self, rnd: RoundState) -> None:
        """Capture oracle state at round end."""
        tick = read_oracle_tick_near(
            rnd.round_end_ts,
            self.oracle_log,
            direction="before",
        )
        if tick:
            rnd.oracle_end = tick.get("price")
            rnd.oracle_end_ts_ms = tick.get("oracle_ts_ms")
        else:
            latest = read_latest_oracle_tick(self.oracle_log)
            if latest:
                rnd.oracle_end = latest.get("price")
                rnd.oracle_end_ts_ms = latest.get("oracle_ts_ms")

    def _collect_spreads(self, rnd: RoundState) -> None:
        """Scan CLOB log for spread data during the round window."""
        if not self.market_log.exists():
            return

        start_ms = int(rnd.round_start_ts * 1000)
        end_ms = int(rnd.round_end_ts * 1000)

        lines = _tail_lines(self.market_log, max_bytes=4_000_000)

        for line in reversed(lines[-10000:]):
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            ts = rec.get("ts_arrival_ms", 0)
            if ts < start_ms:
                break
            if ts > end_ms:
                continue

            rnd.event_count += 1
            asset_id = rec.get("asset_id", "")
            bid = rec.get("best_bid")
            ask = rec.get("best_ask")

            if bid is not None and ask is not None:
                spread = round(ask - bid, 6)
                if asset_id == rnd.up_token:
                    rnd.spreads_up.append(spread)
                elif asset_id == rnd.down_token:
                    rnd.spreads_down.append(spread)

            # Track tick_size_change events
            if rec.get("event_type") == "tick_size_change":
                rnd.tick_size_changes.append({
                    "ts_arrival_ms": ts,
                    "old_tick_size": rec.get("old_tick_size"),
                    "new_tick_size": rec.get("new_tick_size"),
                })

    def _check_market_resolution(self, rnd: RoundState) -> None:
        """Check CLOB log for market_resolved events for this round's market."""
        if not self.market_log.exists():
            return

        end_ms = int(rnd.round_end_ts * 1000)

        lines = _tail_lines(self.market_log)

        for line in reversed(lines[-2000:]):
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if rec.get("event_type") != "market_resolved":
                continue

            ts = rec.get("ts_arrival_ms", 0)
            # Resolution should come near/after round end
            if ts < end_ms - 60_000:  # At least within 60s before end
                break

            # Check if this resolution is for one of our tokens
            winning = rec.get("winning_asset_id", "")
            if winning == rnd.up_token:
                rnd.market_resolved_received = True
                rnd.resolved_outcome = "Up"
                rnd.resolution_source = "market_resolved"
                return
            elif winning == rnd.down_token:
                rnd.market_resolved_received = True
                rnd.resolved_outcome = "Down"
                rnd.resolution_source = "market_resolved"
                return

    def _finalize_round(self, rnd: RoundState) -> Dict[str, Any]:
        """Finalize a round: capture end state, collect spreads, write log."""
        logger.info(
            "[observer] finalizing round %s (%.0fs after end)",
            rnd.slug,
            time.time() - rnd.round_end_ts,
        )

        # Capture oracle end
        self._capture_end_state(rnd)

        # Collect spread data from log
        self._collect_spreads(rnd)

        # Check for market_resolved event
        self._check_market_resolution(rnd)

        # Determine direction
        if not rnd.resolved_outcome:
            direction = rnd.direction_from_oracle()
            if direction and direction != "FLAT":
                rnd.resolved_outcome = "Up" if direction == "UP" else "Down"
                rnd.resolution_source = "oracle_direction"

        rnd.resolved = True
        record = rnd.to_record()

        # Write to rounds log
        _append_jsonl(self.rounds_log, record)
        self._completed_rounds += 1

        logger.info(
            "[observer] round #%d complete: %s direction=%s oracle=%.2f→%.2f "
            "bundle=%.4f net_ev=%.4f events=%d",
            self._completed_rounds,
            rnd.slug,
            record.get("direction", "?"),
            rnd.oracle_start or 0,
            rnd.oracle_end or 0,
            record.get("bundle_cost") or 0,
            record.get("net_ev_after_fee") or 0,
            rnd.event_count,
        )

        return record

    # ----- main observation loop -----

    async def run(self) -> None:
        """Main observation loop — discover, observe, finalize rounds."""
        from prediction.market_discovery import discover_btc_updown_markets

        self._running = True
        logger.info(
            "[observer] Round Observer starting — timeframe=%s log=%s",
            self.timeframe,
            self.rounds_log,
        )
        _ensure_parent(self.rounds_log)
        _ensure_parent(self.env_events_log)

        while self._running:
            try:
                await self._observe_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[observer] cycle error: %s", exc, exc_info=True)
                await asyncio.sleep(10)

            # Check round limit
            if self.max_rounds > 0 and self._completed_rounds >= self.max_rounds:
                logger.info(
                    "[observer] reached max_rounds=%d — stopping",
                    self.max_rounds,
                )
                break

        logger.info(
            "[observer] shutdown — completed %d rounds",
            self._completed_rounds,
        )

    async def _observe_cycle(self) -> None:
        """One observation cycle: find market, track round, finalize."""
        from prediction.market_discovery import discover_btc_updown_markets

        # Discover current market
        snap = discover_btc_updown_markets(
            timeframes=[self.timeframe],
            safety_buffer_s=30,  # Accept markets with at least 30s left
        )

        target = (
            snap.current_15m if self.timeframe == "15m" else snap.current_5m
        )

        if target is None:
            logger.info(
                "[observer] no active %s market — waiting %.0fs",
                self.timeframe,
                self.discovery_poll_s,
            )
            await asyncio.sleep(self.discovery_poll_s)
            return

        now = time.time()
        time_to_end = target.end_ts - now

        # If less than 5 seconds left, skip this window — too late
        if time_to_end < 5:
            logger.debug(
                "[observer] market %s expires in %.0fs — too late, waiting",
                target.slug,
                time_to_end,
            )
            await asyncio.sleep(time_to_end + 2)
            return

        # If already tracking this round, don't re-initialize
        if self._current_round and self._current_round.slug == target.slug:
            pass
        else:
            # Start tracking new round
            # Compute round start from slug timestamp or endDate - duration
            duration_s = 900 if self.timeframe == "15m" else 300
            round_start_ts = target.end_ts - duration_s
            round_start_iso = datetime.fromtimestamp(
                round_start_ts, tz=timezone.utc
            ).isoformat()

            self._current_round = RoundState(
                slug=target.slug,
                question=target.question,
                timeframe=target.timeframe,
                up_token=target.up_token,
                down_token=target.down_token,
                round_start_iso=round_start_iso,
                round_end_iso=target.end_date_utc,
                round_start_ts=round_start_ts,
                round_end_ts=target.end_ts,
                condition_id=target.condition_id,
            )

            logger.info(
                "[observer] tracking round: %s (%.0fs remaining)",
                target.slug,
                time_to_end,
            )

            # Capture start state
            self._capture_start_state(self._current_round)

        rnd = self._current_round
        assert rnd is not None

        # Schedule intraround samples + terminal snapshots + finalization
        now = time.time()
        time_to_end = rnd.round_end_ts - now

        # --- Intraround periodic sampling (every INTRAROUND_INTERVAL_S) ---
        # Sample from now until T-30, then terminal snapshots take over
        first_terminal_offset = max(SNAPSHOT_OFFSETS_S)  # 30
        while self._running:
            now = time.time()
            time_to_end = rnd.round_end_ts - now
            if time_to_end <= first_terminal_offset:
                break  # Hand off to terminal snapshot loop

            sample = self._capture_intraround_sample(rnd)
            rnd.intraround_samples.append(sample)
            bc = sample.get("bundle_cost")
            bc_str = f"{bc:.4f}" if bc is not None else "None"
            logger.debug(
                "[observer] intraround sample #%d: elapsed=%.0fs bundle=%s",
                len(rnd.intraround_samples),
                sample.get("elapsed_s", 0),
                bc_str,
            )

            # Sleep until next sample or until terminal snapshots begin
            now = time.time()
            time_to_first_terminal = rnd.round_end_ts - now - first_terminal_offset
            sleep_s = min(INTRAROUND_INTERVAL_S, max(0, time_to_first_terminal))
            if sleep_s <= 0:
                break
            await asyncio.sleep(sleep_s)

        if not self._running:
            return

        # --- Terminal snapshots at T-30, T-10, T-3 ---
        now = time.time()
        time_to_end = rnd.round_end_ts - now

        # Take snapshots at T-30, T-10, T-3
        for offset in SNAPSHOT_OFFSETS_S:
            label = f"t_minus_{offset}"
            if label in rnd.snapshots:
                continue  # Already captured

            wait = time_to_end - offset
            if wait > 0:
                logger.debug(
                    "[observer] scheduling snapshot %s in %.0fs",
                    label,
                    wait,
                )
                await asyncio.sleep(wait)

                if not self._running:
                    return

                # Recalculate time_to_end after sleep
                now = time.time()
                time_to_end = rnd.round_end_ts - now

                snap_data = self._capture_terminal_snapshot(label, rnd)
                rnd.snapshots[label] = snap_data
                logger.info(
                    "[observer] snapshot %s: bid_up=%s ask_up=%s spread=%s oracle=%.2f",
                    label,
                    snap_data.best_bid_up,
                    snap_data.best_ask_up,
                    snap_data.spread_up,
                    snap_data.oracle_price or 0,
                )
            elif wait > -offset:
                # We missed this snapshot window but round isn't over
                snap_data = self._capture_terminal_snapshot(label, rnd)
                rnd.snapshots[label] = snap_data
                logger.info("[observer] late snapshot %s captured", label)

        # Wait for round end + small buffer for resolution event
        now = time.time()
        remaining = rnd.round_end_ts - now
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Post-round buffer: wait a few seconds for market_resolved event
        await asyncio.sleep(5)

        if not self._running:
            return

        # Finalize
        self._finalize_round(rnd)
        self._current_round = None

    def stop(self) -> None:
        """Signal graceful shutdown."""
        self._running = False


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the Round Observer as a standalone process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    observer = RoundObserver()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, observer.stop)

    try:
        loop.run_until_complete(observer.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("[observer] shutdown complete")


if __name__ == "__main__":
    main()
