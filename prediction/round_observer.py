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

# Assumed latency for EV calculations (ms)
ASSUMED_LATENCY_MS: int = int(os.environ.get("OBSERVER_LATENCY_MS", "250"))

# Fee rate (bps) for net EV calculation — Polymarket taker fee
FEE_RATE_BPS: float = float(os.environ.get("OBSERVER_FEE_BPS", "0"))

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


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # Event counts during round
    event_count: int = 0

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
    try:
        with open(oracle_log, "r") as f:
            # Read last N lines (tail)
            lines = f.readlines()
    except OSError:
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
        with open(oracle_log, "r") as f:
            lines = f.readlines()
        if not lines:
            return None
        # Last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
    except (OSError, json.JSONDecodeError):
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

    try:
        with open(market_log, "r") as f:
            lines = f.readlines()
    except OSError:
        return None

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

        try:
            with open(self.market_log, "r") as f:
                lines = f.readlines()
        except OSError:
            return

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

        try:
            with open(self.market_log, "r") as f:
                lines = f.readlines()
        except OSError:
            return

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

        # Schedule terminal snapshots and finalization
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
