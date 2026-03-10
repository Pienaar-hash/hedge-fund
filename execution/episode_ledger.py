"""
Episode Ledger — Completed Trade Cycle Tracking (v7.9-E2)

This module derives completed trading episodes from execution logs.
It is READ-ONLY observability — it does NOT influence execution decisions.

Purpose:
  - Track completed trade cycles (entry → exit)
  - Prevent "snapshot blindness" (positions_state == [] ≠ no trading)
  - Provide historical participation visibility
  - Support postmortem analysis

An episode is:
  - A position that was opened AND closed
  - Identified by symbol + position side + entry window
  - Contains: entry/exit times, regimes, PnL, fees, exit reason, duration

v7.9-E2 Fills-Faithful Changes:
  - Reads ALL log files (current + rotated) for complete fill history
  - Filters to event_type='order_fill' AND executedQty > 0 (no ghosts)
  - Tracks multi-fill exits properly (exit_fills is real count)
  - Excess exits (more exits than entry qty) are counted as orphans
  - Reconciliation summary: fills_total, fills_consumed, fills_orphaned
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from dateutil import parser as dateparser

from execution.exit_reason_normalizer import normalize_exit_reason as _normalize_exit

logger = logging.getLogger(__name__)

# State file path
EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
EXECUTION_LOG_DIR = Path("logs/execution")
EXECUTION_LOG_PATH = Path("logs/execution/orders_executed.jsonl")
DOCTRINE_LOG_PATH = Path("logs/doctrine_events.jsonl")
DLE_SHADOW_LOG_PATH = Path("logs/execution/dle_shadow_events.jsonl")

# B.4 matching constants
MATCH_WINDOW_NARROW_S = 120.0   # first pass: ±120 seconds
MATCH_WINDOW_WIDE_S = 600.0     # fallback: ±600 seconds
EPISODE_UID_HASH_LEN = 12       # EP_<sha256_12>
EPISODE_LEDGER_SCHEMA_V2 = "episode_ledger_v2"


@dataclass
class Episode:
    """A completed trade cycle: entry → exit."""
    
    episode_id: str
    symbol: str
    side: str  # LONG or SHORT
    
    # Timing
    entry_ts: str
    exit_ts: str
    duration_hours: float
    
    # Execution
    entry_fills: int
    exit_fills: int
    entry_notional: float
    exit_notional: float
    total_qty: float
    avg_entry_price: float
    avg_exit_price: float
    
    # PnL
    gross_pnl: float
    fees: float
    net_pnl: float
    
    # Context
    regime_at_entry: str
    regime_at_exit: str
    exit_reason: str  # Canonical DLE exit reason (TAKE_PROFIT, STOP_LOSS, etc.)
    exit_reason_raw: str = ""  # Original raw exit reason for provenance
    
    # Metadata
    strategy: str = "unknown"
    
    # ── Scoring fields (v7.9-S1: audit-grade joinability) ──────────────
    # These fields make every episode attributable to the intent that
    # triggered it.  Without them, we cannot compute conviction deciles
    # or prove edge.
    intent_id: str = ""          # join key → orders_attempted / score_decomposition
    attempt_id: str = ""         # join key → sizing_snapshots
    confidence: float = 0.0      # registry confidence at entry [0, 1.5]
    hybrid_score: float = 0.0    # composite signal quality [0, 1]
    conviction_score: float = 0.0  # conviction engine output [0, 1]
    conviction_band: str = ""    # very_low / low / medium / high / very_high
    entry_regime_confidence: float = 0.0  # sentinel-X confidence at entry
    expected_edge: float = 0.0   # predicted edge from conviction model
    engine_source: str = ""      # hydra / legacy / unknown
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeLedger:
    """Container for all completed episodes."""
    
    episodes: list[Episode] = field(default_factory=list)
    episodes_v2: list = field(default_factory=list)  # list[EpisodeV2] — B.4
    last_rebuild_ts: str = ""
    stats: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        d: dict = {
            "episodes": [e.to_dict() for e in self.episodes],
            "last_rebuild_ts": self.last_rebuild_ts,
            "stats": self.stats,
            "episode_count": len(self.episodes),
        }
        # B.4: V2 surface (authority-bound episodes)
        if self.episodes_v2:
            d["schema_version"] = EPISODE_LEDGER_SCHEMA_V2
            d["episodes_v2"] = [e.to_dict() for e in self.episodes_v2]
        return d


# ---------------------------------------------------------------------------
# B.4 — Authority binding data structures
# ---------------------------------------------------------------------------

@dataclass
class AuthorityRef:
    """Cross-reference to a DLE authority event."""
    request_id: Optional[str] = None
    decision_id: Optional[str] = None
    permit_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuthorityFlags:
    """Explicit flags for missing / ambiguous binding."""
    entry_missing: bool = False
    exit_missing: bool = False
    entry_ambiguous: bool = False
    exit_ambiguous: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeV2:
    """V1 Episode fields + authority chain + deterministic UID."""

    # --- all V1 fields ---
    episode_id: str
    episode_uid: str  # EP_<sha256_12> — stable across rebuilds
    symbol: str
    side: str
    entry_ts: str
    exit_ts: str
    duration_hours: float
    entry_fills: int
    exit_fills: int
    entry_notional: float
    exit_notional: float
    total_qty: float
    avg_entry_price: float
    avg_exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    regime_at_entry: str
    regime_at_exit: str
    exit_reason: str
    exit_reason_raw: str
    strategy: str

    # --- B.4 authority chain ---
    authority_entry: AuthorityRef = field(default_factory=AuthorityRef)
    authority_exit: AuthorityRef = field(default_factory=AuthorityRef)
    authority_flags: AuthorityFlags = field(default_factory=AuthorityFlags)
    regime_bindable: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Nest authority for cleaner JSON shape
        d["authority"] = {
            "entry": d.pop("authority_entry"),
            "exit": d.pop("authority_exit"),
        }
        return d


# ---------------------------------------------------------------------------
# B.4 — Shadow log index (LINK + DECISION)
# ---------------------------------------------------------------------------

@dataclass
class _LinkRecord:
    """Parsed LINK event for indexing."""
    ts_unix: float
    ts_iso: str
    request_id: str
    decision_id: str
    permit_id: Optional[str]
    symbol: str
    action: str  # ENTRY | EXIT
    strategy: str


@dataclass
class _DecisionRecord:
    """Parsed DECISION event for regime binding."""
    decision_id: str
    context_snapshot: Dict[str, Any]
    provenance: Dict[str, Any]


def _iso_to_unix(ts_iso: str) -> Optional[float]:
    """Convert ISO timestamp to Unix seconds. Returns None on failure."""
    try:
        dt = dateparser.parse(ts_iso)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_shadow_indexes() -> Tuple[
    Dict[Tuple[str, str], List[_LinkRecord]],
    Dict[Tuple[str, str], List[_LinkRecord]],
    Dict[str, _DecisionRecord],
]:
    """
    Load LINK and DECISION events from DLE shadow log.

    Returns:
      links_entry: {(symbol, strategy): [_LinkRecord sorted by ts_unix]}
      links_exit:  {(symbol, strategy): [_LinkRecord sorted by ts_unix]}
      decision_index: {decision_id: _DecisionRecord}

    Fail-open: returns empty indexes if log is missing or unreadable.
    """
    links_entry: Dict[Tuple[str, str], List[_LinkRecord]] = defaultdict(list)
    links_exit: Dict[Tuple[str, str], List[_LinkRecord]] = defaultdict(list)
    decision_index: Dict[str, _DecisionRecord] = {}

    if not DLE_SHADOW_LOG_PATH.exists():
        logger.info("B.4: No shadow log found at %s — authority binding skipped", DLE_SHADOW_LOG_PATH)
        return dict(links_entry), dict(links_exit), decision_index

    try:
        with open(DLE_SHADOW_LOG_PATH, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    evt = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                event_type = evt.get("event_type", "")
                payload = evt.get("payload", {})

                if event_type == "LINK":
                    # Prefer payload.ts (B.4 addition), fall back to top-level ts
                    ts_iso = payload.get("ts") or evt.get("ts", "")
                    ts_unix = _iso_to_unix(ts_iso)
                    if ts_unix is None:
                        continue
                    rec = _LinkRecord(
                        ts_unix=ts_unix,
                        ts_iso=ts_iso,
                        request_id=payload.get("request_id", ""),
                        decision_id=payload.get("decision_id", ""),
                        permit_id=payload.get("permit_id"),
                        symbol=payload.get("symbol", ""),
                        action=payload.get("requested_action", ""),
                        strategy=payload.get("strategy", ""),
                    )
                    key = (rec.symbol, rec.strategy)
                    if rec.action == "ENTRY":
                        links_entry[key].append(rec)
                    elif rec.action == "EXIT":
                        links_exit[key].append(rec)
                    # Also store in wildcard bucket for fallback
                    wk = (rec.symbol, "*")
                    if rec.action == "ENTRY":
                        links_entry[wk].append(rec)
                    elif rec.action == "EXIT":
                        links_exit[wk].append(rec)

                elif event_type == "DECISION":
                    dec_id = payload.get("decision_id", "")
                    if dec_id:
                        decision_index[dec_id] = _DecisionRecord(
                            decision_id=dec_id,
                            context_snapshot=payload.get("context_snapshot", {}),
                            provenance=payload.get("provenance", {}),
                        )
    except Exception as e:
        logger.warning("B.4: Failed to load shadow log: %s — authority binding skipped", e)
        return {}, {}, {}

    # Sort each bucket by ts_unix for binary search
    for bucket in links_entry.values():
        bucket.sort(key=lambda r: r.ts_unix)
    for bucket in links_exit.values():
        bucket.sort(key=lambda r: r.ts_unix)

    logger.info(
        "B.4: Shadow index loaded — %d ENTRY keys, %d EXIT keys, %d DECISION records",
        len(links_entry), len(links_exit), len(decision_index),
    )
    return dict(links_entry), dict(links_exit), decision_index


# ---------------------------------------------------------------------------
# B.4 — Nearest-time matcher (binary search)
# ---------------------------------------------------------------------------

def _find_nearest_link(
    ts_unix: float,
    candidates: List[_LinkRecord],
    strategy_hint: str,
    window_narrow: float = MATCH_WINDOW_NARROW_S,
    window_wide: float = MATCH_WINDOW_WIDE_S,
) -> Tuple[Optional[_LinkRecord], bool]:
    """
    Find the nearest LINK record to *ts_unix* within a time window.

    Returns (matched_record, is_ambiguous).
    - (None, False): no match within window
    - (record, False): unique nearest match
    - (None, True): ambiguous (multiple equidistant candidates)
    """
    if not candidates:
        return None, False

    # Extract sorted ts list for bisect
    ts_list = [c.ts_unix for c in candidates]

    for window in (window_narrow, window_wide):
        lo = bisect.bisect_left(ts_list, ts_unix - window)
        hi = bisect.bisect_right(ts_list, ts_unix + window)
        window_candidates = candidates[lo:hi]
        if not window_candidates:
            continue

        # Sort by distance
        scored = sorted(window_candidates, key=lambda c: abs(c.ts_unix - ts_unix))
        best_dist = abs(scored[0].ts_unix - ts_unix)

        # Collect all ties
        ties = [c for c in scored if abs(abs(c.ts_unix - ts_unix) - best_dist) < 0.001]

        if len(ties) == 1:
            return ties[0], False

        # Multiple ties — try strategy disambiguation
        exact_strat = [c for c in ties if c.strategy == strategy_hint]
        if len(exact_strat) == 1:
            return exact_strat[0], False
        elif len(exact_strat) > 1:
            return None, True  # ambiguous even within same strategy
        else:
            return None, True  # ambiguous across strategies

    return None, False


# ---------------------------------------------------------------------------
# B.4 — Deterministic episode UID
# ---------------------------------------------------------------------------

def _compute_episode_uid(
    symbol: str,
    side: str,
    entry_ts: str,
    exit_ts: str,
    total_qty: float,
    avg_entry_price: float,
    avg_exit_price: float,
) -> str:
    """
    Stable UID from episode identity fields.
    Deterministic across rebuilds as long as fill aggregation is stable.
    """
    # Round to reasonable tolerance to absorb float jitter
    payload = json.dumps({
        "symbol": symbol,
        "side": side,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "total_qty": round(total_qty, 6),
        "avg_entry_price": round(avg_entry_price, 4),
        "avg_exit_price": round(avg_exit_price, 4),
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:EPISODE_UID_HASH_LEN]
    return f"EP_{digest}"


# ---------------------------------------------------------------------------
# B.4 — Regime binding from DECISION context_snapshot
# ---------------------------------------------------------------------------

def _resolve_regime(
    authority_ref: AuthorityRef,
    decision_index: Dict[str, _DecisionRecord],
) -> str:
    """
    Pull regime from DECISION v2 context_snapshot for the given authority ref.
    Returns regime string or "unknown".
    """
    if not authority_ref.decision_id:
        return "unknown"
    dec = decision_index.get(authority_ref.decision_id)
    if dec is None:
        return "unknown"
    cs = dec.context_snapshot
    if not isinstance(cs, dict):
        return "unknown"
    regime = cs.get("regime")
    return str(regime) if regime else "unknown"


# ---------------------------------------------------------------------------
# B.4 — Build V2 episodes with authority binding
# ---------------------------------------------------------------------------

def _bind_authority(
    episodes: List[Episode],
    links_entry: Dict[Tuple[str, str], List[_LinkRecord]],
    links_exit: Dict[Tuple[str, str], List[_LinkRecord]],
    decision_index: Dict[str, _DecisionRecord],
) -> Tuple[List[EpisodeV2], Dict[str, Any]]:
    """
    Convert V1 episodes to V2 with authority chain binding.

    Returns (episodes_v2, authority_stats).
    """
    episodes_v2: List[EpisodeV2] = []
    # Stats accumulators
    entry_bound = 0
    exit_bound = 0
    ambiguous_count = 0
    missing_count = 0
    permit_null_on_entry = 0
    max_delta_entry = 0.0
    max_delta_exit = 0.0

    for ep in episodes:
        entry_ts_unix = _iso_to_unix(ep.entry_ts)
        exit_ts_unix = _iso_to_unix(ep.exit_ts)

        # --- Entry authority ---
        entry_ref = AuthorityRef()
        entry_missing = True
        entry_ambiguous = False

        if entry_ts_unix is not None:
            # Try exact strategy key first, then wildcard
            for strat_key in [(ep.symbol, ep.strategy), (ep.symbol, "*")]:
                bucket = links_entry.get(strat_key, [])
                if not bucket:
                    continue
                match, ambig = _find_nearest_link(entry_ts_unix, bucket, ep.strategy)
                if ambig:
                    entry_ambiguous = True
                    break
                if match is not None:
                    entry_ref = AuthorityRef(
                        request_id=match.request_id,
                        decision_id=match.decision_id,
                        permit_id=match.permit_id,
                    )
                    entry_missing = False
                    delta = abs(match.ts_unix - entry_ts_unix)
                    if delta > max_delta_entry:
                        max_delta_entry = delta
                    if match.permit_id is None:
                        permit_null_on_entry += 1
                    break

        # --- Exit authority ---
        exit_ref = AuthorityRef()
        exit_missing = True
        exit_ambiguous = False

        if exit_ts_unix is not None:
            for strat_key in [(ep.symbol, ep.strategy), (ep.symbol, "*")]:
                bucket = links_exit.get(strat_key, [])
                if not bucket:
                    continue
                match, ambig = _find_nearest_link(exit_ts_unix, bucket, ep.strategy)
                if ambig:
                    exit_ambiguous = True
                    break
                if match is not None:
                    exit_ref = AuthorityRef(
                        request_id=match.request_id,
                        decision_id=match.decision_id,
                        permit_id=match.permit_id,
                    )
                    exit_missing = False
                    delta = abs(match.ts_unix - exit_ts_unix)
                    if delta > max_delta_exit:
                        max_delta_exit = delta
                    break

        # --- Regime binding from DECISION context_snapshot ---
        regime_entry = _resolve_regime(entry_ref, decision_index)
        regime_exit = _resolve_regime(exit_ref, decision_index)
        regime_bindable = (regime_entry != "unknown" or regime_exit != "unknown")

        # --- Deterministic UID ---
        episode_uid = _compute_episode_uid(
            ep.symbol, ep.side, ep.entry_ts, ep.exit_ts,
            ep.total_qty, ep.avg_entry_price, ep.avg_exit_price,
        )

        # --- Flags ---
        flags = AuthorityFlags(
            entry_missing=entry_missing,
            exit_missing=exit_missing,
            entry_ambiguous=entry_ambiguous,
            exit_ambiguous=exit_ambiguous,
        )

        # --- Stats ---
        if not entry_missing:
            entry_bound += 1
        if not exit_missing:
            exit_bound += 1
        if entry_ambiguous or exit_ambiguous:
            ambiguous_count += 1
        if entry_missing or exit_missing:
            missing_count += 1

        ep_v2 = EpisodeV2(
            episode_id=ep.episode_id,
            episode_uid=episode_uid,
            symbol=ep.symbol,
            side=ep.side,
            entry_ts=ep.entry_ts,
            exit_ts=ep.exit_ts,
            duration_hours=ep.duration_hours,
            entry_fills=ep.entry_fills,
            exit_fills=ep.exit_fills,
            entry_notional=ep.entry_notional,
            exit_notional=ep.exit_notional,
            total_qty=ep.total_qty,
            avg_entry_price=ep.avg_entry_price,
            avg_exit_price=ep.avg_exit_price,
            gross_pnl=ep.gross_pnl,
            fees=ep.fees,
            net_pnl=ep.net_pnl,
            regime_at_entry=regime_entry,
            regime_at_exit=regime_exit,
            exit_reason=ep.exit_reason,
            exit_reason_raw=ep.exit_reason_raw,
            strategy=ep.strategy,
            authority_entry=entry_ref,
            authority_exit=exit_ref,
            authority_flags=flags,
            regime_bindable=regime_bindable,
        )
        episodes_v2.append(ep_v2)

    total = len(episodes)
    authority_stats = {
        "entry_coverage_pct": round(entry_bound / total * 100, 1) if total else 0.0,
        "exit_coverage_pct": round(exit_bound / total * 100, 1) if total else 0.0,
        "ambiguous_count": ambiguous_count,
        "missing_count": missing_count,
        "permit_null_on_executed_entry_count": permit_null_on_entry,
        "max_time_delta_s_entry": round(max_delta_entry, 2),
        "max_time_delta_s_exit": round(max_delta_exit, 2),
        "episodes_total": total,
        "entry_bound": entry_bound,
        "exit_bound": exit_bound,
    }

    return episodes_v2, authority_stats


def _parse_ts(ts_val) -> Optional[datetime]:
    """Parse timestamp from various formats."""
    if ts_val is None:
        return None
    try:
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(ts_val, tz=timezone.utc)
        return dateparser.parse(str(ts_val))
    except Exception:
        return None


def _load_execution_log() -> list[dict]:
    """Load all order fill events from execution logs (current + rotated).

    v7.9-E2: Reads all ``orders_executed*.jsonl`` files in the log
    directory so that entries from rotated files are not lost.  Records
    are deduplicated by ``(symbol, positionSide, side, ts_fill_first)``
    and filtered to real fills (``event_type == 'order_fill'`` AND
    ``executedQty > 0``).
    """
    log_files: list[Path] = []
    if EXECUTION_LOG_DIR.exists():
        # Rotated files first (oldest to newest), then current
        rotated = sorted(
            EXECUTION_LOG_DIR.glob("orders_executed.*.jsonl"),
            key=lambda p: p.name,
            reverse=True,          # .2, .1 → oldest first
        )
        log_files.extend(rotated)
    if EXECUTION_LOG_PATH.exists():
        log_files.append(EXECUTION_LOG_PATH)

    fills: list[dict] = []
    seen: set[tuple] = set()  # dedup key

    for path in log_files:
        try:
            with open(path, "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Must be a real fill: event_type=order_fill AND qty > 0
                    if event.get("event_type") != "order_fill":
                        continue
                    qty = float(event.get("executedQty", 0) or 0)
                    if qty <= 0:
                        continue
                    # Dedup across rotated + current overlap
                    dedup_key = (
                        event.get("symbol", ""),
                        event.get("positionSide", ""),
                        event.get("side", ""),
                        str(event.get("ts_fill_first", "")),
                        str(event.get("orderId", "")),
                    )
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    fills.append(event)
        except OSError:
            continue

    return fills


def _load_doctrine_events() -> dict[str, str]:
    """Load regime at time of doctrine events (for regime context)."""
    # This is a simplified lookup - in production would need more sophisticated
    # regime tracking. For now, return empty dict and use 'unknown'.
    return {}


def _extract_exit_reason(fill: dict) -> str:
    """Extract exit reason from fill metadata.
    
    Priority order:
    1. Explicit metadata.exit.reason (from exit_scanner)
    2. Strategy name hints (tp_exit, sl_exit, etc.)
    3. Intent source field (auto_reduce = position_flip)
    4. Attempt ID suffix (_reduce = flip)
    5. Unknown fallback
    """
    meta = fill.get("metadata", {})
    exit_info = meta.get("exit", {})
    reason = exit_info.get("reason", "")
    
    # Normalize to lowercase for matching
    reason_lower = reason.lower() if reason else ""
    
    # 1. Explicit exit reason from exit_scanner
    if reason_lower in ("tp", "sl", "thesis", "regime_flip"):
        return reason_lower
    
    # 2. Check for strategy hints
    strategy = str(meta.get("strategy", "") or "").lower()
    if "exit" in strategy:
        if "tp" in strategy:
            return "tp"
        if "sl" in strategy:
            return "sl"
    
    # 3. Check intent source field (screener auto_reduce = position flip)
    source = str(fill.get("source", "") or "").lower()
    if source == "auto_reduce":
        return "position_flip"
    
    # 4. Check attempt_id suffix (flip reduce operations)
    attempt_id = str(fill.get("attempt_id", "") or "")
    if attempt_id.endswith("_reduce"):
        return "position_flip"
    
    # 5. Check if this is a reduceOnly close triggered by new signal (signal flip)
    if fill.get("reduceOnly") or fill.get("reduce_only"):
        # reduceOnly without explicit reason = signal-driven close
        return "signal_close"
    
    return "unknown"


def _extract_strategy(fill: dict) -> str:
    """Extract strategy name from fill metadata."""
    meta = fill.get("metadata", {})
    return meta.get("strategy", "unknown")


def _extract_scoring_fields(entry_fill: dict) -> dict:
    """Extract scoring/joinability fields from entry fill metadata.

    v7.9-S1: Every episode must carry the scoring context that existed
    at entry time.  These fields flow from:
      screener -> intent metadata -> fill metadata -> episode

    Returns dict suitable for unpacking into Episode kwargs.
    """
    meta = entry_fill.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return {
        "intent_id": str(entry_fill.get("intent_id", "") or ""),
        "attempt_id": str(entry_fill.get("attempt_id", "") or ""),
        "confidence": float(meta.get("confidence", 0) or 0),
        "hybrid_score": float(meta.get("hybrid_score", 0) or 0),
        "conviction_score": float(meta.get("conviction_score", 0) or 0),
        "conviction_band": str(meta.get("conviction_band", "") or ""),
        "entry_regime_confidence": float(
            meta.get("entry_regime_confidence", 0) or 0
        ),
        "expected_edge": float(meta.get("expected_edge", 0) or 0),
        "engine_source": str(meta.get("source", "") or ""),
    }


def _extract_regime_at_entry(entry_fill: dict) -> str:
    """Extract regime at entry from fill metadata."""
    meta = entry_fill.get("metadata", {})
    if not isinstance(meta, dict):
        return "unknown"
    return str(meta.get("entry_regime", "unknown") or "unknown")


def _compute_metadata_pnl(
    fills: list[dict],
    since_date: Optional[str],
    until_date: Optional[str],
) -> dict:
    """
    Compute realized PnL from exit fill metadata.
    
    This is an independent estimator that uses entry_price stored in exit metadata.
    More robust than episode matching for partial fills and scaling.
    
    Returns dict with: exits, gross_pnl, fees, net_pnl
    """
    realized_pnl = 0.0
    total_fees = 0.0
    exit_count = 0
    exits_with_entry_price = 0
    
    for f in fills:
        # Only process exits (reduceOnly fills)
        if not f.get("reduceOnly"):
            continue
        
        # Check if exit is in window
        ts = _parse_ts(f.get("ts"))
        if ts:
            date_str = ts.strftime("%Y-%m-%d")
            if since_date and date_str < since_date:
                continue
            if until_date and date_str > until_date:
                continue
        
        exit_count += 1
        total_fees += float(f.get("fee_total", 0) or 0)
        
        # Extract entry price from metadata
        meta = f.get("metadata", {})
        exit_info = meta.get("exit", {}) or meta.get("tp_sl", {}) or {}
        entry_price = exit_info.get("entry_price") or meta.get("entry_price")
        
        if entry_price:
            exits_with_entry_price += 1
            exit_price = float(f.get("avgPrice", 0) or 0)
            qty = float(f.get("executedQty", 0) or 0)
            pos_side = f.get("positionSide", "")
            
            if pos_side == "LONG":
                pnl = (exit_price - float(entry_price)) * qty
            else:  # SHORT
                pnl = (float(entry_price) - exit_price) * qty
            realized_pnl += pnl
    
    return {
        "exits": exit_count,
        "exits_with_entry_price": exits_with_entry_price,
        "gross_pnl": round(realized_pnl, 2),
        "fees": round(total_fees, 2),
        "net_pnl": round(realized_pnl - total_fees, 2),
    }


def build_episode_ledger(
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    lookback_days: int = 7,
) -> EpisodeLedger:
    """
    Rebuild episode ledger from execution logs.
    
    Algorithm:
      1. Load fills (with lookback buffer to capture entries before window)
      2. Group by (symbol, positionSide)
      3. Match entries to exits by cumulative quantity
      4. Create episode for each completed cycle
      5. Filter episodes by EXIT timestamp in [since_date, until_date]
    
    Window semantics: "episodes ending in window" — an episode is included
    if its exit_ts falls within the window, regardless of when entry occurred.
    """
    fills = _load_execution_log()
    
    if not fills:
        return EpisodeLedger(
            last_rebuild_ts=datetime.now(timezone.utc).isoformat(),
            stats={"total_fills": 0, "episodes_found": 0},
        )
    
    # Compute lookback date for fill filtering (entry could be before window)
    fill_since_date: Optional[str] = None
    if since_date:
        try:
            since_dt = datetime.strptime(since_date, "%Y-%m-%d")
            lookback_dt = since_dt - timedelta(days=lookback_days)
            fill_since_date = lookback_dt.strftime("%Y-%m-%d")
        except ValueError:
            fill_since_date = since_date  # fallback to exact date
    
    # Pre-filter fills with lookback buffer (for performance, not semantics)
    if fill_since_date or until_date:
        filtered = []
        for f in fills:
            ts = _parse_ts(f.get("ts"))
            if ts is None:
                continue
            date_str = ts.strftime("%Y-%m-%d")
            if fill_since_date and date_str < fill_since_date:
                continue
            if until_date and date_str > until_date:
                continue
            filtered.append(f)
        fills = filtered
    
    # Group fills by (symbol, positionSide)
    # positionSide: LONG = bought to open, sold to close
    #               SHORT = sold to open, bought to close
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    
    for f in fills:
        symbol = f.get("symbol", "")
        pos_side = f.get("positionSide", "")
        if not symbol or not pos_side:
            continue
        groups[(symbol, pos_side)].append(f)
    
    episodes = []
    episode_counter = 0
    # v7.9-E2: reconciliation counters
    recon_fills_total = 0
    recon_fills_consumed = 0
    recon_fills_orphaned = 0
    
    for (symbol, pos_side), group_fills in groups.items():
        # Sort by timestamp
        group_fills.sort(key=lambda x: x.get("ts") or x.get("ts_fill_first") or "")
        
        # Track position state
        open_qty = 0.0
        entry_fills: list[dict] = []
        exit_fills_accum: list[dict] = []  # v7.9-E2: accumulate exit fills
        
        for f in group_fills:
            recon_fills_total += 1
            qty = float(f.get("executedQty", 0))
            is_reduce = f.get("reduceOnly", False)
            side = f.get("side", "")
            
            # Determine if this is entry or exit
            # LONG position: BUY = entry, SELL = exit
            # SHORT position: SELL = entry, BUY = exit
            if pos_side == "LONG":
                is_entry = (side == "BUY" and not is_reduce)
                is_exit = (side == "SELL" or is_reduce)
            else:  # SHORT
                is_entry = (side == "SELL" and not is_reduce)
                is_exit = (side == "BUY" or is_reduce)
            
            if is_entry:
                open_qty += qty
                entry_fills.append(f)
                recon_fills_consumed += 1
            
            elif is_exit and open_qty > 0:
                # This is an exit — consume against open_qty
                exit_qty = min(qty, open_qty)
                open_qty -= exit_qty
                exit_fills_accum.append(f)
                recon_fills_consumed += 1
                
                # If position is now closed (or very close), create episode
                if open_qty < 0.0001 and entry_fills:
                    episode_counter += 1
                    
                    # Calculate entry stats
                    entry_notional = sum(
                        float(e.get("avgPrice", 0)) * float(e.get("executedQty", 0))
                        for e in entry_fills
                    )
                    entry_qty = sum(float(e.get("executedQty", 0)) for e in entry_fills)
                    entry_fees = sum(float(e.get("fee_total", 0) or 0) for e in entry_fills)
                    avg_entry = entry_notional / entry_qty if entry_qty > 0 else 0
                    
                    # v7.9-E2: exit stats from ALL accumulated exit fills
                    exit_notional = sum(
                        float(x.get("avgPrice", 0)) * float(x.get("executedQty", 0))
                        for x in exit_fills_accum
                    )
                    exit_qty_total = sum(float(x.get("executedQty", 0)) for x in exit_fills_accum)
                    exit_fees = sum(float(x.get("fee_total", 0) or 0) for x in exit_fills_accum)
                    avg_exit = exit_notional / exit_qty_total if exit_qty_total > 0 else 0
                    
                    # PnL calculation — use min of entry_qty and exit_qty for trade qty
                    trade_qty = min(entry_qty, exit_qty_total) if exit_qty_total > 0 else entry_qty
                    if pos_side == "LONG":
                        gross_pnl = (avg_exit - avg_entry) * trade_qty
                    else:  # SHORT
                        gross_pnl = (avg_entry - avg_exit) * trade_qty
                    
                    total_fees = entry_fees + exit_fees
                    net_pnl = gross_pnl - total_fees
                    
                    # Timing
                    entry_ts = entry_fills[0].get("ts") or entry_fills[0].get("ts_fill_first") or ""
                    last_exit = exit_fills_accum[-1] if exit_fills_accum else f
                    exit_ts = last_exit.get("ts") or last_exit.get("ts_fill_first") or ""
                    
                    entry_dt = _parse_ts(entry_ts)
                    exit_dt = _parse_ts(exit_ts)
                    duration_hours = 0.0
                    if entry_dt and exit_dt:
                        duration_hours = (exit_dt - entry_dt).total_seconds() / 3600

                    # Exit reason from last exit fill
                    _raw_reason = _extract_exit_reason(last_exit)
                    _norm = _normalize_exit(_raw_reason, source="episode_ledger")
                    
                    episode = Episode(
                        episode_id=f"EP_{episode_counter:04d}",
                        symbol=symbol,
                        side=pos_side,
                        entry_ts=entry_ts,
                        exit_ts=exit_ts,
                        duration_hours=round(duration_hours, 2),
                        entry_fills=len(entry_fills),
                        exit_fills=len(exit_fills_accum),
                        entry_notional=round(entry_notional, 2),
                        exit_notional=round(exit_notional, 2),
                        total_qty=round(trade_qty, 6),
                        avg_entry_price=round(avg_entry, 4),
                        avg_exit_price=round(avg_exit, 4),
                        gross_pnl=round(gross_pnl, 4),
                        fees=round(total_fees, 4),
                        net_pnl=round(net_pnl, 4),
                        regime_at_entry=_extract_regime_at_entry(entry_fills[0]) if entry_fills else "unknown",
                        regime_at_exit="unknown",
                        exit_reason=_norm.canonical,
                        exit_reason_raw=_norm.raw,
                        strategy=_extract_strategy(entry_fills[0]) if entry_fills else "unknown",
                        **(_extract_scoring_fields(entry_fills[0]) if entry_fills else {}),
                    )
                    episodes.append(episode)
                    
                    # Reset for next episode
                    entry_fills = []
                    exit_fills_accum = []
                    open_qty = 0.0
            
            elif is_exit:
                # v7.9-E2: exit without open position — orphan fill
                recon_fills_orphaned += 1
    
    # Filter episodes by EXIT timestamp in window (the canonical semantics)
    # "Episodes ending in window" — entry can be before, but exit must be in range
    if since_date or until_date:
        window_episodes = []
        for ep in episodes:
            exit_dt = _parse_ts(ep.exit_ts)
            if exit_dt is None:
                continue
            exit_date_str = exit_dt.strftime("%Y-%m-%d")
            if since_date and exit_date_str < since_date:
                continue
            if until_date and exit_date_str > until_date:
                continue
            window_episodes.append(ep)
        episodes = window_episodes
    
    # Calculate aggregate stats
    total_gross = sum(e.gross_pnl for e in episodes)
    total_fees = sum(e.fees for e in episodes)
    total_net = sum(e.net_pnl for e in episodes)
    
    winners = [e for e in episodes if e.net_pnl > 0]
    losers = [e for e in episodes if e.net_pnl < 0]
    
    # Compute max drawdown from cumulative PnL
    # Note: This is PnL-based drawdown (from trading peak to trough)
    # Not NAV-based drawdown (which would require starting capital)
    max_dd_pct = 0.0
    max_dd_abs = 0.0
    if episodes:
        # Sort by exit_ts for sequential equity calculation
        sorted_eps = sorted(episodes, key=lambda e: e.exit_ts)
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        for ep in sorted_eps:
            cumulative_pnl += ep.net_pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            dd = peak_pnl - cumulative_pnl
            if dd > max_dd_abs:
                max_dd_abs = dd
        # Express as percentage of NAV-based equity peak
        # Fallback: if we have no NAV context, use 10_000 as reasonable base
        try:
            import json as _json
            from pathlib import Path as _Path
            _nav_path = _Path("logs/state/nav_state.json")
            _nav_base = float(_json.loads(_nav_path.read_text()).get("total_equity", 10000)) if _nav_path.exists() else 10000
        except Exception:
            _nav_base = 10000
        _equity_peak = _nav_base + peak_pnl
        if _equity_peak > 0 and max_dd_abs > 0:
            max_dd_pct = round((max_dd_abs / _equity_peak) * 100, 2)
        else:
            max_dd_pct = 0.0
    
    # Metadata-based PnL estimator (independent cross-check)
    # Uses entry_price from exit fill metadata — robust for partial fills
    meta_pnl = _compute_metadata_pnl(fills, since_date, until_date)
    
    stats = {
        "total_fills": len(fills),
        "episodes_found": len(episodes),
        "total_gross_pnl": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net_pnl": round(total_net, 2),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(episodes) * 100, 1) if episodes else 0,
        "avg_duration_hours": round(sum(e.duration_hours for e in episodes) / len(episodes), 1) if episodes else 0,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_abs": round(max_dd_abs, 2),
        "exit_reasons": {
            reason: len([e for e in episodes if e.exit_reason == reason])
            for reason in sorted(set(e.exit_reason for e in episodes))
        },
        # Cross-check from exit metadata (more robust for partial fills)
        "metadata_pnl": meta_pnl,
        # v7.9-E2: Reconciliation summary
        "reconciliation": {
            "fills_total": recon_fills_total,
            "fills_consumed": recon_fills_consumed,
            "fills_orphaned": recon_fills_orphaned,
            "log_files_read": len(
                list(EXECUTION_LOG_DIR.glob("orders_executed*.jsonl"))
            ) if EXECUTION_LOG_DIR.exists() else 0,
        },
    }
    
    # -----------------------------------------------------------------------
    # B.4: Authority binding (shadow — never blocks, fail-open)
    # -----------------------------------------------------------------------
    episodes_v2: list = []
    try:
        links_entry_idx, links_exit_idx, decision_idx = _load_shadow_indexes()
        if links_entry_idx or links_exit_idx:
            episodes_v2, authority_stats = _bind_authority(
                episodes, links_entry_idx, links_exit_idx, decision_idx,
            )
            stats["authority"] = authority_stats
            logger.info(
                "B.4: Authority binding complete — entry %.0f%%, exit %.0f%%, %d missing, %d ambiguous",
                authority_stats["entry_coverage_pct"],
                authority_stats["exit_coverage_pct"],
                authority_stats["missing_count"],
                authority_stats["ambiguous_count"],
            )
        else:
            logger.info("B.4: No LINK events in shadow log — authority binding skipped")
    except Exception as e:
        logger.warning("B.4: Authority binding failed (shadow-safe): %s", e)

    return EpisodeLedger(
        episodes=episodes,
        episodes_v2=episodes_v2,
        last_rebuild_ts=datetime.now(timezone.utc).isoformat(),
        stats=stats,
    )


def save_episode_ledger(ledger: EpisodeLedger) -> None:
    """Write episode ledger to state file."""
    EPISODE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EPISODE_LEDGER_PATH, "w") as f:
        json.dump(ledger.to_dict(), f, indent=2)
    logger.info(f"Episode ledger saved: {len(ledger.episodes)} episodes")


def load_episode_ledger() -> Optional[EpisodeLedger]:
    """Load episode ledger from state file."""
    if not EPISODE_LEDGER_PATH.exists():
        return None
    
    try:
        with open(EPISODE_LEDGER_PATH, "r") as f:
            data = json.load(f)
        
        episodes = [Episode(**e) for e in data.get("episodes", [])]
        # B.4: Load V2 episodes if present (best-effort)
        episodes_v2: list = []
        for raw in data.get("episodes_v2", []):
            try:
                # Flatten nested authority back to dataclass shape
                auth = raw.pop("authority", {})
                raw["authority_entry"] = AuthorityRef(**auth.get("entry", {}))
                raw["authority_exit"] = AuthorityRef(**auth.get("exit", {}))
                raw["authority_flags"] = AuthorityFlags(**raw.pop("authority_flags", {}))
                episodes_v2.append(EpisodeV2(**raw))
            except Exception:
                continue  # skip malformed v2 entries
        return EpisodeLedger(
            episodes=episodes,
            episodes_v2=episodes_v2,
            last_rebuild_ts=data.get("last_rebuild_ts", ""),
            stats=data.get("stats", {}),
        )
    except Exception as e:
        logger.error(f"Failed to load episode ledger: {e}")
        return None


def rebuild_and_save(
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
) -> EpisodeLedger:
    """Convenience function: rebuild and save ledger."""
    ledger = build_episode_ledger(since_date, until_date)
    save_episode_ledger(ledger)
    return ledger


def print_ledger_summary(ledger: EpisodeLedger) -> None:
    """Print human-readable summary of episode ledger."""
    stats = ledger.stats
    print("=" * 70)
    print("EPISODE LEDGER SUMMARY")
    print("=" * 70)
    print(f"Last Rebuild:     {ledger.last_rebuild_ts}")
    print(f"Total Episodes:   {stats.get('episodes_found', 0)}")
    print(f"Total Fills:      {stats.get('total_fills', 0)}")
    print()
    print("PnL BREAKDOWN")
    print("-" * 40)
    print(f"  Gross PnL:      ${stats.get('total_gross_pnl', 0):+,.2f}")
    print(f"  Total Fees:     ${stats.get('total_fees', 0):,.2f}")
    print(f"  Net PnL:        ${stats.get('total_net_pnl', 0):+,.2f}")
    print()
    print("PERFORMANCE")
    print("-" * 40)
    print(f"  Winners:        {stats.get('winners', 0)}")
    print(f"  Losers:         {stats.get('losers', 0)}")
    print(f"  Win Rate:       {stats.get('win_rate', 0):.1f}%")
    print(f"  Avg Duration:   {stats.get('avg_duration_hours', 0):.1f}h")
    print()
    print("EXIT REASONS")
    print("-" * 40)
    reasons = stats.get("exit_reasons", {})
    for reason, count in reasons.items():
        print(f"  {reason:12}  {count}")
    print()
    
    # Metadata-based cross-check (more robust for partial fills)
    meta_pnl = stats.get("metadata_pnl", {})
    if meta_pnl:
        print("METADATA PnL (exit-based cross-check)")
        print("-" * 40)
        print(f"  Exit fills:     {meta_pnl.get('exits', 0)} ({meta_pnl.get('exits_with_entry_price', 0)} with entry_price)")
        print(f"  Gross PnL:      ${meta_pnl.get('gross_pnl', 0):+,.2f}")
        print(f"  Total Fees:     ${meta_pnl.get('fees', 0):,.2f}")
        print(f"  Net PnL:        ${meta_pnl.get('net_pnl', 0):+,.2f}")
        print()
    
    # v7.9-E2: Reconciliation summary
    recon = stats.get("reconciliation", {})
    if recon:
        print("RECONCILIATION (v7.9-E2)")
        print("-" * 40)
        print(f"  Fills total:    {recon.get('fills_total', 0)}")
        print(f"  Fills consumed: {recon.get('fills_consumed', 0)}")
        print(f"  Fills orphaned: {recon.get('fills_orphaned', 0)}")
        print(f"  Log files read: {recon.get('log_files_read', 0)}")
        print()


# CLI entry point
if __name__ == "__main__":
    import sys
    
    since = sys.argv[1] if len(sys.argv) > 1 else None
    until = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Building episode ledger (since={since}, until={until})...")
    ledger = rebuild_and_save(since, until)
    print_ledger_summary(ledger)
