"""Dynamic Market Discovery — Polymarket 15-minute BTC Up/Down markets.

Discovers the current and upcoming 15-minute (and optionally 5-minute)
Bitcoin Up/Down binary markets via the Gamma REST API.  These markets
rotate every cycle (15m or 5m) with **brand-new** market_ids and
outcome token_ids each time.

How it works
------------
1. Poll ``GET /events`` on gamma-api.polymarket.com
2. Filter for ``"bitcoin up or down"`` in title (case-insensitive)
3. Filter by slug prefix (``btc-updown-15m-`` or ``btc-updown-5m-``)
4. Return the currently-active window (end_time > now + safety_buffer)
5. Extract UP token (index 0) and DOWN token (index 1) from ``clobTokenIds``

Rotation lifecycle
------------------
* Each 15-minute window gets a **new** event → market → token pair.
* Consumers must re-discover tokens before the previous window expires.
* This module is polled on a timer (default 60s) by the CLOB client.

Design invariants
-----------------
* **Read-only** — only queries Gamma, never writes orders or state.
* **Isolated** — no imports from ``execution/`` or ``dashboard/``.
* **Deterministic** — returns ``DiscoveredMarket`` dataclass.
* **Resilient** — HTTP failures return ``None``, never raise.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prediction.market_discovery")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAMMA_BASE = os.environ.get(
    "GAMMA_API_BASE", "https://gamma-api.polymarket.com"
)

# Slug prefix filters
SLUG_15M = "btc-updown-15m-"
SLUG_5M = "btc-updown-5m-"

# Title substring (case-insensitive)
TITLE_FILTER = "bitcoin up or down"

# Default to 15-minute markets; set DISCOVERY_TIMEFRAMES=5m,15m for both
_RAW_TIMEFRAMES = os.environ.get("DISCOVERY_TIMEFRAMES", "15m")
ENABLED_TIMEFRAMES: List[str] = [
    t.strip() for t in _RAW_TIMEFRAMES.split(",") if t.strip()
]

# Safety margin: skip markets expiring in less than this many seconds.
# Gives time to subscribe, receive at least a few ticks, and gracefully
# unsubscribe before resolution.
EXPIRY_SAFETY_BUFFER_S: float = float(
    os.environ.get("DISCOVERY_EXPIRY_BUFFER_S", "120")
)

# Maximum market lookahead: skip markets starting more than this far
# in the future (seconds).  0 = no limit.
MAX_LOOKAHEAD_S: float = float(
    os.environ.get("DISCOVERY_MAX_LOOKAHEAD_S", "0")
)

# HTTP timeout for Gamma API
HTTP_TIMEOUT_S: float = float(os.environ.get("DISCOVERY_HTTP_TIMEOUT_S", "15"))

# How many events to request per poll
GAMMA_LIMIT: int = int(os.environ.get("DISCOVERY_GAMMA_LIMIT", "200"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscoveredMarket:
    """A single discovered BTC Up/Down binary market."""

    event_id: str
    market_id: str  # Gamma market ID (not condition_id)
    condition_id: str
    slug: str
    question: str
    timeframe: str  # "15m" or "5m"
    up_token: str  # clobTokenIds[0]
    down_token: str  # clobTokenIds[1]
    end_date_utc: str  # ISO-8601
    end_ts: float  # Unix seconds
    remaining_s: float  # Seconds until expiry (at discovery time)


@dataclass
class DiscoverySnapshot:
    """Result of one discovery poll cycle."""

    ts: str  # ISO-8601 timestamp of poll
    markets: List[DiscoveredMarket] = field(default_factory=list)
    current_15m: Optional[DiscoveredMarket] = None
    current_5m: Optional[DiscoveredMarket] = None
    error: Optional[str] = None
    raw_event_count: int = 0  # Total events fetched before filtering
    btc_updown_count: int = 0  # Events matching title filter


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _gamma_get(
    path: str,
    params: Optional[Dict[str, str]] = None,
    timeout: float = HTTP_TIMEOUT_S,
) -> Optional[Any]:
    """GET from Gamma API.  Returns parsed JSON or None on failure."""
    url = GAMMA_BASE.rstrip("/") + "/" + path.lstrip("/")
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    req = urllib.request.Request(url, headers={"User-Agent": "hedge-fund/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
        logger.warning("[discovery] Gamma GET %s failed: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_iso(dt_str: str) -> float:
    """Parse an ISO-8601 datetime string to Unix timestamp (seconds)."""
    # Handle Z suffix and +00:00
    s = dt_str.rstrip("Z")
    try:
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _slug_to_timeframe(slug: str) -> Optional[str]:
    """Extract timeframe from slug, or None if not a BTC updown slug."""
    if slug.startswith(SLUG_15M):
        return "15m"
    if slug.startswith(SLUG_5M):
        return "5m"
    return None


def _parse_tokens(raw: Any) -> List[str]:
    """Parse clobTokenIds which may be a JSON string or list."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ---------------------------------------------------------------------------
# Core discovery
# ---------------------------------------------------------------------------
def discover_btc_updown_markets(
    timeframes: Optional[List[str]] = None,
    safety_buffer_s: float = EXPIRY_SAFETY_BUFFER_S,
    max_lookahead_s: float = MAX_LOOKAHEAD_S,
) -> DiscoverySnapshot:
    """Poll Gamma and return all active BTC Up/Down markets.

    Parameters
    ----------
    timeframes
        List of timeframes to include (``["15m"]``, ``["5m"]``,
        ``["5m", "15m"]``).  Defaults to ``ENABLED_TIMEFRAMES``.
    safety_buffer_s
        Minimum remaining seconds before expiry.
    max_lookahead_s
        Maximum seconds into the future for market start.  0 = no limit.

    Returns
    -------
    DiscoverySnapshot
        Contains all matching markets sorted by ``end_ts``, plus shortcuts
        to the *current* (soonest-expiring valid) 15m and 5m markets.
    """
    tf_set = set(timeframes or ENABLED_TIMEFRAMES)
    now = time.time()

    snap = DiscoverySnapshot(
        ts=datetime.now(timezone.utc).isoformat(),
    )

    # Fetch events sorted by newest start date first
    data = _gamma_get("events", params={
        "active": "true",
        "closed": "false",
        "limit": str(GAMMA_LIMIT),
        "order": "startDate",
        "ascending": "false",
    })

    if data is None:
        snap.error = "gamma_fetch_failed"
        return snap

    if not isinstance(data, list):
        snap.error = f"unexpected_response_type:{type(data).__name__}"
        return snap

    snap.raw_event_count = len(data)
    markets: List[DiscoveredMarket] = []

    for event in data:
        if not isinstance(event, dict):
            continue

        title = event.get("title", "")
        if TITLE_FILTER not in title.lower():
            continue

        snap.btc_updown_count += 1
        slug = event.get("slug", "")
        timeframe = _slug_to_timeframe(slug)

        if timeframe is None or timeframe not in tf_set:
            continue

        event_id = str(event.get("id", ""))

        for mkt in event.get("markets", []):
            if not isinstance(mkt, dict):
                continue

            end_str = mkt.get("endDate", "")
            end_ts = _parse_iso(end_str)
            remaining = end_ts - now

            # Skip already-expired or about-to-expire
            if remaining < safety_buffer_s:
                continue

            # Skip too far in the future (if configured)
            if max_lookahead_s > 0 and remaining > max_lookahead_s:
                continue

            tokens = _parse_tokens(mkt.get("clobTokenIds"))
            if len(tokens) < 2:
                logger.debug("[discovery] skipping market with <2 tokens: %s", slug)
                continue

            market_id = str(mkt.get("id", ""))
            condition_id = mkt.get("conditionId", mkt.get("condition_id", ""))
            question = mkt.get("question", title)

            dm = DiscoveredMarket(
                event_id=event_id,
                market_id=market_id,
                condition_id=condition_id,
                slug=slug,
                question=question,
                timeframe=timeframe,
                up_token=tokens[0],
                down_token=tokens[1],
                end_date_utc=end_str,
                end_ts=end_ts,
                remaining_s=round(remaining, 1),
            )
            markets.append(dm)

    # Sort by soonest expiry first
    markets.sort(key=lambda m: m.end_ts)
    snap.markets = markets

    # Pick the *current* (soonest-expiring still-valid) for each timeframe
    for m in markets:
        if m.timeframe == "15m" and snap.current_15m is None:
            snap.current_15m = m
        elif m.timeframe == "5m" and snap.current_5m is None:
            snap.current_5m = m

    logger.info(
        "[discovery] poll: %d raw events, %d btc-updown, %d valid markets "
        "(current_15m=%s, current_5m=%s)",
        snap.raw_event_count,
        snap.btc_updown_count,
        len(markets),
        snap.current_15m.slug if snap.current_15m else "none",
        snap.current_5m.slug if snap.current_5m else "none",
    )
    return snap


def get_current_tokens(
    timeframe: str = "15m",
    safety_buffer_s: float = EXPIRY_SAFETY_BUFFER_S,
) -> Optional[DiscoveredMarket]:
    """Convenience: get the current active market for one timeframe.

    Returns None if no valid market found or Gamma unreachable.
    """
    snap = discover_btc_updown_markets(
        timeframes=[timeframe],
        safety_buffer_s=safety_buffer_s,
    )
    if timeframe == "15m":
        return snap.current_15m
    return snap.current_5m


# ---------------------------------------------------------------------------
# Self-test / standalone usage
# ---------------------------------------------------------------------------
def main() -> None:
    """Standalone discovery check."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    timeframes = ["15m", "5m"]
    snap = discover_btc_updown_markets(timeframes=timeframes, safety_buffer_s=0)

    print(f"\nDiscovery snapshot @ {snap.ts}")
    print(f"  Raw events:      {snap.raw_event_count}")
    print(f"  BTC Up/Down:     {snap.btc_updown_count}")
    print(f"  Valid markets:   {len(snap.markets)}")
    if snap.error:
        print(f"  ERROR: {snap.error}")

    if snap.current_15m:
        m = snap.current_15m
        print(f"\n  Current 15m: {m.question}")
        print(f"    slug:      {m.slug}")
        print(f"    remaining: {m.remaining_s / 60:.0f} min")
        print(f"    UP:        {m.up_token[:20]}...")
        print(f"    DOWN:      {m.down_token[:20]}...")
    else:
        print("\n  No active 15m market found")

    if snap.current_5m:
        m = snap.current_5m
        print(f"\n  Current 5m:  {m.question}")
        print(f"    slug:      {m.slug}")
        print(f"    remaining: {m.remaining_s / 60:.0f} min")
        print(f"    UP:        {m.up_token[:20]}...")
        print(f"    DOWN:      {m.down_token[:20]}...")
    else:
        print("\n  No active 5m market found")

    # Show next 5 markets
    if snap.markets:
        print(f"\n  --- All {len(snap.markets)} valid markets (by expiry) ---")
        for m in snap.markets[:10]:
            print(f"    {m.timeframe:3s}  {m.remaining_s/60:6.0f}m  {m.question}")

    sys.exit(0 if snap.markets else 1)


if __name__ == "__main__":
    main()
