"""CLOB Market Client — Polymarket orderbook + trade stream ingestion.

Layer 2 of the Binary Sleeve data plane.  Connects to Polymarket's CLOB
WebSocket, subscribes to configured market asset_ids, and persists every
event as append-only JSONL.

Supports two modes:

1. **Static** — fixed asset_ids (default, or from ``CLOB_ASSET_IDS`` env).
2. **Dynamic discovery** — auto-discovers the current 15-minute BTC Up/Down
   market on Polymarket and rotates subscriptions every cycle.  Enabled via
   ``CLOB_DISCOVERY_MODE=1`` or by passing ``discovery_mode=True``.

Design invariants
-----------------
* **Append-only** — never mutates ``MARKET_LOG``.
* **One event per line** — JSONL, each line is a self-contained JSON object.
* **Deterministic** — captures server event + local arrival timestamps.
* **Isolated** — no imports from ``execution/`` or ``dashboard/``.
* **Standalone** — runnable via ``python -m prediction.clob_market_client``.

Event types captured
--------------------
* ``best_bid_ask``   — top-of-book bid/ask/spread (requires custom_feature_enabled)
* ``last_trade_price`` — trade execution with price/side/size
* ``tick_size_change`` — market tick size changed (old/new sizes)
* ``market_resolved`` — market resolved with winning asset (requires custom_feature_enabled)
* ``book``           — full orderbook snapshot (logged as summary only)
* ``price_change``   — level delta (logged)

Log paths
---------
* Events:    ``logs/prediction/clob_market.jsonl``
* Anomalies: ``logs/execution/environment_events.jsonl``
* Health:    ``logs/prediction/clob_market_health.jsonl``

References
----------
* CLOB WS endpoint: ``wss://ws-subscriptions-clob.polymarket.com/ws/market``
* Docs: ``docs.polymarket.com/api-reference/wss/market``
* Subscribe: ``{type: "market", assets_ids: [...], custom_feature_enabled: true}``
* Keepalive: text ``ping`` every 10s, server replies ``pong``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import websockets
    import websockets.asyncio.client as ws_client
except ImportError:
    websockets = None  # type: ignore[assignment]
    ws_client = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MARKET_LOG = _PROJECT_ROOT / "logs" / "prediction" / "clob_market.jsonl"
HEALTH_LOG = _PROJECT_ROOT / "logs" / "prediction" / "clob_market_health.jsonl"
ENV_EVENTS_LOG = _PROJECT_ROOT / "logs" / "execution" / "environment_events.jsonl"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
WS_URI = os.environ.get(
    "CLOB_WS_URI", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
)

# Default asset IDs — BTC $150k markets (YES tokens only, most liquid side).
# These can be overridden via CLOB_ASSET_IDS env var (comma-separated).
# Markets: "Will Bitcoin hit $150k by March 31, 2026?" + "... June 30, 2026?"
_DEFAULT_ASSET_IDS = [
    # BTC $150k by March 31, 2026 — YES token
    "46866868857194367945413771860582064655745092128562966218540356888709464260149",
    # BTC $150k by June 30, 2026 — YES token
    "13915689317269078219168496739008737517740566192006337297676041270492637394586",
]

ASSET_IDS: List[str] = (
    os.environ.get("CLOB_ASSET_IDS", "").split(",")
    if os.environ.get("CLOB_ASSET_IDS")
    else _DEFAULT_ASSET_IDS
)

# Discovery mode — auto-discover 15m BTC Up/Down markets
DISCOVERY_MODE: bool = os.environ.get("CLOB_DISCOVERY_MODE", "0") == "1"
DISCOVERY_POLL_INTERVAL_S: float = float(
    os.environ.get("CLOB_DISCOVERY_POLL_S", "60")
)
DISCOVERY_TIMEFRAME: str = os.environ.get("CLOB_DISCOVERY_TIMEFRAME", "15m")

PING_INTERVAL_S: float = float(os.environ.get("CLOB_PING_INTERVAL_S", "10"))
STALE_EVENT_THRESHOLD_S: float = float(
    os.environ.get("CLOB_STALE_EVENT_S", "120")
)
MAX_RECONNECT_FAILURES: int = int(
    os.environ.get("CLOB_MAX_RECONNECT_FAILURES", "20")
)
RECONNECT_BASE_DELAY_S: float = 1.0
RECONNECT_MAX_DELAY_S: float = 60.0
HEALTH_EMIT_INTERVAL_S: float = float(
    os.environ.get("CLOB_HEALTH_INTERVAL_S", "60")
)

# Event types we track for ingestion metrics
TRACKED_EVENT_TYPES = frozenset({
    "best_bid_ask",
    "last_trade_price",
    "tick_size_change",
    "market_resolved",
    "book",
    "price_change",
})

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("prediction.clob_market")


# ---------------------------------------------------------------------------
# Anomaly types
# ---------------------------------------------------------------------------
class AnomalyType:
    STALE = "clob_stale"
    RECONNECT_STORM = "clob_reconnect_storm"
    WS_ERROR = "clob_ws_error"
    UNKNOWN_EVENT = "clob_unknown_event"
    SUBSCRIBE_ERROR = "clob_subscribe_error"


# ---------------------------------------------------------------------------
# Writer helpers (append-only, atomic per line)
# ---------------------------------------------------------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a single JSON line.  Thread-safe for single-writer."""
    _ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Health accumulator
# ---------------------------------------------------------------------------
class HealthAccumulator:
    """Collects per-window event stats and emits periodic summaries."""

    def __init__(self) -> None:
        self.event_counts: Dict[str, int] = {}
        self.spreads: List[float] = []
        self.trade_prices: List[float] = []
        self.anomaly_count: int = 0
        self.total_events: int = 0
        self.window_start: float = time.time()

    def record_event(self, event_type: str) -> None:
        self.total_events += 1
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

    def record_spread(self, spread: float) -> None:
        self.spreads.append(spread)

    def record_trade_price(self, price: float) -> None:
        self.trade_prices.append(price)

    def record_anomaly(self) -> None:
        self.anomaly_count += 1

    def emit_and_reset(self) -> Dict[str, Any]:
        now = time.time()
        elapsed_s = now - self.window_start

        summary: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "window_s": round(elapsed_s, 2),
            "total_events": self.total_events,
            "event_counts": dict(self.event_counts),
            "anomaly_count": self.anomaly_count,
        }

        if self.spreads:
            sorted_s = sorted(self.spreads)
            summary["spread"] = {
                "mean": round(statistics.mean(sorted_s), 6),
                "median": round(statistics.median(sorted_s), 6),
                "min": round(sorted_s[0], 6),
                "max": round(sorted_s[-1], 6),
                "samples": len(sorted_s),
            }
        else:
            summary["spread"] = None

        if self.trade_prices:
            summary["last_trade_price"] = round(self.trade_prices[-1], 6)
            summary["trade_count"] = len(self.trade_prices)
        else:
            summary["last_trade_price"] = None
            summary["trade_count"] = 0

        summary["event_frequency_hz"] = (
            round(self.total_events / elapsed_s, 3) if elapsed_s > 0 else 0
        )

        # Reset
        self.event_counts = {}
        self.spreads = []
        self.trade_prices = []
        self.anomaly_count = 0
        self.total_events = 0
        self.window_start = now

        return summary


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------
class CLOBMarketClient:
    """Async WebSocket client for Polymarket CLOB market data stream.

    Supports two modes:

    * **Static** — subscribes to a fixed list of ``asset_ids`` for the
      lifetime of the connection.
    * **Discovery** — polls Gamma every ``discovery_poll_s`` to discover
      the current 15-minute (or 5-minute) BTC Up/Down market and
      dynamically rotates WS subscriptions when the market window changes.
    """

    def __init__(
        self,
        ws_uri: str = WS_URI,
        asset_ids: Optional[List[str]] = None,
        market_log: Path = MARKET_LOG,
        health_log: Path = HEALTH_LOG,
        env_events_log: Path = ENV_EVENTS_LOG,
        *,
        discovery_mode: bool = DISCOVERY_MODE,
        discovery_poll_s: float = DISCOVERY_POLL_INTERVAL_S,
        discovery_timeframe: str = DISCOVERY_TIMEFRAME,
    ) -> None:
        self.ws_uri = ws_uri
        self.asset_ids = asset_ids if asset_ids is not None else list(ASSET_IDS)
        self.market_log = market_log
        self.health_log = health_log
        self.env_events_log = env_events_log

        # Discovery mode fields
        self.discovery_mode = discovery_mode
        self.discovery_poll_s = discovery_poll_s
        self.discovery_timeframe = discovery_timeframe
        self._current_slug: Optional[str] = None
        self._current_market: Optional[Any] = None  # DiscoveredMarket
        self._subscribed_ids: List[str] = []  # Currently subscribed asset IDs
        self._ws_ref: Optional[Any] = None  # Reference to active WS connection
        self._rotation_count: int = 0

        self._last_event_time: float = 0.0
        self._consecutive_reconnects: int = 0
        self._running: bool = False
        self._event_counter: int = 0

        # Latest state per asset (for querying)
        self._best_bid: Dict[str, float] = {}
        self._best_ask: Dict[str, float] = {}
        self._last_spread: Dict[str, float] = {}
        self._last_trade: Dict[str, float] = {}

        self.health = HealthAccumulator()

    # ----- public query methods -----

    def get_best_bid(self, asset_id: str) -> Optional[float]:
        """Latest best bid price for an asset, or None."""
        return self._best_bid.get(asset_id)

    def get_best_ask(self, asset_id: str) -> Optional[float]:
        """Latest best ask price for an asset, or None."""
        return self._best_ask.get(asset_id)

    def get_spread(self, asset_id: str) -> Optional[float]:
        """Latest spread for an asset, or None."""
        return self._last_spread.get(asset_id)

    def get_last_trade_price(self, asset_id: str) -> Optional[float]:
        """Latest trade price for an asset, or None."""
        return self._last_trade.get(asset_id)

    def get_current_market_slug(self) -> Optional[str]:
        """Current market slug in discovery mode, or None."""
        return self._current_slug

    def get_current_market(self) -> Optional[Any]:
        """Current DiscoveredMarket in discovery mode, or None."""
        return self._current_market

    def get_rotation_count(self) -> int:
        """Number of market rotations performed."""
        return self._rotation_count

    def get_subscribed_ids(self) -> List[str]:
        """Currently subscribed asset IDs."""
        return list(self._subscribed_ids)

    # ----- anomaly logging -----

    def _log_anomaly(self, anomaly_type: str, detail: Dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": anomaly_type,
            "source": "clob_market_client",
            **detail,
        }
        _append_jsonl(self.env_events_log, event)
        self.health.record_anomaly()
        logger.warning("[clob] anomaly %s: %s", anomaly_type, detail)

    # ----- event processing -----

    def _process_message(self, raw_msg: str) -> List[Dict[str, Any]]:
        """Route an incoming CLOB WS message.

        CLOB WS sends JSON arrays of events::

            [
                {"event_type": "best_bid_ask", "asset_id": "...",
                 "best_bid": "0.65", "best_ask": "0.67", ...},
                ...
            ]

        Or single event objects. We handle both.

        The ``pong`` text response to our ``ping`` is silently ignored.
        """
        arrival_ms = int(time.time() * 1000)
        results: List[Dict[str, Any]] = []

        # pong keepalive response — ignore silently
        if raw_msg.strip().lower() == "pong":
            self._last_event_time = time.time()
            return results

        try:
            parsed = json.loads(raw_msg)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("[clob] non-JSON message: %s", e)
            return results

        # Normalise to list
        events: List[Dict[str, Any]]
        if isinstance(parsed, list):
            events = parsed
        elif isinstance(parsed, dict):
            events = [parsed]
        else:
            return results

        for ev in events:
            if not isinstance(ev, dict):
                continue
            recs = self._process_event(ev, arrival_ms=arrival_ms)
            results.extend(recs)

        return results

    def _process_event(
        self,
        ev: Dict[str, Any],
        *,
        arrival_ms: int,
    ) -> List[Dict[str, Any]]:
        """Process a single CLOB event and write to log.

        Returns list of logged records (may expand ``price_change`` arrays).
        """
        event_type = ev.get("event_type", "")

        # Connection alive proof
        self._last_event_time = time.time()

        if event_type not in TRACKED_EVENT_TYPES:
            if event_type:
                self._log_anomaly(
                    AnomalyType.UNKNOWN_EVENT,
                    {"event_type": event_type, "keys": list(ev.keys())},
                )
            return []

        # ----- price_change: nested array format -----
        # Real format: {event_type, market, timestamp, price_changes: [{asset_id, price, size, side, best_bid, best_ask}, ...]}
        if event_type == "price_change":
            return self._process_price_changes(ev, arrival_ms=arrival_ms)

        asset_id = ev.get("asset_id", "")
        self._event_counter += 1
        self.health.record_event(event_type)

        # Build base record
        record: Dict[str, Any] = {
            "ts_arrival_ms": arrival_ms,
            "event_type": event_type,
            "asset_id": asset_id,
            "seq": self._event_counter,
            "source": "CLOB",
        }

        # ----- best_bid_ask -----
        if event_type == "best_bid_ask":
            bid = _safe_float(ev.get("best_bid"))
            ask = _safe_float(ev.get("best_ask"))
            timestamp = ev.get("timestamp")
            record["best_bid"] = bid
            record["best_ask"] = ask
            record["timestamp"] = timestamp
            if bid is not None and ask is not None:
                spread = round(ask - bid, 6)
                record["spread"] = spread
                self._best_bid[asset_id] = bid
                self._best_ask[asset_id] = ask
                self._last_spread[asset_id] = spread
                self.health.record_spread(spread)

        # ----- last_trade_price -----
        elif event_type == "last_trade_price":
            price = _safe_float(ev.get("price"))
            record["price"] = price
            record["side"] = ev.get("side")
            record["size"] = ev.get("size")
            record["fee_rate_bps"] = ev.get("fee_rate_bps")
            record["timestamp"] = ev.get("timestamp")
            if price is not None:
                self._last_trade[asset_id] = price
                self.health.record_trade_price(price)

        # ----- tick_size_change -----
        elif event_type == "tick_size_change":
            record["old_tick_size"] = ev.get("old_tick_size")
            record["new_tick_size"] = ev.get("new_tick_size")
            record["timestamp"] = ev.get("timestamp")

        # ----- market_resolved -----
        elif event_type == "market_resolved":
            record["winning_asset_id"] = ev.get("winning_asset_id")
            record["winning_outcome"] = ev.get("winning_outcome")
            record["timestamp"] = ev.get("timestamp")

        # ----- book (snapshot) -----
        elif event_type == "book":
            # Full orderbook can be large; log summary only
            bids = ev.get("bids", [])
            asks = ev.get("asks", [])
            record["bid_levels"] = len(bids) if isinstance(bids, list) else 0
            record["ask_levels"] = len(asks) if isinstance(asks, list) else 0
            record["market"] = ev.get("market")
            record["timestamp"] = ev.get("timestamp")
            # Extract top-of-book if available
            if bids and isinstance(bids[0], dict):
                record["top_bid"] = _safe_float(bids[0].get("price"))
            if asks and isinstance(asks[0], dict):
                record["top_ask"] = _safe_float(asks[0].get("price"))

        # Append to market log
        _append_jsonl(self.market_log, record)

        return [record]

    def _process_price_changes(
        self,
        ev: Dict[str, Any],
        *,
        arrival_ms: int,
    ) -> List[Dict[str, Any]]:
        """Expand a ``price_change`` event into per-asset records.

        Real CLOB WS format::

            {
                "event_type": "price_change",
                "market": "0x...",
                "timestamp": "1234567890000",
                "price_changes": [
                    {"asset_id": "...", "price": "0.17", "size": "100",
                     "side": "SELL", "best_bid": "0.16", "best_ask": "0.17"},
                    ...
                ]
            }

        Each item in ``price_changes`` becomes a separate logged record.
        """
        changes = ev.get("price_changes", [])
        timestamp = ev.get("timestamp")
        market = ev.get("market")
        results: List[Dict[str, Any]] = []

        if not isinstance(changes, list) or not changes:
            # Fallback: flat format (forward compatibility)
            asset_id = ev.get("asset_id", "")
            self._event_counter += 1
            self.health.record_event("price_change")
            record: Dict[str, Any] = {
                "ts_arrival_ms": arrival_ms,
                "event_type": "price_change",
                "asset_id": asset_id,
                "seq": self._event_counter,
                "source": "CLOB",
                "price": _safe_float(ev.get("price")),
                "side": ev.get("side"),
                "size": ev.get("size"),
                "best_bid": _safe_float(ev.get("best_bid")),
                "best_ask": _safe_float(ev.get("best_ask")),
                "market": market,
                "timestamp": timestamp,
            }
            _append_jsonl(self.market_log, record)
            return [record]

        for item in changes:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id", "")
            self._event_counter += 1
            self.health.record_event("price_change")
            price = _safe_float(item.get("price"))
            bid = _safe_float(item.get("best_bid"))
            ask = _safe_float(item.get("best_ask"))
            record = {
                "ts_arrival_ms": arrival_ms,
                "event_type": "price_change",
                "asset_id": asset_id,
                "seq": self._event_counter,
                "source": "CLOB",
                "price": price,
                "side": item.get("side"),
                "size": item.get("size"),
                "best_bid": bid,
                "best_ask": ask,
                "market": market,
                "timestamp": timestamp,
            }
            # Update internal state from price_change best_bid/best_ask
            if bid is not None and ask is not None:
                spread = round(ask - bid, 6)
                self._best_bid[asset_id] = bid
                self._best_ask[asset_id] = ask
                self._last_spread[asset_id] = spread
                self.health.record_spread(spread)
            _append_jsonl(self.market_log, record)
            results.append(record)

        return results

    # ----- subscription message -----

    @staticmethod
    def _subscribe_message(asset_ids: List[str]) -> str:
        """Build the CLOB market subscribe payload.

        Sends initial subscription with ``custom_feature_enabled`` to receive
        ``best_bid_ask``, ``new_market``, and ``market_resolved`` events.
        """
        return json.dumps({
            "type": "market",
            "assets_ids": asset_ids,
            "custom_feature_enabled": True,
        })

    @staticmethod
    def _dynamic_subscribe(asset_ids: List[str]) -> str:
        """Build a dynamic subscribe message (add more assets at runtime)."""
        return json.dumps({
            "assets_ids": asset_ids,
            "operation": "subscribe",
        })

    @staticmethod
    def _dynamic_unsubscribe(asset_ids: List[str]) -> str:
        """Build a dynamic unsubscribe message."""
        return json.dumps({
            "assets_ids": asset_ids,
            "operation": "unsubscribe",
        })

    # ----- discovery + rotation -----

    def _log_rotation(self, detail: Dict[str, Any]) -> None:
        """Log a market rotation event to environment_events.jsonl."""
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "clob_market_rotation",
            "source": "clob_market_client",
            **detail,
        }
        _append_jsonl(self.env_events_log, event)
        logger.info(
            "[clob] market rotation #%d: %s",
            self._rotation_count,
            detail.get("new_slug", "?"),
        )

    async def _rotate_subscription(
        self, new_market: Any, ws: Any
    ) -> None:
        """Unsubscribe from old tokens and subscribe to new ones.

        Parameters
        ----------
        new_market
            A ``DiscoveredMarket`` from the discovery module.
        ws
            Active WebSocket connection.
        """
        old_ids = list(self._subscribed_ids)
        new_ids = [new_market.up_token, new_market.down_token]

        # Don't rotate if tokens are identical (market still current)
        if set(old_ids) == set(new_ids):
            return

        self._rotation_count += 1

        # Unsubscribe old
        if old_ids:
            try:
                await ws.send(self._dynamic_unsubscribe(old_ids))
                logger.info(
                    "[clob] unsubscribed %d old token(s) (%s)",
                    len(old_ids),
                    self._current_slug or "?",
                )
            except Exception as exc:
                logger.warning("[clob] unsubscribe failed: %s", exc)

        # Subscribe new
        try:
            await ws.send(self._dynamic_subscribe(new_ids))
            self._subscribed_ids = new_ids
            old_slug = self._current_slug
            self._current_slug = new_market.slug
            self._current_market = new_market
            logger.info(
                "[clob] subscribed to %s — UP=%s... DOWN=%s... (%.0fm remaining)",
                new_market.slug,
                new_market.up_token[:16],
                new_market.down_token[:16],
                new_market.remaining_s / 60,
            )
        except Exception as exc:
            logger.error("[clob] subscribe to new market failed: %s", exc)
            self._log_anomaly(
                AnomalyType.SUBSCRIBE_ERROR,
                {"slug": new_market.slug, "error": str(exc)},
            )
            return

        self._log_rotation({
            "rotation_number": self._rotation_count,
            "old_slug": old_slug,
            "new_slug": new_market.slug,
            "new_question": new_market.question,
            "new_up_token": new_market.up_token,
            "new_down_token": new_market.down_token,
            "remaining_s": new_market.remaining_s,
            "timeframe": new_market.timeframe,
        })

    async def _discovery_loop(self, ws: Any) -> None:
        """Periodically poll Gamma for the current 15m market and rotate.

        Runs only in discovery mode.  Polls every ``discovery_poll_s``
        and triggers ``_rotate_subscription`` when the market window changes.
        """
        from prediction.market_discovery import discover_btc_updown_markets

        logger.info(
            "[clob] discovery loop started — timeframe=%s poll_interval=%.0fs",
            self.discovery_timeframe,
            self.discovery_poll_s,
        )

        while self._running:
            try:
                snap = discover_btc_updown_markets(
                    timeframes=[self.discovery_timeframe],
                )

                target = (
                    snap.current_15m
                    if self.discovery_timeframe == "15m"
                    else snap.current_5m
                )

                if target is None:
                    logger.warning(
                        "[clob] discovery: no active %s market found "
                        "(raw=%d, btc_updown=%d, error=%s)",
                        self.discovery_timeframe,
                        snap.raw_event_count,
                        snap.btc_updown_count,
                        snap.error,
                    )
                elif target.slug != self._current_slug:
                    # New market window — rotate
                    await self._rotate_subscription(target, ws)
                else:
                    logger.debug(
                        "[clob] discovery: market unchanged (%s, %.0fm left)",
                        self._current_slug,
                        target.remaining_s / 60,
                    )

            except Exception as exc:
                logger.warning("[clob] discovery poll error: %s", exc)

            await asyncio.sleep(self.discovery_poll_s)

    # ----- main loop -----

    async def _ping_loop(self, ws: Any) -> None:
        """Send lowercase ``ping`` text every PING_INTERVAL_S.

        CLOB WS expects text ``ping`` messages (not WebSocket control frames).
        Server responds with text ``pong``.
        """
        while self._running:
            try:
                await ws.send("ping")
            except Exception:
                return
            await asyncio.sleep(PING_INTERVAL_S)

    async def _stale_monitor(self) -> None:
        """Detect when no event arrives for STALE_EVENT_THRESHOLD_S."""
        while self._running:
            await asyncio.sleep(STALE_EVENT_THRESHOLD_S)
            if self._last_event_time > 0:
                elapsed = time.time() - self._last_event_time
                if elapsed > STALE_EVENT_THRESHOLD_S:
                    self._log_anomaly(
                        AnomalyType.STALE,
                        {
                            "elapsed_s": round(elapsed, 2),
                            "threshold_s": STALE_EVENT_THRESHOLD_S,
                        },
                    )

    async def _health_emitter(self) -> None:
        """Emit periodic health summary to HEALTH_LOG."""
        while self._running:
            await asyncio.sleep(HEALTH_EMIT_INTERVAL_S)
            summary = self.health.emit_and_reset()
            # Enrich with discovery-mode metadata
            if self.discovery_mode:
                summary["discovery_mode"] = True
                summary["current_slug"] = self._current_slug
                summary["rotation_count"] = self._rotation_count
                if self._current_market:
                    summary["market_remaining_s"] = round(
                        self._current_market.end_ts - time.time(), 1
                    )
            summary["subscribed_ids"] = len(self._subscribed_ids)
            _append_jsonl(self.health_log, summary)
            logger.info("[clob] health: %s", json.dumps(summary))

    async def _connect_and_stream(self) -> None:
        """Single connection lifecycle."""
        logger.info("[clob] connecting to %s", self.ws_uri)

        async with ws_client.connect(
            self.ws_uri,
            ping_interval=None,  # We manage ping ourselves
            close_timeout=10,
        ) as ws:
            self._consecutive_reconnects = 0
            self._ws_ref = ws

            if self.discovery_mode:
                # Discovery mode: do initial discovery, subscribe to current
                # market, then start periodic re-discovery loop.
                logger.info(
                    "[clob] connected — discovery mode (%s)",
                    self.discovery_timeframe,
                )
                from prediction.market_discovery import (
                    discover_btc_updown_markets,
                )

                snap = discover_btc_updown_markets(
                    timeframes=[self.discovery_timeframe],
                )
                target = (
                    snap.current_15m
                    if self.discovery_timeframe == "15m"
                    else snap.current_5m
                )
                if target:
                    initial_ids = [target.up_token, target.down_token]
                    await ws.send(self._subscribe_message(initial_ids))
                    self._subscribed_ids = initial_ids
                    self._current_slug = target.slug
                    self._current_market = target
                    logger.info(
                        "[clob] initial subscribe — %s (%.0fm remaining)",
                        target.slug,
                        target.remaining_s / 60,
                    )
                else:
                    # No market available right now — subscribe empty, let
                    # discovery loop pick up once one appears.
                    logger.warning(
                        "[clob] no active market at connect time — "
                        "waiting for discovery loop"
                    )
                    await ws.send(self._subscribe_message([]))
            else:
                # Static mode: subscribe to configured asset_ids
                logger.info(
                    "[clob] connected — subscribing to %d asset(s)",
                    len(self.asset_ids),
                )
                await ws.send(self._subscribe_message(self.asset_ids))
                self._subscribed_ids = list(self.asset_ids)

            # Start ping loop
            ping_task = asyncio.create_task(self._ping_loop(ws))

            # Start discovery loop if in discovery mode
            discovery_task: Optional[asyncio.Task[None]] = None
            if self.discovery_mode:
                discovery_task = asyncio.create_task(self._discovery_loop(ws))

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break
                    if isinstance(raw_msg, bytes):
                        raw_msg = raw_msg.decode("utf-8", errors="replace")
                    records = self._process_message(raw_msg)
                    if records:
                        for r in records:
                            logger.debug(
                                "[clob] %s asset=%s",
                                r.get("event_type"),
                                r.get("asset_id", "?")[:16] + "...",
                            )
            finally:
                ping_task.cancel()
                if discovery_task is not None:
                    discovery_task.cancel()
                for t in [ping_task] + (
                    [discovery_task] if discovery_task else []
                ):
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                self._ws_ref = None

    async def run(self) -> None:
        """Main entry — connect with exponential backoff reconnection."""
        if websockets is None:
            raise ImportError(
                "websockets is required: pip install websockets"
            )
        if not self.discovery_mode and not self.asset_ids:
            raise ValueError("No asset_ids configured — cannot subscribe")

        self._running = True

        if self.discovery_mode:
            logger.info(
                "[clob] CLOB Market Client starting — DISCOVERY MODE "
                "uri=%s timeframe=%s poll=%.0fs",
                self.ws_uri,
                self.discovery_timeframe,
                self.discovery_poll_s,
            )
        else:
            logger.info(
                "[clob] CLOB Market Client starting — STATIC MODE "
                "uri=%s assets=%d",
                self.ws_uri,
                len(self.asset_ids),
            )
            logger.info(
                "[clob] asset_ids: %s",
                ", ".join(a[:16] + "..." for a in self.asset_ids),
            )
        logger.info(
            "[clob] logs: events=%s health=%s anomalies=%s",
            self.market_log,
            self.health_log,
            self.env_events_log,
        )

        # Ensure log directories
        _ensure_parent(self.market_log)
        _ensure_parent(self.health_log)
        _ensure_parent(self.env_events_log)

        # Background tasks
        stale_task = asyncio.create_task(self._stale_monitor())
        health_task = asyncio.create_task(self._health_emitter())

        try:
            while self._running:
                try:
                    await self._connect_and_stream()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._consecutive_reconnects += 1
                    delay = min(
                        RECONNECT_BASE_DELAY_S * (2 ** min(self._consecutive_reconnects, 6)),
                        RECONNECT_MAX_DELAY_S,
                    )
                    logger.warning(
                        "[clob] connection lost (%s), reconnect %d/%d in %.1fs: %s",
                        type(exc).__name__,
                        self._consecutive_reconnects,
                        MAX_RECONNECT_FAILURES,
                        delay,
                        exc,
                    )
                    self._log_anomaly(
                        AnomalyType.WS_ERROR,
                        {
                            "error": str(exc),
                            "reconnect_count": self._consecutive_reconnects,
                        },
                    )

                    if self._consecutive_reconnects >= MAX_RECONNECT_FAILURES:
                        self._log_anomaly(
                            AnomalyType.RECONNECT_STORM,
                            {
                                "count": self._consecutive_reconnects,
                                "max": MAX_RECONNECT_FAILURES,
                            },
                        )
                        logger.error(
                            "[clob] reconnect storm (%d failures) — stopping",
                            self._consecutive_reconnects,
                        )
                        break

                    await asyncio.sleep(delay)
        finally:
            self._running = False
            stale_task.cancel()
            health_task.cancel()
            for t in (stale_task, health_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    def stop(self) -> None:
        """Signal graceful shutdown."""
        self._running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the CLOB market client as a standalone process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    client = CLOBMarketClient()

    if client.discovery_mode:
        logger.info(
            "[clob] Discovery mode ON — timeframe=%s poll=%.0fs",
            client.discovery_timeframe,
            client.discovery_poll_s,
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, client.stop)

    try:
        loop.run_until_complete(client.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("[clob] shutdown complete")


if __name__ == "__main__":
    main()
