"""RTDS Oracle Client — Chainlink BTC/USD price stream ingestion.

Layer 1 of the Binary Sleeve data plane.  Connects to Polymarket's RTDS
WebSocket, subscribes to ``crypto_prices_chainlink`` filtered to BTC/USD,
and persists every tick as append-only JSONL.

Design invariants
-----------------
* **Append-only** — never mutates ``ORACLE_LOG``.
* **One event per line** — JSONL, each line is a self-contained JSON object.
* **Deterministic** — captures both oracle timestamp and local arrival time.
* **Isolated** — no imports from ``execution/`` or ``dashboard/``.
* **Standalone** — runnable via ``python -m prediction.rtds_oracle_client``.

Log paths
---------
* Ticks:     ``logs/prediction/rtds_oracle.jsonl``
* Anomalies: ``logs/execution/environment_events.jsonl``
* Health:    ``logs/prediction/rtds_oracle_health.jsonl``

References
----------
* Research report: ``research/polymarket/deep-research-report.md``
* RTDS endpoint: ``wss://ws-live-data.polymarket.com``
* Topic: ``crypto_prices_chainlink``, symbol ``btc/usd``
* Format: batch array ``payload.data = [{timestamp, value}, ...]``
* Keepalive: text PING every 5 s
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

ORACLE_LOG = _PROJECT_ROOT / "logs" / "prediction" / "rtds_oracle.jsonl"
HEALTH_LOG = _PROJECT_ROOT / "logs" / "prediction" / "rtds_oracle_health.jsonl"
ENV_EVENTS_LOG = _PROJECT_ROOT / "logs" / "execution" / "environment_events.jsonl"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
WS_URI = os.environ.get(
    "RTDS_WS_URI", "wss://ws-live-data.polymarket.com"
)
RTDS_TOPIC = "crypto_prices_chainlink"
RTDS_SYMBOL = "btc/usd"

PING_INTERVAL_S: float = float(os.environ.get("RTDS_PING_INTERVAL_S", "5"))
STALE_TICK_THRESHOLD_S: float = float(
    os.environ.get("RTDS_STALE_TICK_S", "120")
)
MAX_RECONNECT_FAILURES: int = int(
    os.environ.get("RTDS_MAX_RECONNECT_FAILURES", "20")
)
RECONNECT_BASE_DELAY_S: float = 1.0
RECONNECT_MAX_DELAY_S: float = 60.0
HEALTH_EMIT_INTERVAL_S: float = float(
    os.environ.get("RTDS_HEALTH_INTERVAL_S", "60")
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("prediction.rtds_oracle")


# ---------------------------------------------------------------------------
# Anomaly types
# ---------------------------------------------------------------------------
class AnomalyType:
    GAP = "rtds_gap"
    TIME_REGRESSION = "rtds_time_regression"
    DUPLICATE_SEQ = "rtds_duplicate_seq"
    STALE = "rtds_stale"
    RECONNECT_STORM = "rtds_reconnect_storm"
    WS_ERROR = "rtds_ws_error"


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
    """Collects per-window latency / gap stats and emits periodic summaries."""

    def __init__(self) -> None:
        self.latencies_ms: List[float] = []
        self.gaps_ms: List[float] = []
        self.tick_count: int = 0
        self.anomaly_count: int = 0
        self.window_start: float = time.time()

    def record_tick(self, latency_ms: float, gap_ms: Optional[float]) -> None:
        self.tick_count += 1
        self.latencies_ms.append(latency_ms)
        if gap_ms is not None:
            self.gaps_ms.append(gap_ms)

    def record_anomaly(self) -> None:
        self.anomaly_count += 1

    def emit_and_reset(self) -> Dict[str, Any]:
        now = time.time()
        elapsed_s = now - self.window_start

        summary: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "window_s": round(elapsed_s, 2),
            "tick_count": self.tick_count,
            "anomaly_count": self.anomaly_count,
        }

        if self.latencies_ms:
            sorted_lat = sorted(self.latencies_ms)
            summary["latency_ms"] = {
                "mean": round(statistics.mean(sorted_lat), 2),
                "median": round(statistics.median(sorted_lat), 2),
                "p95": round(
                    sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) >= 20 else sorted_lat[-1],
                    2,
                ),
                "max": round(sorted_lat[-1], 2),
                "min": round(sorted_lat[0], 2),
            }
        else:
            summary["latency_ms"] = None

        if self.gaps_ms:
            sorted_gap = sorted(self.gaps_ms)
            summary["gap_ms"] = {
                "mean": round(statistics.mean(sorted_gap), 2),
                "max": round(sorted_gap[-1], 2),
                "p95": round(
                    sorted_gap[int(len(sorted_gap) * 0.95)] if len(sorted_gap) >= 20 else sorted_gap[-1],
                    2,
                ),
            }
            summary["tick_frequency_hz"] = (
                round(self.tick_count / elapsed_s, 3) if elapsed_s > 0 else 0
            )
        else:
            summary["gap_ms"] = None
            summary["tick_frequency_hz"] = 0

        # Reset
        self.latencies_ms = []
        self.gaps_ms = []
        self.tick_count = 0
        self.anomaly_count = 0
        self.window_start = now

        return summary


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------
class RTDSOracleClient:
    """Async WebSocket client for RTDS Chainlink BTC/USD stream."""

    def __init__(
        self,
        ws_uri: str = WS_URI,
        oracle_log: Path = ORACLE_LOG,
        health_log: Path = HEALTH_LOG,
        env_events_log: Path = ENV_EVENTS_LOG,
    ) -> None:
        self.ws_uri = ws_uri
        self.oracle_log = oracle_log
        self.health_log = health_log
        self.env_events_log = env_events_log

        self._last_oracle_ts_ms: Optional[int] = None
        self._last_seq: Optional[int] = None
        self._last_tick_time: float = 0.0
        self._seq_counter: int = 0
        self._consecutive_reconnects: int = 0
        self._running: bool = False

        self.health = HealthAccumulator()

    # ----- anomaly logging -----

    def _log_anomaly(self, anomaly_type: str, detail: Dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": anomaly_type,
            "source": "rtds_oracle_client",
            **detail,
        }
        _append_jsonl(self.env_events_log, event)
        self.health.record_anomaly()
        logger.warning("[rtds] anomaly %s: %s", anomaly_type, detail)

    # ----- tick processing -----

    def _process_message(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Route an RTDS message — may contain a batch of ticks.

        Real RTDS format (observed 2026-02-22)::

            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "timestamp": <envelope_ms>,
                "payload": {
                    "symbol": "btc/usd",
                    "data": [
                        {"timestamp": <ms>, "value": <float>},
                        ...
                    ]
                }
            }

        Without a server-side symbol filter the RTDS sends ~1 Hz heartbeat
        messages for every symbol, most with ``data: []``.  These keep the
        connection alive and we refresh ``_last_tick_time`` so the stale
        monitor stays quiet.  Only messages where ``payload.symbol``
        contains 'btc' are further processed for tick extraction.

        Falls back to single-tick parsing for forward compatibility.
        """
        arrival_ms = int(time.time() * 1000)
        results: List[Dict[str, Any]] = []

        payload = msg.get("payload")

        # Any valid RTDS message (even empty heartbeats) proves the
        # connection is alive → refresh stale monitor clock.
        if isinstance(payload, dict):
            self._last_tick_time = time.time()

        # --- Batch format: payload.data is a list of {timestamp, value} ---
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            symbol_raw = str(payload.get("symbol", "")).lower().replace("_", "/")
            if "btc" not in symbol_raw:
                return results
            for item in payload["data"]:
                if not isinstance(item, dict):
                    continue
                rec = self._process_tick(item, arrival_ms=arrival_ms, symbol_hint=symbol_raw)
                if rec is not None:
                    results.append(rec)
            return results

        # --- Single-tick fallback (forward compatibility) ---
        rec = self._process_tick(msg, arrival_ms=arrival_ms)
        if rec is not None:
            results.append(rec)
        return results

    def _process_tick(
        self,
        msg: Dict[str, Any],
        *,
        arrival_ms: Optional[int] = None,
        symbol_hint: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Parse a single tick item, validate, write, return record or None.

        Called either with a batch data item ``{"timestamp": ..., "value": ...}``
        or with a full single-tick message (fallback path).
        """
        if arrival_ms is None:
            arrival_ms = int(time.time() * 1000)

        # --- Extract fields defensively ---
        # Batch items are flat: {timestamp, value}
        # Single messages may be nested: {payload: {symbol, timestamp, value}}
        inner = msg
        if isinstance(msg.get("payload"), dict):
            inner = msg["payload"]

        price = (
            inner.get("value")
            or inner.get("price")
            or inner.get("p")
        )
        oracle_ts_ms = (
            inner.get("timestamp")
            or inner.get("t")
            or msg.get("timestamp")
        )
        seq = inner.get("seq") or inner.get("sequence")

        # Symbol filter — only BTC/USD
        symbol_raw = (
            symbol_hint
            or inner.get("symbol")
            or inner.get("s")
            or inner.get("pair")
            or msg.get("symbol")
            or ""
        )
        symbol = str(symbol_raw).lower().replace("_", "/")

        if not price or not oracle_ts_ms:
            return None

        if "btc" not in symbol:
            return None

        price = float(price)
        oracle_ts_ms = int(oracle_ts_ms)
        latency_ms = arrival_ms - oracle_ts_ms

        # --- Sequence validation ---
        self._seq_counter += 1
        local_seq = self._seq_counter

        if seq is not None:
            seq = int(seq)
            if self._last_seq is not None and seq == self._last_seq:
                self._log_anomaly(
                    AnomalyType.DUPLICATE_SEQ,
                    {"seq": seq, "price": price},
                )
                return None
            self._last_seq = seq

        # --- Time regression check ---
        if self._last_oracle_ts_ms is not None:
            if oracle_ts_ms < self._last_oracle_ts_ms:
                self._log_anomaly(
                    AnomalyType.TIME_REGRESSION,
                    {
                        "oracle_ts_ms": oracle_ts_ms,
                        "prev_oracle_ts_ms": self._last_oracle_ts_ms,
                        "delta_ms": oracle_ts_ms - self._last_oracle_ts_ms,
                    },
                )
                # Still log the tick but flag it
                pass

        # --- Gap detection ---
        gap_ms: Optional[float] = None
        if self._last_oracle_ts_ms is not None:
            gap_ms = float(oracle_ts_ms - self._last_oracle_ts_ms)
            if gap_ms > STALE_TICK_THRESHOLD_S * 1000:
                self._log_anomaly(
                    AnomalyType.GAP,
                    {
                        "gap_ms": gap_ms,
                        "threshold_ms": STALE_TICK_THRESHOLD_S * 1000,
                        "oracle_ts_ms": oracle_ts_ms,
                    },
                )

        self._last_oracle_ts_ms = oracle_ts_ms
        self._last_tick_time = time.time()

        # Build record
        record: Dict[str, Any] = {
            "ts_arrival_ms": arrival_ms,
            "oracle_ts_ms": oracle_ts_ms,
            "symbol": "BTC/USD",
            "price": price,
            "latency_ms": latency_ms,
            "seq": seq if seq is not None else local_seq,
            "source": "RTDS",
        }

        # Append to oracle log
        _append_jsonl(self.oracle_log, record)
        self.health.record_tick(float(latency_ms), gap_ms)

        return record

    # ----- subscription message -----

    @staticmethod
    @staticmethod
    def _subscribe_message() -> str:
        """Build the RTDS subscribe payload per documented protocol.

        Subscribes to **all** symbols on the topic (no filters).
        RTDS with symbol filter only delivers the initial backfill batch
        then goes silent; without filter it sends ~1 Hz heartbeats for
        every symbol, including price updates in the ``data`` array.
        Client-side symbol filtering in ``_process_message`` handles
        restricting to BTC/USD.
        """
        return json.dumps(
            {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": RTDS_TOPIC,
                        "type": "*",
                    }
                ],
            }
        )

    # ----- main loop -----

    async def _ping_loop(self, ws: Any) -> None:
        """Send PING every PING_INTERVAL_S as required by RTDS spec.

        RTDS expects text 'PING' messages (not WebSocket control frames).
        """
        while self._running:
            try:
                await ws.send("PING")
            except Exception:
                return
            await asyncio.sleep(PING_INTERVAL_S)

    async def _stale_monitor(self) -> None:
        """Detect when no tick arrives for STALE_TICK_THRESHOLD_S."""
        while self._running:
            await asyncio.sleep(STALE_TICK_THRESHOLD_S)
            if self._last_tick_time > 0:
                elapsed = time.time() - self._last_tick_time
                if elapsed > STALE_TICK_THRESHOLD_S:
                    self._log_anomaly(
                        AnomalyType.STALE,
                        {
                            "elapsed_s": round(elapsed, 2),
                            "threshold_s": STALE_TICK_THRESHOLD_S,
                        },
                    )

    async def _health_emitter(self) -> None:
        """Emit periodic health summary to HEALTH_LOG."""
        while self._running:
            await asyncio.sleep(HEALTH_EMIT_INTERVAL_S)
            summary = self.health.emit_and_reset()
            _append_jsonl(self.health_log, summary)
            logger.info("[rtds] health: %s", json.dumps(summary))

    async def _connect_and_stream(self) -> None:
        """Single connection lifecycle."""
        logger.info("[rtds] connecting to %s", self.ws_uri)

        async with ws_client.connect(
            self.ws_uri,
            ping_interval=None,  # We manage PING ourselves
            close_timeout=10,
        ) as ws:
            self._consecutive_reconnects = 0
            logger.info("[rtds] connected — subscribing to %s/%s", RTDS_TOPIC, RTDS_SYMBOL)

            # Subscribe
            await ws.send(self._subscribe_message())

            # Start ping loop
            ping_task = asyncio.create_task(self._ping_loop(ws))

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break
                    try:
                        msg = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug("[rtds] non-JSON message: %s", e)
                        continue

                    self._process_message(msg)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

    async def run(self) -> None:
        """Main entry — connect with exponential backoff reconnection."""
        if websockets is None:
            raise ImportError(
                "websockets is required: pip install websockets"
            )

        self._running = True
        logger.info(
            "[rtds] RTDS Oracle Client starting — uri=%s topic=%s symbol=%s",
            self.ws_uri,
            RTDS_TOPIC,
            RTDS_SYMBOL,
        )
        logger.info(
            "[rtds] logs: ticks=%s health=%s anomalies=%s",
            self.oracle_log,
            self.health_log,
            self.env_events_log,
        )

        # Ensure log directories
        _ensure_parent(self.oracle_log)
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
                        "[rtds] connection lost (%s), reconnect %d/%d in %.1fs: %s",
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
                            "[rtds] reconnect storm (%d failures) — stopping",
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
# Standalone entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the RTDS oracle client as a standalone process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    client = RTDSOracleClient()

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
        logger.info("[rtds] shutdown complete")


if __name__ == "__main__":
    main()
