"""Tests for prediction.rtds_oracle_client — Layer 1 oracle pipe.

Covers:
    - Tick processing (happy path, filtering, schema)
    - Anomaly detection (gap, time regression, duplicate seq)
    - Health accumulator (stats, reset)
    - Log writing (append-only JSONL invariant)
    - Subscribe message format
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from prediction.rtds_oracle_client import (
    AnomalyType,
    HealthAccumulator,
    RTDSOracleClient,
    _append_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_logs(tmp_path: Path):
    """Create temp log paths for isolated testing."""
    return {
        "oracle": tmp_path / "rtds_oracle.jsonl",
        "health": tmp_path / "rtds_oracle_health.jsonl",
        "env": tmp_path / "environment_events.jsonl",
    }


@pytest.fixture()
def client(tmp_logs) -> RTDSOracleClient:
    """Client with temp log paths — no real WS connection."""
    return RTDSOracleClient(
        ws_uri="wss://localhost:9999/fake",
        oracle_log=tmp_logs["oracle"],
        health_log=tmp_logs["health"],
        env_events_log=tmp_logs["env"],
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# _append_jsonl
# ---------------------------------------------------------------------------
class TestAppendJsonl:
    def test_creates_file_and_parent(self, tmp_path: Path):
        target = tmp_path / "sub" / "deep" / "log.jsonl"
        _append_jsonl(target, {"a": 1})
        assert target.exists()
        lines = _read_jsonl(target)
        assert len(lines) == 1
        assert lines[0] == {"a": 1}

    def test_appends_multiple(self, tmp_path: Path):
        target = tmp_path / "log.jsonl"
        _append_jsonl(target, {"n": 1})
        _append_jsonl(target, {"n": 2})
        _append_jsonl(target, {"n": 3})
        lines = _read_jsonl(target)
        assert len(lines) == 3
        assert [l["n"] for l in lines] == [1, 2, 3]

    def test_one_line_per_event(self, tmp_path: Path):
        target = tmp_path / "log.jsonl"
        _append_jsonl(target, {"k": "v"})
        raw = target.read_text()
        assert raw.count("\n") == 1  # exactly one newline


# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------
class TestProcessTick:
    def test_happy_path_rtds_format(self, client: RTDSOracleClient, tmp_logs):
        """Real RTDS message format: {topic, type, timestamp, payload: {symbol, data: [...]}}."""
        msg = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1708598123456,
            "payload": {
                "symbol": "btc/usd",
                "data": [
                    {"timestamp": 1708598122000, "value": 67500.0},
                    {"timestamp": 1708598123000, "value": 67501.0},
                ],
            },
        }
        results = client._process_message(msg)
        assert len(results) == 2
        assert results[0]["symbol"] == "BTC/USD"
        assert results[0]["price"] == 67500.0
        assert results[1]["price"] == 67501.0
        assert results[0]["source"] == "RTDS"

        # Verify written to log
        lines = _read_jsonl(tmp_logs["oracle"])
        assert len(lines) == 2
        assert lines[0]["price"] == 67500.0
        assert lines[1]["price"] == 67501.0

    def test_single_tick_nested_payload(self, client: RTDSOracleClient, tmp_logs):
        """Single-tick format: {payload: {symbol, timestamp, value}} (no data array)."""
        msg = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1708598123456,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 1708598123000,
                "value": 67500.0,
            },
        }
        result = client._process_tick(msg)
        assert result is not None
        assert result["symbol"] == "BTC/USD"
        assert result["price"] == 67500.0
        assert result["oracle_ts_ms"] == 1708598123000
        assert result["source"] == "RTDS"
        assert "ts_arrival_ms" in result
        assert "latency_ms" in result

    def test_flat_payload_fallback(self, client: RTDSOracleClient, tmp_logs):
        """Flat message without nested payload."""
        msg = {
            "symbol": "btc/usd",
            "price": 67500.0,
            "timestamp": 1708598123000,
            "seq": 100,
        }
        result = client._process_tick(msg)
        assert result is not None
        assert result["price"] == 67500.0
        assert result["seq"] == 100

    def test_batch_data_array(self, client: RTDSOracleClient, tmp_logs):
        """Batch format with payload.data array (observed from live RTDS)."""
        msg = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1708598123456,
            "payload": {
                "symbol": "btc/usd",
                "data": [
                    {"timestamp": 1708598120000, "value": 67000.0},
                    {"timestamp": 1708598121000, "value": 67001.0},
                    {"timestamp": 1708598122000, "value": 67002.0},
                ],
            },
        }
        results = client._process_message(msg)
        assert len(results) == 3
        assert [r["price"] for r in results] == [67000.0, 67001.0, 67002.0]

    def test_nested_payload_field(self, client: RTDSOracleClient):
        """RTDS may nest deeper in a 'payload' field."""
        msg = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1708598125456,
            "payload": {
                "symbol": "btc/usd",
                "value": 69000.0,
                "timestamp": 1708598125000,
            },
        }
        result = client._process_tick(msg)
        assert result is not None
        assert result["price"] == 69000.0

    def test_filters_non_btc(self, client: RTDSOracleClient, tmp_logs):
        msg = {
            "symbol": "eth/usd",
            "price": 3500.0,
            "timestamp": 1708598126000,
        }
        result = client._process_tick(msg)
        assert result is None
        assert not tmp_logs["oracle"].exists()

    def test_filters_non_btc_batch(self, client: RTDSOracleClient, tmp_logs):
        """Batch with non-BTC symbol should produce no results."""
        msg = {
            "payload": {
                "symbol": "eth/usd",
                "data": [{"timestamp": 1708598126000, "value": 3500.0}],
            }
        }
        results = client._process_message(msg)
        assert len(results) == 0

    def test_skips_control_message(self, client: RTDSOracleClient, tmp_logs):
        """Messages without price/timestamp are silently skipped."""
        msg = {"type": "subscribed", "channel": "crypto_prices_chainlink"}
        result = client._process_tick(msg)
        assert result is None

    def test_alternative_field_names(self, client: RTDSOracleClient):
        """Support shorthand field names p/t/s."""
        msg = {"s": "btc/usd", "p": 67000.0, "t": 1708598127000}
        result = client._process_tick(msg)
        assert result is not None
        assert result["price"] == 67000.0

    def test_value_field_canonical(self, client: RTDSOracleClient):
        """The canonical RTDS field is 'value', not 'price'."""
        msg = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1708598130000,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 1708598129500,
                "value": 67999.99,
            },
        }
        result = client._process_tick(msg)
        assert result is not None
        assert result["price"] == 67999.99

    def test_underscore_symbol_normalised(self, client: RTDSOracleClient):
        """btc_usd should be normalised to btc/usd and pass filter."""
        msg = {"symbol": "BTC_USD", "price": 66000.0, "timestamp": 1708598128000}
        result = client._process_tick(msg)
        assert result is not None


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
class TestAnomalyDetection:
    def test_gap_detection(self, client: RTDSOracleClient, tmp_logs):
        """Gap > STALE_TICK_THRESHOLD_S triggers anomaly."""
        import prediction.rtds_oracle_client as mod
        orig = mod.STALE_TICK_THRESHOLD_S
        mod.STALE_TICK_THRESHOLD_S = 2.0  # 2 second threshold
        try:
            client._process_tick(
                {"symbol": "btc/usd", "value": 67000.0, "timestamp": 1000000}
            )
            # 3 seconds later
            client._process_tick(
                {"symbol": "btc/usd", "value": 67001.0, "timestamp": 1003000}
            )
            anomalies = _read_jsonl(tmp_logs["env"])
            gap_events = [e for e in anomalies if e["event"] == AnomalyType.GAP]
            assert len(gap_events) == 1
            assert gap_events[0]["gap_ms"] == 3000.0
        finally:
            mod.STALE_TICK_THRESHOLD_S = orig

    def test_time_regression(self, client: RTDSOracleClient, tmp_logs):
        """Oracle time going backwards triggers anomaly."""
        client._process_tick(
            {"symbol": "btc/usd", "value": 67000.0, "timestamp": 2000000}
        )
        client._process_tick(
            {"symbol": "btc/usd", "value": 67001.0, "timestamp": 1999000}
        )
        anomalies = _read_jsonl(tmp_logs["env"])
        regression = [e for e in anomalies if e["event"] == AnomalyType.TIME_REGRESSION]
        assert len(regression) == 1
        assert regression[0]["delta_ms"] == -1000

    def test_duplicate_seq(self, client: RTDSOracleClient, tmp_logs):
        """Duplicate sequence number triggers anomaly and drops tick."""
        client._process_tick(
            {"symbol": "btc/usd", "value": 67000.0, "timestamp": 3000000, "seq": 50}
        )
        result = client._process_tick(
            {"symbol": "btc/usd", "value": 67001.0, "timestamp": 3001000, "seq": 50}
        )
        assert result is None  # dropped
        anomalies = _read_jsonl(tmp_logs["env"])
        dup_events = [e for e in anomalies if e["event"] == AnomalyType.DUPLICATE_SEQ]
        assert len(dup_events) == 1

    def test_no_anomaly_on_normal_ticks(self, client: RTDSOracleClient, tmp_logs):
        """Sequential, non-gapped ticks produce no anomalies."""
        for i in range(5):
            client._process_tick(
                {"symbol": "btc/usd", "value": 67000.0 + i, "timestamp": 4000000 + i * 500, "seq": i}
            )
        assert not tmp_logs["env"].exists() or len(_read_jsonl(tmp_logs["env"])) == 0


# ---------------------------------------------------------------------------
# Health accumulator
# ---------------------------------------------------------------------------
class TestHealthAccumulator:
    def test_empty_emit(self):
        h = HealthAccumulator()
        summary = h.emit_and_reset()
        assert summary["tick_count"] == 0
        assert summary["latency_ms"] is None
        assert summary["gap_ms"] is None

    def test_stats_correct(self):
        h = HealthAccumulator()
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for i, lat in enumerate(latencies):
            gap = float(i * 100) if i > 0 else None
            h.record_tick(lat, gap)

        summary = h.emit_and_reset()
        assert summary["tick_count"] == 5
        assert summary["latency_ms"]["mean"] == 30.0
        assert summary["latency_ms"]["min"] == 10.0
        assert summary["latency_ms"]["max"] == 50.0

    def test_reset_clears(self):
        h = HealthAccumulator()
        h.record_tick(10.0, None)
        h.record_tick(20.0, 100.0)
        h.record_anomaly()
        h.emit_and_reset()

        # After reset
        summary = h.emit_and_reset()
        assert summary["tick_count"] == 0
        assert summary["anomaly_count"] == 0

    def test_p95_with_many_samples(self):
        h = HealthAccumulator()
        for i in range(100):
            h.record_tick(float(i), float(i * 10) if i > 0 else None)
        summary = h.emit_and_reset()
        assert summary["tick_count"] == 100
        # p95 of 0..99 = 95.0
        assert summary["latency_ms"]["p95"] == 95.0


# ---------------------------------------------------------------------------
# Subscribe message
# ---------------------------------------------------------------------------
class TestSubscribeMessage:
    def test_format(self):
        msg = json.loads(RTDSOracleClient._subscribe_message())
        assert msg["action"] == "subscribe"
        subs = msg["subscriptions"]
        assert len(subs) == 1
        assert subs[0]["topic"] == "crypto_prices_chainlink"
        assert subs[0]["type"] == "*"
        # No symbol filter — client-side filtering handles BTC/USD
        assert "filters" not in subs[0]


class TestHeartbeat:
    def test_empty_data_refreshes_stale_clock(self, client: RTDSOracleClient):
        """Empty-data heartbeats (from non-BTC or empty batches) refresh _last_tick_time."""
        client._last_tick_time = 0
        msg = {
            "payload": {
                "symbol": "eth/usd",
                "data": [],
            }
        }
        client._process_message(msg)
        assert client._last_tick_time > 0

    def test_btc_empty_batch_refreshes_stale_clock(self, client: RTDSOracleClient):
        """BTC empty-data heartbeat also refreshes stale clock."""
        client._last_tick_time = 0
        msg = {
            "payload": {
                "symbol": "btc/usd",
                "data": [],
            }
        }
        results = client._process_message(msg)
        assert len(results) == 0  # no ticks
        assert client._last_tick_time > 0  # but stale clock refreshed


# ---------------------------------------------------------------------------
# Sequential tick logging integrity
# ---------------------------------------------------------------------------
class TestLogIntegrity:
    def test_multiple_ticks_append_in_order(self, client: RTDSOracleClient, tmp_logs):
        prices = [67000.0, 67001.0, 67002.0, 67003.0, 67004.0]
        for i, p in enumerate(prices):
            client._process_tick(
                {"symbol": "btc/usd", "value": p, "timestamp": 5000000 + i * 500, "seq": 200 + i}
            )
        lines = _read_jsonl(tmp_logs["oracle"])
        assert len(lines) == 5
        assert [l["price"] for l in lines] == prices

    def test_seq_increments_when_no_upstream_seq(self, client: RTDSOracleClient, tmp_logs):
        """When upstream provides no seq, client assigns local seq."""
        client._process_tick(
            {"symbol": "btc/usd", "value": 67000.0, "timestamp": 6000000}
        )
        client._process_tick(
            {"symbol": "btc/usd", "value": 67001.0, "timestamp": 6001000}
        )
        lines = _read_jsonl(tmp_logs["oracle"])
        assert len(lines) == 2
        assert lines[0]["seq"] != lines[1]["seq"]  # unique
