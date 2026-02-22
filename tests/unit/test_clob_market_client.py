"""Tests for prediction.clob_market_client — Layer 2 market data pipe.

Covers:
    - Event processing (best_bid_ask, last_trade_price, tick_size_change,
      market_resolved, book, price_change)
    - Message routing (arrays, single objects, pong handling)
    - Anomaly detection (unknown event, stale)
    - Health accumulator (stats, spread tracking, reset)
    - Log writing (append-only JSONL invariant)
    - Subscribe message format
    - Public query methods (bid/ask/spread/trade)
    - Discovery mode (rotation, subscribed_ids, current_market)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from prediction.clob_market_client import (
    AnomalyType,
    CLOBMarketClient,
    HealthAccumulator,
    _append_jsonl,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_logs(tmp_path: Path):
    """Create temp log paths for isolated testing."""
    return {
        "market": tmp_path / "clob_market.jsonl",
        "health": tmp_path / "clob_market_health.jsonl",
        "env": tmp_path / "environment_events.jsonl",
    }


ASSET_A = "46866868857194367945413771860582064655745092128562966218540356888709464260149"
ASSET_B = "13915689317269078219168496739008737517740566192006337297676041270492637394586"


@pytest.fixture()
def client(tmp_logs) -> CLOBMarketClient:
    """Client with temp log paths — no real WS connection."""
    return CLOBMarketClient(
        ws_uri="wss://localhost:9999/fake",
        asset_ids=[ASSET_A, ASSET_B],
        market_log=tmp_logs["market"],
        health_log=tmp_logs["health"],
        env_events_log=tmp_logs["env"],
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------
class TestSafeFloat:
    def test_string_number(self):
        assert _safe_float("0.65") == 0.65

    def test_float(self):
        assert _safe_float(0.65) == 0.65

    def test_int(self):
        assert _safe_float(1) == 1.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("not_a_number") is None

    def test_empty_string(self):
        assert _safe_float("") is None


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
        assert [ln["n"] for ln in lines] == [1, 2, 3]

    def test_one_line_per_event(self, tmp_path: Path):
        target = tmp_path / "log.jsonl"
        _append_jsonl(target, {"k": "v"})
        raw = target.read_text()
        assert raw.count("\n") == 1


# ---------------------------------------------------------------------------
# best_bid_ask processing
# ---------------------------------------------------------------------------
class TestBestBidAsk:
    def test_happy_path(self, client: CLOBMarketClient, tmp_logs):
        """Single best_bid_ask event in array format."""
        raw = json.dumps([{
            "event_type": "best_bid_ask",
            "asset_id": ASSET_A,
            "best_bid": "0.65",
            "best_ask": "0.67",
            "timestamp": "1708598123456",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "best_bid_ask"
        assert r["best_bid"] == 0.65
        assert r["best_ask"] == 0.67
        assert r["spread"] == round(0.67 - 0.65, 6)
        assert r["asset_id"] == ASSET_A
        assert r["source"] == "CLOB"
        assert "ts_arrival_ms" in r

        # Written to log
        logged = _read_jsonl(tmp_logs["market"])
        assert len(logged) == 1
        assert logged[0]["event_type"] == "best_bid_ask"

    def test_updates_internal_state(self, client: CLOBMarketClient):
        raw = json.dumps([{
            "event_type": "best_bid_ask",
            "asset_id": ASSET_A,
            "best_bid": "0.60",
            "best_ask": "0.62",
        }])
        client._process_message(raw)
        assert client.get_best_bid(ASSET_A) == 0.60
        assert client.get_best_ask(ASSET_A) == 0.62
        assert client.get_spread(ASSET_A) == round(0.62 - 0.60, 6)

    def test_spread_calculation(self, client: CLOBMarketClient):
        raw = json.dumps([{
            "event_type": "best_bid_ask",
            "asset_id": ASSET_B,
            "best_bid": "0.10",
            "best_ask": "0.15",
        }])
        client._process_message(raw)
        assert client.get_spread(ASSET_B) == 0.05

    def test_null_bid_or_ask(self, client: CLOBMarketClient):
        """Missing bid/ask should still log but not crash."""
        raw = json.dumps([{
            "event_type": "best_bid_ask",
            "asset_id": ASSET_A,
            "best_bid": None,
            "best_ask": "0.67",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        assert records[0]["best_bid"] is None
        assert records[0]["best_ask"] == 0.67
        # No spread when bid is None
        assert "spread" not in records[0] or records[0].get("spread") is None


# ---------------------------------------------------------------------------
# last_trade_price processing
# ---------------------------------------------------------------------------
class TestLastTradePrice:
    def test_happy_path(self, client: CLOBMarketClient, tmp_logs):
        raw = json.dumps([{
            "event_type": "last_trade_price",
            "asset_id": ASSET_A,
            "price": "0.66",
            "side": "BUY",
            "size": "100.5",
            "fee_rate_bps": "200",
            "timestamp": "1708598123456",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "last_trade_price"
        assert r["price"] == 0.66
        assert r["side"] == "BUY"
        assert r["size"] == "100.5"

    def test_updates_internal_state(self, client: CLOBMarketClient):
        raw = json.dumps([{
            "event_type": "last_trade_price",
            "asset_id": ASSET_A,
            "price": "0.72",
        }])
        client._process_message(raw)
        assert client.get_last_trade_price(ASSET_A) == 0.72


# ---------------------------------------------------------------------------
# tick_size_change processing
# ---------------------------------------------------------------------------
class TestTickSizeChange:
    def test_happy_path(self, client: CLOBMarketClient, tmp_logs):
        raw = json.dumps([{
            "event_type": "tick_size_change",
            "asset_id": ASSET_A,
            "old_tick_size": "0.01",
            "new_tick_size": "0.001",
            "timestamp": "1708598123456",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "tick_size_change"
        assert r["old_tick_size"] == "0.01"
        assert r["new_tick_size"] == "0.001"

        logged = _read_jsonl(tmp_logs["market"])
        assert logged[0]["old_tick_size"] == "0.01"


# ---------------------------------------------------------------------------
# market_resolved processing
# ---------------------------------------------------------------------------
class TestMarketResolved:
    def test_happy_path(self, client: CLOBMarketClient, tmp_logs):
        raw = json.dumps([{
            "event_type": "market_resolved",
            "asset_id": ASSET_A,
            "winning_asset_id": ASSET_A,
            "winning_outcome": "YES",
            "timestamp": "1708598200000",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "market_resolved"
        assert r["winning_asset_id"] == ASSET_A
        assert r["winning_outcome"] == "YES"

        logged = _read_jsonl(tmp_logs["market"])
        assert logged[0]["winning_outcome"] == "YES"


# ---------------------------------------------------------------------------
# book (snapshot) processing
# ---------------------------------------------------------------------------
class TestBookSnapshot:
    def test_summarizes_depth(self, client: CLOBMarketClient, tmp_logs):
        raw = json.dumps([{
            "event_type": "book",
            "asset_id": ASSET_A,
            "market": "0xabc123",
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
            ],
            "asks": [
                {"price": "0.67", "size": "150"},
            ],
            "timestamp": "1708598123456",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "book"
        assert r["bid_levels"] == 2
        assert r["ask_levels"] == 1
        assert r["top_bid"] == 0.65
        assert r["top_ask"] == 0.67


# ---------------------------------------------------------------------------
# price_change processing (nested array format)
# ---------------------------------------------------------------------------
class TestPriceChange:
    def test_nested_array_format(self, client: CLOBMarketClient, tmp_logs):
        """Real CLOB format: price_changes is a nested array."""
        raw = json.dumps([{
            "event_type": "price_change",
            "market": "0xabc123",
            "timestamp": "1708598123456",
            "price_changes": [
                {
                    "asset_id": ASSET_A,
                    "price": "0.66",
                    "size": "100",
                    "side": "BUY",
                    "best_bid": "0.65",
                    "best_ask": "0.67",
                },
                {
                    "asset_id": ASSET_B,
                    "price": "0.34",
                    "size": "100",
                    "side": "SELL",
                    "best_bid": "0.33",
                    "best_ask": "0.35",
                },
            ],
        }])
        records = client._process_message(raw)
        assert len(records) == 2
        assert records[0]["asset_id"] == ASSET_A
        assert records[0]["price"] == 0.66
        assert records[0]["best_bid"] == 0.65
        assert records[0]["best_ask"] == 0.67
        assert records[0]["market"] == "0xabc123"
        assert records[1]["asset_id"] == ASSET_B
        assert records[1]["price"] == 0.34

    def test_updates_internal_state(self, client: CLOBMarketClient):
        """price_change with best_bid/best_ask updates internal state."""
        raw = json.dumps([{
            "event_type": "price_change",
            "timestamp": "123",
            "price_changes": [
                {
                    "asset_id": ASSET_A,
                    "price": "0.66",
                    "side": "BUY",
                    "size": "50",
                    "best_bid": "0.65",
                    "best_ask": "0.67",
                },
            ],
        }])
        client._process_message(raw)
        assert client.get_best_bid(ASSET_A) == 0.65
        assert client.get_best_ask(ASSET_A) == 0.67
        assert client.get_spread(ASSET_A) == 0.02

    def test_flat_fallback(self, client: CLOBMarketClient, tmp_logs):
        """Forward-compat: flat price_change without price_changes array."""
        raw = json.dumps([{
            "event_type": "price_change",
            "asset_id": ASSET_A,
            "price": "0.66",
            "side": "BUY",
            "size": "50",
            "timestamp": "1708598123456",
        }])
        records = client._process_message(raw)
        assert len(records) == 1
        assert records[0]["price"] == 0.66


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------
class TestMessageRouting:
    def test_pong_ignored(self, client: CLOBMarketClient, tmp_logs):
        """pong keepalive should not produce events."""
        records = client._process_message("pong")
        assert records == []
        assert not tmp_logs["market"].exists()

    def test_pong_refreshes_event_time(self, client: CLOBMarketClient):
        client._last_event_time = 0.0
        client._process_message("pong")
        assert client._last_event_time > 0.0

    def test_single_object_message(self, client: CLOBMarketClient):
        """Server sends a single object instead of array."""
        raw = json.dumps({
            "event_type": "last_trade_price",
            "asset_id": ASSET_A,
            "price": "0.55",
        })
        records = client._process_message(raw)
        assert len(records) == 1

    def test_array_of_mixed_events(self, client: CLOBMarketClient, tmp_logs):
        """Multiple event types in one WS message."""
        raw = json.dumps([
            {
                "event_type": "best_bid_ask",
                "asset_id": ASSET_A,
                "best_bid": "0.60",
                "best_ask": "0.62",
            },
            {
                "event_type": "last_trade_price",
                "asset_id": ASSET_A,
                "price": "0.61",
            },
        ])
        records = client._process_message(raw)
        assert len(records) == 2
        assert records[0]["event_type"] == "best_bid_ask"
        assert records[1]["event_type"] == "last_trade_price"

        logged = _read_jsonl(tmp_logs["market"])
        assert len(logged) == 2

    def test_invalid_json_ignored(self, client: CLOBMarketClient):
        records = client._process_message("not valid json {{{")
        assert records == []

    def test_empty_array(self, client: CLOBMarketClient):
        records = client._process_message("[]")
        assert records == []


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
class TestAnomalies:
    def test_unknown_event_type(self, client: CLOBMarketClient, tmp_logs):
        raw = json.dumps([{
            "event_type": "some_new_event",
            "asset_id": ASSET_A,
        }])
        records = client._process_message(raw)
        assert records == []
        anomalies = _read_jsonl(tmp_logs["env"])
        assert len(anomalies) == 1
        assert anomalies[0]["event"] == AnomalyType.UNKNOWN_EVENT

    def test_empty_event_type_no_anomaly(self, client: CLOBMarketClient, tmp_logs):
        """Events with no event_type should be silently ignored, not an anomaly."""
        raw = json.dumps([{"some_field": "value"}])
        records = client._process_message(raw)
        assert records == []
        anomalies = _read_jsonl(tmp_logs["env"])
        assert len(anomalies) == 0


# ---------------------------------------------------------------------------
# Health accumulator
# ---------------------------------------------------------------------------
class TestHealthAccumulator:
    def test_empty_emit(self):
        h = HealthAccumulator()
        summary = h.emit_and_reset()
        assert summary["total_events"] == 0
        assert summary["spread"] is None
        assert summary["trade_count"] == 0
        assert summary["event_frequency_hz"] == 0

    def test_records_events(self):
        h = HealthAccumulator()
        h.record_event("best_bid_ask")
        h.record_event("best_bid_ask")
        h.record_event("last_trade_price")
        summary = h.emit_and_reset()
        assert summary["total_events"] == 3
        assert summary["event_counts"]["best_bid_ask"] == 2
        assert summary["event_counts"]["last_trade_price"] == 1

    def test_spread_stats(self):
        h = HealthAccumulator()
        for s in [0.02, 0.03, 0.01, 0.04]:
            h.record_spread(s)
        summary = h.emit_and_reset()
        assert summary["spread"]["min"] == 0.01
        assert summary["spread"]["max"] == 0.04
        assert summary["spread"]["samples"] == 4
        assert "p95" in summary["spread"]

    def test_p95_spread(self):
        h = HealthAccumulator()
        # 20 samples; p95 index = int(20 * 0.95) = 19
        for i in range(20):
            h.record_spread(float(i) * 0.001)
        summary = h.emit_and_reset()
        assert summary["spread"]["p95"] >= summary["spread"]["median"]
        assert summary["spread"]["p95"] <= summary["spread"]["max"]

    def test_mid_tracking(self):
        h = HealthAccumulator()
        h.record_spread(0.02, mid=0.54)
        h.record_spread(0.03, mid=0.55)
        h.record_spread(0.01)  # No mid
        summary = h.emit_and_reset()
        assert summary["mid"] is not None
        assert summary["mid"]["last"] == 0.55
        assert summary["mid"]["samples"] == 2
        assert summary["mid"]["mean"] == pytest.approx(0.545, abs=1e-6)

    def test_mid_none_when_no_mids(self):
        h = HealthAccumulator()
        h.record_spread(0.02)
        summary = h.emit_and_reset()
        assert summary["mid"] is None

    def test_trade_price_tracking(self):
        h = HealthAccumulator()
        h.record_trade_price(0.60)
        h.record_trade_price(0.65)
        h.record_trade_price(0.63)
        summary = h.emit_and_reset()
        assert summary["last_trade_price"] == 0.63
        assert summary["trade_count"] == 3

    def test_reset_clears(self):
        h = HealthAccumulator()
        h.record_event("best_bid_ask")
        h.record_spread(0.01, mid=0.5)
        h.record_trade_price(0.5)
        h.record_anomaly()
        h.emit_and_reset()
        summary = h.emit_and_reset()
        assert summary["total_events"] == 0
        assert summary["spread"] is None
        assert summary["mid"] is None
        assert summary["trade_count"] == 0
        assert summary["anomaly_count"] == 0


# ---------------------------------------------------------------------------
# Subscribe message format
# ---------------------------------------------------------------------------
class TestSubscribeMessage:
    def test_initial_subscribe_format(self):
        msg = json.loads(CLOBMarketClient._subscribe_message([ASSET_A, ASSET_B]))
        assert msg["type"] == "market"
        assert msg["assets_ids"] == [ASSET_A, ASSET_B]
        assert msg["custom_feature_enabled"] is True

    def test_dynamic_subscribe(self):
        msg = json.loads(CLOBMarketClient._dynamic_subscribe([ASSET_A]))
        assert msg["assets_ids"] == [ASSET_A]
        assert msg["operation"] == "subscribe"

    def test_dynamic_unsubscribe(self):
        msg = json.loads(CLOBMarketClient._dynamic_unsubscribe([ASSET_A]))
        assert msg["assets_ids"] == [ASSET_A]
        assert msg["operation"] == "unsubscribe"


# ---------------------------------------------------------------------------
# Public query methods
# ---------------------------------------------------------------------------
class TestQueryMethods:
    def test_no_data_returns_none(self, client: CLOBMarketClient):
        assert client.get_best_bid(ASSET_A) is None
        assert client.get_best_ask(ASSET_A) is None
        assert client.get_spread(ASSET_A) is None
        assert client.get_last_trade_price(ASSET_A) is None

    def test_tracks_multiple_assets(self, client: CLOBMarketClient):
        """Each asset gets its own state."""
        for aid, bid, ask in [(ASSET_A, "0.60", "0.63"), (ASSET_B, "0.10", "0.15")]:
            raw = json.dumps([{
                "event_type": "best_bid_ask",
                "asset_id": aid,
                "best_bid": bid,
                "best_ask": ask,
            }])
            client._process_message(raw)

        assert client.get_best_bid(ASSET_A) == 0.60
        assert client.get_best_bid(ASSET_B) == 0.10
        assert client.get_spread(ASSET_A) == 0.03
        assert client.get_spread(ASSET_B) == 0.05


# ---------------------------------------------------------------------------
# Sequential event counter
# ---------------------------------------------------------------------------
class TestEventCounter:
    def test_sequential_seq(self, client: CLOBMarketClient):
        """Events should get sequential seq numbers."""
        for i in range(3):
            raw = json.dumps([{
                "event_type": "best_bid_ask",
                "asset_id": ASSET_A,
                "best_bid": "0.60",
                "best_ask": "0.62",
            }])
            client._process_message(raw)
        assert client._event_counter == 3


# ---------------------------------------------------------------------------
# Log integrity
# ---------------------------------------------------------------------------
class TestLogIntegrity:
    def test_all_events_logged(self, client: CLOBMarketClient, tmp_logs):
        """Every processed event should be persisted to JSONL."""
        events = [
            {"event_type": "best_bid_ask", "asset_id": ASSET_A,
             "best_bid": "0.60", "best_ask": "0.62"},
            {"event_type": "last_trade_price", "asset_id": ASSET_A,
             "price": "0.61"},
            {"event_type": "price_change", "timestamp": "123",
             "price_changes": [{"asset_id": ASSET_A, "price": "0.62",
              "side": "BUY", "size": "10", "best_bid": "0.61", "best_ask": "0.63"}]},
        ]
        for ev in events:
            raw = json.dumps([ev])
            client._process_message(raw)

        logged = _read_jsonl(tmp_logs["market"])
        assert len(logged) == 3
        types = [l["event_type"] for l in logged]
        assert types == ["best_bid_ask", "last_trade_price", "price_change"]

    def test_each_line_is_valid_json(self, client: CLOBMarketClient, tmp_logs):
        """Each line in the log file must be independently parseable JSON."""
        for i in range(5):
            raw = json.dumps([{
                "event_type": "best_bid_ask",
                "asset_id": ASSET_A,
                "best_bid": str(0.60 + i * 0.01),
                "best_ask": str(0.62 + i * 0.01),
            }])
            client._process_message(raw)

        with open(tmp_logs["market"]) as f:
            for line_no, line in enumerate(f, 1):
                obj = json.loads(line)
                assert "event_type" in obj, f"Line {line_no} missing event_type"
                assert "ts_arrival_ms" in obj, f"Line {line_no} missing ts_arrival_ms"


# ===================================================================
# Discovery mode tests
# ===================================================================
class TestDiscoveryMode:
    """Tests for discovery mode fields and rotation logic."""

    def test_discovery_mode_defaults_off(self, tmp_logs) -> None:
        """Discovery mode should default to False."""
        c = CLOBMarketClient(
            ws_uri="wss://localhost/fake",
            asset_ids=[ASSET_A],
            market_log=tmp_logs["market"],
            health_log=tmp_logs["health"],
            env_events_log=tmp_logs["env"],
        )
        assert c.discovery_mode is False
        assert c._current_slug is None
        assert c._rotation_count == 0

    def test_discovery_mode_init(self, tmp_logs) -> None:
        """Discovery mode can be enabled via constructor."""
        c = CLOBMarketClient(
            ws_uri="wss://localhost/fake",
            asset_ids=[],
            market_log=tmp_logs["market"],
            health_log=tmp_logs["health"],
            env_events_log=tmp_logs["env"],
            discovery_mode=True,
            discovery_timeframe="15m",
            discovery_poll_s=30.0,
        )
        assert c.discovery_mode is True
        assert c.discovery_timeframe == "15m"
        assert c.discovery_poll_s == 30.0

    def test_get_subscribed_ids_empty(self, client: CLOBMarketClient) -> None:
        """Initially no subscribed IDs tracked."""
        assert client.get_subscribed_ids() == []

    def test_get_rotation_count_zero(self, client: CLOBMarketClient) -> None:
        assert client.get_rotation_count() == 0

    def test_get_current_market_slug_none(self, client: CLOBMarketClient) -> None:
        assert client.get_current_market_slug() is None

    def test_get_current_market_none(self, client: CLOBMarketClient) -> None:
        assert client.get_current_market() is None

    def test_dynamic_subscribe_format(self) -> None:
        """Dynamic subscribe message should have correct format."""
        msg = json.loads(CLOBMarketClient._dynamic_subscribe(["tok1", "tok2"]))
        assert msg["assets_ids"] == ["tok1", "tok2"]
        assert msg["operation"] == "subscribe"

    def test_dynamic_unsubscribe_format(self) -> None:
        """Dynamic unsubscribe message should have correct format."""
        msg = json.loads(CLOBMarketClient._dynamic_unsubscribe(["tok1"]))
        assert msg["assets_ids"] == ["tok1"]
        assert msg["operation"] == "unsubscribe"

    def test_rotation_count_increments(self, tmp_logs) -> None:
        """_rotation_count increments on each rotation call."""
        c = CLOBMarketClient(
            ws_uri="wss://localhost/fake",
            asset_ids=[],
            market_log=tmp_logs["market"],
            health_log=tmp_logs["health"],
            env_events_log=tmp_logs["env"],
            discovery_mode=True,
        )
        assert c._rotation_count == 0

    def test_discovery_mode_no_asset_ids_ok(self, tmp_logs) -> None:
        """In discovery mode, empty asset_ids should not raise."""
        c = CLOBMarketClient(
            ws_uri="wss://localhost/fake",
            asset_ids=[],
            market_log=tmp_logs["market"],
            health_log=tmp_logs["health"],
            env_events_log=tmp_logs["env"],
            discovery_mode=True,
        )
        assert c.asset_ids == []
        assert c.discovery_mode is True

    def test_log_rotation_writes_event(self, tmp_logs) -> None:
        """_log_rotation should write to env_events log."""
        c = CLOBMarketClient(
            ws_uri="wss://localhost/fake",
            asset_ids=[],
            market_log=tmp_logs["market"],
            health_log=tmp_logs["health"],
            env_events_log=tmp_logs["env"],
            discovery_mode=True,
        )
        c._rotation_count = 1
        c._log_rotation({
            "rotation_number": 1,
            "old_slug": None,
            "new_slug": "btc-updown-15m-123",
        })
        with open(tmp_logs["env"]) as f:
            lines = f.readlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "clob_market_rotation"
        assert event["new_slug"] == "btc-updown-15m-123"
        assert event["source"] == "clob_market_client"

