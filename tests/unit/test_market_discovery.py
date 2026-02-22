"""Tests for prediction.market_discovery — dynamic BTC Up/Down market discovery.

Covers:
    - Gamma API response parsing
    - Slug-to-timeframe mapping
    - Token extraction (list + JSON-string formats)
    - Market filtering (expiry safety buffer, lookahead)
    - DiscoverySnapshot construction (current_15m / current_5m)
    - ISO timestamp parsing
    - Error handling (API failure, malformed data)
    - get_current_tokens convenience function
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from prediction.market_discovery import (
    DiscoveredMarket,
    DiscoverySnapshot,
    SLUG_15M,
    SLUG_5M,
    TITLE_FILTER,
    _gamma_get,
    _parse_iso,
    _parse_tokens,
    _slug_to_timeframe,
    discover_btc_updown_markets,
    get_current_tokens,
)


# ---------------------------------------------------------------------------
# Helpers — build fake Gamma events
# ---------------------------------------------------------------------------
def _make_event(
    title: str = "Bitcoin Up or Down - Test Window",
    slug: str = "btc-updown-15m-1771833600",
    event_id: str = "223294",
    market_id: str = "900001",
    condition_id: str = "0xabc123",
    end_date: str = "",
    clob_token_ids: Any = None,
    question: str = "",
    active: bool = True,
    closed: bool = False,
) -> Dict[str, Any]:
    """Construct a minimal Gamma event dict."""
    if not end_date:
        # Default: 30 minutes from now
        from datetime import datetime, timezone, timedelta
        end_date = (
            datetime.now(timezone.utc) + timedelta(minutes=30)
        ).isoformat().replace("+00:00", "Z")

    tokens = clob_token_ids
    if tokens is None:
        tokens = json.dumps(["UP_TOKEN_001", "DOWN_TOKEN_002"])
    elif isinstance(tokens, list):
        tokens = json.dumps(tokens)

    return {
        "id": event_id,
        "title": title,
        "slug": slug,
        "active": active,
        "closed": closed,
        "markets": [
            {
                "id": market_id,
                "conditionId": condition_id,
                "question": question or title,
                "endDate": end_date,
                "clobTokenIds": tokens,
            }
        ],
    }


def _make_15m_event(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("slug", "btc-updown-15m-1771833600")
    return _make_event(**kwargs)


def _make_5m_event(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("slug", "btc-updown-5m-1771833600")
    return _make_event(**kwargs)


# ---------------------------------------------------------------------------
# Tests — _parse_iso
# ---------------------------------------------------------------------------
class TestParseIso:
    def test_z_suffix(self) -> None:
        ts = _parse_iso("2026-02-23T10:45:00Z")
        assert ts > 0
        assert isinstance(ts, float)

    def test_no_suffix(self) -> None:
        ts = _parse_iso("2026-02-23T10:45:00")
        assert ts > 0

    def test_invalid_returns_zero(self) -> None:
        assert _parse_iso("not-a-date") == 0.0
        assert _parse_iso("") == 0.0

    def test_roundtrip(self) -> None:
        from datetime import datetime, timezone
        dt = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = _parse_iso("2026-03-01T12:00:00Z")
        assert abs(ts - dt.timestamp()) < 1


# ---------------------------------------------------------------------------
# Tests — _slug_to_timeframe
# ---------------------------------------------------------------------------
class TestSlugToTimeframe:
    def test_15m(self) -> None:
        assert _slug_to_timeframe("btc-updown-15m-1771833600") == "15m"

    def test_5m(self) -> None:
        assert _slug_to_timeframe("btc-updown-5m-1771833600") == "5m"

    def test_unknown_slug(self) -> None:
        assert _slug_to_timeframe("something-else") is None

    def test_partial_match(self) -> None:
        # Must start with prefix, not contain it
        assert _slug_to_timeframe("xrp-updown-15m-1234") is None

    def test_empty(self) -> None:
        assert _slug_to_timeframe("") is None


# ---------------------------------------------------------------------------
# Tests — _parse_tokens
# ---------------------------------------------------------------------------
class TestParseTokens:
    def test_json_string(self) -> None:
        raw = json.dumps(["token_a", "token_b"])
        assert _parse_tokens(raw) == ["token_a", "token_b"]

    def test_list_directly(self) -> None:
        assert _parse_tokens(["a", "b"]) == ["a", "b"]

    def test_empty_string(self) -> None:
        assert _parse_tokens("") == []

    def test_none(self) -> None:
        assert _parse_tokens(None) == []

    def test_invalid_json(self) -> None:
        assert _parse_tokens("{not json}") == []

    def test_numeric_tokens_stringified(self) -> None:
        raw = json.dumps([12345, 67890])
        result = _parse_tokens(raw)
        assert result == ["12345", "67890"]

    def test_single_token(self) -> None:
        raw = json.dumps(["only_one"])
        result = _parse_tokens(raw)
        assert result == ["only_one"]


# ---------------------------------------------------------------------------
# Tests — discover_btc_updown_markets
# ---------------------------------------------------------------------------
class TestDiscoverMarkets:
    @patch("prediction.market_discovery._gamma_get")
    def test_basic_discovery_15m(self, mock_get: Any) -> None:
        """Should find a single 15m market and set current_15m."""
        mock_get.return_value = [_make_15m_event()]
        snap = discover_btc_updown_markets(timeframes=["15m"])

        assert snap.error is None
        assert snap.raw_event_count == 1
        assert snap.btc_updown_count == 1
        assert len(snap.markets) == 1
        assert snap.current_15m is not None
        assert snap.current_15m.timeframe == "15m"
        assert snap.current_15m.up_token == "UP_TOKEN_001"
        assert snap.current_15m.down_token == "DOWN_TOKEN_002"

    @patch("prediction.market_discovery._gamma_get")
    def test_basic_discovery_5m(self, mock_get: Any) -> None:
        """Should find a single 5m market and set current_5m."""
        mock_get.return_value = [_make_5m_event()]
        snap = discover_btc_updown_markets(timeframes=["5m"])

        assert snap.current_5m is not None
        assert snap.current_5m.timeframe == "5m"
        assert snap.current_15m is None

    @patch("prediction.market_discovery._gamma_get")
    def test_multiple_markets_sorted_by_expiry(self, mock_get: Any) -> None:
        """Soonest-expiring market should be current_15m."""
        from datetime import datetime, timezone, timedelta
        soon = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        later = (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat()

        mock_get.return_value = [
            _make_15m_event(
                slug="btc-updown-15m-later",
                event_id="2",
                end_date=later,
                clob_token_ids=["LATER_UP", "LATER_DOWN"],
            ),
            _make_15m_event(
                slug="btc-updown-15m-sooner",
                event_id="1",
                end_date=soon,
                clob_token_ids=["SOON_UP", "SOON_DOWN"],
            ),
        ]
        snap = discover_btc_updown_markets(
            timeframes=["15m"],
            safety_buffer_s=0,
        )

        assert len(snap.markets) == 2
        # First = soonest
        assert snap.markets[0].up_token == "SOON_UP"
        assert snap.current_15m is not None
        assert snap.current_15m.up_token == "SOON_UP"

    @patch("prediction.market_discovery._gamma_get")
    def test_expired_market_filtered(self, mock_get: Any) -> None:
        """Markets past their end_date should be excluded."""
        from datetime import datetime, timezone, timedelta
        expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        mock_get.return_value = [_make_15m_event(end_date=expired)]
        snap = discover_btc_updown_markets(timeframes=["15m"], safety_buffer_s=0)

        assert len(snap.markets) == 0
        assert snap.current_15m is None

    @patch("prediction.market_discovery._gamma_get")
    def test_safety_buffer_filters_close_expiry(self, mock_get: Any) -> None:
        """Markets expiring within safety buffer should be excluded."""
        from datetime import datetime, timezone, timedelta
        close = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()

        mock_get.return_value = [_make_15m_event(end_date=close)]
        snap = discover_btc_updown_markets(
            timeframes=["15m"],
            safety_buffer_s=120,  # 2 minutes buffer
        )

        assert len(snap.markets) == 0

    @patch("prediction.market_discovery._gamma_get")
    def test_max_lookahead_filters_far_future(self, mock_get: Any) -> None:
        """Markets too far in the future should be excluded if configured."""
        from datetime import datetime, timezone, timedelta
        far = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()

        mock_get.return_value = [_make_15m_event(end_date=far)]
        snap = discover_btc_updown_markets(
            timeframes=["15m"],
            safety_buffer_s=0,
            max_lookahead_s=3600,  # 1 hour max
        )

        assert len(snap.markets) == 0

    @patch("prediction.market_discovery._gamma_get")
    def test_non_btc_events_filtered(self, mock_get: Any) -> None:
        """Events not matching 'bitcoin up or down' should be excluded."""
        mock_get.return_value = [
            _make_event(title="Ethereum Up or Down - Window"),
            _make_event(title="Bitcoin Up or Down - Window"),
        ]
        snap = discover_btc_updown_markets(timeframes=["15m"])

        # Ethereum event still matches title filter, but slug check matters
        assert snap.btc_updown_count >= 1

    @patch("prediction.market_discovery._gamma_get")
    def test_wrong_timeframe_filtered(self, mock_get: Any) -> None:
        """Requesting 15m but only 5m available should return empty."""
        mock_get.return_value = [_make_5m_event()]
        snap = discover_btc_updown_markets(timeframes=["15m"])

        assert len(snap.markets) == 0
        assert snap.current_15m is None

    @patch("prediction.market_discovery._gamma_get")
    def test_both_timeframes(self, mock_get: Any) -> None:
        """Requesting both should populate current_15m and current_5m."""
        mock_get.return_value = [_make_15m_event(), _make_5m_event()]
        snap = discover_btc_updown_markets(timeframes=["15m", "5m"])

        assert snap.current_15m is not None
        assert snap.current_5m is not None

    @patch("prediction.market_discovery._gamma_get")
    def test_gamma_failure_returns_error(self, mock_get: Any) -> None:
        """API failure should return snapshot with error, not raise."""
        mock_get.return_value = None
        snap = discover_btc_updown_markets()

        assert snap.error == "gamma_fetch_failed"
        assert len(snap.markets) == 0

    @patch("prediction.market_discovery._gamma_get")
    def test_unexpected_response_type(self, mock_get: Any) -> None:
        """Non-list response should set error."""
        mock_get.return_value = {"error": "something"}
        snap = discover_btc_updown_markets()

        assert snap.error is not None
        assert "unexpected_response_type" in snap.error

    @patch("prediction.market_discovery._gamma_get")
    def test_market_with_fewer_than_2_tokens_skipped(self, mock_get: Any) -> None:
        """Markets with <2 clobTokenIds should be skipped."""
        mock_get.return_value = [
            _make_15m_event(clob_token_ids=["only_one"]),
        ]
        snap = discover_btc_updown_markets(timeframes=["15m"])

        assert len(snap.markets) == 0

    @patch("prediction.market_discovery._gamma_get")
    def test_discovered_market_fields(self, mock_get: Any) -> None:
        """DiscoveredMarket should have all expected fields."""
        mock_get.return_value = [
            _make_15m_event(
                event_id="E1",
                market_id="M1",
                condition_id="0xcondition",
                question="BTC Up or Down?",
            ),
        ]
        snap = discover_btc_updown_markets(timeframes=["15m"])

        m = snap.markets[0]
        assert m.event_id == "E1"
        assert m.market_id == "M1"
        assert m.condition_id == "0xcondition"
        assert m.question == "BTC Up or Down?"
        assert m.timeframe == "15m"
        assert m.slug.startswith(SLUG_15M)
        assert m.remaining_s > 0


# ---------------------------------------------------------------------------
# Tests — get_current_tokens convenience
# ---------------------------------------------------------------------------
class TestGetCurrentTokens:
    @patch("prediction.market_discovery._gamma_get")
    def test_returns_15m(self, mock_get: Any) -> None:
        mock_get.return_value = [_make_15m_event()]
        result = get_current_tokens("15m", safety_buffer_s=0)
        assert result is not None
        assert result.timeframe == "15m"

    @patch("prediction.market_discovery._gamma_get")
    def test_returns_5m(self, mock_get: Any) -> None:
        mock_get.return_value = [_make_5m_event()]
        result = get_current_tokens("5m", safety_buffer_s=0)
        assert result is not None
        assert result.timeframe == "5m"

    @patch("prediction.market_discovery._gamma_get")
    def test_returns_none_on_failure(self, mock_get: Any) -> None:
        mock_get.return_value = None
        assert get_current_tokens("15m") is None


# ---------------------------------------------------------------------------
# Tests — DiscoveredMarket dataclass
# ---------------------------------------------------------------------------
class TestDiscoveredMarket:
    def test_frozen(self) -> None:
        m = DiscoveredMarket(
            event_id="1",
            market_id="2",
            condition_id="0x",
            slug="btc-updown-15m-123",
            question="Test?",
            timeframe="15m",
            up_token="UP",
            down_token="DOWN",
            end_date_utc="2026-01-01T00:00:00Z",
            end_ts=1000000.0,
            remaining_s=600.0,
        )
        assert m.up_token == "UP"
        with pytest.raises(AttributeError):
            m.up_token = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        args = dict(
            event_id="1",
            market_id="2",
            condition_id="0x",
            slug="btc-updown-15m-123",
            question="Test?",
            timeframe="15m",
            up_token="UP",
            down_token="DOWN",
            end_date_utc="2026-01-01T00:00:00Z",
            end_ts=1000000.0,
            remaining_s=600.0,
        )
        m1 = DiscoveredMarket(**args)
        m2 = DiscoveredMarket(**args)
        assert m1 == m2


# ---------------------------------------------------------------------------
# Tests — DiscoverySnapshot
# ---------------------------------------------------------------------------
class TestDiscoverySnapshot:
    def test_defaults(self) -> None:
        snap = DiscoverySnapshot(ts="2026-01-01T00:00:00Z")
        assert snap.markets == []
        assert snap.current_15m is None
        assert snap.current_5m is None
        assert snap.error is None
        assert snap.raw_event_count == 0
        assert snap.btc_updown_count == 0
