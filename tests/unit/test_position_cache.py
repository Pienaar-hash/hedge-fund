"""Tests for execution.position_cache — TTL-based position cache."""

import time
from unittest.mock import MagicMock

import pytest

from execution.position_cache import PositionCache


# ── TTL behaviour ─────────────────────────────────────────────────────

class TestPositionCacheTTL:
    def test_first_call_fetches(self):
        fetch = MagicMock(return_value=[{"symbol": "BTCUSDT"}])
        cache = PositionCache(ttl_s=1.0)
        result = cache.get(fetch)
        assert result == [{"symbol": "BTCUSDT"}]
        assert fetch.call_count == 1

    def test_second_call_within_ttl_is_cached(self):
        fetch = MagicMock(return_value=[{"symbol": "BTCUSDT"}])
        cache = PositionCache(ttl_s=10.0)
        cache.get(fetch)
        cache.get(fetch)
        assert fetch.call_count == 1

    def test_expired_ttl_triggers_refetch(self, monkeypatch):
        counter = {"n": 0}

        def _fetch():
            counter["n"] += 1
            return [{"symbol": "BTCUSDT", "call": counter["n"]}]

        cache = PositionCache(ttl_s=0.0)  # always stale
        r1 = cache.get(_fetch)
        r2 = cache.get(_fetch)
        assert counter["n"] == 2
        assert r1[0]["call"] == 1
        assert r2[0]["call"] == 2

    def test_none_return_normalised_to_empty_list(self):
        fetch = MagicMock(return_value=None)
        cache = PositionCache(ttl_s=1.0)
        result = cache.get(fetch)
        assert result == []


# ── invalidate ────────────────────────────────────────────────────────

class TestPositionCacheInvalidate:
    def test_invalidate_forces_refetch(self):
        fetch = MagicMock(return_value=[{"symbol": "ETHUSDT"}])
        cache = PositionCache(ttl_s=60.0)  # very long TTL
        cache.get(fetch)
        assert fetch.call_count == 1

        cache.invalidate()
        cache.get(fetch)
        assert fetch.call_count == 2

    def test_invalidate_clears_peek(self):
        fetch = MagicMock(return_value=[{"symbol": "ETHUSDT"}])
        cache = PositionCache(ttl_s=60.0)
        cache.get(fetch)
        assert cache.peek() is not None

        cache.invalidate()
        assert cache.peek() is None


# ── peek / age ────────────────────────────────────────────────────────

class TestPositionCachePeekAge:
    def test_peek_returns_none_before_first_fetch(self):
        cache = PositionCache()
        assert cache.peek() is None

    def test_peek_returns_cached_data(self):
        fetch = MagicMock(return_value=[{"symbol": "XRPUSDT"}])
        cache = PositionCache(ttl_s=60.0)
        cache.get(fetch)
        assert cache.peek() == [{"symbol": "XRPUSDT"}]

    def test_age_inf_before_first_fetch(self):
        cache = PositionCache()
        assert cache.age_s() == float("inf")

    def test_age_increases_after_fetch(self):
        fetch = MagicMock(return_value=[])
        cache = PositionCache(ttl_s=60.0)
        cache.get(fetch)
        # age should be very small right after fetch
        assert cache.age_s() < 1.0


# ── defensive guards ─────────────────────────────────────────────────

class TestPositionCacheGuards:
    def test_non_callable_raises_assertion(self):
        cache = PositionCache()
        with pytest.raises(AssertionError, match="callable"):
            cache.get("not_a_function")  # type: ignore[arg-type]

    def test_non_callable_none_raises_assertion(self):
        cache = PositionCache()
        with pytest.raises(AssertionError, match="callable"):
            cache.get(None)  # type: ignore[arg-type]
