"""
Thin TTL cache for exchange position snapshots.

Reduces redundant ``get_positions()`` calls within a single executor loop
iteration while guaranteeing freshness via short TTL and explicit
invalidation after confirmed fills.

Thread-safety: **not required** — executor is single-threaded.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

__all__ = ["PositionCache", "POSITION_CACHE"]


class PositionCache:
    """TTL-governed cache for ``get_positions()`` results.

    Parameters
    ----------
    ttl_s : float
        Maximum age (seconds) before the cached value is considered stale
        and a fresh fetch is triggered.  Default 1.0 s.
    """

    def __init__(self, ttl_s: float = 1.0) -> None:
        self._ttl_s = ttl_s
        self._data: Optional[List[Dict[str, Any]]] = None
        self._ts: float = 0.0  # monotonic timestamp of last fetch

    # ── public API ────────────────────────────────────────────────────

    def get(self, fetch_fn: Callable[[], Any]) -> List[Dict[str, Any]]:
        """Return cached positions or fetch fresh ones if stale/empty.

        Parameters
        ----------
        fetch_fn : callable
            Typically ``exchange_utils.get_positions``.  Must return an
            iterable of position dicts (or ``None``).
        """
        assert callable(fetch_fn), f"fetch_fn must be callable, got {type(fetch_fn)}"
        now = time.monotonic()
        if self._data is not None and (now - self._ts) < self._ttl_s:
            return self._data
        self._data = list(fetch_fn() or [])
        self._ts = now
        return self._data

    def invalidate(self) -> None:
        """Force next ``get()`` to perform a fresh fetch."""
        self._data = None
        self._ts = 0.0

    def peek(self) -> Optional[List[Dict[str, Any]]]:
        """Return the cached value without fetching.  May be ``None``."""
        return self._data

    def age_s(self) -> float:
        """Seconds since last successful fetch (monotonic)."""
        if self._ts == 0.0:
            return float("inf")
        return time.monotonic() - self._ts


# Module-level singleton used by the executor.
POSITION_CACHE = PositionCache()
