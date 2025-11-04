from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

FetchCallable = Callable[[], Awaitable[Any] | Any]
SinkCallable = Callable[[str, Any], Awaitable[None] | None]

__all__ = [
    "CacheEntry",
    "AsyncCacheManager",
    "gather_once",
]


@dataclass(slots=True)
class CacheEntry:
    """Metadata captured for every cache refresh run."""

    name: str
    payload: Any
    refreshed_at: float
    latency_ms: float
    ok: bool
    error: Optional[str] = None


class AsyncCacheManager:
    """Coordinate asynchronous cache refresh jobs for the dashboard."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: Dict[str, CacheEntry] = {}

    def get_entry(self, name: str) -> Optional[CacheEntry]:
        return self._entries.get(name)

    async def refresh(self, name: str, fetcher: FetchCallable, sink: Optional[SinkCallable] = None) -> CacheEntry:
        started = time.perf_counter()
        error: Optional[str] = None
        try:
            payload = await _maybe_await(fetcher())
            ok = True
        except Exception as exc:  # pragma: no cover - defensive, hard to trigger deterministically
            payload = None
            ok = False
            error = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000.0
        entry = CacheEntry(
            name=name,
            payload=payload,
            refreshed_at=time.time(),
            latency_ms=latency_ms,
            ok=ok,
            error=error,
        )
        if sink is not None:
            try:
                await _maybe_await(sink(name, payload))
            except Exception:  # pragma: no cover - downstream logging only
                pass
        async with self._lock:
            self._entries[name] = entry
        return entry

    async def periodic_refresh(
        self,
        name: str,
        fetcher: FetchCallable,
        interval: float,
        *,
        sink: Optional[SinkCallable] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        while True:
            await self.refresh(name, fetcher, sink=sink)
            if stop_event is not None and stop_event.is_set():
                return
            try:
                if stop_event is None:
                    await asyncio.sleep(interval)
                else:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                    return
            except asyncio.TimeoutError:
                continue


async def gather_once(fetchers: Mapping[str, FetchCallable]) -> Dict[str, CacheEntry]:
    manager = AsyncCacheManager()
    tasks = [manager.refresh(name, fetcher) for name, fetcher in fetchers.items()]
    results = await asyncio.gather(*tasks)
    return {entry.name: entry for entry in results}


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def _demo() -> None:  # pragma: no cover
    async def sample_fetcher(delay: float, value: Any) -> Any:
        await asyncio.sleep(delay)
        return value

    async def main() -> None:
        entries = await gather_once(
            {
                "doctor": lambda: sample_fetcher(0.1, {"status": "ok"}),
                "router_health": lambda: sample_fetcher(0.2, {"latency": 45}),
            }
        )
        for name, entry in entries.items():
            print(f"{name}: ok={entry.ok} latency={entry.latency_ms:.2f}ms payload={entry.payload}")

    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    _demo()

