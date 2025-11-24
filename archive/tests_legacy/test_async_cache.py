from __future__ import annotations

import asyncio
from typing import Any, Dict

from dashboard.async_cache import AsyncCacheManager, gather_once


def test_gather_once_runs_fetchers_concurrently() -> None:
    async def runner() -> Dict[str, Any]:
        async def fetch_fast() -> Dict[str, Any]:
            return {"status": "fast"}

        async def fetch_slow() -> Dict[str, Any]:
            await asyncio.sleep(0.05)
            return {"status": "slow"}

        return await gather_once({"fast": fetch_fast, "slow": fetch_slow})

    entries = asyncio.run(runner())
    assert set(entries.keys()) == {"fast", "slow"}
    assert entries["fast"].payload["status"] == "fast"
    assert entries["slow"].payload["status"] == "slow"
    assert entries["slow"].latency_ms >= 40.0


def test_async_cache_manager_periodic_refresh_stop_event() -> None:
    async def runner() -> Dict[str, Any]:
        manager = AsyncCacheManager()
        stop_event = asyncio.Event()
        calls = 0

        async def fetcher() -> Dict[str, Any]:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return {"runs": calls}

        task = asyncio.create_task(manager.periodic_refresh("demo", fetcher, interval=0.05, stop_event=stop_event))
        await asyncio.sleep(0.12)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        entry = manager.get_entry("demo")
        return {"entry": entry, "calls": calls}

    result = asyncio.run(runner())
    entry = result["entry"]
    assert entry is not None
    assert entry.payload["runs"] >= 1
    assert result["calls"] >= 1
