#!/usr/bin/env python3
"""
polymarket_insiders.py
Realtime insider-trade detector for Polymarket using the RTDS WebSocket feed.
No authentication required.
"""

from __future__ import annotations

import asyncio
import json
import csv
import os
import random
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List
import websockets

# ───────────────────────────────────────────────────────────────
URI = "wss://ws-live-data.polymarket.com/ws/"
LOG_DIR = "logs"

THRESHOLDS = {
    "unique_markets": 3,          # max markets per wallet
    "trade_size_usd": 10_000,     # min USD size per trade
    "concentration_ratio": 0.5,   # ratio of this trade / total wallet volume
}

HEADERS = {
    "Origin": "https://polymarket.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
}
# ───────────────────────────────────────────────────────────────


_SNAPSHOT_CACHE: List[Dict[str, str | float]] = []

async def run_once():
    """Connect once to RTDS and stream live trades."""
    async with websockets.connect(
        URI,
        ping_interval=25,
        ping_timeout=20,
        additional_headers=list(HEADERS.items()),  # correct param for websockets 15.x
        max_queue=None,
    ) as ws:
        sub = {"type": "subscribe", "channel": "trades", "symbols": []}
        await ws.send(json.dumps(sub))
        print("✅ Connected – waiting for trades...")

        trades = []
        stats = defaultdict(lambda: {"markets": set(), "volume": 0})
        os.makedirs(LOG_DIR, exist_ok=True)

        async for msg in ws:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "trade":
                continue

            trade = data.get("data", {})
            maker = trade.get("maker")
            market = (trade.get("market") or {}).get("id")
            size_usd = float(trade.get("sizeUsd") or 0)
            ts = datetime.now(timezone.utc)

            if not maker or not market:
                continue

            stats[maker]["markets"].add(market)
            stats[maker]["volume"] += size_usd

            # apply heuristic
            if (
                size_usd >= THRESHOLDS["trade_size_usd"]
                and len(stats[maker]["markets"]) <= THRESHOLDS["unique_markets"]
            ):
                conc = (
                    size_usd / stats[maker]["volume"]
                    if stats[maker]["volume"]
                    else 0
                )
                if conc >= THRESHOLDS["concentration_ratio"]:
                    print(
                        f"🚨 {maker[:10]} | ${size_usd:,.0f} | "
                        f"{len(stats[maker]['markets'])} mkts | "
                        f"{conc*100:.0f}% conc | "
                        f"{ts.strftime('%H:%M:%S')} | {market}"
                    )
                    trades.append(
                        {
                            "maker": maker,
                            "market": market,
                            "sizeUsd": size_usd,
                            "timestamp": ts.isoformat(),
                            "unique_markets": len(stats[maker]["markets"]),
                            "concentration_ratio": round(conc, 3),
                        }
                    )

                    # periodic CSV save
                    if len(trades) % 10 == 0:
                        path = os.path.join(LOG_DIR, "insiders_live.csv")
                        with open(path, "w", newline="") as fh:
                            writer = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                            writer.writeheader()
                            writer.writerows(trades)
                        print(f"📁 {len(trades)} trades logged → {path}")


def get_polymarket_snapshot() -> List[Dict[str, str | float]]:
    """
    Return cached Polymarket insider snapshots for dashboard usage.
    """
    return list(_SNAPSHOT_CACHE)


async def main():
    """Reconnect automatically if socket closes."""
    while True:
        try:
            await run_once()
        except websockets.ConnectionClosedError as e:
            print(f"⚠️  Connection closed ({e}); retrying in 5 s…")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"⚠️  Error: {e}; reconnecting in 10 s…")
            await asyncio.sleep(10 + random.random() * 5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
