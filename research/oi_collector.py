"""
Daily OI snapshot collector — passive data collection for future research.

Run once per day via cron. Appends one JSON line per symbol to
data/oi_history.jsonl. After 90+ days this file provides enough history
to run research/oi_momentum.py properly (Binance retains only 30 days
of openInterestHist; this collector builds the full archive locally).

Cron example (run at 00:05 UTC daily):
    5 0 * * * cd /root/hedge-fund && python3 -m research.oi_collector

Each line written:
    {"ts_ms": 1234567890000, "symbol": "BTCUSDT", "oi": 123456.78, "date": "2026-06-04"}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

_BASE_URL = "https://fapi.binance.com"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
_OUT_PATH = Path("data/oi_history.jsonl")


def _get(path: str, params: dict) -> list | dict:
    try:
        import requests as _requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc
    r = _requests.get(_BASE_URL + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_current_oi(symbol: str) -> float:
    """Return current open interest (sumOpenInterest) for symbol."""
    data = _get("/fapi/v1/openInterest", {"symbol": symbol})
    return float(data["openInterest"])


def collect(symbols: list[str] = _DEFAULT_SYMBOLS) -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with _OUT_PATH.open("a") as f:
        for sym in symbols:
            try:
                oi = fetch_current_oi(sym)
                record = {"ts_ms": ts_ms, "symbol": sym, "oi": oi, "date": date_str}
                f.write(json.dumps(record) + "\n")
                print(f"  {sym}: OI={oi:,.0f}")
            except Exception as exc:
                print(f"  {sym}: ERROR — {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily OI snapshot collector")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    args = parser.parse_args()
    print(f"Collecting OI snapshots — {datetime.now(timezone.utc).date()}")
    collect(args.symbols)
    print(f"Appended to {_OUT_PATH}")
