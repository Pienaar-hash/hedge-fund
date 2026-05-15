from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.shadow_soak_v8 import (
    _extract_live_order_timestamp,
    _parse_iso,
    _read_live_orders,
    _read_shadow_signals,
    _side_to_order_side,
)

_SORT_TS_FALLBACK = datetime.max.replace(tzinfo=timezone.utc)


@dataclass
class _SignalPoint:
    symbol: str
    side: str
    ts: datetime | None
    ts_raw: str | None
    raw: dict[str, Any]


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_live_orders(logs_dir: Path) -> dict[str, list[_SignalPoint]]:
    streams: dict[str, list[_SignalPoint]] = {}
    for order in _read_live_orders(logs_dir, order_type="executed"):
        symbol = str(order.get("symbol") or "").strip()
        side = _side_to_order_side(str(order.get("side") or "").strip().upper())
        if not symbol or side is None:
            continue
        ts_raw = _extract_live_order_timestamp(order)
        ts = _parse_iso(ts_raw)
        streams.setdefault(symbol, []).append(
            _SignalPoint(
                symbol=symbol,
                side=side,
                ts=ts,
                ts_raw=str(ts_raw) if ts_raw is not None else None,
                raw=order,
            )
        )
    for symbol in streams:
        streams[symbol].sort(key=lambda p: (p.ts is None, p.ts or _SORT_TS_FALLBACK))
    return streams


def _normalize_shadow_signals(replay_dir: Path) -> dict[str, list[_SignalPoint]]:
    streams: dict[str, list[_SignalPoint]] = {}
    signals, _ = _read_shadow_signals(replay_dir)
    for signal in signals:
        symbol = str(signal.get("symbol") or "").strip()
        side = _side_to_order_side(str(signal.get("side") or "").strip().upper())
        if not symbol or side is None:
            continue
        ts_raw = signal.get("entry_ts") or signal.get("ts")
        ts = _parse_iso(ts_raw)
        streams.setdefault(symbol, []).append(
            _SignalPoint(
                symbol=symbol,
                side=side,
                ts=ts,
                ts_raw=str(ts_raw) if ts_raw is not None else None,
                raw=signal,
            )
        )
    for symbol in streams:
        streams[symbol].sort(key=lambda p: (p.ts is None, p.ts or _SORT_TS_FALLBACK))
    return streams


def _candidate_ts(candidate: dict[str, Any] | None) -> datetime:
    if candidate is None:
        return _SORT_TS_FALLBACK
    live = candidate.get("live") or {}
    shadow = candidate.get("shadow") or {}
    return _parse_iso(live.get("ts")) or _parse_iso(shadow.get("ts")) or _SORT_TS_FALLBACK


def find_first_divergence(logs_dir: str | Path, replay_dir: str | Path) -> dict[str, Any]:
    live_streams = _normalize_live_orders(Path(logs_dir))
    shadow_streams = _normalize_shadow_signals(Path(replay_dir))

    first: dict[str, Any] | None = None
    symbols = sorted(set(live_streams) | set(shadow_streams))
    for symbol in symbols:
        live = live_streams.get(symbol, [])
        shadow = shadow_streams.get(symbol, [])
        m = min(len(live), len(shadow))

        for idx in range(m):
            lp = live[idx]
            sp = shadow[idx]
            if lp.side != sp.side:
                candidate = {
                    "type": "side_mismatch",
                    "symbol": symbol,
                    "stream_index": idx,
                    "live": {"side": lp.side, "ts": _to_iso(lp.ts), "ts_raw": lp.ts_raw},
                    "shadow": {"side": sp.side, "ts": _to_iso(sp.ts), "ts_raw": sp.ts_raw},
                    "abs_ts_delta_s": abs((lp.ts - sp.ts).total_seconds()) if lp.ts and sp.ts else None,
                }
                if first is None or _candidate_ts(candidate) < _candidate_ts(first):
                    first = candidate
                break

        candidate: dict[str, Any] | None = None
        if len(live) > len(shadow):
            lp = live[len(shadow)]
            candidate = {
                "type": "missing_shadow_event",
                "symbol": symbol,
                "stream_index": len(shadow),
                "live": {"side": lp.side, "ts": _to_iso(lp.ts), "ts_raw": lp.ts_raw},
                "shadow": None,
                "abs_ts_delta_s": None,
            }
        elif len(shadow) > len(live):
            sp = shadow[len(live)]
            candidate = {
                "type": "missing_live_event",
                "symbol": symbol,
                "stream_index": len(live),
                "live": None,
                "shadow": {"side": sp.side, "ts": _to_iso(sp.ts), "ts_raw": sp.ts_raw},
                "abs_ts_delta_s": None,
            }
        if candidate is not None and (first is None or _candidate_ts(candidate) < _candidate_ts(first)):
            first = candidate

    return {
        "divergence_found": first is not None,
        "first_divergence": first,
        "symbols_compared": symbols,
        "live_event_count": sum(len(v) for v in live_streams.values()),
        "shadow_event_count": sum(len(v) for v in shadow_streams.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find first replay-vs-live divergence point for Phase 5 root-cause research."
    )
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--replay-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = find_first_divergence(logs_dir=args.logs_dir, replay_dir=args.replay_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
