#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "execution"


def _parse_iso(ts: str) -> Optional[float]:
    try:
        cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _coerce_ts(record: Dict[str, Any]) -> Optional[float]:
    for key in ("ts", "timestamp", "time", "t", "local_ts"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            try:
                return float(value)
            except ValueError:
                parsed = _parse_iso(value)
                if parsed is not None:
                    return parsed
    return None


def _load_jsonl(path: Path, since: Optional[float], malformed_counter: Counter) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    malformed_counter[path.name] += 1
                    continue
                if not isinstance(record, dict):
                    malformed_counter[path.name] += 1
                    continue
                ts = _coerce_ts(record)
                if ts is None:
                    malformed_counter[path.name] += 1
                    continue
                if since is not None and ts < since:
                    continue
                record["_ts"] = ts
                record["_line"] = idx
                yield record
    except FileNotFoundError:
        return
    except Exception:
        return


def _bucket_ts(ts: float, bucket_seconds: int = 60) -> int:
    return int(ts // bucket_seconds)


def _extract_strategy(record: Dict[str, Any]) -> str:
    for key in ("strategy", "strategy_name", "strategyId", "strategy_id", "source"):
        value = record.get(key)
        if value:
            return str(value)
    payload = record.get("nav_snapshot") or record.get("intent") or {}
    if isinstance(payload, dict):
        for key in ("strategy", "strategy_name", "strategyId"):
            value = payload.get(key)
            if value:
                return str(value)
    return "unknown"


def _extract_symbol(record: Dict[str, Any]) -> str:
    symbol = record.get("symbol") or record.get("pair")
    if symbol:
        return str(symbol).upper()
    payload = record.get("intent") or record.get("veto_detail") or {}
    if isinstance(payload, dict):
        value = payload.get("symbol")
        if value:
            return str(value).upper()
    return "UNKNOWN"


def _extract_request_id(record: Dict[str, Any]) -> Optional[str]:
    req_id = record.get("request_id") or record.get("client_order_id") or record.get("clientOrderId")
    if req_id:
        return str(req_id)
    payload = record.get("intent") or record.get("veto_detail") or record.get("normalized") or {}
    if isinstance(payload, dict):
        for key in ("request_id", "client_order_id", "clientOrderId"):
            value = payload.get(key)
            if value:
                return str(value)
    return None


def _group_key(record: Dict[str, Any]) -> tuple[str, str, Optional[str], int]:
    strategy = _extract_strategy(record)
    symbol = _extract_symbol(record)
    req_id = _extract_request_id(record)
    bucket = _bucket_ts(record["_ts"])
    return strategy, symbol, req_id, bucket


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


@dataclass
class AttemptChain:
    attempt: Dict[str, Any]
    vetos: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[Dict[str, Any]] = field(default_factory=list)

    def latency_ms(self) -> Optional[float]:
        if not self.orders:
            return None
        first_order_ts = min(order["_ts"] for order in self.orders)
        return max(0.0, (first_order_ts - self.attempt["_ts"]) * 1000.0)

    def is_resolved(self) -> bool:
        return bool(self.vetos or self.orders)

    def has_gap(self, window_seconds: float = 300.0) -> bool:
        if self.is_resolved():
            return False
        latest_ts = self.attempt["_ts"]
        return (datetime.now(timezone.utc).timestamp() - latest_ts) > window_seconds


def _load_chains(since: Optional[float]) -> tuple[Dict[tuple[str, str, int], AttemptChain], Counter]:
    malformed = Counter()

    attempts: Dict[tuple[str, str, int], AttemptChain] = {}
    attempts_with_req: Dict[tuple[str, str, str], AttemptChain] = {}

    for record in _load_jsonl(LOG_DIR / "orders_attempted.jsonl", since, malformed):
        key = _group_key(record)
        chains_key = (key[0], key[1], key[3])
        chain = AttemptChain(attempt=record)
        attempts[chains_key] = chain
        if key[2]:
            attempts_with_req[(key[0], key[1], key[2])] = chain

    def attach(record: Dict[str, Any], attr: str) -> None:
        key = _group_key(record)
        req_key = (key[0], key[1], key[2])
        if key[2] and req_key in attempts_with_req:
            chain = attempts_with_req[req_key]
        else:
            bucket_key = (key[0], key[1], key[3])
            chain = attempts.get(bucket_key)
        if chain is None:
            return
        getattr(chain, attr).append(record)

    for record in _load_jsonl(LOG_DIR / "risk_vetoes.jsonl", since, malformed):
        attach(record, "vetos")

    for record in _load_jsonl(LOG_DIR / "orders_executed.jsonl", since, malformed):
        attach(record, "orders")

    for record in _load_jsonl(LOG_DIR / "position_state.jsonl", since, malformed):
        attach(record, "positions")

    return attempts, malformed


def _summaries(chains: Iterable[AttemptChain]) -> Dict[str, Any]:
    total_attempts = 0
    resolved = 0
    latencies: List[float] = []
    gaps: List[AttemptChain] = []

    for chain in chains:
        total_attempts += 1
        if chain.is_resolved():
            resolved += 1
        latency = chain.latency_ms()
        if latency is not None:
            latencies.append(latency)
        if chain.has_gap():
            gaps.append(chain)

    coverage = (resolved / total_attempts) * 100.0 if total_attempts else 0.0

    latency_stats = {
        "p50_ms": _percentile(latencies, 0.5) if latencies else math.nan,
        "p90_ms": _percentile(latencies, 0.9) if latencies else math.nan,
        "p99_ms": _percentile(latencies, 0.99) if latencies else math.nan,
    }

    gaps_summary = [
        {
            "strategy": _extract_strategy(chain.attempt),
            "symbol": _extract_symbol(chain.attempt),
            "ts": datetime.fromtimestamp(chain.attempt["_ts"], tz=timezone.utc).isoformat(),
        }
        for chain in gaps
    ]

    return {
        "total_attempts": total_attempts,
        "resolved_attempts": resolved,
        "coverage_pct": coverage,
        "latency": latency_stats,
        "gaps": gaps_summary,
    }


def _chains_as_dict(chains: Iterable[AttemptChain]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for chain in chains:
        attempt_ts = chain.attempt["_ts"]
        items.append(
            {
                "attempt": chain.attempt,
                "vetos": chain.vetos,
                "orders": chain.orders,
                "positions": chain.positions,
                "latency_ms": chain.latency_ms(),
                "resolved": chain.is_resolved(),
                "attempt_iso": datetime.fromtimestamp(attempt_ts, tz=timezone.utc).isoformat(),
            }
        )
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct execution chains from JSONL logs.")
    parser.add_argument(
        "--since",
        help="ISO8601 timestamp; only consider entries at or after this time (UTC).",
    )
    parser.add_argument(
        "--json",
        help="Optional path to write reconstructed chains as JSON.",
    )
    args = parser.parse_args(argv)

    since_ts: Optional[float] = None
    if args.since:
        since_ts = _parse_iso(args.since)
        if since_ts is None:
            print(f"Could not parse --since timestamp: {args.since}", file=sys.stderr)
            return 1

    chains_map, malformed = _load_chains(since_ts)
    chains = list(chains_map.values())

    summary = _summaries(chains)

    print("=== Execution Replay Summary ===")
    print(f"Attempts: {summary['total_attempts']}")
    print(f"Resolved: {summary['resolved_attempts']} ({summary['coverage_pct']:.1f}%)")
    latency = summary["latency"]
    print("Latency ms (attempt → order):")
    for pct in ("p50_ms", "p90_ms", "p99_ms"):
        value = latency[pct]
        value_display = f"{value:.1f}" if value == value else "n/a"
        print(f"  {pct.replace('_', '').upper()}: {value_display}")
    gaps = summary["gaps"]
    print(f"Gaps (>5m with no veto/order): {len(gaps)}")
    for gap in gaps[:10]:
        print(f"  - {gap['strategy']} {gap['symbol']} @ {gap['ts']}")

    if malformed:
        print("Malformed records encountered:")
        for name, count in malformed.items():
            print(f"  {name}: {count}")

    if args.json:
        try:
            output = {
                "summary": summary,
                "chains": _chains_as_dict(chains),
                "malformed": dict(malformed),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "since": args.since,
            }
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2, default=str)
            print(f"Wrote JSON chains to {args.json}")
        except Exception as exc:
            print(f"Failed to write JSON output: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
