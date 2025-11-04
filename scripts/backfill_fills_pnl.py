#!/usr/bin/env python3
"""
Backfill fill and PnL events from Binance REST endpoints.

Reads legacy ACK-only logs and reconstructs `order_fill` and `order_close`
events. Defaults to dry-run (prints sample events). Use `--apply` to write
to `logs/execution/orders_events_backfilled.jsonl`.
"""

from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import requests

from execution.pnl_tracker import Fill as PnlFill
from execution.pnl_tracker import PositionTracker

DEFAULT_SOURCE_GLOB = "logs/execution/audit_orders_*.jsonl"
DEFAULT_OUTPUT = Path("logs/execution/orders_events_backfilled.jsonl")
BASE_URL = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")


def _iso_from_ms(value: Any) -> str:
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()
    if ms > 1e12:
        ms /= 1000.0
    return datetime.fromtimestamp(ms, tz=timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num


def _identifier(record: Mapping[str, Any]) -> str:
    order_id = record.get("orderId") or record.get("order_id")
    client_id = record.get("clientOrderId") or record.get("client_order_id")
    if order_id:
        return str(order_id)
    if client_id:
        return str(client_id)
    return f"anon_{hash(tuple(sorted(record.items())))}"


def _iter_json_lines(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    yield payload
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[backfill] failed to read {path}: {exc}", file=sys.stderr)


def _collect_ack_sources(pattern: str) -> List[Path]:
    if "*" in pattern or "?" in pattern:
        return sorted(Path(".").glob(pattern))
    path = Path(pattern)
    if path.exists():
        return [path]
    fallback = Path("logs/execution/orders_executed.jsonl")
    return [fallback] if fallback.exists() else []


def _extract_ack(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    event = str(record.get("event_type") or record.get("event") or "").lower()
    if event and event not in {"order_ack", "order_executed"}:
        return None
    order_id = record.get("orderId") or record.get("order_id")
    client_id = record.get("clientOrderId") or record.get("client_order_id")
    symbol = record.get("symbol")
    side = record.get("side")
    if not order_id or not symbol or not side:
        return None
    ack = {
        "orderId": int(order_id),
        "clientOrderId": client_id,
        "symbol": str(symbol).upper(),
        "side": str(side).upper(),
        "positionSide": record.get("positionSide"),
        "reduceOnly": record.get("reduceOnly"),
        "request_qty": _safe_float(record.get("request_qty") or record.get("qty") or record.get("quantity")),
        "order_type": str(record.get("order_type") or record.get("type") or "MARKET").upper(),
        "status": str(record.get("status") or "UNKNOWN").upper(),
        "ts_ack": record.get("ts_ack") or record.get("ts"),
    }
    return ack


def iter_ack_records(pattern: str) -> Iterator[Dict[str, Any]]:
    sources = _collect_ack_sources(pattern)
    for path in sources:
        for record in _iter_json_lines(path):
            ack = _extract_ack(record)
            if ack:
                yield ack


class BinanceRestClient:
    def __init__(self, api_key: str, api_secret: str) -> None:
        if not api_key or not api_secret:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        self.api_key = api_key
        self.api_secret = api_secret.encode()

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {k: v for k, v in params.items() if v is not None}
        payload["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={payload[k]}" for k in payload)
        signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    def get(self, path: str, params: Dict[str, Any]) -> Any:
        signed = self._signed_params(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        response = requests.get(f"{BASE_URL}{path}", params=signed, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def fetch_order(self, symbol: str, order_id: int, client_order_id: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol, "orderId": order_id}
        if client_order_id and not order_id:
            params["origClientOrderId"] = client_order_id
        return self.get("/fapi/v1/order", params)

    def fetch_trades(self, symbol: str, order_id: int) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "orderId": order_id}
        data = self.get("/fapi/v1/userTrades", params)
        if isinstance(data, list):
            return data
        return []


def build_fill_event(
    ack: Mapping[str, Any],
    trades: Sequence[Mapping[str, Any]],
    order_status: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not trades:
        return None
    executed_qty = 0.0
    notional = 0.0
    fee_total = 0.0
    fee_asset = None
    trade_ids: List[int] = []
    ts_first = None
    ts_last = None
    for trade in trades:
        qty = _safe_float(trade.get("qty")) or 0.0
        price = _safe_float(trade.get("price")) or 0.0
        executed_qty += qty
        notional += qty * price
        fee_total += _safe_float(trade.get("commission")) or 0.0
        fee_asset = fee_asset or trade.get("commissionAsset")
        trade_ids.append(int(trade.get("id", 0)))
        ts = trade.get("time")
        if ts_first is None or ts < ts_first:
            ts_first = ts
        if ts_last is None or ts > ts_last:
            ts_last = ts
    if executed_qty <= 0.0:
        return None
    avg_price = notional / executed_qty if executed_qty else 0.0
    status = str(order_status.get("status") or ack.get("status") or "UNKNOWN").upper()
    fill_event: Dict[str, Any] = {
        "event_type": "order_fill",
        "symbol": ack["symbol"],
        "side": ack["side"],
        "ts_fill_first": _iso_from_ms(ts_first),
        "ts_fill_last": _iso_from_ms(ts_last),
        "orderId": ack["orderId"],
        "clientOrderId": ack.get("clientOrderId"),
        "executedQty": executed_qty,
        "avgPrice": avg_price,
        "fee_total": fee_total,
        "feeAsset": fee_asset or order_status.get("commissionAsset") or "USDT",
        "tradeIds": sorted(trade_ids),
        "status": status,
    }
    if ack.get("attempt_id"):
        fill_event["attempt_id"] = ack.get("attempt_id")
    if ack.get("intent_id"):
        fill_event["intent_id"] = ack.get("intent_id")
    if ack.get("positionSide"):
        fill_event["positionSide"] = ack.get("positionSide")
    if ack.get("reduceOnly"):
        fill_event["reduceOnly"] = True
    return fill_event


def build_close_events(
    tracker: PositionTracker,
    ack: Mapping[str, Any],
    trades: Sequence[Mapping[str, Any]],
    fill_event: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    close_events: List[Dict[str, Any]] = []
    results = []
    for trade in trades:
        qty = _safe_float(trade.get("qty")) or 0.0
        price = _safe_float(trade.get("price")) or 0.0
        if qty <= 0.0 or price <= 0.0:
            continue
        commission = _safe_float(trade.get("commission")) or 0.0
        pnl_fill = PnlFill(
            symbol=ack["symbol"],
            side=ack["side"],
            qty=qty,
            price=price,
            fee=commission,
            position_side=ack.get("positionSide"),
            reduce_only=bool(ack.get("reduceOnly")),
        )
        result = tracker.apply_fill(pnl_fill)
        if result:
            results.append(result)
    if not results:
        return close_events
    total_realized = sum(item.realized_pnl for item in results)
    total_fees = sum(item.fees for item in results)
    close_event: Dict[str, Any] = {
        "event_type": "order_close",
        "symbol": ack["symbol"],
        "ts_close": fill_event.get("ts_fill_last"),
        "orderId": ack["orderId"],
        "clientOrderId": ack.get("clientOrderId"),
        "realizedPnlUsd": total_realized,
        "fees_total": total_fees,
        "position_size_before": results[0].position_before,
        "position_size_after": results[-1].position_after,
    }
    if ack.get("attempt_id"):
        close_event["attempt_id"] = ack.get("attempt_id")
    if ack.get("intent_id"):
        close_event["intent_id"] = ack.get("intent_id")
    if ack.get("positionSide"):
        close_event["positionSide"] = ack.get("positionSide")
    close_events.append(close_event)
    return close_events


def backfill(
    sources: Iterable[Dict[str, Any]],
    client: BinanceRestClient,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tracker = PositionTracker()
    fills: List[Dict[str, Any]] = []
    closes: List[Dict[str, Any]] = []
    for idx, ack in enumerate(sources):
        if limit is not None and idx >= limit:
            break
        try:
            order_status = client.fetch_order(ack["symbol"], ack["orderId"], ack.get("clientOrderId"))
            trades = client.fetch_trades(ack["symbol"], ack["orderId"])
        except requests.HTTPError as exc:
            print(f"[backfill] REST error for {ack['symbol']} order {ack['orderId']}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"[backfill] error fetching order {ack['orderId']}: {exc}", file=sys.stderr)
            continue
        fill_event = build_fill_event(ack, trades, order_status)
        if not fill_event:
            continue
        fills.append(fill_event)
        closes.extend(build_close_events(tracker, ack, trades, fill_event))
    return fills, closes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill fill and PnL events from Binance REST")
    parser.add_argument("--source", default=DEFAULT_SOURCE_GLOB, help="Glob or file path for ACK logs")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N orders")
    parser.add_argument("--apply", action="store_true", help="Write events to output file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    args = parser.parse_args(argv)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        print("[backfill] BINANCE_API_KEY / BINANCE_API_SECRET must be set", file=sys.stderr)
        return 1

    client = BinanceRestClient(api_key, api_secret)
    ack_iterator = iter_ack_records(args.source)
    fills, closes = backfill(ack_iterator, client, limit=args.limit)
    if not fills:
        print("[backfill] No fills reconstructed")
        return 0

    events = fills + closes
    events.sort(key=lambda ev: ev.get("ts_fill_last") or ev.get("ts_close") or "")

    if args.apply:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")
        print(f"[backfill] wrote {len(events)} events to {output_path}")
    else:
        print(f"[backfill] reconstructed {len(events)} events (dry-run, showing first 5)")
        for event in events[:5]:
            print(json.dumps(event, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
