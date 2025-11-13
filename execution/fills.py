"""
Fills and order submission helpers sourced from execution logs.

These utilities aggregate data from logs/execution/orders_executed.jsonl
and logs/execution/orders_attempted.jsonl so higher-level telemetry
can derive rolling notionals, fees, and slippage.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
ORDER_EVENTS_PATH = LOG_DIR / "orders_executed.jsonl"
ORDER_ATTEMPTS_PATH = LOG_DIR / "orders_attempted.jsonl"
_READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num


def _to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    return None


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
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
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _recent_window(records: List[Dict[str, Any]], window_days: int, ts_key: str) -> List[Dict[str, Any]]:
    cutoff = time.time() - max(window_days, 0) * 86400.0
    filtered: List[Dict[str, Any]] = []
    for record in records:
        ts = _to_epoch(record.get(ts_key) or record.get("ts"))
        if ts is None or ts >= cutoff:
            record["_ts"] = ts
            filtered.append(record)
    return filtered


def get_recent_fills(symbol: Optional[str] = None, window_days: int = 7) -> List[Dict[str, Any]]:
    """Return recent fill events as lightweight dicts."""
    records = _read_jsonl(ORDER_EVENTS_PATH, _READ_LIMIT)
    fills: List[Dict[str, Any]] = []
    cutoff_records = _recent_window(records, window_days, "ts_fill_last")
    sym_filter = str(symbol or "").upper()
    for record in cutoff_records:
        event_type = str(record.get("event_type") or record.get("event") or "").lower()
        if event_type not in {"order_fill", ""}:
            continue
        sym = str(record.get("symbol") or "").upper()
        if sym_filter and sym != sym_filter:
            continue
        qty = _to_float(record.get("executedQty") or record.get("qty") or record.get("origQty")) or 0.0
        price = _to_float(record.get("avgPrice") or record.get("price")) or 0.0
        notional = abs(qty) * abs(price)
        fills.append(
            {
                "symbol": sym,
                "side": record.get("side"),
                "qty": qty,
                "price": price,
                "notional": notional,
                "fee": _to_float(record.get("fee_total")),
                "fee_asset": record.get("feeAsset"),
                "slippage_bps": _to_float(record.get("slippage_bps")),
                "mid_before": _to_float(record.get("mark_price") or record.get("mark")),
                "ts": record.get("_ts"),
                "attempt_id": record.get("attempt_id"),
            }
        )
    return fills


def get_recent_orders(symbol: Optional[str] = None, window_days: int = 7) -> List[Dict[str, Any]]:
    """Return recently submitted orders (attempts) with estimated notionals."""
    records = _read_jsonl(ORDER_ATTEMPTS_PATH, _READ_LIMIT)
    cutoff_records = _recent_window(records, window_days, "local_ts")
    orders: List[Dict[str, Any]] = []
    sym_filter = str(symbol or "").upper()
    for record in cutoff_records:
        sym = str(record.get("symbol") or "").upper()
        if sym_filter and sym != sym_filter:
            continue
        qty = _to_float(record.get("qty")) or 0.0
        price = _to_float(record.get("price_hint") or record.get("price"))
        if price is None or price <= 0:
            price = _to_float(record.get("mark_price"))
        notional = abs(qty) * abs(price or 0.0)
        orders.append(
            {
                "symbol": sym,
                "qty": qty,
                "price": price,
                "notional": notional,
                "ts": record.get("_ts"),
                "attempt_id": record.get("attempt_id"),
            }
        )
    return orders


__all__ = ["get_recent_fills", "get_recent_orders"]
