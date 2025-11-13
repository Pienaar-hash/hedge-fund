"""
Trade log accessors backed by execution events.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
ORDER_EVENTS_PATH = LOG_DIR / "orders_executed.jsonl"
_READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


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


def get_recent_trades(symbol: str, limit: int = 300) -> List[Dict[str, Any]]:
    """Return recent trade close events with realized PnL."""
    sym = str(symbol or "").upper()
    if not sym:
        return []
    records = _read_jsonl(ORDER_EVENTS_PATH, _READ_LIMIT)
    trades: List[Dict[str, Any]] = []
    for record in reversed(records):
        event_type = str(record.get("event_type") or record.get("event") or "").lower()
        if event_type != "order_close":
            continue
        rec_symbol = str(record.get("symbol") or "").upper()
        if rec_symbol != sym:
            continue
        pnl = record.get("realizedPnlUsd") or record.get("realized_pnl_usd")
        try:
            pnl_val = float(pnl)
        except (TypeError, ValueError):
            continue
        trades.append(
            {
                "symbol": rec_symbol,
                "realized_pnl": pnl_val,
                "fees": float(record.get("fees_total") or 0.0),
                "ts": _to_epoch(record.get("ts_close") or record.get("ts")),
            }
        )
        if len(trades) >= limit:
            break
    return trades


__all__ = ["get_recent_trades"]
