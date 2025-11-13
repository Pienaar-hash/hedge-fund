"""
Router metrics helpers backed by executor JSONL logs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
ROUTER_METRICS_PATH = Path(os.getenv("ROUTER_METRICS_PATH") or (LOG_DIR / "order_metrics.jsonl"))
READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


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
            return float(text)
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
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _recent_window(records: List[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
    cutoff = time.time() - max(window_days, 0) * 86400.0
    filtered: List[Dict[str, Any]] = []
    for record in records:
        ts = _to_epoch(record.get("ts"))
        if ts is None or ts >= cutoff:
            record["_ts"] = ts
            filtered.append(record)
    return filtered


def get_recent_router_events(symbol: Optional[str] = None, window_days: int = 7) -> List[Dict[str, Any]]:
    records = _read_jsonl(ROUTER_METRICS_PATH, READ_LIMIT)
    filtered = _recent_window(records, window_days)
    sym_filter = (symbol or "").upper()
    if not sym_filter:
        return filtered
    return [record for record in filtered if str(record.get("symbol") or "").upper() == sym_filter]


__all__ = ["get_recent_router_events"]
