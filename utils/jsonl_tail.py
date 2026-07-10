from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def file_signature(path: Path) -> Tuple[float, int]:
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except FileNotFoundError:
        return (0.0, 0)


def read_tail_jsonl(
    path: Path,
    *,
    max_bytes: int,
    max_lines: int,
    window_seconds: Optional[float] = None,
    now_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read()
    except Exception:
        return []

    cutoff: Optional[float] = None
    if window_seconds is not None:
        ref_now = time.time() if now_ts is None else now_ts
        cutoff = ref_now - window_seconds

    records: List[Dict[str, Any]] = []
    for line in chunk.decode(errors="ignore").splitlines()[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if cutoff is not None:
            ts = payload.get("ts") or payload.get("timestamp") or 0
            try:
                if float(ts) < cutoff:
                    continue
            except (TypeError, ValueError):
                pass
        records.append(payload)
    return records
