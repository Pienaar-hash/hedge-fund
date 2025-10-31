"""Thread-safe JSONL logging utilities with size-based rotation."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import gzip
import json
import math
import os
import shutil
import socket
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, List, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
_HOSTNAME = socket.gethostname()


class JsonlLogger:
    """Minimal JSONL logger with atomic writes and size rotation."""

    def __init__(self, path: Path, max_bytes: int, backup_count: int) -> None:
        self._path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.RLock()
        self._archive_root = self._path.parent / "archive"

    def write(self, record: Mapping[str, Any] | None) -> None:
        """Persist the provided record as a JSONL line."""
        payload = safe_dump(record or {})
        line = json.dumps(payload, ensure_ascii=False)
        encoded = f"{line}\n".encode("utf-8")

        with self._lock:
            self._ensure_dirs()
            self._rotate_if_needed(len(encoded))
            self._atomic_append(encoded)

    def _ensure_dirs(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_root.mkdir(parents=True, exist_ok=True)

    def _indexed_path(self, index: int) -> Path:
        if index == 0:
            return self._path
        suffix = self._path.suffix
        base_name = self._path.name
        base = base_name[: -len(suffix)] if suffix else base_name
        rotated = f"{base}.{index}{suffix}"
        return self._path.with_name(rotated)

    def _rotate_if_needed(self, incoming_len: int) -> None:
        if self.backup_count <= 0 or self.max_bytes <= 0 or not self._path.exists():
            return
        try:
            current_size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if current_size + incoming_len <= self.max_bytes:
            return

        for idx in range(self.backup_count, 0, -1):
            src = self._indexed_path(idx - 1)
            if not src.exists():
                continue
            dst = self._indexed_path(idx)
            os.replace(src, dst)

        oldest = self._indexed_path(self.backup_count)
        if self.backup_count > 0 and oldest.exists():
            self._archive_file(oldest)

    def _archive_file(self, path: Path) -> None:
        archive_dir = self._archive_root
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{path.name}.gz"
        try:
            with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink(missing_ok=True)
            if __debug__:
                print(f"[log_utils] archived {path} -> {archive_path}")
        except Exception as exc:
            if __debug__:
                print(f"[log_utils] archive failed for {path}: {exc}")

    def _atomic_append(self, data: bytes) -> None:
        directory = self._path.parent
        fd, temp_path = tempfile.mkstemp(
            dir=str(directory), prefix=self._path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as tmp:
                if self._path.exists():
                    with self._path.open("rb") as src:
                        shutil.copyfileobj(src, tmp)
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_path, self._path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def get_logger(
    path: str, max_bytes: int = 10_000_000, backup_count: int = 5
) -> JsonlLogger:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    return JsonlLogger(target, max_bytes=max_bytes, backup_count=backup_count)


def append_jsonl(path: Path, record: Mapping[str, Any] | None) -> None:
    """Append a JSON serializable record to a JSONL file, creating parents if needed."""
    payload = safe_dump(record or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def log_event(logger: JsonlLogger, event_type: str, payload: Mapping[str, Any] | None) -> None:
    event: MutableMapping[str, Any] = safe_dump(payload or {})
    event.update(
        {
            "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "event_type": event_type,
            "pid": os.getpid(),
            "hostname": _HOSTNAME,
        }
    )
    logger.write(event)


def safe_dump(obj: Any) -> MutableMapping[str, Any]:
    """Return a dict that can be JSON-serialized by coercing complex objects."""

    def coerce(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): coerce(v) for k, v in value.items()}
        if dataclasses.is_dataclass(value):
            return coerce(dataclasses.asdict(value))
        if isinstance(value, (list, tuple)):
            return [coerce(v) for v in value]
        if isinstance(value, set):
            return [coerce(v) for v in value]
        if isinstance(value, _dt.datetime):
            item = value
            if item.tzinfo is None:
                item = item.replace(tzinfo=_dt.timezone.utc)
            return item.astimezone(_dt.timezone.utc).isoformat()
        if isinstance(value, _dt.date):
            return value.isoformat()
        if isinstance(value, _dt.time):
            if value.tzinfo is None:
                return value.isoformat()
            return value.astimezone(_dt.timezone.utc).isoformat()
        if hasattr(value, "__dict__"):
            return {str(k): coerce(v) for k, v in vars(value).items()}
        return repr(value)

    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {str(k): coerce(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return safe_dump(dataclasses.asdict(obj))
    if hasattr(obj, "__dict__"):
        return safe_dump(vars(obj))
    return {"value": coerce(obj)}


def percentile(values: Sequence[float], p: float) -> float:
    """Compute the percentile for a sequence of floats with linear interpolation."""
    if not values:
        return 0.0
    if not math.isfinite(p):
        raise ValueError("percentile requires finite p value")
    pct = max(0.0, min(100.0, float(p)))
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return 0.0
    if pct == 0.0:
        return ordered[0]
    if pct == 100.0:
        return ordered[-1]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


class AggWindow:
    """Fixed-size rolling window to compute percentile aggregates."""

    def __init__(self, capacity: int = 200) -> None:
        if capacity <= 0:
            raise ValueError("AggWindow capacity must be positive")
        self._capacity = int(capacity)
        self._values: Deque[float] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()

    def add(self, value: float | None) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric):
            return
        with self._lock:
            self._values.append(numeric)

    def snapshot(self) -> Mapping[str, float]:
        with self._lock:
            data: List[float] = list(self._values)
        if not data:
            return {"count": 0, "p50": 0.0, "p95": 0.0}
        return {
            "count": len(data),
            "p50": percentile(data, 50.0),
            "p95": percentile(data, 95.0),
        }

    def values(self) -> List[float]:
        with self._lock:
            return list(self._values)


if __name__ == "__main__":
    test_path = Path(__file__).with_name("_log_utils_self_test.jsonl")
    logger = get_logger(str(test_path))
    log_event(logger, "self_test_start", {"message": "hello"})
    log_event(logger, "self_test_end", {"sequence": 2})
    with test_path.open("r", encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    print(f"Lines logged: {count}")
