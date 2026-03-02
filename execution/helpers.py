"""Pure utility helpers extracted from executor_live.py.

All functions in this module are stateless and side-effect-free (except
``git_commit`` which shells out to ``git describe``).  They must never
import from ``dashboard/`` or write to any state file.

Part of v7.9 architecture repair sprint — Phase 1 extraction.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, List, Optional


# ── Type conversion helpers ──────────────────────────────────────────


def to_float(value: Any) -> Optional[float]:
    """Coerce *value* to ``float``, returning ``None`` on failure."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def ms_to_iso(value: Any) -> Optional[str]:
    """Convert a millisecond or second epoch timestamp to ISO-8601 UTC string."""
    try:
        if value is None:
            return None
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1e12:
        val /= 1000.0
    try:
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except Exception:
        return None


def iso_to_ts(value: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 string to a Unix epoch float (seconds)."""
    if not value:
        return None
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


def normalize_status(status: Any) -> str:
    """Normalize an order status string (e.g. ``CANCELLED`` → ``CANCELED``)."""
    if not status:
        return "UNKNOWN"
    try:
        value = str(status).upper()
    except Exception:
        return "UNKNOWN"
    if value == "CANCELLED":
        return "CANCELED"
    return value


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> str:
    """Default serializer for ``json.dumps`` — handles datetime objects."""
    try:
        if isinstance(value, (datetime,)):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


# ── ID / environment helpers ─────────────────────────────────────────


def mk_id(prefix: str) -> str:
    """Generate a short random ID with the given prefix."""
    base = prefix.strip("_") or "id"
    return f"{base}_{uuid.uuid4().hex[:10]}"


def git_commit() -> str:
    """Return the current git tag/SHA via ``git describe``."""
    try:
        return (
            subprocess.check_output(["git", "describe", "--tags", "--always"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def truthy_env(name: str, default: str = "0") -> bool:
    """Return ``True`` if environment variable *name* is truthy (1/true/yes)."""
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def read_dry_run_flag() -> bool:
    """Read the ``DRY_RUN`` environment variable."""
    return truthy_env("DRY_RUN")


def resolve_env(default: str = "dev") -> str:
    """Resolve the runtime environment from ``ENV`` / ``ENVIRONMENT``."""
    raw = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip()
    if not raw:
        return default
    return raw


def coerce_veto_reasons(raw: Any) -> List[str]:
    """Normalize a veto-reasons value into a list of strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw if item]
    return [str(raw)]
