"""v6-only NAV helpers."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
SYNCED_STATE_PATH = Path(os.getenv("SYNCED_STATE_PATH") or (STATE_DIR / "synced_state.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _to_epoch_seconds(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:
            val /= 1000.0
        return val
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            if txt.isdigit():
                return _to_epoch_seconds(float(txt))
        except Exception:
            pass
        try:
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            return datetime.fromisoformat(txt).astimezone(timezone.utc).timestamp()
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def load_nav_state() -> Tuple[Dict[str, Any], str]:
    """
    v6: single canonical NAV state file: logs/state/nav_state.json
    Returns (payload, source_name).
    """
    payload = _load_json(NAV_STATE_PATH)
    return payload, NAV_STATE_PATH.name


def load_synced_state() -> Dict[str, Any]:
    """
    v6: synced_state.json mirrors executor state (nav, positions, caps).
    """
    return _load_json(SYNCED_STATE_PATH)


def nav_state_age_seconds(nav_state: Dict[str, Any]) -> Optional[float]:
    """
    Compute age of the nav_state snapshot in seconds, based on the freshest
    timestamp we can find in the document.
    """
    if not isinstance(nav_state, dict) or not nav_state:
        return None

    candidates: List[Any] = []

    for key in ("updated_at", "ts", "updated_ts"):
        if key in nav_state:
            candidates.append(nav_state.get(key))

    series = nav_state.get("series")
    if isinstance(series, list) and series:
        for entry in reversed(series):
            if not isinstance(entry, dict):
                continue
            for key in ("t", "ts"):
                if key in entry:
                    candidates.append(entry.get(key))
            break

    now = time.time()
    for raw in candidates:
        ts_val = _to_epoch_seconds(raw)
        if ts_val is not None:
            return max(0.0, now - float(ts_val))

    return None


def signal_attempts_summary(lines: List[str]) -> str:
    """
    Compact screener tail summary for the Signals tab.
    """
    if not lines:
        return "No screener attempts recorded yet."

    attempted = 0
    emitted = 0
    submitted = 0

    for line in lines:
        if "attempted=" in line:
            attempted += 1
        if "emitted=" in line:
            emitted += 1
        if "submitted=" in line:
            submitted += 1

    parts: List[str] = []
    if attempted:
        parts.append(f"attempt lines={attempted}")
    if emitted:
        parts.append(f"emitted lines={emitted}")
    if submitted:
        parts.append(f"submitted lines={submitted}")

    if not parts:
        return f"{len(lines)} screener log lines."

    return " · ".join(parts)


__all__ = [
    "load_nav_state",
    "load_synced_state",
    "nav_state_age_seconds",
    "signal_attempts_summary",
]
