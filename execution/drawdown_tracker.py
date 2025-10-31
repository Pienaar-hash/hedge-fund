from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone, tzinfo
from typing import Any, Dict, Optional

PEAK_STATE_PATH = os.path.join("logs", "cache", "peak_state.json")


def _ensure_dir() -> None:
    try:
        os.makedirs(os.path.dirname(PEAK_STATE_PATH), exist_ok=True)
    except Exception:
        pass


def _as_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
        number = float(value)
        if not math.isfinite(number):
            return 0.0
        return number
    except Exception:
        return 0.0


def _resolve_timezone(name: str | None) -> tzinfo:
    if not name or str(name).upper() == "UTC":
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return timezone.utc
    try:
        return ZoneInfo(str(name))
    except Exception:
        return timezone.utc


def load_peak_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback = dict(default or {})
    try:
        with open(PEAK_STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return fallback


def save_peak_state(state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    payload = dict(state)
    payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("ts", time.time())
    _ensure_dir()
    try:
        with open(PEAK_STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    except Exception:
        pass


def compute_intraday_drawdown(
    nav_usd: Any,
    realized_pnl_usd_today: Any,
    *,
    reset_timezone: str = "UTC",
    now: Optional[datetime] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_state = dict(state or load_peak_state())
    tz = _resolve_timezone(reset_timezone)
    now_dt = now.astimezone(tz) if now is not None else datetime.now(tz)
    day_key = now_dt.strftime("%Y-%m-%d")
    prev_day = base_state.get("day") or base_state.get("date")

    nav_value = _as_float(nav_usd)
    stored_nav = _as_float(base_state.get("nav") or base_state.get("nav_usd"))
    if nav_value <= 0.0 and stored_nav > 0.0:
        nav_value = stored_nav

    realized_today = _as_float(realized_pnl_usd_today)
    if realized_today == 0.0:
        realized_cached = _as_float(base_state.get("realized_pnl_today"))
        if realized_cached != 0.0:
            realized_today = realized_cached

    previous_peak = _as_float(base_state.get("peak") or base_state.get("peak_equity"))
    if prev_day != day_key or previous_peak <= 0.0:
        peak = max(nav_value, previous_peak, 0.0)
    else:
        peak = max(previous_peak, nav_value)

    if peak <= 0.0:
        dd_abs = 0.0
        dd_pct = 0.0
    else:
        dd_abs = max(0.0, peak - max(nav_value, 0.0))
        dd_pct = (dd_abs / peak) * 100.0 if peak > 0 else 0.0

    updated = {
        "day": day_key,
        "peak": peak,
        "peak_equity": peak,
        "nav": nav_value,
        "nav_usd": nav_value,
        "realized_pnl_today": realized_today,
        "dd_abs": dd_abs,
        "dd_pct": dd_pct,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ts": time.time(),
        "reset_timezone": reset_timezone or "UTC",
    }
    return updated


def mirror_peak_state_to_firestore(
    state: Dict[str, Any],
    db: Any,
    *,
    env: Optional[str] = None,
) -> None:
    if not isinstance(state, dict) or not state:
        return
    if db is None:
        return
    if getattr(db, "_is_noop", False):
        return
    try:
        env_name = env or os.getenv("ENV", "dev")
        if str(env_name).lower() == "prod":
            allow = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
            if allow not in {"1", "true", "yes"}:
                raise RuntimeError("drawdown_tracker refuses to write with ENV=prod without ALLOW_PROD_WRITE=1")
        doc = (
            db.collection("hedge")
            .document(env_name)
            .collection("risk")
            .document("peak_state")
        )
        payload = dict(state)
        payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("ts", time.time())
        doc.set(payload, merge=True)
    except Exception:
        pass


__all__ = [
    "compute_intraday_drawdown",
    "load_peak_state",
    "mirror_peak_state_to_firestore",
    "save_peak_state",
]
