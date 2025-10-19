from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

MAX_EVENTS_PER_SYMBOL = 500
ROLLING_WINDOWS = (20, 50, 100)
HISTORY_WINDOW = 60
LOG_DIR = Path("logs") / "ml"
METRICS_PATH = LOG_DIR / "live_metrics.json"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload.setdefault("symbols", {})
                return payload
        except Exception:
            pass
    return {"symbols": {}}


def _trim_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(events) <= MAX_EVENTS_PER_SYMBOL:
        return events
    return events[-MAX_EVENTS_PER_SYMBOL:]


def _rolling_metrics(events: List[Dict[str, Any]], window: int) -> Dict[str, float | None]:
    if not events:
        return {"hit_rate": None, "avg_prob": None, "count": 0}
    data = events[-window:]
    if not data:
        return {"hit_rate": None, "avg_prob": None, "count": 0}
    hits = sum(int(evt.get("hit", 0)) for evt in data)
    count = len(data)
    avg_prob = sum(float(evt.get("prob", 0.0) or 0.0) for evt in data) / count
    hit_rate = hits / count if count else None
    return {
        "hit_rate": hit_rate,
        "avg_prob": avg_prob,
        "count": count,
        "calibration_gap": (hit_rate - avg_prob) if hit_rate is not None else None,
    }


def _history(events: List[Dict[str, Any]], window: int) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    if not events:
        return history
    hits_running = 0.0
    probs_running = 0.0
    buffer: List[Dict[str, Any]] = []
    for evt in events[-window:]:
        buffer.append(evt)
        hits_running += int(evt.get("hit", 0))
        probs_running += float(evt.get("prob", 0.0) or 0.0)
        if len(buffer) > window:
            popped = buffer.pop(0)
            hits_running -= int(popped.get("hit", 0))
            probs_running -= float(popped.get("prob", 0.0) or 0.0)
        count = len(buffer)
        if count == 0:
            continue
        history.append(
            {
                "ts": float(evt.get("ts", time.time())),
                "hit_rate": hits_running / count if count else 0.0,
                "avg_prob": probs_running / count if count else 0.0,
            }
        )
    return history[-window:]


def update_live_metrics(
    symbol: str,
    probability: float,
    outcome: int,
    *,
    timestamp: float | None = None,
) -> Dict[str, Any]:
    """Append a live prediction outcome and persist rolling calibration metrics."""
    _ensure_dir()
    state = _load_state()
    ts = float(timestamp or time.time())
    symbol_key = str(symbol).upper()
    symbols: MutableMapping[str, Any] = state.setdefault("symbols", {})
    symbol_state: Dict[str, Any] = symbols.setdefault(symbol_key, {})
    events: List[Dict[str, Any]] = symbol_state.setdefault("events", [])
    events.append(
        {
            "ts": ts,
            "prob": float(probability),
            "hit": int(outcome),
        }
    )
    symbol_state["events"] = _trim_events(events)

    windows: Dict[str, Dict[str, float | None]] = {}
    for window in ROLLING_WINDOWS:
        windows[str(window)] = _rolling_metrics(symbol_state["events"], window)
    symbol_state["windows"] = windows
    symbol_state["history"] = _history(symbol_state["events"], HISTORY_WINDOW)
    symbol_state["updated_at"] = ts
    symbol_state["event_count"] = len(symbol_state["events"])

    state["updated_at"] = _now_iso()
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return symbol_state


def load_live_metrics() -> Dict[str, Any]:
    """Return cached live metrics (without mutating)."""
    _ensure_dir()
    return _load_state()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # type: ignore[name-defined]


__all__ = ["update_live_metrics", "load_live_metrics"]
