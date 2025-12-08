from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from execution.router_metrics import compute_maker_reliability

STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
ROUTER_HEALTH_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
RISK_SNAPSHOT_PATH = Path(os.getenv("RISK_SNAPSHOT_STATE_PATH") or (STATE_DIR / "risk_snapshot.json"))

_ROUTER_CACHE: MutableMapping[str, Any] = {"path": None, "mtime": 0.0, "data": {}}
_RISK_CACHE: MutableMapping[str, Any] = {"path": None, "mtime": 0.0, "data": {}}


def _load_json(path: Path, cache: MutableMapping[str, Any]) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return {}
    if cache.get("path") == str(path) and cache.get("mtime") == mtime and cache.get("data"):
        return cache["data"]
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        data = {}
    cache["path"] = str(path)
    cache["mtime"] = mtime
    cache["data"] = data if isinstance(data, Mapping) else {}
    return cache["data"]


def load_router_health_snapshot(path: Path | str | None = None) -> Mapping[str, Any]:
    target = Path(path) if path else ROUTER_HEALTH_PATH
    return dict(_load_json(target, _ROUTER_CACHE))


def load_risk_snapshot(path: Path | str | None = None) -> Mapping[str, Any]:
    target = Path(path) if path else RISK_SNAPSHOT_PATH
    return dict(_load_json(target, _RISK_CACHE))


def _normalize_risk_mode(value: Any) -> str:
    if value is None:
        return "OK"
    text = str(value).strip().upper()
    return text or "OK"


def _effective_reliability(reliability: float, risk_mode: str) -> float:
    if risk_mode == "WARN":
        return max(0.0, reliability - 0.1)
    return reliability


def _offset_multiplier(reliability: float) -> float:
    if reliability >= 0.8:
        return 1.0
    if reliability >= 0.5:
        return 0.7
    return 0.4


def _normalize_min_offset(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.5


def suggest_autotune_for_symbol(
    symbol: str,
    base_offset_bps: float,
    *,
    router_health: Mapping[str, Any] | None = None,
    risk_snapshot: Mapping[str, Any] | None = None,
    min_offset_bps: float | None = None,
) -> Mapping[str, Any]:
    router_health = router_health or load_router_health_snapshot()
    risk_snapshot = risk_snapshot or load_risk_snapshot()
    stats = {}
    if isinstance(router_health, Mapping):
        stats = router_health.get("symbol_stats") or {}
        if isinstance(stats, Mapping):
            stats = dict(stats.get(symbol.upper()) or {})
        else:
            stats = {}
    risk_mode = _normalize_risk_mode(risk_snapshot.get("risk_mode"))
    reliability = float(stats.get("maker_reliability") or stats.get("maker_reliability_score") or 0.0)
    reliability = max(0.0, min(1.0, reliability))
    effective_reliability = _effective_reliability(reliability, risk_mode)
    multiplier = _offset_multiplier(reliability if risk_mode != "DEFENSIVE" else effective_reliability)
    offset = float(base_offset_bps or 0.0) * multiplier
    if risk_mode == "DEFENSIVE":
        offset *= 0.5
    minimum = _normalize_min_offset(min_offset_bps) if min_offset_bps is not None else 0.5
    adaptive_offset = max(minimum, offset)

    threshold = 0.75 if risk_mode == "DEFENSIVE" else 0.6
    maker_first = bool(effective_reliability >= threshold)
    if risk_mode == "HALTED":
        maker_first = False

    return {
        "adaptive_offset_bps": adaptive_offset,
        "maker_reliability": reliability,
        "maker_first": maker_first,
        "risk_mode": risk_mode,
        "effective_reliability": effective_reliability,
    }
