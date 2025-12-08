from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_pnl_attribution(path: str | Path = "logs/state/pnl_attribution.json") -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def safe_get_block(snapshot: dict, key: str, default: Any | None = None) -> Any:
    if not isinstance(snapshot, dict):
        return {} if default is None else default
    value = snapshot.get(key, default)
    if value is None:
        return {} if default is None else default
    return value


def get_factor_buckets(snapshot: dict[str, Any]) -> dict[str, Any]:
    regimes = snapshot.get("regimes") or {}
    return regimes.get("factors") or {}


def get_hybrid_deciles(snapshot: dict[str, Any]) -> dict[str, Any]:
    return get_factor_buckets(snapshot).get("hybrid_score_decile", {}) or {}


def get_trend_strength_buckets(snapshot: dict[str, Any]) -> dict[str, Any]:
    return get_factor_buckets(snapshot).get("trend_strength_bucket", {}) or {}


def get_carry_regimes(snapshot: dict[str, Any]) -> dict[str, Any]:
    return get_factor_buckets(snapshot).get("carry_regime", {}) or {}


def get_exits_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    exits = snapshot.get("exits") or {}
    summary = exits.get("summary") or {}
    return {
        "total_exits": summary.get("total_exits", 0),
        "tp_hits": summary.get("tp_hits", 0),
        "sl_hits": summary.get("sl_hits", 0),
        "tp_ratio": summary.get("tp_ratio", 0.0),
        "avg_rr_tp": summary.get("avg_rr_tp"),
        "avg_rr_sl": summary.get("avg_rr_sl"),
        "avg_exit_pnl": summary.get("avg_exit_pnl", 0.0),
        "total_exit_pnl": summary.get("total_exit_pnl", 0.0),
    }


def get_exits_by_strategy(snapshot: dict[str, Any]) -> dict[str, Any]:
    exits = snapshot.get("exits") or {}
    return exits.get("by_strategy", {}) or {}


def get_exits_by_symbol(snapshot: dict[str, Any]) -> dict[str, Any]:
    exits = snapshot.get("exits") or {}
    return exits.get("by_symbol", {}) or {}


def get_exits_regimes(snapshot: dict[str, Any]) -> dict[str, Any]:
    exits = snapshot.get("exits") or {}
    return exits.get("regimes", {}) or {}
