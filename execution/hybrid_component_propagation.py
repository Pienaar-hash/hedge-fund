from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from execution.log_utils import append_jsonl

_LOG_PATH = Path("logs/execution/hybrid_component_propagation.jsonl")
_TS_BUCKET_S = 300


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_ts(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            t = float(value)
            return t / 1000.0 if t > 1e12 else t
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None
    return None


def _norm_side(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().upper()
    if s in ("BUY", "LONG"):
        return "LONG"
    if s in ("SELL", "SHORT"):
        return "SHORT"
    return s


def _stable_correlation_key(payload: Mapping[str, Any], *, strategy: str, source_head: str) -> str:
    symbol = str(payload.get("symbol", "") or "").upper()
    side = _norm_side(
        payload.get("side")
        or payload.get("signal")
        or payload.get("direction")
        or payload.get("net_side")
        or payload.get("positionSide")
    )
    strat = str(strategy or payload.get("strategy") or payload.get("source") or "")
    head = str(source_head or payload.get("source_head") or "")
    score = _safe_float(payload.get("hybrid_score"))
    if score is None:
        score = _safe_float(payload.get("score"))
    nav_pct = _safe_float(payload.get("nav_pct"))
    if nav_pct is None:
        meta = payload.get("metadata")
        if isinstance(meta, Mapping):
            nav_pct = _safe_float(meta.get("nav_pct"))

    ts_source = (
        _safe_ts(payload.get("signal_ts"))
        or _safe_ts(payload.get("local_ts"))
        or _safe_ts(payload.get("ts"))
        or time.time()
    )
    ts_bucket = int(ts_source // _TS_BUCKET_S)

    raw = "|".join(
        [
            symbol,
            side,
            strat,
            head,
            "" if score is None else f"{score:.6f}",
            "" if nav_pct is None else f"{nav_pct:.6f}",
            str(ts_bucket),
        ]
    )
    return "corr_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def trace_hybrid_component_propagation(
    *,
    origin_stage: str,
    intent: Optional[Mapping[str, Any]] = None,
    symbol: str = "",
    intent_id: str = "",
    strategy: str = "",
    source_head: str = "",
    merge_path: str = "",
    intent_type: str = "",
    hybrid_components: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append a fail-open telemetry record for hybrid component propagation.

    This function is observability-only and must never affect execution flow.
    """
    try:
        payload = dict(intent or {})
        comp = dict(hybrid_components or payload.get("hybrid_components") or {})
        keys = sorted(str(k) for k in comp.keys())
        intent_id_val = intent_id or str(payload.get("intent_id") or payload.get("attempt_id") or "")
        intent_id_present = bool(intent_id_val)
        correlation_key = intent_id_val if intent_id_present else _stable_correlation_key(
            payload,
            strategy=strategy,
            source_head=source_head,
        )
        rec = {
            "ts": time.time(),
            "symbol": symbol or str(payload.get("symbol", "") or "").upper(),
            "intent_id": intent_id_val,
            "intent_id_present": intent_id_present,
            "correlation_key": correlation_key,
            "strategy": strategy or str(payload.get("strategy") or payload.get("source") or ""),
            "source_head": source_head or str(payload.get("source_head") or ""),
            "origin_stage": origin_stage,
            "has_hybrid_components": bool(comp),
            "hybrid_component_keys": keys,
            "hybrid_expectancy": _safe_float(comp.get("expectancy")),
            "merge_path": merge_path,
            "intent_type": intent_type or str(payload.get("type") or payload.get("event_type") or ""),
        }
        append_jsonl(_LOG_PATH, rec)
    except Exception:
        pass


__all__ = ["trace_hybrid_component_propagation"]
