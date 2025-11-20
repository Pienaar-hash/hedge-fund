from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
ORDER_EVENTS_PATH = Path("logs/execution/orders_executed.jsonl")

LOG = logging.getLogger("dash.router")

__all__ = ["RouterHealthData", "load_router_health", "_load_order_events"]

@dataclass
class RouterHealthData:
    trades: pd.DataFrame
    per_symbol: pd.DataFrame
    pnl_curve: pd.DataFrame
    summary: Dict[str, Any]
    overlays: Dict[str, pd.DataFrame] = field(default_factory=dict)


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts_val = float(value)
        return ts_val / 1000.0 if ts_val > 1e12 else ts_val
    if isinstance(value, str):
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
    return None


def _default_summary(updated_ts: Optional[float] = None) -> Dict[str, Any]:
    summary = {
        "count": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "cum_pnl": 0.0,
        "fill_rate_pct": 0.0,
        "fees_total": 0.0,
        "realized_pnl": 0.0,
        "avg_confidence": None,
        "confidence_weighted_cum_pnl": 0.0,
        "normalized_sharpe": 0.0,
        "volatility_scale": 1.0,
        "rolling_sharpe_last": 0.0,
    }
    if updated_ts is not None:
        summary["updated_ts"] = updated_ts
    return summary


def _empty_router_health(updated_ts: Optional[float] = None) -> RouterHealthData:
    trades_df = pd.DataFrame(
        columns=[
            "time",
            "symbol",
            "pnl_usd",
            "attempt_id",
            "intent_id",
            "signal",
            "doctor_confidence",
            "fees_total",
        ]
    )
    pnl_curve_df = pd.DataFrame(columns=["time", "cum_pnl", "hit_rate", "confidence_weighted_cum_pnl", "rolling_sharpe"])
    return RouterHealthData(
        trades=trades_df,
        per_symbol=pd.DataFrame(),
        pnl_curve=pnl_curve_df,
        summary=_default_summary(updated_ts),
        overlays={},
    )


def _mean_from_column(frame: pd.DataFrame, column: str, scale: float | None = None) -> Optional[float]:
    if column not in frame.columns:
        return None
    try:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
    except Exception:
        return None
    if series.empty:
        return None
    val = float(series.mean())
    if scale is not None:
        val *= scale
    return val


def _build_router_health(payload: Mapping[str, Any]) -> RouterHealthData:
    symbols = payload.get("symbols")
    updated_ts = _parse_ts(payload.get("updated_ts") or payload.get("updated_at"))
    if not isinstance(symbols, list):
        return _empty_router_health(updated_ts)
    per_symbol_df = pd.DataFrame.from_records(symbols) if symbols else pd.DataFrame()
    summary = _default_summary(updated_ts)
    summary["count"] = int(len(per_symbol_df))
    maker_fill_pct = _mean_from_column(per_symbol_df, "maker_fill_rate", scale=100.0)
    if maker_fill_pct is not None:
        summary["fill_rate_pct"] = maker_fill_pct
    fallback_pct = _mean_from_column(per_symbol_df, "fallback_rate", scale=100.0)
    if fallback_pct is not None:
        summary["fallback_rate_pct"] = fallback_pct
    pnl_curve_df = pd.DataFrame(columns=["time", "cum_pnl", "hit_rate", "confidence_weighted_cum_pnl", "rolling_sharpe"])
    trades_df = pd.DataFrame(
        columns=[
            "time",
            "symbol",
            "pnl_usd",
            "attempt_id",
            "intent_id",
            "signal",
            "doctor_confidence",
            "fees_total",
        ]
    )
    return RouterHealthData(
        trades=trades_df,
        per_symbol=per_symbol_df,
        pnl_curve=pnl_curve_df,
        summary=summary,
        overlays={},
    )


def _load_order_events(limit: int = 0) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Minimal stub for doctor compatibility; returns empty lists (orders not used)."""
    _ = limit
    return [], [], []


def _load_state_router_health(path: Optional[Path] = None) -> Optional[RouterHealthData]:
    target = path or ROUTER_HEALTH_STATE_PATH
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text())
    except Exception:
        LOG.debug("[dash] failed to read router health state from %s", target)
        return None
    if not isinstance(payload, Mapping):
        return None
    return _build_router_health(payload)


def load_router_health(
    window: int = 0,
    *,
    signal_path: Path | None = None,
    order_path: Path | None = None,
    snapshot: Optional[Mapping[str, Any]] = None,
    trades_snapshot: Optional[Mapping[str, Any]] = None,
) -> RouterHealthData:
    _ = window
    _ = signal_path
    _ = order_path
    _ = trades_snapshot
    state_view = _load_state_router_health()
    if state_view is not None:
        return state_view
    if snapshot and isinstance(snapshot, Mapping):
        return _build_router_health(snapshot)
    return _empty_router_health()
