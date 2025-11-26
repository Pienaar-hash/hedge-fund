"""v6 router health with policy overlays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from dashboard.live_helpers import (
    load_router_policy_v6,
    load_router_suggestions_v6,
)


@dataclass
class RouterHealthData:
    per_symbol: pd.DataFrame
    trades: pd.DataFrame
    pnl_curve: pd.DataFrame
    summary: Dict[str, Any]
    overlays: Dict[str, Any]


def is_empty_router_health(data: RouterHealthData) -> bool:
    if data is None:
        return True
    if not isinstance(data.summary, dict):
        return True
    if data.summary.get("count", 0) == 0 and data.per_symbol.empty and data.pnl_curve.empty:
        return True
    return False


def _to_dataframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, list):
        df = pd.DataFrame([row for row in payload if isinstance(row, dict)])
    else:
        df = pd.DataFrame()

    if "value" in df.columns:
        try:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["value"] = df["value"].fillna(0.0).astype(float)
        except Exception:
            df["value"] = df["value"]
    return df


def load_router_health(
    window: int = 300,
    snapshot: Optional[Dict[str, Any]] = None,
    trades_snapshot: Optional[Dict[str, Any]] = None,
) -> RouterHealthData:
    """
    v6 router health loader with policy overlays.
    """
    summary: Dict[str, Any] = {}
    per_symbol = pd.DataFrame()
    pnl_curve = pd.DataFrame()
    overlays: Dict[str, Any] = {}

    if isinstance(snapshot, dict):
        if isinstance(snapshot.get("summary"), dict):
            summary = dict(snapshot["summary"])
        elif snapshot.get("updated_ts") is not None:
            summary = {"updated_ts": snapshot.get("updated_ts")}
        per_symbol = _to_dataframe(snapshot.get("per_symbol"))
        if per_symbol.empty and isinstance(snapshot.get("symbols"), list):
            per_symbol = _to_dataframe(snapshot.get("symbols"))
        pnl_curve = _to_dataframe(snapshot.get("pnl_curve"))

    if not pnl_curve.empty:
        if "time" in pnl_curve.columns:
            try:
                pnl_curve["time"] = pd.to_datetime(pnl_curve["time"], utc=True, errors="coerce")
                pnl_curve = pnl_curve.sort_values("time")
                if window > 0:
                    pnl_curve = pnl_curve.tail(window)
            except Exception:
                pnl_curve = pnl_curve.tail(window)
        else:
            pnl_curve = pnl_curve.tail(window)

    policy_state = load_router_policy_v6() or {}
    policy_symbols = policy_state.get("symbols")
    policy_df = pd.DataFrame()

    if isinstance(policy_symbols, list):
        policy_df = pd.DataFrame(
            [row for row in policy_symbols if isinstance(row, dict) and row.get("symbol")]
        )
    elif isinstance(policy_state, dict) and "per_symbol" in policy_state:
        policy_df = _to_dataframe(policy_state.get("per_symbol"))

    if not per_symbol.empty and not policy_df.empty:
        policy_df = policy_df.copy()
        policy_df["symbol"] = policy_df["symbol"].astype(str).str.upper()
        per_symbol = per_symbol.copy()
        if "symbol" in per_symbol.columns:
            per_symbol["symbol"] = per_symbol["symbol"].astype(str).str.upper()

        merge_cols = [c for c in ("maker_first", "bias", "quality", "allocator_state") if c in policy_df.columns]
        if merge_cols:
            right = policy_df[["symbol"] + merge_cols].drop_duplicates("symbol")
            per_symbol = per_symbol.merge(right, on="symbol", how="left")

    suggestions = load_router_suggestions_v6()
    overlays["policy_suggestions"] = suggestions

    if isinstance(suggestions, dict):
        summary["policy_stale"] = bool(suggestions.get("stale"))
        if "generated_at" in suggestions:
            summary["policy_generated_at"] = suggestions.get("generated_at")

    defaults = {
        "count": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "cum_pnl": 0.0,
        "fill_rate_pct": 0.0,
        "fees_total": 0.0,
        "realized_pnl": 0.0,
        "confidence_weighted_cum_pnl": 0.0,
        "rolling_sharpe_last": 0.0,
        "normalized_sharpe": 0.0,
        "volatility_scale": 1.0,
    }
    for key, val in defaults.items():
        summary.setdefault(key, val)
    if summary.get("count", 0) == 0 and not per_symbol.empty:
        try:
            summary["count"] = int(len(per_symbol))
        except Exception:
            summary["count"] = 0

    return RouterHealthData(
        per_symbol=per_symbol,
        trades=_to_dataframe(trades_snapshot.get("items")) if isinstance(trades_snapshot, dict) else pd.DataFrame(),
        pnl_curve=pnl_curve,
        summary=summary,
        overlays=overlays,
    )


__all__ = ["RouterHealthData", "load_router_health", "is_empty_router_health", "_load_order_events"]


def _load_order_events(limit: int = 0) -> Tuple[list, list, list]:
    """
    v6 compatibility stub for scripts/doctor.py.
    """
    _ = limit
    return ([], [], [])
