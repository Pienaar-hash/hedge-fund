from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

DEFAULT_HISTORY = 500
SIGNAL_METRICS_PATH = Path("logs/execution/signal_metrics.jsonl")
ORDER_METRICS_PATH = Path("logs/execution/order_metrics.jsonl")


def _tail_jsonl(path: Path, limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e12:
            seconds = seconds / 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception:
            return None
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
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


@dataclass
class RouterHealthData:
    trades: pd.DataFrame
    per_symbol: pd.DataFrame
    pnl_curve: pd.DataFrame
    summary: Dict[str, Any]


def _join_signal_metadata(signals: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    for row in signals:
        attempt_id = row.get("attempt_id")
        if not attempt_id:
            continue
        attempt = str(attempt_id)
        doctor = row.get("doctor") if isinstance(row.get("doctor"), Mapping) else {}
        enriched[attempt] = {
            "symbol": row.get("symbol"),
            "signal": row.get("signal"),
            "doctor_confidence": _to_float((doctor or {}).get("confidence")),
            "ts": _to_datetime(row.get("ts")),
        }
    return enriched


def _normalize_trade(row: Mapping[str, Any], meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pnl_keys = (
        "pnl_at_close_usd",
        "realized_pnl_usd",
        "pnl_usd",
        "pnl",
    )
    pnl_val: Optional[float] = None
    for key in pnl_keys:
        pnl_val = _to_float(row.get(key))
        if pnl_val is not None:
            break
    if pnl_val is None:
        return None
    attempt_id = str(row.get("attempt_id") or "")
    intent_id = str(row.get("intent_id") or "")
    symbol = str(row.get("symbol") or meta.get("symbol") or "").upper()
    ts = _to_datetime(row.get("ts")) or meta.get("ts")
    if not symbol or ts is None:
        return None
    confidence = _to_float(meta.get("doctor_confidence"))
    return {
        "time": ts,
        "symbol": symbol,
        "pnl_usd": pnl_val,
        "attempt_id": attempt_id,
        "intent_id": intent_id,
        "signal": meta.get("signal"),
        "doctor_confidence": confidence,
    }


def load_router_health(
    window: int = DEFAULT_HISTORY,
    *,
    signal_path: Path | None = None,
    order_path: Path | None = None,
) -> RouterHealthData:
    sig_path = signal_path or SIGNAL_METRICS_PATH
    ord_path = order_path or ORDER_METRICS_PATH
    signal_rows = _tail_jsonl(sig_path, window * 2)
    order_rows = _tail_jsonl(ord_path, window * 2)
    signal_meta = _join_signal_metadata(signal_rows)

    trades: list[Dict[str, Any]] = []
    for row in order_rows:
        event = str(row.get("event") or "").lower()
        if event and event != "position_close":
            continue
        attempt_id = str(row.get("attempt_id") or "")
        meta = signal_meta.get(attempt_id, {})
        normalized = _normalize_trade(row, meta)
        if normalized is None:
            continue
        trades.append(normalized)

    trades_df = pd.DataFrame.from_records(trades)
    if trades_df.empty:
        per_symbol_df = pd.DataFrame(columns=["symbol", "count", "win_rate", "avg_pnl", "cum_pnl", "sharpe"])
        pnl_curve_df = pd.DataFrame(columns=["time", "cum_pnl", "hit_rate"])
        summary = {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "cum_pnl": 0.0}
        return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary)

    trades_df = trades_df.sort_values("time", ascending=True).reset_index(drop=True)
    trades_df["is_win"] = trades_df["pnl_usd"] > 0
    trades_df["trade_index"] = trades_df.index + 1
    trades_df["cum_pnl"] = trades_df["pnl_usd"].cumsum()
    trades_df["cum_wins"] = trades_df["is_win"].cumsum()
    trades_df["hit_rate"] = trades_df["cum_wins"] / trades_df["trade_index"]

    pnl_curve_df = trades_df[["time", "cum_pnl", "hit_rate"]].copy()

    per_symbol_records: list[Dict[str, Any]] = []
    for symbol, frame in trades_df.groupby("symbol"):
        count = int(len(frame))
        wins = int(frame["is_win"].sum())
        win_rate = (wins / count) * 100.0 if count else 0.0
        avg_pnl = float(frame["pnl_usd"].mean()) if count else 0.0
        cum_pnl = float(frame["pnl_usd"].sum())
        std = float(frame["pnl_usd"].std(ddof=1)) if count > 1 else 0.0
        sharpe = 0.0
        if std > 0 and count > 1:
            sharpe = (avg_pnl / std) * math.sqrt(count)
        conf_series = frame["doctor_confidence"].dropna()
        avg_conf = float(conf_series.mean()) if not conf_series.empty else None
        per_symbol_records.append(
            {
                "symbol": symbol,
                "count": count,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "cum_pnl": cum_pnl,
                "sharpe": sharpe,
                "avg_confidence": avg_conf,
            }
        )

    per_symbol_df = pd.DataFrame(per_symbol_records)
    per_symbol_df.sort_values("cum_pnl", ascending=False, inplace=True, ignore_index=True)

    summary = {
        "count": int(len(trades_df)),
        "wins": int(trades_df["is_win"].sum()),
        "win_rate": float(trades_df["is_win"].mean() * 100.0),
        "avg_pnl": float(trades_df["pnl_usd"].mean()),
        "cum_pnl": float(trades_df["pnl_usd"].sum()),
        "last_trade_at": trades_df["time"].iloc[-1],
    }

    return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary)


__all__ = ["load_router_health", "RouterHealthData"]
