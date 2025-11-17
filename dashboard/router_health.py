from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import os

import pandas as pd

from execution.metrics_normalizer import confidence_weighted_cumsum, compute_normalized_metrics, rolling_sharpe

DEFAULT_HISTORY = 500
SIGNAL_METRICS_PATH = Path("logs/execution/signal_metrics.jsonl")
ORDER_METRICS_PATH = Path("logs/execution/order_metrics.jsonl")
ORDER_EVENTS_PATH = Path("logs/execution/orders_executed.jsonl")
STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
LOG = logging.getLogger("dash.router")


def _load_state_router_health(path: Optional[Path] = None) -> Optional[RouterHealthData]:
    path = path or ROUTER_HEALTH_STATE_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        return None
    now_iso = datetime.utcnow().isoformat()
    per_symbol_df = pd.DataFrame.from_records(symbols) if symbols else pd.DataFrame()
    trades_df = pd.DataFrame(columns=["time", "symbol", "pnl_usd"])
    pnl_df = pd.DataFrame(columns=["time", "cum_pnl"])
    summary = {
        "updated_ts": payload.get("updated_ts"),
        "generated_at": now_iso,
    }
    return RouterHealthData(
        trades=trades_df,
        per_symbol=per_symbol_df,
        pnl_curve=pnl_df,
        summary=summary,
        overlays={},
    )


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
            seconds /= 1000.0
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
    overlays: Dict[str, pd.DataFrame] = field(default_factory=dict)


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


def _event_identifier(record: Mapping[str, Any]) -> str:
    order_id = record.get("orderId") or record.get("order_id")
    client_id = record.get("clientOrderId") or record.get("client_order_id")
    if order_id:
        return str(order_id)
    if client_id:
        return str(client_id)
    return f"anon_{id(record)}"


def _normalize_fill_event(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_id": record.get("attempt_id"),
        "intent_id": record.get("intent_id"),
        "order_id": record.get("orderId"),
        "client_order_id": record.get("clientOrderId"),
        "symbol": str(record.get("symbol") or "").upper(),
        "side": record.get("side"),
        "executed_qty": _to_float(record.get("executedQty")),
        "avg_price": _to_float(record.get("avgPrice")),
        "status": record.get("status"),
        "fee_total": _to_float(record.get("fee_total")),
        "fee_asset": record.get("feeAsset"),
        "latency_ms": _to_float(record.get("latency_ms")),
        "ts_fill_first": _to_datetime(record.get("ts_fill_first")),
        "ts_fill_last": _to_datetime(record.get("ts_fill_last")),
        "identifier": _event_identifier(record),
    }


def _normalize_close_event(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_id": record.get("attempt_id"),
        "intent_id": record.get("intent_id"),
        "order_id": record.get("orderId"),
        "client_order_id": record.get("clientOrderId"),
        "symbol": str(record.get("symbol") or "").upper(),
        "realized_pnl_usd": _to_float(record.get("realizedPnlUsd")),
        "fees_total": _to_float(record.get("fees_total")),
        "position_before": _to_float(record.get("position_size_before")),
        "position_after": _to_float(record.get("position_size_after")),
        "ts_close": _to_datetime(record.get("ts_close")),
        "identifier": _event_identifier(record),
    }


def _load_order_events(limit: int = 1000) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not ORDER_EVENTS_PATH.exists():
        return [], [], []
    records = _tail_jsonl(ORDER_EVENTS_PATH, limit)
    acks_raw: list[Dict[str, Any]] = []
    fills_raw: list[Dict[str, Any]] = []
    closes_raw: list[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        event = str(record.get("event_type") or record.get("event") or "").lower()
        if event == "order_ack":
            acks_raw.append(dict(record))
        elif event == "order_fill":
            fills_raw.append(_normalize_fill_event(record))
        elif event == "order_close":
            closes_raw.append(_normalize_close_event(record))
        else:  # legacy fall-through
            fills_raw.append(_normalize_fill_event(record))
    return acks_raw, fills_raw, closes_raw


def _normalize_trade(row: Mapping[str, Any], meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pnl_keys = (
        "pnl_at_close_usd",
        "realized_pnl_usd",
        "realizedPnlUsd",
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
    snapshot: Optional[Mapping[str, Any]] = None,
    trades_snapshot: Optional[Mapping[str, Any]] = None,
) -> RouterHealthData:
    if signal_path is None and order_path is None and snapshot is None and trades_snapshot is None:
        state_view = _load_state_router_health()
        if state_view is not None:
            return state_view
    sig_path = signal_path or SIGNAL_METRICS_PATH
    ord_path = order_path or ORDER_METRICS_PATH
    signal_rows = _tail_jsonl(sig_path, window * 2)
    signal_meta = _join_signal_metadata(signal_rows)

    snapshot_items: List[Dict[str, Any]] = []
    snapshot_summary: Optional[Dict[str, Any]] = None
    if snapshot and isinstance(snapshot.get("items"), list):
        for item in snapshot.get("items", []):
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "summary" and snapshot_summary is None:
                snapshot_summary = dict(item)
            else:
                snapshot_items.append(dict(item))

    trades_snapshot_items: List[Dict[str, Any]] = []
    if trades_snapshot and isinstance(trades_snapshot.get("items"), list):
        for item in trades_snapshot.get("items", []):
            if isinstance(item, dict):
                trades_snapshot_items.append(dict(item))

    metrics_rows = _tail_jsonl(ord_path, window * 2)
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for row in metrics_rows:
        attempt = row.get("attempt_id") or row.get("attemptId")
        if attempt:
            metrics_map[str(attempt)] = row

    ack_events, fill_events, close_events = _load_order_events(window * 4)

    trades_records: list[Dict[str, Any]] = []
    fees_by_symbol: Dict[str, float] = {}
    for close in close_events:
        symbol = close.get("symbol")
        ts_close = close.get("ts_close")
        pnl_val = close.get("realized_pnl_usd")
        if not symbol or ts_close is None or pnl_val is None:
            continue
        attempt_id = str(close.get("attempt_id") or "")
        meta = signal_meta.get(attempt_id, {})
        record = {
            "time": ts_close,
            "symbol": symbol,
            "pnl_usd": float(pnl_val),
            "attempt_id": attempt_id,
            "intent_id": close.get("intent_id"),
            "signal": meta.get("signal"),
            "doctor_confidence": meta.get("doctor_confidence"),
            "fees_total": close.get("fees_total"),
        }
        trades_records.append(record)
        fees_val = close.get("fees_total")
        if fees_val is not None:
            fees_by_symbol[symbol] = fees_by_symbol.get(symbol, 0.0) + float(fees_val or 0.0)

    ack_symbol_ids: Dict[str, set[str]] = defaultdict(set)
    for ack in ack_events:
        symbol = str(ack.get("symbol") or "").upper()
        if not symbol:
            continue
        ack_symbol_ids[symbol].add(_event_identifier(ack))

    fill_symbol_ids: Dict[str, set[str]] = defaultdict(set)
    latency_by_symbol: Dict[str, List[float]] = defaultdict(list)
    slippage_by_symbol: Dict[str, List[float]] = defaultdict(list)
    for fill in fill_events:
        symbol = fill.get("symbol")
        if not symbol:
            continue
        identifier = fill.get("identifier")
        fill_symbol_ids[symbol].add(identifier)
        if fill.get("latency_ms") is not None:
            latency_by_symbol[symbol].append(float(fill["latency_ms"]))
        if fill.get("fee_total") is not None:
            fees_by_symbol[symbol] = fees_by_symbol.get(symbol, 0.0) + float(fill["fee_total"] or 0.0)
        attempt_id = str(fill.get("attempt_id") or "")
        if attempt_id and attempt_id in metrics_map:
            prices = metrics_map[attempt_id].get("prices") or {}
            mark = _to_float(prices.get("mark"))
            fill_price = fill.get("avg_price")
            if mark not in (None, 0) and fill_price not in (None, 0):
                slip = ((float(fill_price) - float(mark)) / float(mark)) * 10_000.0
                if str(fill.get("side") or "").upper() == "SELL":
                    slip *= -1.0
                slippage_by_symbol[symbol].append(slip)

    if trades_records:
        trades_df = pd.DataFrame.from_records(trades_records)
        if "attempt_id" in trades_df.columns:
            trades_df = trades_df.drop_duplicates(subset=["attempt_id"], keep="last")
        trades_df = trades_df.sort_values("time", ascending=True).reset_index(drop=True)
        trades_df["is_win"] = trades_df["pnl_usd"] > 0
        trades_df["trade_index"] = trades_df.index + 1
        trades_df["cum_pnl"] = trades_df["pnl_usd"].cumsum()
        trades_df["cum_wins"] = trades_df["is_win"].cumsum()
        trades_df["hit_rate"] = trades_df["cum_wins"] / trades_df["trade_index"]
        if "doctor_confidence" not in trades_df.columns:
            trades_df["doctor_confidence"] = pd.NA
        trades_df["doctor_confidence"] = pd.to_numeric(trades_df["doctor_confidence"], errors="coerce")

        confidence_vals = trades_df["doctor_confidence"].tolist()
        confidence_weighted = confidence_weighted_cumsum(trades_df["pnl_usd"].tolist(), confidence_vals)
        trades_df["confidence_weighted_cum_pnl"] = confidence_weighted

        sharpe_window = min(30, max(3, len(trades_df) // 4 or 3))
        rolling_sharpe_series = rolling_sharpe(trades_df["pnl_usd"], window=sharpe_window)
        pnl_curve_df = trades_df[["time", "cum_pnl", "hit_rate"]].copy()
        pnl_curve_df["confidence_weighted_cum_pnl"] = trades_df["confidence_weighted_cum_pnl"]
        pnl_curve_df["rolling_sharpe"] = rolling_sharpe_series.fillna(0.0)

        rolling_conf_window = min(20, max(3, len(trades_df) // 4 or 3))
        confidence_series = trades_df["doctor_confidence"].ffill().fillna(0.5)
        confidence_curve_df = pd.DataFrame(
            {
                "time": trades_df["time"],
                "confidence": trades_df["doctor_confidence"],
                "rolling_confidence": confidence_series.rolling(
                    window=rolling_conf_window, min_periods=1
                ).mean(),
            }
        )

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
            frame_conf = pd.to_numeric(frame["doctor_confidence"], errors="coerce")
            conf_series = frame_conf.dropna()
            avg_conf = float(conf_series.mean()) if not conf_series.empty else None
            ack_count = len(ack_symbol_ids.get(symbol, set()))
            fill_count = len(fill_symbol_ids.get(symbol, set()))
            fill_rate_pct = (fill_count / ack_count * 100.0) if ack_count else None
            latency_vals = latency_by_symbol.get(symbol, [])
            latency_med = float(pd.Series(latency_vals).median()) if latency_vals else None
            slippage_vals = slippage_by_symbol.get(symbol, [])
            slippage_med = float(pd.Series(slippage_vals).median()) if slippage_vals else None
            fees_total = fees_by_symbol.get(symbol, 0.0)
            normalized = compute_normalized_metrics(
                frame["pnl_usd"].to_numpy(dtype=float),
                target_vol=1.0,
                annualization=max(count, 2),
                window=count,
            )
            conf_weights = frame_conf.fillna(0.5)
            confidence_weighted_pnl = float((frame["pnl_usd"] * conf_weights).sum())
            per_symbol_records.append(
                {
                    "symbol": symbol,
                    "count": count,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "cum_pnl": cum_pnl,
                    "sharpe": sharpe,
                    "avg_confidence": avg_conf,
                    "ack_count": ack_count,
                    "fill_count": fill_count,
                    "fill_rate_pct": fill_rate_pct,
                    "median_latency_ms": latency_med,
                    "median_slippage_bps": slippage_med,
                    "fees_total": fees_total,
                    "realized_pnl": cum_pnl,
                    "confidence_weighted_pnl": confidence_weighted_pnl,
                    "normalized_sharpe": normalized.normalized_sharpe,
                    "volatility_scale": normalized.volatility_scale,
                }
            )

        per_symbol_df = pd.DataFrame.from_records(per_symbol_records)
        overall_norm = compute_normalized_metrics(
            trades_df["pnl_usd"].to_numpy(dtype=float),
            target_vol=1.0,
            annualization=max(len(trades_df), 2),
            window=len(trades_df),
        )
        avg_confidence_series = trades_df["doctor_confidence"].dropna()
        avg_conf = float(avg_confidence_series.mean()) if not avg_confidence_series.empty else None
        latest_conf_weighted = float(trades_df["confidence_weighted_cum_pnl"].iloc[-1])
        rolling_sharpe_last = float(pnl_curve_df["rolling_sharpe"].dropna().iloc[-1]) if not pnl_curve_df.empty else 0.0
        all_latencies = [val for values in latency_by_symbol.values() for val in values]
        all_slippages = [val for values in slippage_by_symbol.values() for val in values]
        total_acks = sum(len(values) for values in ack_symbol_ids.values())
        total_fills = sum(len(values) for values in fill_symbol_ids.values())
        summary = {
            "count": int(len(trades_df)),
            "win_rate": float(trades_df["is_win"].mean()) * 100.0 if not trades_df.empty else 0.0,
            "avg_pnl": float(trades_df["pnl_usd"].mean()) if not trades_df.empty else 0.0,
            "cum_pnl": float(trades_df["pnl_usd"].sum()),
            "fill_rate_pct": (total_fills / total_acks * 100.0) if total_acks else 0.0,
            "median_latency_ms": float(pd.Series(all_latencies).median()) if all_latencies else None,
            "median_slippage_bps": float(pd.Series(all_slippages).median()) if all_slippages else None,
            "fees_total": sum(fees_by_symbol.values()),
            "realized_pnl": float(trades_df["pnl_usd"].sum()),
            "avg_confidence": avg_conf,
            "confidence_weighted_cum_pnl": latest_conf_weighted,
            "normalized_sharpe": overall_norm.normalized_sharpe,
            "volatility_scale": overall_norm.volatility_scale,
            "rolling_sharpe_last": rolling_sharpe_last,
        }
        if snapshot_summary:
            summary["mirror_summary"] = snapshot_summary
        overlays = {
            "confidence": confidence_curve_df,
            "rolling_sharpe": pnl_curve_df[["time", "rolling_sharpe"]].copy(),
        }
        if snapshot_items:
            overlays["mirror_execs"] = pd.DataFrame(snapshot_items)
        return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary, overlays)


    try:
        import inspect

        _ROUTER_LINES, _ROUTER_START = inspect.getsourcelines(load_router_health)
        LOG.info(
            "[runbook] router_health readers: %s:%d-%d",
            __file__,
            _ROUTER_START,
            _ROUTER_START + len(_ROUTER_LINES) - 1,
        )
    except Exception:
        pass

    # Fallback to legacy order_metrics-based data
    trades: list[Dict[str, Any]] = []
    for row in metrics_rows:
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
    if not trades_df.empty and "attempt_id" in trades_df.columns:
        trades_df = trades_df.drop_duplicates(subset=["attempt_id"], keep="last")
    if trades_df.empty:
        per_symbol_df = pd.DataFrame(
            columns=[
                "symbol",
                "count",
                "win_rate",
                "avg_pnl",
                "cum_pnl",
                "sharpe",
                "avg_confidence",
                "confidence_weighted_pnl",
                "normalized_sharpe",
                "volatility_scale",
            ]
        )
        pnl_curve_df = pd.DataFrame(columns=["time", "cum_pnl", "hit_rate", "confidence_weighted_cum_pnl", "rolling_sharpe"])
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
        if snapshot_summary:
            summary["mirror_summary"] = snapshot_summary
        overlays = {
            "confidence": pd.DataFrame(columns=["time", "confidence", "rolling_confidence"]),
            "rolling_sharpe": pd.DataFrame(columns=["time", "rolling_sharpe"]),
        }
        if snapshot_items:
            overlays["mirror_execs"] = pd.DataFrame(snapshot_items)
        if trades_snapshot_items:
            trades_df = pd.DataFrame(trades_snapshot_items)
            return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary, overlays)
        return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary, overlays)
    else:
        trades_df = trades_df.sort_values("time", ascending=True).reset_index(drop=True)
        trades_df["is_win"] = trades_df["pnl_usd"] > 0
        trades_df["trade_index"] = trades_df.index + 1
        trades_df["cum_pnl"] = trades_df["pnl_usd"].cumsum()
        trades_df["cum_wins"] = trades_df["is_win"].cumsum()
        trades_df["hit_rate"] = trades_df["cum_wins"] / trades_df["trade_index"]
        if "doctor_confidence" not in trades_df.columns:
            trades_df["doctor_confidence"] = pd.NA
        trades_df["doctor_confidence"] = pd.to_numeric(trades_df["doctor_confidence"], errors="coerce")
        trades_df["confidence_weighted_cum_pnl"] = confidence_weighted_cumsum(
            trades_df["pnl_usd"].tolist(),
            trades_df["doctor_confidence"].tolist(),
        )

        sharpe_window = min(30, max(3, len(trades_df) // 4 or 3))
        rolling_sharpe_series = rolling_sharpe(trades_df["pnl_usd"], window=sharpe_window)
        pnl_curve_df = trades_df[["time", "cum_pnl", "hit_rate"]].copy()
        pnl_curve_df["confidence_weighted_cum_pnl"] = trades_df["confidence_weighted_cum_pnl"]
        pnl_curve_df["rolling_sharpe"] = rolling_sharpe_series.fillna(0.0)

        confidence_series = trades_df["doctor_confidence"].ffill().fillna(0.5)
        confidence_curve_df = pd.DataFrame(
            {
                "time": trades_df["time"],
                "confidence": trades_df["doctor_confidence"],
                "rolling_confidence": confidence_series.rolling(
                    window=sharpe_window,
                    min_periods=1,
                ).mean(),
            }
        )

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
            frame_conf = pd.to_numeric(frame.get("doctor_confidence"), errors="coerce")
            conf_series = frame_conf.dropna()
            avg_conf = float(conf_series.mean()) if not conf_series.empty else None
            normalized = compute_normalized_metrics(
                frame["pnl_usd"].to_numpy(dtype=float),
                target_vol=1.0,
                annualization=max(count, 2),
                window=count,
            )
            conf_weights = frame_conf.fillna(0.5)
            confidence_weighted_pnl = float((frame["pnl_usd"] * conf_weights).sum())
            per_symbol_records.append(
                {
                    "symbol": symbol,
                    "count": count,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "cum_pnl": cum_pnl,
                    "sharpe": sharpe,
                    "avg_confidence": avg_conf,
                    "confidence_weighted_pnl": confidence_weighted_pnl,
                    "normalized_sharpe": normalized.normalized_sharpe,
                    "volatility_scale": normalized.volatility_scale,
                }
            )

        per_symbol_df = pd.DataFrame.from_records(per_symbol_records)
        overall_norm = compute_normalized_metrics(
            trades_df["pnl_usd"].to_numpy(dtype=float),
            target_vol=1.0,
            annualization=max(len(trades_df), 2),
            window=len(trades_df),
        )
        avg_confidence_series = trades_df["doctor_confidence"].dropna()
        avg_conf = float(avg_confidence_series.mean()) if not avg_confidence_series.empty else None
        latest_conf_weighted = float(trades_df["confidence_weighted_cum_pnl"].iloc[-1])
        rolling_sharpe_last = float(pnl_curve_df["rolling_sharpe"].dropna().iloc[-1]) if not pnl_curve_df.empty else 0.0
        summary = {
            "count": int(len(trades_df)),
            "win_rate": float(trades_df["is_win"].mean()) * 100.0 if not trades_df.empty else 0.0,
            "avg_pnl": float(trades_df["pnl_usd"].mean()) if not trades_df.empty else 0.0,
            "cum_pnl": float(trades_df["pnl_usd"].sum()),
            "fill_rate_pct": 0.0,
            "fees_total": 0.0,
            "realized_pnl": float(trades_df["pnl_usd"].sum()),
            "avg_confidence": avg_conf,
            "confidence_weighted_cum_pnl": latest_conf_weighted,
            "normalized_sharpe": overall_norm.normalized_sharpe,
            "volatility_scale": overall_norm.volatility_scale,
            "rolling_sharpe_last": rolling_sharpe_last,
        }
        overlays = {
            "confidence": confidence_curve_df,
            "rolling_sharpe": pnl_curve_df[["time", "rolling_sharpe"]].copy(),
        }
        return RouterHealthData(trades_df, per_symbol_df, pnl_curve_df, summary, overlays)
