from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TAIL_MAX_BYTES = 128 * 1024
TAIL_MAX_LINES = 800
WINDOW_SECONDS = 86400.0
MAX_ITEMS = 200
SIGNAL_ITEMS_LIMIT = 500


@dataclass
class MirrorPayloads:
    router: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]


_TAIL_CACHE: Dict[Path, Tuple[float, int, List[Dict[str, Any]]]] = {}
_SNAPSHOT_CACHE: Tuple[
    Tuple[Tuple[float, int], Tuple[float, int], Tuple[float, int], Tuple[float, int]],
    MirrorPayloads,
] | None = None


def _file_signature(path: Path) -> Tuple[float, int]:
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except FileNotFoundError:
        return (0.0, 0)


def _load_tail_jsonl(path: Path, *, max_lines: int = TAIL_MAX_LINES) -> List[Dict[str, Any]]:
    sig = _file_signature(path)
    cached = _TAIL_CACHE.get(path)
    if cached and cached[0] == sig[0] and cached[1] == sig[1]:
        return cached[2]
    if sig[1] == 0:
        _TAIL_CACHE[path] = (sig[0], sig[1], [])
        return []
    try:
        with path.open("rb") as handle:
            size = sig[1]
            seek = max(0, size - TAIL_MAX_BYTES)
            handle.seek(seek)
            chunk = handle.read()
    except FileNotFoundError:
        _TAIL_CACHE.pop(path, None)
        return []
    lines = chunk.decode(errors="ignore").splitlines()
    records: List[Dict[str, Any]] = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    _TAIL_CACHE[path] = (sig[0], sig[1], records)
    return records


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _ts_from_record(record: Dict[str, Any]) -> Optional[float]:
    for key in ("ts", "time", "timestamp", "created_at", "t"):
        if key not in record:
            continue
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            val = float(value)
            if val > 1e12:
                val = val / 1000.0
            return val
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                continue
            try:
                if txt.endswith("Z"):
                    txt = txt[:-1] + "+00:00"
                return datetime.fromisoformat(txt).replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
    return None


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _summaries_from_records(
    attempts: Iterable[Dict[str, Any]],
    executed: Iterable[Dict[str, Any]],
    vetoes: Iterable[Dict[str, Any]],
    now_ts: float,
) -> Dict[str, Any]:
    attempts_24h = [row for row in attempts if _within_window(row, now_ts)]
    executed_24h = [row for row in executed if _within_window(row, now_ts)]
    vetoes_24h = [row for row in vetoes if _within_window(row, now_ts)]
    attempt_count = len(attempts_24h)
    executed_count = len(executed_24h)
    veto_count = len(vetoes_24h)
    fill_pct = (executed_count / attempt_count * 100.0) if attempt_count else None
    return {
        "kind": "summary",
        "window_sec": int(WINDOW_SECONDS),
        "attempted": attempt_count,
        "executed": executed_count,
        "vetoed": veto_count,
        "fill_rate_pct": round(fill_pct, 2) if fill_pct is not None else None,
        "updated": _iso(now_ts),
    }


def _within_window(record: Dict[str, Any], now_ts: float) -> bool:
    ts_val = _ts_from_record(record)
    if ts_val is None:
        return False
    return (now_ts - ts_val) <= WINDOW_SECONDS


def _normalize_exec(record: Dict[str, Any]) -> Dict[str, Any]:
    ts_val = _ts_from_record(record)
    qty = (
        _to_float(record.get("executed_qty"))
        or _to_float(record.get("executedQty"))
        or _to_float(record.get("qty"))
        or _to_float(record.get("quantity"))
    )
    price = (
        _to_float(record.get("avg_price"))
        or _to_float(record.get("price"))
        or _to_float(record.get("avgPrice"))
    )
    side = record.get("side") or record.get("position_side") or record.get("order_side")
    status = record.get("status") or record.get("order_status")
    latency = (
        _to_float(record.get("latency_ms"))
        or _to_float(record.get("latencyMs"))
        or _to_float(record.get("latency"))
    )
    return {
        "ts": _iso(ts_val),
        "symbol": (record.get("symbol") or "").upper(),
        "side": side,
        "qty": qty,
        "price": price,
        "status": status,
        "latency_ms": latency,
        "attempt_id": record.get("attempt_id"),
        "order_id": record.get("orderId") or record.get("order_id"),
        "client_order_id": record.get("clientOrderId") or record.get("client_order_id"),
    }


def _normalize_trade(record: Dict[str, Any]) -> Dict[str, Any]:
    ts_val = _ts_from_record(record)
    return {
        "ts": _iso(ts_val),
        "symbol": (record.get("symbol") or "").upper(),
        "side": record.get("side") or record.get("position_side"),
        "qty": (
            _to_float(record.get("executed_qty"))
            or _to_float(record.get("executedQty"))
            or _to_float(record.get("qty"))
        ),
        "price": (
            _to_float(record.get("avg_price"))
            or _to_float(record.get("price"))
            or _to_float(record.get("avgPrice"))
        ),
        "pnl_usd": _to_float(
            record.get("realized_pnl_usd")
            or record.get("pnl_usd")
            or record.get("pnl")
            or record.get("realizedPnlUsd")
        ),
        "attempt_id": record.get("attempt_id"),
        "client_order_id": record.get("clientOrderId") or record.get("client_order_id"),
        "order_id": record.get("orderId") or record.get("order_id"),
        "status": record.get("status"),
    }


def _normalize_signal(record: Dict[str, Any]) -> Dict[str, Any]:
    ts_val = _ts_from_record(record)
    doctor = record.get("doctor") if isinstance(record.get("doctor"), dict) else {}
    return {
        "ts": _iso(ts_val),
        "symbol": (record.get("symbol") or "").upper(),
        "side": record.get("signal") or record.get("side"),
        "timeframe": record.get("timeframe") or record.get("tf"),
        "confidence": _to_float(doctor.get("confidence") or record.get("confidence")),
        "attempt_id": record.get("attempt_id") or record.get("id"),
        "strategy": record.get("strategy"),
        "queue_depth": record.get("queue_depth"),
    }


def _doctor_ok(record: Dict[str, Any]) -> Optional[bool]:
    if isinstance(record.get("doctor"), dict):
        doctor_ok = record["doctor"].get("ok")
        if doctor_ok is not None:
            try:
                return bool(doctor_ok)
            except Exception:
                return None
    if "ok" in record:
        try:
            return bool(record.get("ok"))
        except Exception:
            return None
    return None


def build_signal_items_24h(log_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Produce up to 24h of signal metrics for Firestore publishing.

    Returns a list beginning with a summary row to keep downstream consumers alive.
    """
    base_dir = Path(log_root or Path("logs")).resolve()
    metrics_path = base_dir / "execution" / "signal_metrics.jsonl"
    now_ts = time.time()
    summary_ts = datetime.now(timezone.utc).isoformat()
    records = _load_tail_jsonl(metrics_path, max_lines=SIGNAL_ITEMS_LIMIT * 2)
    if not records:
        return [
            {
                "note": "signals_summary",
                "kind": "signals_summary",
                "attempted": 0,
                "executed": 0,
                "vetoed": 0,
                "ts": summary_ts,
            }
        ]

    sorted_records = sorted(records, key=_ts_from_record, reverse=True)
    recent_records = [row for row in sorted_records if _within_window(row, now_ts)]
    if not recent_records:
        recent_records = sorted_records[:SIGNAL_ITEMS_LIMIT]

    attempted = 0
    executed = 0
    vetoed = 0
    detail_items: List[Dict[str, Any]] = []
    for row in recent_records[:SIGNAL_ITEMS_LIMIT]:
        attempted += 1
        ok_flag = _doctor_ok(row)
        if ok_flag is True:
            executed += 1
        elif ok_flag is False:
            vetoed += 1
        doctor_info = row.get("doctor") if isinstance(row.get("doctor"), dict) else {}
        ts_iso = _iso(_ts_from_record(row))
        item = {
            "note": "signal_event",
            "symbol": (row.get("symbol") or "").upper(),
            "side": row.get("signal") or row.get("side"),
            "attempt_id": row.get("attempt_id") or row.get("id"),
            "ts": ts_iso,
            "ok": ok_flag,
            "confidence": _to_float(
                doctor_info.get("confidence") or row.get("confidence")
            ),
            "queue_depth": row.get("queue_depth"),
            "latency_ms": _to_float(row.get("latency_ms")),
        }
        reasons = doctor_info.get("reasons")
        if isinstance(reasons, list) and reasons:
            item["reasons"] = reasons
        detail_items.append(item)

    summary = {
        "note": "signals_summary",
        "kind": "signals_summary",
        "attempted": attempted,
        "executed": executed,
        "vetoed": vetoed,
        "ts": summary_ts,
    }
    return [summary, *detail_items]


def build_mirror_payloads(log_root: Optional[Path] = None) -> MirrorPayloads:
    """
    Build router/trade/signal payloads from local JSONL logs.
    Results are cached on file signature to keep executor loop lightweight.
    """
    global _SNAPSHOT_CACHE
    base_dir = Path(log_root or Path("logs")).resolve()
    exec_dir = base_dir / "execution"
    attempts_path = exec_dir / "orders_attempted.jsonl"
    executed_path = exec_dir / "orders_executed.jsonl"
    risk_path = exec_dir / "risk_vetoes.jsonl"
    signal_metrics_path = exec_dir / "signal_metrics.jsonl"

    signatures = (
        _file_signature(attempts_path),
        _file_signature(executed_path),
        _file_signature(risk_path),
        _file_signature(signal_metrics_path),
    )
    if _SNAPSHOT_CACHE and _SNAPSHOT_CACHE[0] == signatures:
        return _SNAPSHOT_CACHE[1]

    attempts_rows = _load_tail_jsonl(attempts_path)
    executed_rows = _load_tail_jsonl(executed_path)
    risk_rows = _load_tail_jsonl(risk_path)
    now_ts = time.time()

    if not attempts_rows:
        attempts_rows = _load_tail_jsonl(signal_metrics_path)
    else:
        extra_signals = _load_tail_jsonl(signal_metrics_path)
        if extra_signals:
            # merge unique attempt_ids, latest wins
            merged: Dict[str, Dict[str, Any]] = {}
            for row in attempts_rows + extra_signals:
                attempt_id = str(row.get("attempt_id") or row.get("id") or "")
                if not attempt_id:
                    key = f"{row.get('symbol')}:{_ts_from_record(row)}"
                else:
                    key = attempt_id
                merged[key] = row
            attempts_rows = list(merged.values())

    router_items: List[Dict[str, Any]] = []
    summary = _summaries_from_records(attempts_rows, executed_rows, risk_rows, now_ts)
    router_items.append(summary)
    recent_execs = [
        _normalize_exec(row)
        for row in sorted(executed_rows, key=_ts_from_record, reverse=True)
        if _within_window(row, now_ts)
    ]
    router_items.extend(recent_execs[:50])

    trade_candidates = [
        row
        for row in executed_rows
        if str(row.get("status") or "").upper() in {"FILLED", "PARTIALLY_FILLED", "FILLED_TRADE", "COMPLETED"}
        or row.get("event_type") in {"order_fill", "order_close"}
    ]
    trades = [
        _normalize_trade(row)
        for row in sorted(trade_candidates, key=_ts_from_record, reverse=True)
        if _within_window(row, now_ts)
    ][:MAX_ITEMS]

    signal_summary = dict(summary)
    signal_summary["kind"] = "signals_summary"
    normalized_signals = [
        _normalize_signal(row)
        for row in sorted(attempts_rows, key=_ts_from_record, reverse=True)
        if _within_window(row, now_ts)
    ]
    if not normalized_signals:
        # include last known metrics as-is even if outside window to keep doc alive
        for row in sorted(attempts_rows, key=_ts_from_record, reverse=True)[:MAX_ITEMS]:
            normalized_signals.append(_normalize_signal(row))
    signals = [signal_summary]
    signals.extend(normalized_signals[:MAX_ITEMS])

    payloads = MirrorPayloads(
        router=router_items[:MAX_ITEMS],
        trades=trades,
        signals=signals,
    )
    _SNAPSHOT_CACHE = (signatures, payloads)
    return payloads


__all__ = ["MirrorPayloads", "build_mirror_payloads", "build_signal_items_24h"]
