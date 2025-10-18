"""Helpers for dashboard NAV and screener metrics."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.io.formats.style import Styler

from dashboard.dashboard_utils import fetch_state_document

_UNITS_FMT = "{:.6f}"
_USD_FMT = "{:,.2f}"
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV = os.getenv("ENV", "prod")
LOG_DIR = ROOT_DIR / "logs"
EXEC_LOG_DIR = LOG_DIR / "execution"


def _default_exec_stats() -> Dict[str, Any]:
    return {
        "attempted_24h": 0,
        "executed_24h": 0,
        "vetoes_24h": 0,
        "fill_rate": 0.0,
        "top_vetoes": [],
        "last_heartbeats": {
            "executor_live": None,
            "sync_daemon": None,
        },
    }


def _normalize_exec_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _default_exec_stats()
    if not isinstance(stats, dict):
        return base
    base.update(
        {
            "attempted_24h": stats.get("attempted_24h", base["attempted_24h"]),
            "executed_24h": stats.get("executed_24h", base["executed_24h"]),
            "vetoes_24h": stats.get("vetoes_24h", base["vetoes_24h"]),
            "fill_rate": stats.get("fill_rate", base["fill_rate"]),
            "top_vetoes": stats.get("top_vetoes", base["top_vetoes"]) or [],
            "last_heartbeats": stats.get("last_heartbeats", base["last_heartbeats"]) or base["last_heartbeats"],
        }
    )
    # Ensure heartbeat keys present
    hb = base["last_heartbeats"]
    if "executor_live" not in hb:
        hb["executor_live"] = None
    if "sync_daemon" not in hb:
        hb["sync_daemon"] = None
    # Top veto normalization
    top = []
    for entry in base["top_vetoes"]:
        if not isinstance(entry, dict):
            continue
        reason = str(entry.get("reason", "unknown"))
        try:
            count = int(entry.get("count", 0))
        except Exception:
            continue
        top.append({"reason": reason, "count": count})
    base["top_vetoes"] = top[:5]
    return base


def _safe_fetch_state(doc: str) -> Dict[str, Any]:
    try:
        return fetch_state_document(doc, env=ENV)
    except Exception:
        return {}


def _load_exec_stats_from_firestore() -> Optional[Dict[str, Any]]:
    for doc_name in ("positions", "nav"):
        doc = _safe_fetch_state(doc_name)
        if doc:
            stats = doc.get("exec_stats")
            if isinstance(stats, dict):
                return stats
    return None


def _tail_exec_stats_from_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return None
    for line in reversed(lines[-5000:]):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict):
            stats = record.get("exec_stats")
            if isinstance(stats, dict):
                return stats
    return None


def _parse_timestamp(record: Dict[str, Any]) -> Optional[float]:
    for key in ("ts", "timestamp", "time", "t", "local_ts"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            try:
                return float(value)
            except ValueError:
                try:
                    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
                    dt = datetime.fromisoformat(cleaned)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except Exception:
                    continue
    return None


def _iter_recent_records(path: Path, cutoff: float):
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                ts = _parse_timestamp(record)
                if ts is None or ts < cutoff:
                    continue
                record["_ts"] = ts
                yield record
    except FileNotFoundError:
        return
    except Exception:
        return


def _last_heartbeats_from_logs() -> Dict[str, Optional[str]]:
    path = EXEC_LOG_DIR / "sync_heartbeats.jsonl"
    latest: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                svc = record.get("service")
                if not svc:
                    continue
                ts = _parse_timestamp(record)
                if ts is None:
                    continue
                latest[svc] = max(ts, latest.get(svc, float("-inf")))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    result = {}
    for svc in ("executor_live", "sync_daemon"):
        ts = latest.get(svc)
        result[svc] = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts is not None else None
        )
    return result


def _aggregate_exec_stats_from_logs() -> Dict[str, Any]:
    cutoff = time.time() - 86400.0
    attempted = 0
    executed = 0
    successful = 0
    veto_counts: Counter[str] = Counter()

    attempt_path = EXEC_LOG_DIR / "orders_attempted.jsonl"
    for _ in _iter_recent_records(attempt_path, cutoff) or []:
        attempted += 1

    executed_path = EXEC_LOG_DIR / "orders_executed.jsonl"
    for record in _iter_recent_records(executed_path, cutoff) or []:
        executed += 1
        status = str(record.get("status") or record.get("order_status") or "").upper()
        if status in {"FILLED", "SUCCESS"}:
            successful += 1

    veto_path = EXEC_LOG_DIR / "risk_vetoes.jsonl"
    for record in _iter_recent_records(veto_path, cutoff) or []:
        reason = str(record.get("veto_reason") or record.get("reason") or "unknown")
        veto_counts[reason] += 1

    denominator = attempted if attempted > 0 else (executed if executed > 0 else 0)
    fill_rate = (successful / denominator) if denominator else 0.0

    top_vetoes = [{"reason": reason, "count": count} for reason, count in veto_counts.most_common(5)]
    heartbeats = _last_heartbeats_from_logs()

    return {
        "attempted_24h": attempted,
        "executed_24h": executed,
        "vetoes_24h": sum(veto_counts.values()),
        "fill_rate": fill_rate,
        "top_vetoes": top_vetoes,
        "last_heartbeats": heartbeats,
    }


def load_exec_stats() -> Dict[str, Any]:
    """Return execution statistics for dashboard use."""
    stats = _load_exec_stats_from_firestore()
    if stats:
        return _normalize_exec_stats(stats)

    for fallback in ("positions.jsonl", "nav.jsonl", "treasury.jsonl"):
        local_stats = _tail_exec_stats_from_jsonl(LOG_DIR / fallback)
        if local_stats:
            return _normalize_exec_stats(local_stats)

    aggregated = _aggregate_exec_stats_from_logs()
    return _normalize_exec_stats(aggregated)


def heartbeat_badge(age_seconds: Optional[float]) -> str:
    """Return an HTML badge indicating heartbeat freshness."""
    base_style = (
        "display:inline-block;padding:0.2em 0.6em;border-radius:0.6em;"
        "font-size:0.75rem;font-weight:600;color:#ffffff;"
    )
    if age_seconds is None or (isinstance(age_seconds, float) and age_seconds != age_seconds):
        color = "#7f8c8d"
        text = "n/a"
    else:
        age = float(age_seconds)
        if age <= 60:
            color = "#2ecc71"
        elif age <= 180:
            color = "#f39c12"
        else:
            color = "#e74c3c"

        if age < 60:
            text = f"{int(age)}s"
        elif age < 3600:
            text = f"{int(age // 60)}m"
        else:
            text = f"{int(age // 3600)}h"
    return f'<span style="{base_style}background:{color};">{text}</span>'


def treasury_table_from_summary(summary: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of treasury assets from compute_nav_summary output.

    Expected summary shape:
      {
        "details": {
            "treasury": {
                "treasury": {
                    "BTC": {"qty": 0.1, "val_usdt": 1000.0, ...},
                    ...
                }
            }
        }
      }
    """

    treasury_detail = (summary.get("details") or {}).get("treasury", {})
    holdings = treasury_detail.get("treasury", {}) if isinstance(treasury_detail, dict) else {}
    rows = []
    for asset, info in holdings.items():
        try:
            qty = float(info.get("qty", 0.0) or 0.0)
            usd = float(info.get("val_usdt", 0.0) or 0.0)
        except Exception:
            continue
        rows.append({"Asset": str(asset), "Units": qty, "USD Value": usd})

    if not rows:
        return pd.DataFrame(columns=["Asset", "Units", "USD Value"])

    df = pd.DataFrame(rows)
    df = df.sort_values(by="USD Value", ascending=False).reset_index(drop=True)
    return df


def format_treasury_table(df: pd.DataFrame) -> Styler:
    """Format treasury DataFrame for display."""

    fmt: Dict[str, str] = {}
    if "Units" in df.columns:
        fmt["Units"] = _UNITS_FMT
    if "Qty" in df.columns:
        fmt["Qty"] = _UNITS_FMT
    if "USD Value" in df.columns:
        fmt["USD Value"] = _USD_FMT
    return df.style.format(fmt)


def signal_attempts_summary(lines: List[str]) -> str:
    """Return latest screener attempted/emitted summary string."""

    for line in reversed(lines):
        if "attempted=" not in line or "emitted=" not in line:
            continue
        attempted = _extract_int(line, "attempted")
        emitted = _extract_int(line, "emitted")
        if attempted is None or emitted is None:
            continue
        pct = (emitted / attempted * 100.0) if attempted else 0.0
        pct_display = f" ({pct:.0f}%)" if attempted else ""
        return f"Signals: {attempted} attempted, {emitted} emitted{pct_display}"
    return "Signals: N/A"


def _extract_int(text: str, key: str) -> int | None:
    needle = f"{key}="
    start = text.find(needle)
    if start == -1:
        return None
    start += len(needle)
    end = start
    while end < len(text) and text[end].isdigit():
        end += 1
    try:
        return int(text[start:end])
    except Exception:
        return None


__all__ = [
    "treasury_table_from_summary",
    "format_treasury_table",
    "signal_attempts_summary",
    "load_exec_stats",
    "heartbeat_badge",
]
