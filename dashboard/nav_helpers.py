"""Helpers for dashboard NAV and screener metrics."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.io.formats.style import Styler

_UNITS_FMT = "{:.6f}"
_USD_FMT = "{:,.2f}"
ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.getenv("STATE_DIR") or ROOT_DIR / "logs" / "state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or STATE_DIR / "nav.json")
POSITIONS_STATE_PATH = Path(os.getenv("POSITIONS_STATE_PATH") or STATE_DIR / "positions.json")


def _load_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _default_exec_stats() -> Dict[str, Any]:
    return {
        "attempted_24h": 0,
        "executed_24h": 0,
        "fill_rate": 0.0,
        "vetoes_24h": 0,
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


def _load_exec_stats_from_state() -> Optional[Dict[str, Any]]:
    for path in (POSITIONS_STATE_PATH, NAV_STATE_PATH):
        doc = _load_json_file(path)
        if isinstance(doc, dict):
            stats = doc.get("exec_stats")
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


def load_exec_stats() -> Dict[str, Any]:
    """Return execution statistics sourced from v6 state files."""
    stats = _load_exec_stats_from_state()
    return _normalize_exec_stats(stats)


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


def build_nav_dataframe() -> Tuple[pd.DataFrame, float, str]:
    """
    Build a NAV DataFrame from the v6 nav state file.
    """

    entries: List[Dict[str, Any]] = []

    nav_state = _load_json_file(NAV_STATE_PATH)
    if isinstance(nav_state, dict):
        series = nav_state.get("series") if isinstance(nav_state.get("series"), list) else []
        for record in series or []:
            if not isinstance(record, dict):
                continue
            ts = _parse_timestamp(record)
            nav_val = _coerce_float(
                record.get("equity")
                or record.get("nav")
                or record.get("total_equity")
            )
            if ts is None or nav_val is None:
                continue
            entries.append(
                {
                    "ts": ts,
                    "nav_usd": nav_val,
                    "source": "nav_state",
                    "fresh": True,
                }
            )
        if not entries:
            ts = _parse_timestamp({"ts": nav_state.get("updated_at")})
            nav_val = _coerce_float(
                nav_state.get("total_equity")
                or nav_state.get("portfolio_gross_usd")
                or nav_state.get("nav")
            )
            if ts is not None and nav_val is not None:
                entries.append(
                    {
                        "ts": ts,
                        "nav_usd": nav_val,
                        "source": "nav_state",
                        "fresh": True,
                    }
                )

    if entries:
        df = pd.DataFrame(entries)
        df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
        df["nav_usd"] = pd.to_numeric(df["nav_usd"], errors="coerce")
        df = df[df["nav_usd"].notna()]
    else:
        df = pd.DataFrame(columns=["ts", "nav_usd", "source", "fresh"])

    if df.empty:
        fallback_ts = time.time()
        df = pd.DataFrame(
            [
                {
                    "ts": fallback_ts,
                    "nav_usd": 0.0,
                    "source": "placeholder",
                    "fresh": False,
                }
            ]
        )
        latest_nav = 0.0
        label = "using cached NAV"
    else:
        df = df.sort_values("ts").reset_index(drop=True)
        latest = df.iloc[-1]
        latest_nav = float(latest["nav_usd"])
        label = ""

    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df[["ts", "datetime", "nav_usd", "source", "fresh"]]

    return df, latest_nav, label


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
    "signal_attempts_summary",
    "load_exec_stats",
    "heartbeat_badge",
    "build_nav_dataframe",
]
