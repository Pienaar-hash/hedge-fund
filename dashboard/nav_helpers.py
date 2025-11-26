"""NAV helpers (v6/v7)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
SYNCED_STATE_PATH = Path(os.getenv("SYNCED_STATE_PATH") or (STATE_DIR / "synced_state.json"))
NAV_V7_PATH = Path(os.getenv("NAV_V7_PATH") or (STATE_DIR / "nav.json"))


def safe_float(x):
    """
    Convert x to float, or return None if conversion impossible.
    Accepts int, float, str, None.
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x2 = x.replace(",", "").strip()
            if not x2:
                return None
            return float(x2)
        return None
    except Exception:
        return None


def safe_round(x, nd: int = 2):
    xf = safe_float(x)
    if xf is None:
        return None
    return round(xf, nd)


def safe_format(x, nd: int = 2):
    xf = safe_float(x)
    if xf is None:
        return "–"
    return f"{xf:,.{nd}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _to_epoch_seconds(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:
            val /= 1000.0
        return val
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            if txt.isdigit():
                return _to_epoch_seconds(float(txt))
        except Exception:
            pass
        try:
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            return datetime.fromisoformat(txt).astimezone(timezone.utc).timestamp()
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def load_nav_state() -> Tuple[Dict[str, Any], str]:
    """
    v6: single canonical NAV state file: logs/state/nav_state.json
    Returns (payload, source_name).
    """
    payload = _load_json(NAV_STATE_PATH)
    return payload, NAV_STATE_PATH.name


def load_synced_state() -> Dict[str, Any]:
    """
    v6: synced_state.json mirrors executor state (nav, positions, caps).
    """
    return _load_json(SYNCED_STATE_PATH)


def load_nav_with_aum(state_dir: str | None = None) -> Dict[str, Any]:
    """
    Load nav.json (v7) with AUM block; returns {} on failure.
    """
    base_dir = Path(state_dir) if state_dir else STATE_DIR
    nav_path = base_dir / "nav.json"
    return _load_json(Path(nav_path))


def build_aum_slices(nav_snapshot: Dict[str, Any], usd_zar: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Convert nav['aum'] into chart-ready slices.
    """
    if not isinstance(nav_snapshot, dict):
        return []
    aum = nav_snapshot.get("aum") or {}
    if not isinstance(aum, dict):
        aum = {}
    slices: List[Dict[str, Any]] = []
    labels_seen = set()
    futures_val = safe_float(aum.get("futures"))
    futures_zar = futures_val * safe_float(usd_zar) if (usd_zar is not None and futures_val is not None) else None
    slices.append(
        {
            "label": "Futures",
            "value": futures_val if futures_val is not None else 0.0,
            "qty": None,
            "zar": futures_zar,
        }
    )
    labels_seen.add("Futures")

    off = aum.get("offexchange") if isinstance(aum, dict) else {}
    if isinstance(off, dict):
        for sym, payload in off.items():
            if not isinstance(payload, dict):
                continue
            label = str(sym).upper()
            labels_seen.add(label)
            usd_val = safe_float(payload.get("usd_value"))
            qty_val = safe_float(payload.get("qty"))
            zar_val = (usd_val * safe_float(usd_zar)) if (usd_zar is not None and usd_val is not None) else None
            slices.append(
                {
                    "label": label,
                    "value": usd_val if usd_val is not None else 0.0,
                    "qty": qty_val,
                    "zar": zar_val,
                }
            )

    # Optional: asset-level breakdown from nav_detail/asset_breakdown
    nav_detail = nav_snapshot.get("nav_detail") if isinstance(nav_snapshot.get("nav_detail"), dict) else {}
    asset_breakdown = nav_detail.get("asset_breakdown") if isinstance(nav_detail.get("asset_breakdown"), dict) else {}
    if not asset_breakdown and isinstance(nav_snapshot.get("assets"), dict):
        asset_breakdown = nav_snapshot.get("assets") or {}
    if isinstance(asset_breakdown, dict):
        for asset, usd_val_raw in asset_breakdown.items():
            label = str(asset).upper()
            if label in labels_seen:
                continue
            usd_val = safe_float(usd_val_raw)
            zar_val = (usd_val * safe_float(usd_zar)) if (usd_zar is not None and usd_val is not None) else None
            slices.append(
                {
                    "label": label,
                    "value": usd_val if usd_val is not None else 0.0,
                    "qty": None,
                    "zar": zar_val,
                }
            )
            labels_seen.add(label)
    return slices


def snapshot_age_seconds(payload: Dict[str, Any]) -> Optional[float]:
    """
    Compute age of a snapshot (nav.json or kpis_v7.json) in seconds,
    based on 'ts' / 'updated_ts' / similar fields.
    """
    if not isinstance(payload, dict) or not payload:
        return None

    candidates: List[Any] = []
    for key in ("updated_at", "updated_ts", "ts", "timestamp", "time"):
        if key in payload:
            candidates.append(payload.get(key))

    nested_nav = payload.get("nav") if isinstance(payload.get("nav"), dict) else {}
    if isinstance(nested_nav, dict):
        for key in ("updated_at", "updated_ts", "ts"):
            if key in nested_nav:
                candidates.append(nested_nav.get(key))

    now = time.time()
    for raw in candidates:
        ts_val = _to_epoch_seconds(raw)
        if ts_val is not None:
            return max(0.0, now - float(ts_val))

    return None


def nav_state_age_seconds(nav_state: Dict[str, Any]) -> Optional[float]:
    """
    Compute age of the nav_state snapshot in seconds, based on the freshest
    timestamp we can find in the document.
    """
    if not isinstance(nav_state, dict) or not nav_state:
        return None

    # Always calculate from updated_at first (more accurate than stored age_s)
    updated_at = nav_state.get("updated_at")
    if isinstance(updated_at, (int, float)):
        try:
            return max(0.0, time.time() - float(updated_at))
        except Exception:
            pass

    # Fallback to stored age_s only if updated_at not available
    age_val = nav_state.get("age_s")
    if isinstance(age_val, (int, float)) and age_val > 0:
        try:
            return float(age_val)
        except Exception:
            pass

    candidates: List[Any] = []
    for key in ("updated_at", "ts", "updated_ts"):
        if key in nav_state:
            candidates.append(nav_state.get(key))

    series = nav_state.get("series")
    if isinstance(series, list) and series:
        ts_candidates: List[Any] = []
        for entry in series:
            if not isinstance(entry, dict):
                continue
            if "t" in entry:
                ts_candidates.append(entry.get("t"))
        if ts_candidates:
            candidates.append(max(ts_candidates))

    now = time.time()
    for raw in candidates:
        ts_val = _to_epoch_seconds(raw)
        if ts_val is not None:
            return max(0.0, now - float(ts_val))

    return None


def signal_attempts_summary(lines: List[str]) -> str:
    """
    Compact screener tail summary for the Signals tab.
    """
    if not lines:
        return "No screener attempts recorded yet."

    attempted = 0
    emitted = 0
    submitted = 0

    for line in lines:
        if "attempted=" in line:
            attempted += 1
        if "emitted=" in line:
            emitted += 1
        if "submitted=" in line:
            submitted += 1

    parts: List[str] = []
    if attempted:
        parts.append(f"attempt lines={attempted}")
    if emitted:
        parts.append(f"emitted lines={emitted}")
    if submitted:
        parts.append(f"submitted lines={submitted}")

    if not parts:
        return f"{len(lines)} screener log lines."

    return " · ".join(parts)


__all__ = [
    "build_aum_slices",
    "snapshot_age_seconds",
    "signal_attempts_summary",
    "safe_float",
    "safe_round",
    "safe_format",
]
