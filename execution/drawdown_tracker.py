from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any, Dict, Mapping, Optional, Sequence

PEAK_STATE_PATH = os.path.join("logs", "cache", "peak_state.json")
LOGGER = logging.getLogger("drawdown_tracker")


@dataclass
class NavAnomalyConfig:
    """Config for NAV anomaly detection."""
    enabled: bool = False
    max_multiplier_intraday: float = 3.0
    max_gap_abs_usd: float = 20000.0


@dataclass
class PortfolioDDState:
    """Portfolio drawdown state for circuit breaker checks."""
    current_dd_pct: float  # e.g. 0.032 for 3.2%
    peak_nav_usd: float
    latest_nav_usd: float


def load_nav_anomaly_config(cfg: Optional[Mapping[str, Any]] = None) -> NavAnomalyConfig:
    """
    Resolve NavAnomalyConfig from risk config or provided block.

    Accepts either the full risk config (expects `nav_anomalies` key)
    or the nav_anomalies sub-block directly.
    """
    default = NavAnomalyConfig()
    block: Mapping[str, Any] = {}
    if cfg is None:
        try:
            from execution.risk_loader import load_risk_config

            cfg = load_risk_config()
        except Exception:
            cfg = {}
    if isinstance(cfg, Mapping):
        candidate = cfg.get("nav_anomalies") if isinstance(cfg.get("nav_anomalies"), Mapping) else None
        if candidate is None and cfg:
            candidate = cfg if isinstance(cfg, Mapping) else {}
        if isinstance(candidate, Mapping):
            block = candidate

    try:
        enabled = bool(block.get("enabled", default.enabled))
    except Exception:
        enabled = default.enabled

    def _cfg_float(key: str, fallback: float) -> float:
        try:
            val = float(block.get(key, fallback))
            return val if val > 0 else fallback
        except Exception:
            return fallback

    max_multiplier = _cfg_float("max_multiplier_intraday", default.max_multiplier_intraday)
    max_gap = _cfg_float("max_gap_abs_usd", default.max_gap_abs_usd)
    return NavAnomalyConfig(
        enabled=enabled,
        max_multiplier_intraday=max_multiplier,
        max_gap_abs_usd=max_gap,
    )


def is_nav_anomalous(previous_peak: float, new_nav: float, cfg: NavAnomalyConfig) -> bool:
    """
    Returns True if new_nav is likely bogus relative to previous peak.
    """
    if not cfg or not cfg.enabled:
        return False
    try:
        prev = float(previous_peak)
        nav_val = float(new_nav)
    except Exception:
        return False
    if prev <= 0:
        return False
    if nav_val > prev * cfg.max_multiplier_intraday:
        return True
    if nav_val - prev > cfg.max_gap_abs_usd:
        return True
    return False


def get_portfolio_dd_state(nav_history: Sequence[float]) -> Optional[PortfolioDDState]:
    """
    Given a sequence of NAV observations (in USD), compute:
    - peak NAV
    - latest NAV
    - current drawdown percentage from peak (as fraction, e.g. 0.10 = 10%)

    Return None if nav_history is empty or invalid.
    """
    if not nav_history:
        return None

    # Filter out non-positive values
    valid_navs = [n for n in nav_history if isinstance(n, (int, float)) and n > 0]
    if not valid_navs:
        return None

    peak_nav_usd = max(valid_navs)
    latest_nav_usd = valid_navs[-1]

    if peak_nav_usd <= 0:
        return PortfolioDDState(
            current_dd_pct=0.0,
            peak_nav_usd=0.0,
            latest_nav_usd=latest_nav_usd,
        )

    current_dd_pct = (peak_nav_usd - latest_nav_usd) / peak_nav_usd

    return PortfolioDDState(
        current_dd_pct=max(0.0, current_dd_pct),  # Clamp to non-negative
        peak_nav_usd=peak_nav_usd,
        latest_nav_usd=latest_nav_usd,
    )


def _ensure_dir() -> None:
    try:
        os.makedirs(os.path.dirname(PEAK_STATE_PATH), exist_ok=True)
    except Exception:
        pass


def _as_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
        number = float(value)
        if not math.isfinite(number):
            return 0.0
        return number
    except Exception:
        return 0.0


def _resolve_timezone(name: str | None) -> tzinfo:
    if not name or str(name).upper() == "UTC":
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return timezone.utc
    try:
        return ZoneInfo(str(name))
    except Exception:
        return timezone.utc


def load_peak_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback = dict(default or {})
    try:
        with open(PEAK_STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return fallback


def save_peak_state(state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    payload = dict(state)
    payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("ts", time.time())
    _ensure_dir()
    try:
        with open(PEAK_STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    except Exception:
        pass


def compute_intraday_drawdown(
    nav_usd: Any,
    realized_pnl_usd_today: Any,
    *,
    reset_timezone: str = "UTC",
    now: Optional[datetime] = None,
    state: Optional[Dict[str, Any]] = None,
    nav_anomaly_cfg: Optional[NavAnomalyConfig] = None,
) -> Dict[str, Any]:
    base_state = dict(state or load_peak_state())
    tz = _resolve_timezone(reset_timezone)
    now_dt = now.astimezone(tz) if now is not None else datetime.now(tz)
    day_key = now_dt.strftime("%Y-%m-%d")
    prev_day = base_state.get("day") or base_state.get("date")

    nav_value = _as_float(nav_usd)
    stored_nav = _as_float(base_state.get("nav") or base_state.get("nav_usd"))
    if nav_value <= 0.0 and stored_nav > 0.0:
        nav_value = stored_nav

    realized_today = _as_float(realized_pnl_usd_today)
    if realized_today == 0.0:
        realized_cached = _as_float(base_state.get("realized_pnl_today"))
        if realized_cached != 0.0:
            realized_today = realized_cached

    previous_peak = _as_float(base_state.get("peak") or base_state.get("peak_equity"))
    anomaly_cfg = nav_anomaly_cfg or load_nav_anomaly_config()
    peak_candidate = max(nav_value, previous_peak, 0.0)
    if previous_peak > 0 and nav_value > previous_peak and is_nav_anomalous(previous_peak, nav_value, anomaly_cfg):
        LOGGER.warning(
            "nav_anomaly_detected prev_peak=%s nav=%s max_multiplier=%s max_gap_abs_usd=%s",
            previous_peak,
            nav_value,
            anomaly_cfg.max_multiplier_intraday,
            anomaly_cfg.max_gap_abs_usd,
        )
        peak = previous_peak
    else:
        peak = peak_candidate

    if peak <= 0.0:
        dd_abs = 0.0
        dd_pct = 0.0
    else:
        dd_abs = max(0.0, peak - max(nav_value, 0.0))
        dd_pct = (dd_abs / peak) * 100.0 if peak > 0 else 0.0

    updated = {
        "day": day_key,
        "peak": peak,
        "peak_equity": peak,
        "nav": nav_value,
        "nav_usd": nav_value,
        "realized_pnl_today": realized_today,
        "dd_abs": dd_abs,
        "dd_pct": dd_pct,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ts": time.time(),
        "reset_timezone": reset_timezone or "UTC",
    }
    return updated


def mirror_peak_state_to_firestore(
    state: Dict[str, Any],
    db: Any,
    *,
    env: Optional[str] = None,
) -> None:
    if not isinstance(state, dict) or not state:
        return
    if db is None:
        return
    if getattr(db, "_is_noop", False):
        return
    try:
        env_name = env or os.getenv("ENV", "dev")
        if str(env_name).lower() == "prod":
            allow = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
            if allow not in {"1", "true", "yes"}:
                raise RuntimeError("drawdown_tracker refuses to write with ENV=prod without ALLOW_PROD_WRITE=1")
        doc = (
            db.collection("hedge")
            .document(env_name)
            .collection("risk")
            .document("peak_state")
        )
        payload = dict(state)
        payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("ts", time.time())
        doc.set(payload, merge=True)
    except Exception:
        pass


__all__ = [
    "NavAnomalyConfig",
    "PortfolioDDState",
    "compute_intraday_drawdown",
    "get_portfolio_dd_state",
    "is_nav_anomalous",
    "load_peak_state",
    "load_nav_anomaly_config",
    "mirror_peak_state_to_firestore",
    "save_peak_state",
]
