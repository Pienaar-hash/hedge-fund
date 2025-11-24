"""
v5.9 Execution Hardening — Risk gates
- Per-symbol 7d notional share cap
- Per-symbol daily drawdown kill with cooldown
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, getcontext
import math
from typing import Dict, List, Tuple, Optional, Any, Mapping, Sequence
import json
import logging
import os
import time

from execution.drawdown_tracker import load_peak_state, save_peak_state
from execution.nav import (
    compute_trading_nav,
    compute_gross_exposure_usd,
    get_confirmed_nav,
    get_nav_age as _nav_get_nav_age,
    nav_health_snapshot,
)
from execution.utils import load_json, get_live_positions
from execution.universe_resolver import symbol_min_gross, universe_by_symbol
from .utils.metrics import (
    notional_7d_by_symbol,
    total_notional_7d,
    dd_today_pct,
    is_in_asset_universe,
)
from .utils.toggle import disable_symbol_temporarily, is_symbol_disabled
from execution.exchange_utils import get_balances, is_dry_run
from execution.log_utils import get_logger, log_event, safe_dump
from execution.risk_loader import load_risk_config, load_symbol_caps, normalize_percentage

LOGGER = logging.getLogger("risk_limits")

DEFAULT_SYMBOL_SHARE_CAP = 0.25  # 25% of 7d notional
DEFAULT_SYMBOL_DD_CAP_PCT = 3.0
COOLDOWN_H = 24
REASON_TRADE_EQUITY_CAP = "trade_gt_equity_cap"
REASON_MAX_TRADE_NAV = "max_trade_nav_pct"


_GLOBAL_KEYS = {
    "daily_loss_limit_pct",
    "cooldown_minutes_after_stop",
    "max_trades_per_symbol_per_hour",
    "drawdown_alert_pct",
    "max_nav_drawdown_pct",
    "max_gross_exposure_pct",
    "max_portfolio_gross_nav_pct",
    "max_symbol_exposure_pct",
    "min_notional_usdt",
    "max_trade_nav_pct",
    "max_concurrent_positions",
    "burst_limit",
    "error_circuit",
    "whitelist",
    "nav_freshness_seconds",
    "fail_closed_on_nav_stale",
}

LOG_VETOES = get_logger("logs/execution/risk_vetoes.jsonl")

_NAV_SNAPSHOT_PATHS: List[str] = [
    "logs/cache/nav_confirmed.json",
    "logs/nav_log.json",
    "logs/cache/peak_state.json",
    "cache/nav_confirmed.json",  # legacy fallbacks
    "cache/nav_log.json",
    "cache/peak_state.json",
]

DEFAULT_NAV_FRESHNESS_SECONDS = int(os.environ.get("NAV_FRESHNESS_SECONDS", "90"))
DEFAULT_FAIL_CLOSED_ON_NAV_STALE = os.environ.get("FAIL_CLOSED_ON_NAV_STALE", "1") != "0"
PEAK_STATE_MAX_AGE_SEC = int(os.environ.get("PEAK_STATE_MAX_AGE_SEC", str(24 * 3600)))
DEFAULT_PEAK_STALE_SECONDS = int(os.environ.get("PEAK_STALE_SECONDS", "600") or 600)
DEFAULT_TRADE_EQUITY_NAV_PCT = 15.0


def get_nav_freshness_snapshot() -> Tuple[Optional[float], bool]:
    """
    Return (age_seconds, sources_ok) using the confirmed NAV snapshot.
    Falls back to legacy mtime heuristics when the snapshot is absent.
    """
    health = nav_health_snapshot()
    if health.get("age_s") is not None:
        return float(health.get("age_s") or 0.0), bool(health.get("sources_ok"))
    if health.get("fresh") is False:
        return None, bool(health.get("sources_ok"))
    now = time.time()
    snapshot_candidates = [
        "logs/cache/nav_confirmed.json",
        "cache/nav_confirmed.json",
    ]
    try:
        for path in snapshot_candidates:
            if not os.path.exists(path):
                continue
            age: Optional[float] = None
            sources_ok = True
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
                ts_val = payload.get("ts")
                try:
                    ts_float = float(ts_val)
                    if ts_float > 0.0:
                        age = max(0.0, now - ts_float)
                except Exception:
                    age = None
                sources_ok = bool(payload.get("sources_ok", True))
            except Exception as exc:
                LOGGER.error("[risk] nav_snapshot_read_failed: %s", exc)
                payload = {}
                sources_ok = False
            if age is None:
                try:
                    age = max(0.0, now - os.path.getmtime(path))
                except Exception:
                    age = None
            if age is not None:
                return age, sources_ok

        newest_mtime = 0.0
        for rel_path in _NAV_SNAPSHOT_PATHS:
            if not os.path.exists(rel_path):
                continue
            try:
                newest_mtime = max(newest_mtime, os.path.getmtime(rel_path))
            except Exception:
                continue
        if newest_mtime > 0.0:
            return max(0.0, now - newest_mtime), True
        fallback = _nav_get_nav_age()
        if fallback is not None:
            try:
                return max(0.0, float(fallback)), True
            except Exception:
                return None, False
        return None, False
    except Exception as exc:
        LOGGER.error("[risk] get_nav_freshness_snapshot_failed: %s", exc)
        return None, False


def get_nav_age() -> Optional[float]:
    """Backward-compatible helper returning only the NAV age."""
    age, _ = get_nav_freshness_snapshot()
    return age


def is_nav_fresh(nav_dict: Optional[Dict[str, Any]] = None, threshold_s: Optional[int] = None) -> bool:
    """Return True if NAV data is fresh; threshold defaults to mandatory failsafe."""
    threshold = DEFAULT_NAV_FRESHNESS_SECONDS
    if threshold_s is not None:
        try:
            threshold = int(threshold_s)
        except Exception:
            threshold = DEFAULT_NAV_FRESHNESS_SECONDS
    if threshold <= 0:
        threshold = DEFAULT_NAV_FRESHNESS_SECONDS

    age = None
    sources_ok = None
    if isinstance(nav_dict, dict):
        age = nav_dict.get("age")
        sources_ok = nav_dict.get("sources_ok")
    snap_age, snap_sources_ok = get_nav_freshness_snapshot()
    if age is None:
        age = snap_age
    if sources_ok is None:
        sources_ok = snap_sources_ok

    if age is None:
        LOGGER.warning("[risk] nav_age=n/a -> stale (no snapshot)")
        return False

    sources_ok_bool = bool(sources_ok)
    fresh = (age < threshold) and sources_ok_bool
    LOGGER.info(
        "[risk] nav_age=%.1fs threshold=%ss sources_ok=%s fresh=%s",
        age,
        threshold,
        sources_ok_bool,
        fresh,
    )
    return fresh


def enforce_nav_freshness_or_veto(risk_ctx: Dict[str, Any], nav_dict: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """
    Returns True if NAV meets freshness guardrails; logs veto and returns False when stale.
    `risk_ctx` can provide symbol/strategy/signal_ts/qty/detail/context for veto emission.
    """
    threshold_raw = cfg.get("nav_freshness_seconds")
    try:
        threshold = int(threshold_raw)
    except Exception:
        try:
            threshold = int(float(threshold_raw or 0))
        except Exception:
            threshold = DEFAULT_NAV_FRESHNESS_SECONDS
    if threshold <= 0:
        threshold = DEFAULT_NAV_FRESHNESS_SECONDS

    fail_closed_cfg = cfg.get("fail_closed_on_nav_stale")
    fail_closed = (
        bool(fail_closed_cfg)
        if fail_closed_cfg is not None
        else DEFAULT_FAIL_CLOSED_ON_NAV_STALE
    )

    age = nav_dict.get("age")
    sources_ok = nav_dict.get("sources_ok")
    if age is None or sources_ok is None:
        snap_age, snap_sources_ok = get_nav_freshness_snapshot()
        if age is None:
            age = snap_age
        if sources_ok is None:
            sources_ok = snap_sources_ok

    if age is None:
        LOGGER.warning("[risk] nav_age=n/a -> stale (no snapshot)")
        fresh = False
    else:
        sources_ok_bool = bool(sources_ok)
        fresh = (age < threshold) and sources_ok_bool
        LOGGER.info(
            "[risk] nav_age=%.1fs threshold=%ss sources_ok=%s fresh=%s",
            age,
            threshold,
            sources_ok_bool,
            fresh,
        )

    if fresh:
        return True

    detail_payload = dict(risk_ctx.get("detail") or {})
    if not detail_payload:
        detail_payload = {"thresholds": {"nav_freshness_seconds": threshold}}
    thresholds_payload = detail_payload.setdefault("thresholds", {})
    if "nav_freshness_seconds" not in thresholds_payload:
        thresholds_payload["nav_freshness_seconds"] = threshold
    reasons_payload = detail_payload.setdefault("reasons", [])
    if "nav_stale" not in reasons_payload:
        reasons_payload.append("nav_stale")
    observations_payload = detail_payload.setdefault("observations", {})
    if age is not None:
        try:
            observations_payload.setdefault("nav_age_s", int(max(0.0, float(age))))
        except Exception:
            observations_payload.setdefault("nav_age_s", None)
    observations_payload.setdefault(
        "sources_ok",
        bool(sources_ok) if sources_ok is not None else None,
    )

    context_payload = dict(risk_ctx.get("context") or {})
    if age is not None and "nav_age" not in context_payload:
        context_payload["nav_age"] = age
    context_payload.setdefault("nav_freshness_seconds", threshold)
    context_payload.setdefault(
        "sources_ok",
        bool(sources_ok) if sources_ok is not None else None,
    )

    _emit_veto(
        risk_ctx.get("symbol"),
        "nav_stale",
        detail=detail_payload,
        context=context_payload,
        strategy=risk_ctx.get("strategy"),
        signal_ts=risk_ctx.get("signal_ts"),
        qty=risk_ctx.get("qty"),
    )
    return not fail_closed

REASONS = {
    "kill_switch_triggered": "kill_switch",
    "min_notional": "min_notional",
    "below_min_notional": "min_notional",
    "exceeds_per_trade_cap": "per_trade_cap",
    "exceeds_leverage_cap": "leverage_cap",
    "exceeds_open_notional_cap": "open_notional_cap",
    "too_many_positions": "position_limit",
    "invalid_notional": "invalid_notional",
    "cooldown": "cooldown",
    "daily_loss_limit": "daily_loss",
    "day_loss_limit": "daily_loss",
    "trade_gt_max_trade_nav_pct": "max_trade_nav",
    "trade_gt_10pct_equity": "max_trade_nav",
    REASON_TRADE_EQUITY_CAP: "max_trade_nav",
    REASON_MAX_TRADE_NAV: "max_trade_nav",
    "symbol_cap": "symbol_cap",
    "portfolio_cap": "portfolio_cap",
    "max_gross_nav_pct": "portfolio_cap",
    "trade_rate_limit": "trade_rate_limit",
    "not_whitelisted": "whitelist",
    "circuit_breaker": "circuit_breaker",
    "burst_limit": "burst_limit",
    "side_blocked": "side_blocked",
    "leverage_exceeded": "leverage_cap",
    "max_concurrent": "max_concurrent",
    "tier_cap": "tier_cap",
    "nav_stale": "nav_stale",
}


def _emit_veto(
    symbol: Any,
    reason: str,
    *,
    detail: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    strategy: Optional[str] = None,
    signal_ts: Any = None,
    qty: Any = None,
) -> None:
    try:
        payload = {
            "symbol": symbol,
            "strategy": strategy,
            "veto_reason": REASONS.get(reason, reason or "unknown"),
            "original_reason": reason,
            "veto_detail": detail or {},
            "signal_ts": signal_ts,
            "qty_req": qty,
            "context": context or {},
        }
        log_event(LOG_VETOES, "risk_veto", safe_dump(payload))
    except Exception:
        pass


def _normalize_pct(value: Any) -> float:
    return normalize_percentage(value)


def _normalize_risk_cfg(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Ensure risk config exposes `global` and `per_symbol` sections."""
    if not isinstance(cfg, dict):
        return {"global": {}, "per_symbol": {}}

    out = dict(cfg)

    g = out.get("global")
    if not isinstance(g, dict):
        g = {}
    # hoist known globals if they were at top-level legacy locations
    for key in _GLOBAL_KEYS:
        if key in out and key not in g:
            g[key] = out[key]
    out["global"] = g

    per_symbol = out.get("per_symbol")
    if not isinstance(per_symbol, dict):
        per_symbol = {}
    out["per_symbol"] = per_symbol

    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _peak_from_nav_log(limit: int = 200) -> float:
    path = "logs/nav_log.json"
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return 0.0
    series = data[-limit:] if isinstance(data, list) else [data]
    peak_val = 0.0
    for entry in series:
        if not isinstance(entry, dict):
            continue
        try:
            nav_val = _as_float(entry.get("nav") or entry.get("nav_usd"))
            peak_val = max(peak_val, nav_val)
        except Exception:
            continue
    return peak_val


def _drawdown_snapshot(g_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    state = load_peak_state()
    if not isinstance(state, dict):
        state = {}
    confirmed = get_confirmed_nav()
    nav_cache = confirmed if isinstance(confirmed, dict) else {}
    cached_nav = _as_float(nav_cache.get("nav") or nav_cache.get("nav_usd"))
    cache_ts = _as_float(nav_cache.get("ts"))
    nav_health = nav_health_snapshot()
    nav_sources_ok = bool(nav_health.get("sources_ok", True))
    nav_fresh = bool(nav_health.get("fresh"))

    dd_pct = 0.0
    dd_abs = 0.0
    peak = _as_float(state.get("peak") or state.get("peak_equity") or state.get("peak_nav"))
    nav = _as_float(state.get("nav") or state.get("nav_usd") or cached_nav)
    realized = _as_float(state.get("realized_pnl_today"))

    now = time.time()
    nav_cache_age = None
    if cache_ts > 0.0:
        nav_cache_age = max(0.0, now - cache_ts)

    if nav <= 0.0 and cached_nav > 0.0:
        nav = cached_nav

    try:
        stale_threshold = float((g_cfg or {}).get("peak_stale_seconds", DEFAULT_PEAK_STALE_SECONDS) or DEFAULT_PEAK_STALE_SECONDS)
    except Exception:
        stale_threshold = DEFAULT_PEAK_STALE_SECONDS
    if stale_threshold <= 0:
        stale_threshold = DEFAULT_PEAK_STALE_SECONDS
    peak_ts = _as_float(state.get("peak_ts") or state.get("ts"))
    reset_on_testnet = bool((g_cfg or {}).get("reset_peak_on_testnet"))
    testnet = os.getenv("BINANCE_TESTNET", "0").strip().lower() in {"1", "true", "yes", "on"}
    if reset_on_testnet and testnet and nav > 0.0:
        peak = nav
        peak_ts = now
        state = {
            "peak_nav": nav,
            "peak_ts": now,
            "daily_peak": nav,
            "daily_ts": now,
            "nav": nav,
            "nav_usd": nav,
            "peak": nav,
            "peak_equity": nav,
            "ts": now,
            "updated_at": now,
        }
        save_peak_state(state)

    is_stale = stale_threshold > 0 and (peak_ts <= 0.0 or (now - peak_ts) > stale_threshold)
    if (peak <= 0.0 or is_stale) and nav_sources_ok:
        regenerated = _peak_from_nav_log()
        if regenerated > 0.0:
            peak = regenerated
            peak_ts = now
            state.update(
                {
                    "peak_nav": peak,
                    "peak_ts": now,
                    "daily_peak": peak,
                    "daily_ts": now,
                    "peak": peak,
                    "peak_equity": peak,
                    "ts": now,
                    "stale_healed": True,
                }
            )
            save_peak_state(state)
        elif nav > 0.0 and peak <= 0.0:
            peak = nav
            peak_ts = now

    if peak > 0.0 and nav > 0.0:
        dd_abs = max(0.0, peak - nav)
        dd_pct = (dd_abs / peak) * 100.0 if peak > 0 else 0.0

    daily_peak = _as_float(state.get("daily_peak") or peak)
    if daily_peak > 0.0 and nav > 0.0:
        daily_loss_pct = max(0.0, (daily_peak - nav) / daily_peak * 100.0)
    else:
        daily_loss_pct = 0.0

    state_ts = _as_float(state.get("ts"))
    stale_age = None
    if state_ts > 0.0:
        stale_age = max(0.0, now - state_ts)
        if stale_age > PEAK_STATE_MAX_AGE_SEC:
            LOGGER.warning(
                "[risk] drawdown_state_stale age=%.0fs limit=%ss",
                stale_age,
                PEAK_STATE_MAX_AGE_SEC,
            )
    stale_flags: Dict[str, bool] = {}
    nav_age = nav_health.get("age_s")
    if stale_age is not None and stale_age > PEAK_STATE_MAX_AGE_SEC:
        stale_flags["peak_state_stale"] = True
    if nav <= 0.0 or peak <= 0.0:
        stale_flags["nav_missing"] = True
    if not nav_sources_ok:
        stale_flags["nav_sources_unhealthy"] = True
    try:
        if nav_age is not None and float(nav_age) > DEFAULT_NAV_FRESHNESS_SECONDS:
            stale_flags["nav_cache_stale"] = True
    except Exception:
        pass
    if nav_cache_age is not None and nav_cache_age > DEFAULT_NAV_FRESHNESS_SECONDS:
        stale_flags["nav_cache_stale"] = True
    if not nav_fresh:
        stale_flags.setdefault("nav_cache_stale", True)
    usable = nav > 0.0 and peak > 0.0 and not any(stale_flags.values())
    assets = {}
    detail = nav_cache.get("detail")
    if isinstance(detail, dict):
        assets = detail.get("assets") or {}

    return {
        "dd_pct": dd_pct,
        "dd_abs": dd_abs,
        "peak": peak,
        "nav": nav,
        "realized_today": realized,
        "nav_cache_ts": cache_ts,
        "nav_cache_age": nav_cache_age,
        "peak_state_age": stale_age,
        "stale_flags": stale_flags,
        "usable": usable,
        "nav_health": nav_health,
        "peak_state": state,
        "drawdown": {
            "pct": dd_pct,
            "abs": dd_abs,
            "peak_nav": peak,
            "nav": nav,
        },
        "daily_loss": {
            "pct": daily_loss_pct,
            "daily_peak": daily_peak,
            "nav": nav,
        },
        "assets": assets,
    }

getcontext().prec = 28


@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_trade: float  # e.g., 200.0 USDT
    max_open_notional: float  # e.g., 1000.0 USDT
    max_positions: int  # e.g., 5
    max_leverage: float  # e.g., 5.0
    kill_switch_drawdown_pct: float  # e.g., -10.0 (portfolio)
    min_notional: float = 10.0  # exchange min


class RiskState:
    """Holds rolling state the executor can update each loop.

    Extended with lightweight fields/methods to support risk checks in `check_order`.
    """

    _MAX_STORED_EVENTS = 256

    def __init__(self, snapshot: Mapping[str, Any] | None = None) -> None:
        self.open_notional: float = 0.0
        self.open_positions: int = 0
        self.portfolio_drawdown_pct: float = 0.0
        # New fields for cooldown/circuit breaker support
        self._last_fill_by_symbol: Dict[str, float] = {}
        self._error_timestamps: List[float] = []
        # Attempt timestamps for burst control
        self._order_attempt_ts: List[float] = []
        # Optional daily PnL percent (negative means loss)
        self.daily_pnl_pct: float = 0.0
        if snapshot:
            self.apply_snapshot(snapshot)

    # --- Optional helpers used by check_order ---
    def note_fill(self, symbol: str, ts: float) -> None:
        self._last_fill_by_symbol[str(symbol)] = float(ts)

    def last_fill_ts(self, symbol: str) -> float:
        return float(self._last_fill_by_symbol.get(str(symbol), 0.0) or 0.0)

    def note_error(self, ts: float) -> None:
        self._error_timestamps.append(float(ts))

    def errors_in(self, window_sec: int, now: float) -> int:
        cutoff = float(now) - max(float(window_sec or 0), 0.0)
        kept = [t for t in self._error_timestamps if t >= cutoff]
        self._error_timestamps = kept
        return len(kept)

    def note_attempt(self, ts: float) -> None:
        self._order_attempt_ts.append(float(ts))

    def attempts_in(self, window_sec: int, now: float) -> int:
        cutoff = float(now) - max(float(window_sec or 0), 0.0)
        kept = [t for t in self._order_attempt_ts if t >= cutoff]
        self._order_attempt_ts = kept
        return len(kept)

    # --- Persistence helpers -------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of runtime counters."""
        return {
            "open_notional": float(self.open_notional),
            "open_positions": int(self.open_positions),
            "portfolio_drawdown_pct": float(self.portfolio_drawdown_pct),
            "daily_pnl_pct": float(self.daily_pnl_pct),
            "last_fill_by_symbol": dict(self._last_fill_by_symbol),
            "error_timestamps": list(self._error_timestamps)[-self._MAX_STORED_EVENTS :],
            "order_attempt_ts": list(self._order_attempt_ts)[-self._MAX_STORED_EVENTS :],
        }

    def apply_snapshot(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore counters from a previous snapshot."""
        if not isinstance(snapshot, Mapping):
            return

        def _coerce_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _maybe_float(value: Any) -> Optional[float]:
            try:
                numeric = float(value)
            except Exception:
                return None
            if not math.isfinite(numeric):
                return None
            return numeric

        self.open_notional = _coerce_float(snapshot.get("open_notional"), self.open_notional)
        self.open_positions = int(_coerce_float(snapshot.get("open_positions"), self.open_positions))
        self.portfolio_drawdown_pct = _coerce_float(
            snapshot.get("portfolio_drawdown_pct"), self.portfolio_drawdown_pct
        )
        self.daily_pnl_pct = _coerce_float(snapshot.get("daily_pnl_pct"), self.daily_pnl_pct)

        last_fill = snapshot.get("last_fill_by_symbol")
        if isinstance(last_fill, Mapping):
            restored: Dict[str, float] = {}
            for key, value in last_fill.items():
                sym = str(key).upper()
                restored[sym] = _coerce_float(value, 0.0)
            self._last_fill_by_symbol = restored

        errors = snapshot.get("error_timestamps")
        if isinstance(errors, Sequence):
            restored_errors = []
            for ts in errors:
                numeric = _maybe_float(ts)
                if numeric is None:
                    continue
                restored_errors.append(numeric)
            self._error_timestamps = restored_errors[-self._MAX_STORED_EVENTS :]

        attempts = snapshot.get("order_attempt_ts")
        if isinstance(attempts, Sequence):
            restored_attempts = []
            for ts in attempts:
                numeric = _maybe_float(ts)
                if numeric is None:
                    continue
                restored_attempts.append(numeric)
            self._order_attempt_ts = restored_attempts[-self._MAX_STORED_EVENTS :]

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any] | None) -> "RiskState":
        """Factory helper used by tests/new services."""
        return cls(snapshot=snapshot)


def _can_open_position_legacy(
    symbol: str, notional: float, lev: float, cfg: RiskConfig, st: RiskState
) -> tuple[bool, str]:
    context_base = {
        "requested_notional": float(notional),
        "open_notional": float(getattr(st, "open_notional", 0.0)),
        "open_positions": int(getattr(st, "open_positions", 0)),
        "leverage": float(lev),
    }
    if st.portfolio_drawdown_pct <= cfg.kill_switch_drawdown_pct:
        _emit_veto(
            symbol,
            "kill_switch_triggered",
            detail={
                "kill_switch_drawdown_pct": float(cfg.kill_switch_drawdown_pct),
                "portfolio_drawdown_pct": float(getattr(st, "portfolio_drawdown_pct", 0.0)),
            },
            context=context_base,
        )
        return False, "kill_switch_triggered"
    if notional < cfg.min_notional:
        _emit_veto(
            symbol,
            "min_notional",
            detail={
                "min_notional": float(cfg.min_notional),
                "requested_notional": float(notional),
            },
            context=context_base,
        )
        return False, "min_notional"
    if notional > cfg.max_notional_per_trade:
        _emit_veto(
            symbol,
            "exceeds_per_trade_cap",
            detail={
                "max_notional_per_trade": float(cfg.max_notional_per_trade),
                "requested_notional": float(notional),
            },
            context=context_base,
        )
        return False, "exceeds_per_trade_cap"
    if lev > cfg.max_leverage:
        _emit_veto(
            symbol,
            "exceeds_leverage_cap",
            detail={
                "max_leverage": float(cfg.max_leverage),
                "requested_leverage": float(lev),
            },
            context=context_base,
        )
        return False, "exceeds_leverage_cap"
    if (st.open_notional + notional) > cfg.max_open_notional:
        _emit_veto(
            symbol,
            "exceeds_open_notional_cap",
            detail={
                "max_open_notional": float(cfg.max_open_notional),
                "current_open_notional": float(getattr(st, "open_notional", 0.0)),
                "requested_notional": float(notional),
            },
            context=context_base,
        )
        return False, "exceeds_open_notional_cap"
    if st.open_positions >= cfg.max_positions:
        _emit_veto(
            symbol,
            "too_many_positions",
            detail={
                "max_positions": int(cfg.max_positions),
                "current_positions": int(getattr(st, "open_positions", 0)),
            },
            context=context_base,
        )
        return False, "too_many_positions"
    return True, "ok"


def can_open_position(*args, **kwargs):
    """
    Helper supporting two signatures for backward compatibility:
    - Legacy: can_open_position(symbol, notional, lev, cfg, st) -> (ok, reason)
    - New:    can_open_position(symbol, notional, lev, nav, open_qty, now, cfg, state, current_gross_notional=0.0)

    The new path delegates to check_order (if available) and returns
    ``(ok, first_reason_or_ok)``.
    """
    # Detect legacy 5-positional-args call used by current tests
    if len(args) == 5 and not kwargs:
        symbol, notional, lev, cfg, st = args
        return _can_open_position_legacy(symbol, notional, lev, cfg, st)

    # New signature path (allow both positional and keyword usage)
    symbol = kwargs.get("symbol", args[0] if len(args) > 0 else None)
    notional = kwargs.get("notional", args[1] if len(args) > 1 else None)
    lev = kwargs.get("lev", args[2] if len(args) > 2 else None)
    nav = kwargs.get("nav", args[3] if len(args) > 3 else None)
    open_qty = kwargs.get("open_qty", args[4] if len(args) > 4 else None)
    now = kwargs.get("now", args[5] if len(args) > 5 else None)
    cfg = kwargs.get("cfg", args[6] if len(args) > 6 else None)
    state = kwargs.get("state", args[7] if len(args) > 7 else None)
    current_gross_notional = kwargs.get(
        "current_gross_notional", args[8] if len(args) > 8 else 0.0
    )

    veto, details = check_order(
        symbol=symbol,
        side="LONG",
        requested_notional=notional,
        price=0.0,
        nav=nav,
        open_qty=open_qty,
        now=now,
        cfg=cfg,
        state=state,
        current_gross_notional=current_gross_notional,
    )
    reasons = (details.get("reasons") or []) if isinstance(details, dict) else []
    reason = reasons[0] if reasons else "ok"
    return (not veto), reason


def should_reduce_positions(st: RiskState, cfg: RiskConfig) -> bool:
    return st.portfolio_drawdown_pct <= cfg.kill_switch_drawdown_pct


def explain_limits(cfg: RiskConfig) -> Dict[str, float]:
    return {
        "max_notional_per_trade": cfg.max_notional_per_trade,
        "max_open_notional": cfg.max_open_notional,
        "max_positions": cfg.max_positions,
        "max_leverage": cfg.max_leverage,
        "kill_switch_drawdown_pct": cfg.kill_switch_drawdown_pct,
        "min_notional": cfg.min_notional,
    }


def clamp_order_size(requested_qty: float, step_size: float) -> float:
    """Round *down* to the exchange step size using Decimal to avoid FP drift."""
    if step_size <= 0:
        return float(requested_qty)
    q = Decimal(str(requested_qty))
    step = Decimal(str(step_size))
    steps = (q / step).to_integral_value(rounding=ROUND_DOWN)
    return float(steps * step)


# ---------------- Additional helpers used by the executor ----------------


def will_violate_exposure(
    current_gross: float, add_notional: float, nav: float, max_nav_pct: float
) -> bool:
    cap_frac = _normalize_pct(max_nav_pct)
    limit = float(nav) * max(cap_frac, 0.0)
    total = float(current_gross) + float(add_notional)
    return total > limit


def _cfg_get(cfg: dict, path: List[str], default):
    cur = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check_order(
    symbol: str,
    side: str,
    requested_notional: float,
    price: float,
    nav: float,
    open_qty: float,
    now: float,
    cfg: dict,
    state: RiskState,
    current_gross_notional: float = 0.0,
    lev: float = 1.0,
    # New optional parameters for stricter caps
    open_positions_count: int | None = None,
    tier_name: Optional[str] = None,
    current_tier_gross_notional: float = 0.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Apply per-symbol and global risk checks.

    Returns ``(veto, info)`` where ``veto`` indicates the order should be
    blocked and ``info`` contains structured metadata about the violation.
    """
    if not cfg:
        cfg = load_risk_config()
    cfg = _normalize_risk_cfg(cfg)

    detail_payload: Dict[str, Any] = {}
    nav_fresh_flag: Optional[bool] = None
    sym = str(symbol)
    s_cfg = _cfg_get(cfg, ["per_symbol", sym], {}) or {}
    g_cfg = _cfg_get(cfg, ["global"], {}) or {}
    per_symbol_cfg = cfg.get("per_symbol") if isinstance(cfg, Mapping) else {}
    strategy_name = (
        s_cfg.get("strategy")
        or s_cfg.get("strategy_name")
        or s_cfg.get("strategyId")
        or g_cfg.get("strategy")
    )
    signal_ts = (
        s_cfg.get("signal_ts")
        or s_cfg.get("timestamp")
        or s_cfg.get("ts")
        or g_cfg.get("signal_ts")
    )

    reasons: List[str] = []
    details: Dict[str, Any] = {
        "reasons": reasons,
        "notional": float(requested_notional),
    }
    thresholds: Dict[str, Any] = details.setdefault("thresholds", {})
    dry_run = is_dry_run()

    # Restricted symbols guard
    try:
        restricted = {str(x).upper() for x in (cfg.get("restricted_symbols") or [])}
    except Exception:
        restricted = set()
    if restricted and sym.upper() in restricted:
        details.setdefault("reasons", []).append("restricted_symbol")
        return False, details

    # Non-prod dry-run guard (skip on testnet)
    require_dry = bool((cfg.get("non_prod") or {}).get("require_dry_run"))
    if require_dry and not dry_run:
        if os.getenv("BINANCE_TESTNET"):
            pass
        else:
            details.setdefault("reasons", []).append("non_prod_env_guard")
            return False, details

    # Universe guard (exact symbol match)
    try:
        universe_set = {str(s).upper() for s in universe_by_symbol().keys()}
    except Exception:
        universe_set = set()
    if universe_set and sym.upper() not in universe_set:
        details.setdefault("reasons", []).append("symbol_not_in_universe")
        return False, details

    nav_threshold_raw = g_cfg.get("nav_freshness_seconds")
    try:
        nav_threshold_s = int(float(nav_threshold_raw or 0))
    except Exception:
        nav_threshold_s = 0
    if nav_threshold_s <= 0:
        nav_threshold_s = DEFAULT_NAV_FRESHNESS_SECONDS

    nav_fail_closed_cfg = g_cfg.get("fail_closed_on_nav_stale")
    nav_fail_closed = (
        bool(nav_fail_closed_cfg)
        if nav_fail_closed_cfg is not None
        else DEFAULT_FAIL_CLOSED_ON_NAV_STALE
    )

    nav_health = nav_health_snapshot(nav_threshold_s)
    nav_age = nav_health.get("age_s")
    nav_sources_ok = nav_health.get("sources_ok")
    nav_total = nav_health.get("nav_total")
    nav_age_s: Optional[int] = None
    try:
        if nav_health.get("age_s") is not None:
            nav_age_s = int(max(0.0, float(nav_health.get("age_s"))))
    except Exception:
        nav_age_s = None
    if nav_age_s is not None:
        details["nav_age_s"] = nav_age_s
    else:
        details.setdefault("nav_age_s", None)

    details["nav_sources_ok"] = bool(nav_health.get("sources_ok"))
    thresholds.setdefault("nav_freshness_seconds", float(nav_threshold_s))
    testnet_override_active = bool((cfg.get("_meta") or {}).get("testnet_overrides_active"))
    if testnet_override_active:
        thresholds.setdefault("testnet_overrides_active", True)

    qty_req = None
    try:
        if price not in (None, 0, 0.0) and float(price or 0.0) != 0.0:
            qty_req = float(requested_notional) / float(price)
    except Exception:
        qty_req = None

    try:
        nav_f = float(nav_total) if nav_total is not None else float(nav)
    except Exception:
        nav_f = 0.0
    try:
        current_gross_f = float(current_gross_notional)
    except Exception:
        current_gross_f = 0.0
    try:
        now_f = float(now)
    except Exception:
        now_f = time.time()

    nav_observations: Dict[str, Any] = {
        "nav_age_s": nav_age_s,
        "sources_ok": bool(nav_sources_ok) if nav_sources_ok is not None else None,
    }

    nav_context = {
        "nav": nav_f,
        "requested_notional": float(requested_notional),
        "current_gross_notional": current_gross_f,
        "now": now_f,
        "nav_freshness_seconds": nav_threshold_s,
        "sources_ok": bool(nav_sources_ok) if nav_sources_ok is not None else None,
    }
    nav_context["nav_total"] = nav_f
    if nav_age is not None:
        nav_context["nav_age"] = nav_age

    nav_detail_payload: Dict[str, Any] = {
        "reasons": ["nav_stale"],
        "thresholds": {"nav_freshness_seconds": nav_threshold_s},
        "observations": nav_observations,
    }

    nav_guard_cfg = {
        "nav_freshness_seconds": nav_threshold_s,
        "fail_closed_on_nav_stale": nav_fail_closed,
    }

    nav_age_val: Optional[float] = None
    try:
        if nav_age is not None:
            nav_age_val = float(nav_age)
        elif nav_age_s is not None:
            nav_age_val = float(nav_age_s)
    except Exception:
        nav_age_val = None
    nav_sources_bool = bool(nav_sources_ok) if nav_sources_ok is not None else False
    nav_is_fresh = bool(nav_health.get("fresh")) if nav_health else False
    nav_fresh_flag = nav_is_fresh
    detail_payload["nav_fresh"] = nav_is_fresh
    detail_payload["nav_health"] = nav_health

    nav_ok = enforce_nav_freshness_or_veto(
        {
            "symbol": sym,
            "strategy": strategy_name,
            "signal_ts": signal_ts,
            "qty": qty_req,
            "detail": nav_detail_payload,
            "context": nav_context,
        },
        {"age": nav_age, "sources_ok": nav_sources_ok},
        nav_guard_cfg,
    )

    nav_warning = False
    if not nav_ok:
        if nav_fail_closed:
            reasons.append("nav_stale")
        else:
            nav_warning = True
    elif (not nav_is_fresh) and not nav_fail_closed:
        nav_warning = True

    if nav_warning:
        warnings = details.setdefault("warnings", [])
        if "nav_stale" not in warnings:
            warnings.append("nav_stale")
        detail_payload["nav_fresh"] = False
        nav_fresh_flag = False

    # Block duplicate exposure if a live position already exists
    try:
        pos_amt = 0.0
        live_positions = getattr(state, "live_positions", {}) or {}
        if isinstance(live_positions, dict):
            entry = live_positions.get(sym)
            if isinstance(entry, (int, float)):
                pos_amt = float(entry)
            elif isinstance(entry, dict):
                try:
                    pos_amt = float(entry.get("positionAmt", entry.get("amt", 0.0)) or 0.0)
                except Exception:
                    pos_amt = 0.0

        if abs(pos_amt) <= 0.0:
            client = getattr(state, "client", None)
            if client is not None:
                live = get_live_positions(client)
                for entry in live:
                    if str(entry.get("symbol")) == sym:
                        try:
                            pos_amt = float(entry.get("positionAmt", 0.0) or 0.0)
                        except Exception:
                            pos_amt = 0.0
                        break

        if abs(pos_amt) > 0.0:
            reasons.append("already_in_trade")
            thresholds.setdefault("position_amt", abs(pos_amt))
    except Exception as exc:
        LOGGER.error("[risk] already_in_trade veto check failed: %s", exc)

    # Whitelist guardrail (if provided)
    wl = g_cfg.get("whitelist") or []
    if isinstance(wl, list) and len(wl) > 0:
        wl_set = {str(x).upper() for x in wl}
        if sym.upper() not in wl_set:
            reasons.append("not_whitelisted")

    dd_snapshot = _drawdown_snapshot(g_cfg)
    dd_info = dd_snapshot.get("drawdown") or {}
    dd_pct = _as_float(dd_info.get("pct", dd_snapshot.get("dd_pct", 0.0)))
    dd_peak = _as_float(dd_info.get("peak_nav", dd_snapshot.get("peak", 0.0)))
    dd_nav_snapshot = _as_float(dd_info.get("nav", dd_snapshot.get("nav", 0.0)))
    dd_abs = _as_float(dd_info.get("abs", dd_snapshot.get("dd_abs", 0.0)))
    daily_info = dd_snapshot.get("daily_loss") or {}
    daily_loss_pct = _as_float(daily_info.get("pct"))
    drawdown_usable = bool(dd_snapshot.get("usable", True))
    dd_stale_flags = dd_snapshot.get("stale_flags") or {}
    nav_health_diag = dd_snapshot.get("nav_health") or {}
    peak_state_diag = dd_snapshot.get("peak_state") or {}
    assets_diag = dd_snapshot.get("assets") or {}
    if dd_peak > 0.0 or dd_pct > 0.0:
        LOGGER.info("[risk] drawdown dd=%.1f%% peak=%.2f nav=%.2f", dd_pct, dd_peak, dd_nav_snapshot)
    if dd_abs:
        details["drawdown_abs"] = dd_abs
    details["drawdown_pct"] = dd_pct
    if dd_peak:
        details["drawdown_peak"] = dd_peak
    if dd_nav_snapshot:
        details["drawdown_nav"] = dd_nav_snapshot
    if dd_stale_flags:
        details["drawdown_stale_flags"] = dd_stale_flags

    daily_pnl_state = getattr(state, "daily_pnl_pct", 0.0)
    try:
        daily_pnl_state = float(daily_pnl_state)
    except Exception:
        daily_pnl_state = 0.0
    if daily_pnl_state:
        details.setdefault("daily_pnl_pct_state", daily_pnl_state)
        if daily_pnl_state < 0.0:
            fallback_dd = abs(daily_pnl_state)
            if fallback_dd > dd_pct:
                dd_pct = fallback_dd
                details["drawdown_pct"] = dd_pct
    if daily_loss_pct is not None:
        details["daily_loss_pct"] = daily_loss_pct

    try:
        max_nav_drawdown_pct = float(g_cfg.get("max_nav_drawdown_pct", 0.0) or 0.0)
    except Exception:
        max_nav_drawdown_pct = 0.0
    if drawdown_usable and max_nav_drawdown_pct > 0.0 and dd_pct >= max_nav_drawdown_pct:
        reasons.append("nav_drawdown_limit")
        thresholds.setdefault("max_nav_drawdown_pct", max_nav_drawdown_pct)
        thresholds.setdefault("observed_drawdown_pct", dd_pct)
        thresholds.setdefault("drawdown_nav", dd_nav_snapshot)
        thresholds.setdefault("drawdown_peak", dd_peak)
        LOGGER.warning(
            "[risk] drawdown_exceeded dd=%.1f%% limit=%.1f%% peak=%.2f nav=%.2f",
            dd_pct,
            max_nav_drawdown_pct,
            dd_peak,
            dd_nav_snapshot,
        )

    # Daily loss limit (portfolio) sourced from drawdown tracker.
    try:
        day_lim = float(g_cfg.get("daily_loss_limit_pct", 0.0) or 0.0)
    except Exception:
        day_lim = 0.0
    observed_daily_loss_pct = daily_loss_pct if daily_loss_pct is not None else dd_pct
    if day_lim > 0.0:
        observed_dd = observed_daily_loss_pct
        if daily_pnl_state < 0.0:
            observed_dd = max(observed_dd, abs(daily_pnl_state))
        if observed_dd >= day_lim:
            reasons.append("day_loss_limit")
            thresholds.setdefault("daily_loss_limit_pct", day_lim)
            thresholds.setdefault("observed_daily_drawdown_pct", observed_dd)
            thresholds.setdefault("drawdown_peak", dd_peak)
            thresholds.setdefault("drawdown_nav", dd_nav_snapshot)
            LOGGER.warning("[risk] veto daily_loss_limit dd=%.1f%% limit=%.1f%%", observed_dd, day_lim)
    if not drawdown_usable and dd_stale_flags:
        details.setdefault("warnings", [])
        if "drawdown_stale" not in details["warnings"]:
            details["warnings"].append("drawdown_stale")
        thresholds.setdefault("drawdown_state_stale", dd_stale_flags)
    diag_payload = {
        "nav_health_diag": nav_health_diag,
        "peak_state": peak_state_diag,
        "drawdown": dd_info,
        "daily_loss": daily_info,
    }
    if assets_diag:
        diag_payload["assets"] = assets_diag

    # Per-order notional constraints
    g_min = float(g_cfg.get("min_notional_usdt", 0.0) or 0.0)
    min_notional = max(float(s_cfg.get("min_notional", 0.0) or 0.0), g_min)
    max_order_notional = float(s_cfg.get("max_order_notional", 0.0) or 0.0)
    req_notional = float(requested_notional)

    if min_notional > 0.0 and req_notional < min_notional:
        reasons.append("min_notional")
        thresholds.setdefault("min_notional", float(min_notional))

    if max_order_notional > 0.0 and req_notional > max_order_notional:
        reasons.append("symbol_cap")
        thresholds.setdefault("max_order_notional", float(max_order_notional))

    cap_cfg_raw = (
        s_cfg.get("max_nav_pct", s_cfg.get("symbol_notional_share_cap_pct"))
        if isinstance(s_cfg, Mapping)
        else None
    ) or g_cfg.get("symbol_notional_share_cap_pct")
    caps_map = load_symbol_caps()
    cap_entry = None
    if cap_cfg_raw is None:
        # Respect explicit per-symbol entries; otherwise fall back to loader caps.
        if sym not in (per_symbol_cfg.keys() if isinstance(per_symbol_cfg, Mapping) else {}):
            cap_entry = caps_map.get(sym) if isinstance(caps_map, Mapping) else None
            if cap_entry:
                cap_cfg_raw = cap_entry.get("cap_cfg_raw")
    cap_cfg_normalized = cap_entry.get("cap_cfg_normalized") if isinstance(cap_entry, Mapping) else None
    if cap_cfg_normalized is None:
        cap_cfg_normalized = normalize_percentage(cap_cfg_raw)
    cap_abs = nav_f * cap_cfg_normalized if nav_f > 0.0 else 0.0
    asset_breakdown = {}
    try:
        nav_snapshot = get_confirmed_nav()
        detail = nav_snapshot.get("detail") if isinstance(nav_snapshot, Mapping) else {}
        breakdown = detail.get("breakdown") or detail.get("assets") or {}
        if isinstance(breakdown, Mapping):
            asset_breakdown = dict(breakdown)
    except Exception:
        asset_breakdown = {}
    if cap_abs > 0.0 and (current_gross_f + req_notional) > cap_abs:
        reasons.append("symbol_cap")
        thresholds.setdefault("symbol_notional_cap", float(cap_abs))
        detail_payload.setdefault("cap_cfg_raw", cap_cfg_raw)
        detail_payload.setdefault("cap_cfg_normalized", cap_cfg_normalized)
        detail_payload.setdefault("nav_total", nav_f)
        detail_payload.setdefault("asset_breakdown", asset_breakdown)

    # Open quantity cap (applies to increasing long exposure)
    max_open_qty = s_cfg.get("max_open_qty", None)
    if max_open_qty is not None:
        try:
            max_open_qty_f = float(max_open_qty)
        except Exception:
            max_open_qty_f = None
        if max_open_qty_f is not None:
            if (
                str(side).upper() in ("BUY", "LONG")
                and float(open_qty) >= max_open_qty_f
            ):
                reasons.append("symbol_cap")
                thresholds.setdefault("max_open_qty", float(max_open_qty_f))

    # Side block (optional)
    try:
        blocked_sides = {str(x).upper() for x in (s_cfg.get("block_sides") or [])}
        if blocked_sides and str(side).upper() in blocked_sides:
            reasons.append("side_blocked")
    except Exception:
        pass

    # Leverage cap (per-symbol or global)
    try:
        lev_cap = float(
            s_cfg.get("max_leverage", g_cfg.get("max_leverage", 0.0)) or 0.0
        )
    except Exception:
        lev_cap = 0.0
    if lev_cap > 0.0 and float(lev or 0.0) > lev_cap:
        reasons.append("leverage_exceeded")
        thresholds.setdefault("max_leverage", float(lev_cap))

    # Per-symbol cooldown after last fill
    cooldown_sec = int(float(s_cfg.get("cooldown_sec", 0) or 0))
    if cooldown_sec > 0:
        last_fill = getattr(state, "last_fill_ts", lambda _s: 0.0)(sym)
        if last_fill > 0.0:
            cooldown_until = last_fill + float(cooldown_sec)
            if float(now) < cooldown_until:
                reasons.append("cooldown")
                details["cooldown_until"] = float(cooldown_until)
                thresholds.setdefault("cooldown_sec", cooldown_sec)
                thresholds.setdefault("last_fill_ts", float(last_fill))

    # Error circuit breaker (global)
    err_cfg = g_cfg.get("error_circuit", {}) or {}
    max_errors = int(float(err_cfg.get("max_errors", 0) or 0))
    window_sec = int(float(err_cfg.get("window_sec", 0) or 0))
    if max_errors > 0 and window_sec > 0:
        errors_in = getattr(state, "errors_in", lambda _w, _n: 0)
        if errors_in(window_sec, float(now)) >= max_errors:
            reasons.append("circuit_breaker")
            thresholds.setdefault("error_circuit", {"max_errors": max_errors, "window_sec": window_sec})

    # Burst limit on order attempts (global)
    burst_cfg = g_cfg.get("burst_limit", {}) or {}
    try:
        burst_max = int(float(burst_cfg.get("max_orders", 0) or 0))
        burst_win = int(float(burst_cfg.get("window_sec", 0) or 0))
    except Exception:
        burst_max = 0
        burst_win = 0
    if burst_max > 0 and burst_win > 0:
        attempts_in = getattr(state, "attempts_in", lambda _w, _n: 0)
        if attempts_in(burst_win, float(now)) >= burst_max:
            reasons.append("burst_limit")
            thresholds.setdefault("burst_limit", {"max_orders": burst_max, "window_sec": burst_win})

    # Per-trade NAV cap
    def _normalized_pct(value: Any, default: float = 0.0) -> float:
        try:
            pct_val = float(value)
        except Exception:
            pct_val = default
        return _normalize_pct(pct_val if pct_val is not None else default)

    max_trade_pct_raw = g_cfg.get("max_trade_nav_pct", 0.0)
    max_trade_pct = _normalize_pct(max_trade_pct_raw)
    equity_nav_pct_raw = (
        g_cfg.get("trade_equity_nav_pct")
        or g_cfg.get("equity_nav_pct")
        or g_cfg.get("equity_clamp_nav_pct")
        or 0.0
    )
    equity_nav_pct = _normalize_pct(equity_nav_pct_raw)
    thresholds.setdefault("trade_equity_nav_pct", float(equity_nav_pct))
    thresholds.setdefault("max_trade_nav_pct", float(max_trade_pct))
    trade_nav_obs_pct = None
    if nav_f > 0.0:
        trade_nav_obs_pct = req_notional / nav_f
    details["trade_equity_nav_obs"] = trade_nav_obs_pct
    details["max_trade_nav_obs"] = trade_nav_obs_pct
    if nav_f > 0.0:
        if equity_nav_pct > 0.0:
            equity_limit = nav_f * equity_nav_pct
            if req_notional > equity_limit:
                if REASON_TRADE_EQUITY_CAP not in reasons:
                    reasons.append(REASON_TRADE_EQUITY_CAP)
        if max_trade_pct > 0.0:
            trade_limit = nav_f * max_trade_pct
            if req_notional > trade_limit:
                reasons.append(REASON_MAX_TRADE_NAV)

    # Gross exposure cap (global) — accept legacy/new keys
    max_gross_nav_pct_raw = (
        (g_cfg.get("max_portfolio_gross_nav_pct")
         if (g_cfg.get("max_portfolio_gross_nav_pct") is not None)
         else g_cfg.get("max_gross_nav_pct", 0.0))
    )
    max_gross_nav_pct = _normalize_pct(max_gross_nav_pct_raw)
    if max_gross_nav_pct > 0.0:
        if will_violate_exposure(
            float(current_gross_notional), req_notional, float(nav_f), max_gross_nav_pct
        ):
            reasons.append("portfolio_cap")
            thresholds.setdefault("max_gross_exposure_pct", max_gross_nav_pct)

    # Max concurrent positions (global)
    try:
        max_conc = int(float(g_cfg.get("max_concurrent_positions", 0) or 0))
    except Exception:
        max_conc = 0
    if max_conc > 0 and open_positions_count is not None:
        if int(open_positions_count) >= max_conc:
            reasons.append("max_concurrent")
            thresholds.setdefault("max_concurrent_positions", max_conc)

    # Per-tier soft budget per-symbol (gross as % NAV)
    if tier_name:
        try:
            tiers_cfg = g_cfg.get("tiers") or {}
            t_cfg = tiers_cfg.get(str(tier_name)) or {}
            per_sym_pct = _normalize_pct(t_cfg.get("per_symbol_nav_pct", 0.0) or 0.0)
        except Exception:
            per_sym_pct = 0.0
        if per_sym_pct > 0.0 and float(nav) > 0.0:
            cap_abs = float(nav) * per_sym_pct
            # current exposure for this symbol/tier + request
            cur = float(current_tier_gross_notional)
            if (cur + req_notional) > cap_abs:
                reasons.append("tier_cap")
                thresholds.setdefault("tier_cap", {"tier": tier_name, "per_symbol_nav_pct": per_sym_pct})

    if not reasons:
        if details.get("warnings"):
            detail_payload = dict(detail_payload)
            detail_payload["warnings"] = list(details.get("warnings") or [])
            if "drawdown_stale_flags" in details:
                detail_payload["drawdown_stale_flags"] = details.get("drawdown_stale_flags")
        detail_payload = {**detail_payload, **diag_payload}
        detail_payload["reasons"] = list(reasons)
        return False, detail_payload

    try:
        qty_req = (
            float(req_notional / float(price))
            if price not in (None, 0, 0.0) and float(price or 0.0) != 0.0
            else None
        )
    except Exception:
        qty_req = None
    try:
        nav_f = float(nav)
    except Exception:
        nav_f = 0.0
    try:
        current_gross_f = float(current_gross_notional)
    except Exception:
        current_gross_f = 0.0
    try:
        open_qty_f = float(open_qty)
    except Exception:
        open_qty_f = 0.0
    try:
        lev_f = float(lev)
    except Exception:
        lev_f = 0.0
    try:
        now_f = float(now)
    except Exception:
        now_f = time.time()
    try:
        tier_gross_f = float(current_tier_gross_notional)
    except Exception:
        tier_gross_f = 0.0
    context = {
        "nav": nav_f,
        "requested_notional": req_notional,
        "current_gross_notional": current_gross_f,
        "open_qty": open_qty_f,
        "lev": lev_f,
        "now": now_f,
        "open_positions_count": open_positions_count,
        "tier": tier_name,
        "current_tier_gross_notional": tier_gross_f,
    }
    if nav_f > 0.0:
        context["post_trade_exposure_pct"] = ((current_gross_f + req_notional) / nav_f)
    nav_flag_snapshot = detail_payload.get("nav_fresh")
    detail_payload = {
        "gate": "risk_limits",
        "limit": reasons[0],
        "reasons": list(reasons),
        "thresholds": thresholds,
        "notional": float(req_notional),
        **diag_payload,
    }
    detail_payload.setdefault("cap_cfg_raw", cap_cfg_raw)
    detail_payload.setdefault("cap_cfg_normalized", cap_cfg_normalized)
    detail_payload.setdefault("nav_total", nav_f)
    detail_payload.setdefault("asset_breakdown", asset_breakdown)
    if nav_flag_snapshot is not None:
        detail_payload["nav_fresh"] = nav_flag_snapshot
    if not reasons:
        reasons.append("unknown_early_veto")
    value = thresholds.get(reasons[0])
    if value is not None:
        detail_payload["value"] = value
    extra = {
        k: v
        for k, v in details.items()
        if k not in ("reasons", "thresholds")
    }
    if extra:
        detail_payload["observations"] = extra
    _emit_veto(
        symbol,
        reasons[0],
        detail=detail_payload,
        context=context,
        strategy=strategy_name,
        signal_ts=signal_ts,
        qty=qty_req,
    )
    detail_payload["reasons"] = list(reasons)
    return True, detail_payload


def _effective_guard_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(cfg, Mapping):
        return cfg
    loaded = load_risk_config()
    return loaded if isinstance(loaded, Mapping) else {}


def _normalize_ratio(value: Any, default: float) -> float:
    normalized = normalize_percentage(value)
    return normalized if normalized > 0 else default


def _normalize_pct_value(value: Any, default: float) -> float:
    normalized = normalize_percentage(value)
    return normalized if normalized > 0 else default


def _guard_thresholds(symbol: str, cfg: Mapping[str, Any]) -> tuple[float, float]:
    sym_key = str(symbol).upper()
    global_cfg = cfg.get("global") if isinstance(cfg, Mapping) else {}
    if not isinstance(global_cfg, Mapping):
        global_cfg = {}
    per_symbol_cfg = cfg.get("per_symbol") if isinstance(cfg, Mapping) else {}
    entry = per_symbol_cfg.get(sym_key) if isinstance(per_symbol_cfg, Mapping) else {}
    share_raw = (
        (entry or {}).get("symbol_notional_share_cap_pct")
        or global_cfg.get("symbol_notional_share_cap_pct")
        or DEFAULT_SYMBOL_SHARE_CAP
    )
    dd_raw = (
        (entry or {}).get("symbol_drawdown_cap_pct")
        or global_cfg.get("symbol_drawdown_cap_pct")
        or DEFAULT_SYMBOL_DD_CAP_PCT
    )
    return _normalize_ratio(share_raw, DEFAULT_SYMBOL_SHARE_CAP), _normalize_pct_value(
        dd_raw, DEFAULT_SYMBOL_DD_CAP_PCT
    )


def symbol_notional_guard(symbol: str, cfg: Mapping[str, Any] | None = None) -> bool:
    if not is_in_asset_universe(symbol):
        return False
    total = total_notional_7d() or 0.0
    if total <= 0:
        return True
    share = (notional_7d_by_symbol(symbol) or 0.0) / max(total, 1e-9)
    share_cap, _ = _guard_thresholds(symbol, _effective_guard_cfg(cfg))
    return share <= share_cap


def symbol_dd_guard(symbol: str, cfg: Mapping[str, Any] | None = None) -> bool:
    if is_symbol_disabled(symbol):
        return False
    if not is_in_asset_universe(symbol):
        return False
    _, dd_cap = _guard_thresholds(symbol, _effective_guard_cfg(cfg))
    dd_raw = dd_today_pct(symbol) or 0.0
    dd = dd_raw / 100.0
    if dd <= -dd_cap:
        disable_symbol_temporarily(symbol, ttl_hours=COOLDOWN_H, reason="dd_cap_hit")
        return False
    return True


# Deprecated: retained as a no-op shim to avoid import errors while sizing refactor proceeds.
class RiskGate:  # pragma: no cover - compatibility shim
    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def allowed_gross_notional(self, symbol: str, gross_usd: float, now_ts: float | None = None) -> tuple[bool, str]:
        return True, ""
