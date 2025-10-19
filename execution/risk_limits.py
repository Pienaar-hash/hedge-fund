from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import time

from execution.drawdown_tracker import load_peak_state
from execution.nav import (
    compute_trading_nav,
    compute_gross_exposure_usd,
    get_nav_age,
    is_nav_fresh,
)
from execution.utils import load_json, get_live_positions
from execution.exchange_utils import get_balances

LOGGER = logging.getLogger("risk_limits")
from execution.log_utils import get_logger, log_event, safe_dump


_GLOBAL_KEYS = {
    "daily_loss_limit_pct",
    "cooldown_minutes_after_stop",
    "max_trades_per_symbol_per_hour",
    "drawdown_alert_pct",
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

REASONS = {
    "kill_switch_triggered": "kill_switch",
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
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < pct <= 1.0:
        return pct * 100.0
    return pct


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


def _drawdown_snapshot() -> Dict[str, float]:
    state = load_peak_state()
    if not isinstance(state, dict):
        state = {}
    dd_pct = _as_float(state.get("dd_pct"))
    dd_abs = _as_float(state.get("dd_abs"))
    peak = _as_float(state.get("peak") or state.get("peak_equity"))
    nav = _as_float(state.get("nav") or state.get("nav_usd"))
    realized = _as_float(state.get("realized_pnl_today"))
    return {
        "dd_pct": dd_pct,
        "dd_abs": dd_abs,
        "peak": peak,
        "nav": nav,
        "realized_today": realized,
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

    def __init__(self) -> None:
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
            "below_min_notional",
            detail={
                "min_notional": float(cfg.min_notional),
                "requested_notional": float(notional),
            },
            context=context_base,
        )
        return False, "below_min_notional"
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

    The new path delegates to check_order (if available) and returns (ok, first_reason_or_ok).
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

    ok, details = check_order(
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
    return ok, (details.get("reasons") or ["ok"])[0]


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
    cap_frac = float(max_nav_pct) / 100.0
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
) -> Tuple[bool, dict]:
    """Apply per-symbol and global risk checks.

    Returns (ok, details) where details has keys: reasons: list[str], notional: float, cooldown_until?: float
    """
    cfg = _normalize_risk_cfg(cfg)

    sym = str(symbol)
    s_cfg = _cfg_get(cfg, ["per_symbol", sym], {}) or {}
    g_cfg = _cfg_get(cfg, ["global"], {}) or {}
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

    nav_threshold_raw = g_cfg.get("nav_freshness_seconds")
    try:
        nav_threshold_s = float(nav_threshold_raw or 0.0)
    except Exception:
        nav_threshold_s = 0.0
    nav_fail_closed = bool(g_cfg.get("fail_closed_on_nav_stale", False))

    nav_age = get_nav_age()
    nav_age_s: Optional[int] = None
    if isinstance(nav_age, (int, float)):
        try:
            nav_age_s = int(max(0.0, float(nav_age)))
        except Exception:
            nav_age_s = None
    if nav_age_s is not None:
        details["nav_age_s"] = nav_age_s

    nav_status_known = nav_threshold_s > 0.0
    nav_fresh = {
        "fresh": bool(is_nav_fresh(nav_threshold_s if nav_status_known else None)),
        "age": nav_age,
    }

    age_for_log = (
        f"{float(nav_age):.1f}s" if isinstance(nav_age, (int, float)) else "unknown"
    )
    if nav_status_known or nav_age is not None:
        if nav_fresh["fresh"]:
            LOGGER.info("[risk] nav_ok (age=%s, fresh=%s)", age_for_log, nav_fresh["fresh"])
        else:
            log_msg = "[risk] veto nav_stale (age=%s)" if nav_fail_closed else "[risk] nav_stale (age=%s) (fail-open)"
            LOGGER.warning(log_msg, age_for_log)

    if nav_status_known and not nav_fresh["fresh"] and nav_fail_closed:
        reasons.append("nav_stale")
        thresholds.setdefault("nav_freshness_seconds", float(nav_threshold_s))
        if nav_age_s is None:
            details.setdefault("nav_age_s", None)
        qty_req = None
        try:
            if price not in (None, 0, 0.0) and float(price or 0.0) != 0.0:
                qty_req = float(requested_notional) / float(price)
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
            now_f = float(now)
        except Exception:
            now_f = time.time()
        context = {
            "nav": nav_f,
            "requested_notional": float(requested_notional),
            "current_gross_notional": current_gross_f,
            "now": now_f,
            "nav_age_s": nav_age_s,
            "nav_freshness_seconds": nav_threshold_s,
        }
        detail_payload: Dict[str, Any] = {
            "reasons": list(reasons),
            "thresholds": thresholds,
        }
        extra = {
            k: v
            for k, v in details.items()
            if k not in ("reasons", "thresholds")
        }
        if extra:
            detail_payload["observations"] = extra
        _emit_veto(
            symbol,
            "nav_stale",
            detail=detail_payload,
            context=context,
            strategy=strategy_name,
            signal_ts=signal_ts,
            qty=qty_req,
        )
        return False, details

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

    dd_snapshot = _drawdown_snapshot()
    dd_pct = dd_snapshot.get("dd_pct", 0.0)
    dd_peak = dd_snapshot.get("peak", 0.0)
    dd_nav_snapshot = dd_snapshot.get("nav", 0.0)
    dd_abs = dd_snapshot.get("dd_abs", 0.0)
    if dd_peak > 0.0 or dd_pct > 0.0:
        LOGGER.info("[risk] drawdown dd=%.1f%% peak=%.2f nav=%.2f", dd_pct, dd_peak, dd_nav_snapshot)
    if dd_abs:
        details["drawdown_abs"] = dd_abs
    details["drawdown_pct"] = dd_pct
    if dd_peak:
        details["drawdown_peak"] = dd_peak
    if dd_nav_snapshot:
        details["drawdown_nav"] = dd_nav_snapshot

    # Daily loss limit (portfolio) sourced from drawdown tracker.
    try:
        day_lim = float(g_cfg.get("daily_loss_limit_pct", 0.0) or 0.0)
    except Exception:
        day_lim = 0.0
    if day_lim > 0.0:
        if dd_pct >= day_lim:
            reasons.append("daily_loss_limit")
            thresholds.setdefault("daily_loss_limit_pct", day_lim)
            thresholds.setdefault("observed_daily_drawdown_pct", dd_pct)
            thresholds.setdefault("drawdown_peak", dd_peak)
            thresholds.setdefault("drawdown_nav", dd_nav_snapshot)
            LOGGER.warning("[risk] veto daily_loss_limit dd=%.1f%% limit=%.1f%%", dd_pct, day_lim)

    # Per-order notional constraints
    g_min = float(g_cfg.get("min_notional_usdt", 0.0) or 0.0)
    min_notional = max(float(s_cfg.get("min_notional", 0.0) or 0.0), g_min)
    max_order_notional = float(s_cfg.get("max_order_notional", 0.0) or 0.0)
    req_notional = float(requested_notional)

    if min_notional > 0.0 and req_notional < min_notional:
        reasons.append("below_min_notional")
        thresholds.setdefault("min_notional", float(min_notional))

    if max_order_notional > 0.0 and req_notional > max_order_notional:
        reasons.append("symbol_cap")
        thresholds.setdefault("max_order_notional", float(max_order_notional))

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
    try:
        max_trade_pct = float(g_cfg.get("max_trade_nav_pct", 10.0) or 0.0)
    except Exception:
        max_trade_pct = 0.0
    if max_trade_pct > 0.0 and float(nav) > 0.0:
        if req_notional > float(nav) * (max_trade_pct / 100.0):
            reasons.append("trade_gt_max_trade_nav_pct")
            reasons.append("trade_gt_10pct_equity")
            thresholds.setdefault("max_trade_nav_pct", max_trade_pct)

    # Gross exposure cap (global) — accept legacy/new keys
    max_gross_nav_pct = float(
        (g_cfg.get("max_portfolio_gross_nav_pct")
         if (g_cfg.get("max_portfolio_gross_nav_pct") is not None)
         else g_cfg.get("max_gross_nav_pct", 0.0))
        or 0.0
    )
    if max_gross_nav_pct > 0.0:
        if will_violate_exposure(
            float(current_gross_notional), req_notional, float(nav), max_gross_nav_pct
        ):
            # keep legacy reason plus standardized alias
            reasons.append("max_gross_nav_pct")
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
            per_sym_pct = float(t_cfg.get("per_symbol_nav_pct", 0.0) or 0.0)
        except Exception:
            per_sym_pct = 0.0
        if per_sym_pct > 0.0 and float(nav) > 0.0:
            cap_abs = float(nav) * (per_sym_pct / 100.0)
            # current exposure for this symbol/tier + request
            cur = float(current_tier_gross_notional)
            if (cur + req_notional) > cap_abs:
                reasons.append("tier_cap")
                thresholds.setdefault("tier_cap", {"tier": tier_name, "per_symbol_nav_pct": per_sym_pct})

    ok = len(reasons) == 0
    if not ok and reasons:
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
            context["post_trade_exposure_pct"] = ((current_gross_f + req_notional) / nav_f) * 100.0
        detail_payload: Dict[str, Any] = {
            "reasons": list(reasons),
            "thresholds": thresholds,
        }
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
    return ok, details


# ---------------- Canonical gross-notional gate (shared taxonomy) ----------------

class RiskGate:
    """Canonical gross-notional checks used by executor and screener.

    Expects a cfg that contains keys under `sizing` and `risk` namespaces. Tests
    may pass a minimal dict; production can adapt existing config to this shape.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        self.sizing = dict(self.cfg.get("sizing", {}) or {})
        self.risk = dict(self.cfg.get("risk", {}) or {})
        self.nav_provider: Optional[Any] = None
        try:
            self.min_notional = float(self.sizing.get("min_notional_usdt", 5.0))
        except Exception:
            self.min_notional = 5.0
        try:
            self.min_gross_usd_per_order = float(
                (self.sizing.get("min_gross_usd_per_order", 0.0)) or 0.0
            )
        except Exception:
            self.min_gross_usd_per_order = 0.0
        per_symbol_floor = self.sizing.get("per_symbol_min_gross_usd") or {}
        self.per_symbol_min_gross_usd: Dict[str, float] = {}
        if isinstance(per_symbol_floor, dict):
            for key, value in per_symbol_floor.items():
                try:
                    self.per_symbol_min_gross_usd[str(key).upper()] = float(value)
                except Exception:
                    continue
        self._last_stop_ts = 0.0
        self._trade_counts: Dict[tuple[str, int], int] = {}

    # --- overridable helpers (tests may subclass) ---
    def _portfolio_nav(self) -> float:
        # Best-effort NAV; tests may override
        provider = self.nav_provider
        if provider is not None:
            try:
                return float(provider.current_nav_usd())
            except Exception:
                pass
        cfg: Dict[str, Any] = {}
        try:
            cfg = load_json("config/strategy_config.json") or {}
        except Exception:
            cfg = {}
        try:
            nav_val, _ = compute_trading_nav(cfg)
            if nav_val and nav_val > 0:
                return float(nav_val)
        except Exception:
            pass
        try:
            bal = get_balances()
            if isinstance(bal, dict):
                return float(bal.get("USDT", bal.get("walletBalance", 0.0)) or 0.0)
            if isinstance(bal, list):
                for entry in bal:
                    try:
                        if entry.get("asset") == "USDT":
                            return float(entry.get("balance") or entry.get("walletBalance") or 0.0)
                    except Exception:
                        continue
        except Exception:
            pass
        return float(cfg.get("capital_base_usdt", 0.0) or 0.0)

    def _gross_exposure_pct(self) -> float:
        nav = float(self._portfolio_nav())
        if nav <= 0:
            return 0.0
        gross = float(self._portfolio_gross_usd())
        return (gross / nav) * 100.0

    def _symbol_exposure_pct(self, symbol: str) -> float:
        nav = float(self._portfolio_nav())
        if nav <= 0:
            return 0.0
        sym = str(symbol).upper()
        gross_map: Dict[str, float] = {}
        provider = self.nav_provider
        if provider is not None:
            try:
                getter = getattr(provider, "symbol_gross_usd", None)
                if callable(getter):
                    gross_map = getter()
            except Exception:
                gross_map = {}
        if not gross_map:
            try:
                from execution.nav import compute_symbol_gross_usd

                gross_map = compute_symbol_gross_usd()
            except Exception:
                gross_map = {}
        sym_gross = float(gross_map.get(sym, 0.0) or 0.0)
        if sym_gross <= 0:
            return 0.0
        return (sym_gross / nav) * 100.0

    def _portfolio_gross_usd(self) -> float:
        provider = self.nav_provider
        if provider is not None:
            try:
                return float(provider.current_gross_usd())
            except Exception:
                pass
        try:
            return float(compute_gross_exposure_usd())
        except Exception:
            return 0.0

    def _now_hour_key(self) -> int:
        return int(time.time() // 3600)

    def _daily_loss_pct(self) -> float:
        snapshot = _drawdown_snapshot()
        dd_pct = snapshot.get("dd_pct", 0.0)
        if dd_pct and dd_pct > 0.0:
            return float(dd_pct)
        peak = snapshot.get("peak", 0.0)
        if peak <= 0:
            return 0.0
        try:
            curr = float(self._portfolio_nav())
        except Exception:
            curr = snapshot.get("nav", 0.0)
        if curr is None or peak <= 0:
            return 0.0
        return 100.0 * max(0.0, (peak - float(curr)) / peak)

    # NEW: canonical gross-notional checker used by both screener & executor
    def allowed_gross_notional(
        self, symbol: str, gross_usd: float, now_ts: float | None = None
    ) -> tuple[bool, str]:
        """
        Decide if a NEW order with the given gross USD notional may be opened.
        Returns (allowed: bool, veto_reason: str) where veto_reason == "" if allowed.
        Veto reasons (taxonomy): cooldown, daily_loss_limit, portfolio_cap, symbol_cap,
        below_min_notional, trade_rate_limit
        """
        now_ts = float(now_ts or time.time())
        try:
            gross_value = float(gross_usd)
        except (TypeError, ValueError):
            _emit_veto(
                symbol,
                "invalid_notional",
                detail={"raw_gross_usd": gross_usd},
                context={"requested_gross_usd": gross_usd, "now": now_ts},
            )
            return False, "invalid_notional"

        base_context = {"requested_gross_usd": gross_value, "now": now_ts}

        # cooldown after stop
        try:
            cd_min = float(self.risk.get("cooldown_minutes_after_stop", 60) or 0)
        except Exception:
            cd_min = 60.0
        if cd_min > 0 and (now_ts - float(self._last_stop_ts)) < (60.0 * cd_min):
            cooldown_elapsed = now_ts - float(self._last_stop_ts)
            _emit_veto(
                symbol,
                "cooldown",
                detail={
                    "cooldown_minutes_after_stop": cd_min,
                    "elapsed_since_stop_sec": cooldown_elapsed,
                },
                context={**base_context, "last_stop_ts": float(self._last_stop_ts)},
            )
            return False, "cooldown"

        # daily loss stop
        try:
            day_lim = float(self.risk.get("daily_loss_limit_pct", 5) or 0)
        except Exception:
            day_lim = 5.0
        daily_loss_pct = self._daily_loss_pct()
        if day_lim > 0 and daily_loss_pct >= day_lim:
            _emit_veto(
                symbol,
                "daily_loss_limit",
                detail={
                    "daily_loss_limit_pct": day_lim,
                    "observed_daily_loss_pct": daily_loss_pct,
                },
                context=base_context,
            )
            self._last_stop_ts = now_ts
            return False, "daily_loss_limit"

        provider = self.nav_provider
        if provider is not None:
            try:
                provider.refresh()
            except Exception:
                pass
        raw_nav = float(self._portfolio_nav())
        nav = max(raw_nav, 1.0)

        guard_multiplier = 0.8 if os.environ.get("EVENT_GUARD", "0") == "1" else 1.0
        nav_context = {
            **base_context,
            "nav": raw_nav,
            "guard_multiplier": guard_multiplier,
        }

        sym_key = str(symbol).upper()
        floor = float(self.min_gross_usd_per_order or 0.0)
        floor = max(floor, float(self.per_symbol_min_gross_usd.get(sym_key, 0.0)))
        min_floor = max(float(self.min_notional), floor)
        if min_floor > 0.0 and gross_value + 1e-9 < min_floor:
            _emit_veto(
                symbol,
                "below_min_notional",
                detail={
                    "min_gross_usd": min_floor,
                    "requested_gross_usd": gross_value,
                },
                context=nav_context,
            )
            return False, "below_min_notional"

        max_trade_pct = _normalize_pct(self.sizing.get("max_trade_nav_pct", 0.0)) * guard_multiplier
        if max_trade_pct > 0.0 and nav > 0.0:
            if gross_value > nav * (max_trade_pct / 100.0):
                _emit_veto(
                    symbol,
                    "trade_gt_max_trade_nav_pct",
                    detail={
                        "max_trade_nav_pct": max_trade_pct,
                        "requested_gross_usd": gross_value,
                        "nav": nav,
                    },
                    context=nav_context,
                )
                return False, "trade_gt_max_trade_nav_pct"

        additional_pct = (gross_value / nav) * 100.0 if nav > 0 else float("inf")

        max_sym_pct = _normalize_pct(self.sizing.get("max_symbol_exposure_pct", 50)) * guard_multiplier
        if max_sym_pct > 0:
            symbol_pct = float(self._symbol_exposure_pct(symbol))
            if symbol_pct + additional_pct > max_sym_pct:
                _emit_veto(
                    symbol,
                    "symbol_cap",
                    detail={
                        "max_symbol_exposure_pct": max_sym_pct,
                        "current_symbol_pct": symbol_pct,
                        "incoming_pct": additional_pct,
                    },
                    context={**nav_context, "symbol_pct": symbol_pct},
                )
                return False, "symbol_cap"

        current_gross = float(self._portfolio_gross_usd())
        if current_gross <= 0.0 and nav > 0.0:
            try:
                current_gross = max(0.0, float(self._gross_exposure_pct())) * nav / 100.0
            except Exception:
                current_gross = 0.0
        max_gross_pct = _normalize_pct(self.sizing.get("max_gross_exposure_pct", 150)) * guard_multiplier
        if nav > 0 and max_gross_pct > 0:
            current_pct = (current_gross / nav) * 100.0
            if current_pct + additional_pct > max_gross_pct:
                _emit_veto(
                    symbol,
                    "portfolio_cap",
                    detail={
                        "max_gross_exposure_pct": max_gross_pct,
                        "current_portfolio_pct": current_pct,
                        "incoming_pct": additional_pct,
                    },
                    context={
                        **nav_context,
                        "current_portfolio_pct": current_pct,
                        "current_gross_usd": current_gross,
                    },
                )
                return False, "portfolio_cap"

        # rate limit per symbol/hour
        key = (str(symbol), int(self._now_hour_key()))
        self._trade_counts[key] = int(self._trade_counts.get(key, 0)) + 1
        try:
            max_trades = int(float(self.risk.get("max_trades_per_symbol_per_hour", 6) or 0))
        except Exception:
            max_trades = 6
        if max_trades > 0 and self._trade_counts[key] > max_trades:
            _emit_veto(
                symbol,
                "trade_rate_limit",
                detail={
                    "max_trades_per_symbol_per_hour": max_trades,
                    "observed_count": self._trade_counts[key],
                },
                context=nav_context,
            )
            return False, "trade_rate_limit"

        return True, ""
