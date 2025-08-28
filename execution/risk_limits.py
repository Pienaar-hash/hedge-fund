from dataclasses import dataclass
from typing import Dict, List, Tuple
from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 28

@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_trade: float           # e.g., 200.0 USDT
    max_open_notional: float                # e.g., 1000.0 USDT
    max_positions: int                      # e.g., 5
    max_leverage: float                     # e.g., 5.0
    kill_switch_drawdown_pct: float         # e.g., -10.0 (portfolio)
    min_notional: float = 10.0              # exchange min

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

def _can_open_position_legacy(symbol: str, notional: float, lev: float,
                              cfg: RiskConfig, st: RiskState) -> tuple[bool, str]:
    if st.portfolio_drawdown_pct <= cfg.kill_switch_drawdown_pct:
        return False, "kill_switch_triggered"
    if notional < cfg.min_notional:
        return False, "below_min_notional"
    if notional > cfg.max_notional_per_trade:
        return False, "exceeds_per_trade_cap"
    if lev > cfg.max_leverage:
        return False, "exceeds_leverage_cap"
    if (st.open_notional + notional) > cfg.max_open_notional:
        return False, "exceeds_open_notional_cap"
    if st.open_positions >= cfg.max_positions:
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
    symbol = kwargs.get('symbol', args[0] if len(args) > 0 else None)
    notional = kwargs.get('notional', args[1] if len(args) > 1 else None)
    lev = kwargs.get('lev', args[2] if len(args) > 2 else None)
    nav = kwargs.get('nav', args[3] if len(args) > 3 else None)
    open_qty = kwargs.get('open_qty', args[4] if len(args) > 4 else None)
    now = kwargs.get('now', args[5] if len(args) > 5 else None)
    cfg = kwargs.get('cfg', args[6] if len(args) > 6 else None)
    state = kwargs.get('state', args[7] if len(args) > 7 else None)
    current_gross_notional = kwargs.get('current_gross_notional', args[8] if len(args) > 8 else 0.0)

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

def will_violate_exposure(current_gross: float, add_notional: float, nav: float, max_nav_pct: float) -> bool:
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
) -> Tuple[bool, dict]:
    """Apply per-symbol and global risk checks.

    Returns (ok, details) where details has keys: reasons: list[str], notional: float, cooldown_until?: float
    """
    reasons: List[str] = []
    details: Dict[str, float | List[str]] = {"reasons": reasons, "notional": float(requested_notional)}

    sym = str(symbol)
    s_cfg = _cfg_get(cfg, ["per_symbol", sym], {}) or {}
    g_cfg = _cfg_get(cfg, ["global"], {}) or {}

    # Per-order notional constraints
    min_notional = float(s_cfg.get("min_notional", 0.0) or 0.0)
    max_order_notional = float(s_cfg.get("max_order_notional", 0.0) or 0.0)
    req_notional = float(requested_notional)

    if min_notional > 0.0 and req_notional < min_notional:
        reasons.append("min_notional")

    if max_order_notional > 0.0 and req_notional > max_order_notional:
        reasons.append("max_order_notional")

    # Open quantity cap (applies to increasing long exposure)
    max_open_qty = s_cfg.get("max_open_qty", None)
    if max_open_qty is not None:
        try:
            max_open_qty_f = float(max_open_qty)
        except Exception:
            max_open_qty_f = None
        if max_open_qty_f is not None:
            if str(side).upper() in ("BUY", "LONG") and float(open_qty) >= max_open_qty_f:
                reasons.append("max_open_qty")

    # Per-symbol cooldown after last fill
    cooldown_sec = int(float(s_cfg.get("cooldown_sec", 0) or 0))
    if cooldown_sec > 0:
        last_fill = getattr(state, "last_fill_ts", lambda _s: 0.0)(sym)
        if last_fill > 0.0:
            cooldown_until = last_fill + float(cooldown_sec)
            if float(now) < cooldown_until:
                reasons.append("cooldown")
                details["cooldown_until"] = float(cooldown_until)

    # Error circuit breaker (global)
    err_cfg = g_cfg.get("error_circuit", {}) or {}
    max_errors = int(float(err_cfg.get("max_errors", 0) or 0))
    window_sec = int(float(err_cfg.get("window_sec", 0) or 0))
    if max_errors > 0 and window_sec > 0:
        errors_in = getattr(state, "errors_in", lambda _w, _n: 0)
        if errors_in(window_sec, float(now)) >= max_errors:
            reasons.append("circuit_breaker")

    # Gross exposure cap (global)
    max_gross_nav_pct = float(g_cfg.get("max_gross_nav_pct", 0.0) or 0.0)
    if max_gross_nav_pct > 0.0:
        if will_violate_exposure(float(current_gross_notional), req_notional, float(nav), max_gross_nav_pct):
            reasons.append("max_gross_nav_pct")

    ok = len(reasons) == 0
    return ok, details
