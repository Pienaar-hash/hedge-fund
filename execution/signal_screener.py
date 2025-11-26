#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple, Mapping, Optional

import os
from .exchange_utils import get_klines, get_price, get_symbol_filters
from .orderbook_features import evaluate_entry_gate
from .signal_generator import (
    normalize_intent as generator_normalize_intent,
    allow_trade,
)
from .universe_resolver import (
    resolve_allowed_symbols,
    symbol_tier,
    is_listed_on_futures,
    symbol_min_gross,
    symbol_min_notional,
    symbol_target_leverage,
)
from .nav import PortfolioSnapshot, nav_health_snapshot
from execution.v6_flags import get_flags, log_v6_flag_snapshot
from execution.risk_limits import check_order

try:
    from .ml.predict import score_symbol as _score_symbol
except Exception:  # pragma: no cover - optional dependency
    _score_symbol = None

LOG_TAG = "[screener]"
LOGGER = logging.getLogger("signal_screener")
LOGGER.setLevel(logging.INFO)
_DEDUP_CACHE: "OrderedDict[Tuple[str, str, str, str], float]" = OrderedDict()
_DEDUP_MAX_SIZE = 2048
_ENTRY_GATE_NAME = "orderbook"
ORDERBOOK_ALIGNMENT_THRESHOLD = 0.20
DEFAULT_Z_MIN = 0.8
DEFAULT_RSI_BAND = (30.0, 70.0)
RISK_ENGINE_V6_ENABLED = get_flags().risk_engine_v6_enabled
_RISK_ENGINE_V6 = None


def _tf_seconds(tf: str | None) -> float:
    if not tf:
        return 60.0
    tf = tf.strip().lower()
    if not tf:
        return 60.0
    digits = ""
    unit = ""
    for ch in tf:
        if ch.isdigit():
            digits += ch
        else:
            unit += ch
    try:
        value = float(digits or 0)
    except Exception:
        value = 0.0
    if value <= 0:
        value = 1.0
    unit = unit or "m"
    if unit in ("s", "sec", "secs"):
        mult = 1.0
    elif unit in ("m", "min", "mins"):
        mult = 60.0
    elif unit in ("h", "hr", "hrs"):
        mult = 3600.0
    elif unit in ("d", "day", "days"):
        mult = 86400.0
    else:
        mult = 60.0
    return max(value * mult, 1.0)


def _dedupe_key(symbol: str, timeframe: str, side: str, candle_close: Any) -> Tuple[str, str, str, str]:
    return (str(symbol).upper(), str(timeframe).lower(), str(side).upper(), str(candle_close))


def _dedupe_prune(now: float) -> None:
    while _DEDUP_CACHE:
        _, expires_at = next(iter(_DEDUP_CACHE.items()))
        if expires_at > now:
            break
        _DEDUP_CACHE.popitem(last=False)
    while len(_DEDUP_CACHE) > _DEDUP_MAX_SIZE:
        _DEDUP_CACHE.popitem(last=False)


def _entry_gate_result(symbol: str, side: str, *, enabled: bool) -> tuple[bool, Dict[str, Any]]:
    """Call the orderbook gate while guarding return semantics."""

    try:
        raw = evaluate_entry_gate(symbol, side, enabled=enabled)
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"gate": _ENTRY_GATE_NAME, "ok": True, "error": str(exc)}

    if not isinstance(raw, tuple) or len(raw) != 2:
        return False, {"gate": _ENTRY_GATE_NAME, "ok": True, "error": "invalid_gate_return"}

    veto, info = raw
    if not isinstance(info, dict):
        info = {"detail": info}
    info.setdefault("gate", _ENTRY_GATE_NAME)
    info.setdefault("symbol", str(symbol).upper())
    if "ok" not in info:
        info["ok"] = not bool(veto)
    return bool(veto), info


def _normalize_pct(value: Any) -> float:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < pct <= 1.0:
        return pct * 100.0
    return pct


def _nav_fraction(value: Any) -> float:
    """Normalize NAV-based pct inputs; 10 -> 0.10, 0.02 -> 0.02."""
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pct <= 0.0:
        return 0.0
    if pct > 1.0:
        return min(pct / 100.0, 1.0)
    return pct


def _extract_trade_nav_caps(risk_cfg: Mapping[str, Any] | None) -> Tuple[float, float]:
    if not isinstance(risk_cfg, Mapping):
        return 0.0, 0.0
    global_cfg = risk_cfg.get("global")
    if not isinstance(global_cfg, Mapping):
        global_cfg = {}
    trade_equity = _normalize_pct(global_cfg.get("trade_equity_nav_pct") or 0.0)
    max_trade = _normalize_pct(global_cfg.get("max_trade_nav_pct") or 0.0)
    return trade_equity, max_trade


def _positions_by_side(
    positions: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for raw in positions or []:
        try:
            sym = str(raw.get("symbol", "")).upper()
            if not sym:
                continue
            side = str(raw.get("positionSide", "BOTH")).upper()
            if side not in ("LONG", "SHORT"):
                continue
            qty = float(raw.get("qty", raw.get("positionAmt", 0.0)) or 0.0)
            if qty == 0.0:
                continue
            abs_qty = abs(qty)
            mark = float(raw.get("markPrice") or raw.get("entryPrice") or 0.0)
            entry = float(raw.get("entryPrice") or mark or 0.0)
            notional = abs_qty * abs(mark)
            out.setdefault(sym, {})[side] = {
                "qty": abs_qty,
                "mark": abs(mark),
                "entry": abs(entry),
                "notional": notional,
            }
        except Exception:
            continue
    return out


def _reduce_plan(
    symbol: str,
    signal: str,
    timeframe: str,
    positions: Dict[str, Dict[str, float]],
    fallback_price: float,
    nav_usd: float,
    min_notional: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Return (reduce_intents, notional_delta)."""
    desired_side = "LONG" if signal == "BUY" else "SHORT"
    opposite_side = "SHORT" if desired_side == "LONG" else "LONG"
    opp = positions.get(opposite_side, {}) if positions else {}
    qty = float(opp.get("qty", 0.0) or 0.0)
    if qty <= 0.0:
        return [], 0.0
    mark = float(opp.get("mark", 0.0) or 0.0)
    mark = mark if mark > 0 else float(fallback_price)
    if mark <= 0:
        return [], 0.0
    notional = float(opp.get("notional", qty * mark) or (qty * mark))
    reduce_signal = "BUY" if opposite_side == "SHORT" else "SELL"
    intent = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": reduce_signal,
        "reduceOnly": True,
        "positionSide": opposite_side,
        "price": mark,
        "per_trade_nav_pct": (notional / nav_usd) if nav_usd > 0 else 0.0,
        "leverage": 1.0,
        "min_notional": min_notional,
        "source": "auto_reduce",
    }
    return [intent], notional

_ = get_flags()
try:
    log_v6_flag_snapshot(LOGGER)
except Exception:
    LOGGER.debug("v6 flag snapshot logging failed", exc_info=True)


def _strategy_params(entry: Mapping[str, Any]) -> Dict[str, Any]:
    params = entry.get("params")
    if isinstance(params, Mapping):
        return dict(params)
    return {}
def _load_strategy_list() -> List[Dict[str, Any]]:
    scfg = json.load(open("config/strategy_config.json"))
    raw = scfg.get("strategies") or []
    if isinstance(raw, dict):
        iterable = list(raw.values())
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []
    lst = []
    for entry in iterable:
        if not isinstance(entry, dict) or not entry.get("enabled"):
            continue
        strat = dict(entry)
        strat["params"] = _strategy_params(entry)
        lst.append(strat)
    # Prefer universe_resolver for allowed symbols (handles listing + throttles)
    try:
        allowed, _ = resolve_allowed_symbols()
        allow_set = {s.upper() for s in (allowed or [])}
        if allow_set:
            lst = [
                s
                for s in lst
                if isinstance(s, dict)
                and str(s.get("symbol", "")).upper() in allow_set
            ]
    except Exception:
        # If resolver fails (offline), fall through; we'll veto not_listed later
        pass
    return lst


def _load_registry_entries() -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.load(open("config/strategy_registry.json"))
    except Exception:
        return {}
    raw = payload.get("strategies") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Mapping):
        return {}
    entries: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            entries[str(key)] = dict(value)
    return entries


def _strategy_concurrency_budget() -> int:
    registry = _load_registry_entries()
    total = 0
    if registry:
        for entry in registry.values():
            if not entry.get("enabled") or entry.get("sandbox"):
                continue
            try:
                val = int(float(entry.get("max_concurrent", 0) or 0))
            except Exception:
                val = 0
            if val <= 0:
                val = 1
            total += val
    if total <= 0:
        strategies = _load_strategy_list()
        for entry in strategies:
            if not entry.get("enabled"):
                continue
            total += 1
    return max(total, 0)


def _load_risk_cfg() -> Dict[str, Any]:
    from execution.risk_loader import load_risk_config

    cfg = load_risk_config()
    if not isinstance(cfg, dict):
        cfg = {}
    derived = _strategy_concurrency_budget()
    if derived > 0:
        global_cfg = cfg.setdefault("global", {})
        try:
            existing = int(float(global_cfg.get("max_concurrent_positions") or 0))
        except Exception:
            existing = 0
        if existing <= 0 or derived < existing:
            global_cfg["max_concurrent_positions"] = derived
    return cfg


def _load_pairs_cfg() -> Dict[str, Any]:
    try:
        payload = json.load(open(os.getenv("PAIRS_UNIVERSE_CONFIG", "config/pairs_universe.json")))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _get_risk_engine_v6(_cfg: Optional[Mapping[str, Any]] = None):
    return _RISK_ENGINE_V6


def would_emit(
    symbol: str,
    side: str,
    *,
    notional: float = 10.0,
    lev: float = 20.0,
    nav: float = 0.0,
    open_positions_count: int = 0,
    current_gross_notional: float = 0.0,
    current_tier_gross_notional: float = 0.0,
    orderbook_gate: bool = True,
    timeframe: str | None = None,
    candle_close_ts: float | None = None,
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Return (would_emit, reasons, extra_info) for a hypothetical entry now.

    - Applies: listing check, orderbook gate, and risk caps (portfolio, tier, concurrent).
    - Uses gross notional = notional * lev for caps.
    - Optionally de-duplicates intents per (symbol, timeframe, side, candle_close_ts).
    """
    sym = str(symbol).upper()
    extra: Dict[str, Any] = {}
    reasons: List[str] = []

    if not is_listed_on_futures(sym):
        return False, ["not_listed"], extra

    veto, info = _entry_gate_result(sym, side, enabled=orderbook_gate)
    metric = float(info.get("metric", 0.0) or 0.0)
    if veto:
        reasons.append("ob_adverse")
        extra["metric"] = metric
    else:
        if (side.upper() in ("BUY", "LONG") and metric > ORDERBOOK_ALIGNMENT_THRESHOLD) or (
            side.upper() in ("SELL", "SHORT") and metric < -ORDERBOOK_ALIGNMENT_THRESHOLD
        ):
            extra["flag"] = "ob_aligned"
            extra["metric"] = metric

    ml_cfg = {}
    base_cfg = {}
    try:
        base_cfg = json.load(open("config/strategy_config.json"))
        ml_cfg = base_cfg.get("ml", {}) or {}
    except Exception:
        ml_cfg = {}

    if ml_cfg.get("enabled") and _score_symbol is not None:
        try:
            ml_symbol = sym if sym.endswith("USDT") else f"{sym}USDT"
            ml_result = _score_symbol(base_cfg, ml_symbol)
            extra["ml"] = ml_result
            prob = ml_result.get("p")
            threshold = float(ml_cfg.get("prob_threshold", 0.0) or 0.0)
            if prob is not None:
                prob = float(prob)
                extra["ml_p"] = prob
                if prob < threshold:
                    reasons.append(f"ml_p<{threshold:.2f}")
        except Exception as exc:  # pragma: no cover - ML optional
            extra["ml_error"] = str(exc)

    overall_ok = len(reasons) == 0

    if overall_ok and timeframe and candle_close_ts is not None:
        now_ts = time.time()
        _dedupe_prune(now_ts)
        key = _dedupe_key(sym, timeframe, side, candle_close_ts)
        expires_at = _DEDUP_CACHE.get(key)
        if expires_at and expires_at > now_ts:
            reasons.append("dedupe")
            overall_ok = False
        else:
            ttl = max(_tf_seconds(timeframe) * 3.0, 60.0)
            _DEDUP_CACHE[key] = now_ts + ttl
            _DEDUP_CACHE.move_to_end(key)

    return overall_ok, reasons, extra


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-i] - closes[-i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


def _zscore(closes: List[float], lookback: int = 20) -> float:
    if len(closes) < lookback:
        return 0.0
    seg = closes[-lookback:]
    mean = sum(seg) / lookback
    var = sum((x - mean) ** 2 for x in seg) / lookback
    std = (var**0.5) or 1.0
    return (closes[-1] - mean) / std


def _trend_filter(closes: List[float], period: int = 20) -> str:
    """Determine trend direction based on SMA crossover.
    
    Returns:
        'BULL' if price > SMA (uptrend)
        'BEAR' if price < SMA (downtrend)
        'NEUTRAL' if insufficient data
    """
    if len(closes) < period:
        return "NEUTRAL"
    sma = sum(closes[-period:]) / period
    if closes[-1] > sma * 1.005:  # 0.5% buffer to reduce noise
        return "BULL"
    elif closes[-1] < sma * 0.995:
        return "BEAR"
    return "NEUTRAL"


def _position_pnl_pct(
    positions: Dict[str, Dict[str, float]],
    signal: str,
    current_price: float,
) -> Optional[float]:
    """Calculate unrealized PnL % for existing position in same direction as signal.
    
    Returns None if no position exists in the signal direction.
    """
    # Signal direction matches position side we'd be adding to
    position_side = "LONG" if signal == "BUY" else "SHORT"
    pos = positions.get(position_side, {})
    if not pos:
        return None
    
    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return None
    
    # Get entry price from mark (which is used as proxy in _positions_by_side)
    # Note: mark here is actually markPrice, we need entryPrice for accurate PnL
    # For now, use notional/qty as approximate entry
    notional = float(pos.get("notional", 0.0) or 0.0)
    if notional <= 0 or current_price <= 0:
        return None
    
    # mark stored is markPrice; approximate entry from stored data
    mark = float(pos.get("mark", 0.0) or 0.0)
    if mark <= 0:
        return None
    
    # For LONG: pnl = (current - entry) / entry
    # For SHORT: pnl = (entry - current) / entry
    # Since we only have mark (current), we can't compute exact PnL
    # Use mark as approximate entry and compare to live price
    if position_side == "LONG":
        pnl_pct = (current_price - mark) / mark if mark > 0 else 0.0
    else:  # SHORT
        pnl_pct = (mark - current_price) / mark if mark > 0 else 0.0
    
    return pnl_pct


def generate_signals_from_config() -> Iterable[Dict[str, Any]]:
    try:
        strategies = _load_strategy_list()
    except Exception as e:
        print(f"{LOG_TAG} error loading config: {e}")
        return IntentBatch([], 0)

    try:
        base_cfg = json.load(open("config/strategy_config.json"))
    except Exception:
        base_cfg = {}

    ml_cfg = (base_cfg.get("ml") or {})
    ml_enabled = bool(ml_cfg.get("enabled")) and _score_symbol is not None
    out: List[Dict[str, Any]] = []
    attempted = 0
    dbg = os.getenv("DEBUG_SIGNALS", "0").lower() in ("1", "true", "yes")

    # Prepare allowed set and tier map (best effort)
    try:
        allowed_syms, tier_by = resolve_allowed_symbols()
        allowed_set = {s.upper() for s in (allowed_syms or [])}
    except Exception:
        allowed_set = set()
        tier_by = {}

    # Risk cfg + portfolio snapshot (best effort)
    rlc = _load_risk_cfg()
    kill_switch = os.environ.get("KILL_SWITCH", "0").lower() in ("1", "true", "yes", "on")
    snapshot = PortfolioSnapshot(base_cfg)
    try:
        nav_override = float(os.getenv("SCREENER_NAV", "0") or 0.0)
    except Exception:
        nav_override = 0.0
    nav_health = nav_health_snapshot()
    nav = float(nav_override or nav_health.get("nav_total") or 0.0)
    nav_age = nav_health.get("age_s")
    nav_sources_ok = bool(nav_health.get("sources_ok", True))
    nav_unknown = nav <= 0.0 or not nav_sources_ok
    nav_detail_flag = {
        "nav_unknown": nav_unknown,
        "nav_age_s": nav_age,
        "nav_sources_ok": nav_sources_ok,
    }

    try:
        current_portfolio_gross = float(snapshot.current_gross_usd())
    except Exception:
        current_portfolio_gross = 0.0
    open_positions_count = 0
    tier_gross: Dict[str, float] = {}
    positions_by_symbol: Dict[str, Dict[str, Dict[str, float]]] = {}
    try:
        from .exchange_utils import get_positions

        pos = list(get_positions() or [])
        positions_by_symbol = _positions_by_side(pos)
        gross_from_positions = 0.0
        for p in pos:
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
            if abs(qty) <= 0:
                continue
            mark = float(p.get("markPrice") or p.get("entryPrice") or 0.0)
            if mark <= 0:
                mark = abs(float(p.get("entryPrice") or 0.0))
            g = abs(qty) * abs(mark)
            gross_from_positions += g
            open_positions_count += 1
            symp = str(p.get("symbol", "")).upper()
            t = tier_by.get(symp) or symbol_tier(symp) or "?"
            tier_gross[t] = tier_gross.get(t, 0.0) + g
        if gross_from_positions > 0:
            current_portfolio_gross = gross_from_positions
    except Exception:
        pass
    ml_threshold = float(ml_cfg.get("prob_threshold", 0.0) or 0.0)

    def _decide_signal(z: float, rsi: float, entry_cfg: Mapping[str, Any], entry_forced: bool) -> Optional[str]:
        band_raw = entry_cfg.get("band") or DEFAULT_RSI_BAND
        low, high = DEFAULT_RSI_BAND
        try:
            if isinstance(band_raw, (list, tuple)) and len(band_raw) >= 2:
                low = float(band_raw[0])
                high = float(band_raw[1])
        except Exception:
            low, high = DEFAULT_RSI_BAND
        zmin_raw = entry_cfg.get("zmin", DEFAULT_Z_MIN)
        try:
            zmin = float(zmin_raw)
        except Exception:
            zmin = DEFAULT_Z_MIN
        if z <= -zmin:
            return "BUY"
        if z >= zmin:
            return "SELL"
        if rsi <= low:
            return "BUY"
        if rsi >= high:
            return "SELL"
        if entry_forced:
            return "BUY" if z <= 0 else "SELL"
        return None
    for scfg in strategies:
        attempted += 1
        sym = scfg.get("symbol")
        sym_key = str(sym).upper()
        tf = scfg.get("timeframe", "15m")
        params = scfg.get("params") or {}
        per_trade_nav_pct = _nav_fraction(params.get("per_trade_nav_pct"))
        lev = float(params.get("leverage") or symbol_target_leverage(sym_key) or 1.0)
        if lev <= 0:
            lev = 1.0
        per_symbol_cfg = {}
        try:
            per_symbol_cfg = ((rlc.get("per_symbol") or {}) if isinstance(rlc, Mapping) else {}).get(sym_key, {}) or {}
        except Exception:
            per_symbol_cfg = {}
        entry_forced = (params.get("entry", {}) or {}).get("type") == "always_on"
        if kill_switch:
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["kill_switch_on"]}}'
            )
            continue
        if allowed_set and sym_key not in allowed_set:
            print(f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["not_listed"]}}')
            continue
        try:
            kl = get_klines(sym, tf, limit=150)
            closes = [row[4] for row in kl]
            price = get_price(sym)
            rsi = _rsi(closes, 14)
            z = _zscore(closes, 20)
            trend = _trend_filter(closes, 20)
            filters = get_symbol_filters(sym)
            notional_filter = (
                filters.get("MIN_NOTIONAL")
                or filters.get("NOTIONAL")
                or {}
            )
            exch_min_notional = float(
                (notional_filter.get("minNotional")
                or notional_filter.get("notional")
                or 0.0)
            )
            lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE") or {}
            step = float(lot.get("stepSize", 0.0) or 0.0)
            min_qty = float(lot.get("minQty", 0.0) or 0.0)
            qty_floor = max(min_qty, step)
            min_qty_notional = price * qty_floor if price > 0 else 0.0
        except Exception as e:
            print(f"{LOG_TAG} {sym} {tf} error: {e}")
            continue
        sym_min_notional = symbol_min_notional(sym_key)
        sym_floor = max(symbol_min_gross(sym_key), 0.0)
        min_notional = max(sym_floor, sym_min_notional, exch_min_notional, min_qty_notional)
        requested_notional = max(nav * per_trade_nav_pct, min_notional) if nav > 0 else min_notional
        try:
            print(
                f"[sigdbg] sym={sym} tf={tf} price={price:.4f} nav={nav:.4f} "
                f"per_trade_nav_pct={per_trade_nav_pct:.4f} lev={lev:.2f} "
                f"min_notional={min_notional:.4f} open_gross={current_portfolio_gross:.4f}"
            )
        except Exception:
            LOGGER.debug("sigdbg log failed", exc_info=True)

        # Execution hardening gates
        if not allow_trade(sym_key):
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["signal_gate"]}}'
            )
            continue

        entry_cfg = params.get("entry") or {}
        signal = _decide_signal(z, rsi, entry_cfg, entry_forced)
        if signal is None:
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","z":{round(z, 4)},"rsi":{round(rsi, 1)},"veto":["no_cross"]}}'
            )
            continue

        # Trend filter: block counter-trend entries
        is_counter_trend = False
        if signal == "SELL" and trend == "BULL":
            is_counter_trend = True
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","signal":"{signal}","trend":"{trend}","veto":["counter_trend"]}}'
            )
            continue
        if signal == "BUY" and trend == "BEAR":
            is_counter_trend = True
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","signal":"{signal}","trend":"{trend}","veto":["counter_trend"]}}'
            )
            continue

        # No-add-to-loser: block adding to underwater positions
        sym_positions = positions_by_symbol.get(sym_key, {})
        position_side = "LONG" if signal == "BUY" else "SHORT"
        existing_pos = sym_positions.get(position_side, {})
        if existing_pos:
            entry_px = float(existing_pos.get("entry", 0.0) or 0.0)
            if entry_px > 0 and price > 0:
                if position_side == "LONG":
                    unrealized_pct = (price - entry_px) / entry_px
                else:  # SHORT
                    unrealized_pct = (entry_px - price) / entry_px
                # Block if position is more than 2% underwater
                if unrealized_pct < -0.02:
                    print(
                        f'[decision] {{"symbol":"{sym}","tf":"{tf}","signal":"{signal}","unrealized_pct":{round(unrealized_pct*100, 2)},"veto":["no_add_to_loser"]}}'
                    )
                    continue
        # Optional orderbook entry gate (veto/boost)
        feat = params.get("features", {}) if isinstance(params, dict) else {}
        ob_enabled = bool(feat.get("orderbook_gate"))
        veto, info = _entry_gate_result(sym, signal, enabled=ob_enabled)
        if veto:
            if dbg:
                print(f"[sigdbg] {sym} tf={tf} ob_imbalance={float(info.get('metric',0.0)):.3f} veto")
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["ob_adverse"],"metric":{round(float(info.get("metric",0.0)),3)}}}'
            )
            continue
        else:
            m = float(info.get("metric", 0.0) or 0.0)
            if (signal == "BUY" and m > ORDERBOOK_ALIGNMENT_THRESHOLD) or (
                signal == "SELL" and m < -ORDERBOOK_ALIGNMENT_THRESHOLD
            ):
                print(
                    f'[decision] {{"symbol":"{sym}","tf":"{tf}","flag":"ob_aligned","metric":{round(m,3)}}}'
                )

        ml_prob = None
        if ml_enabled:
            try:
                ml_sym = sym if sym.endswith("USDT") else f"{sym}USDT"
                ml_result = _score_symbol(base_cfg, ml_sym)
                ml_prob = ml_result.get("p")
                if ml_prob is not None:
                    ml_prob = float(ml_prob)
                    if ml_prob < ml_threshold:
                        print(
                            f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["ml_prob"],"p":{round(ml_prob,4)}}}'
                        )
                        continue
                if ml_result.get("error") and dbg:
                    print(f"{LOG_TAG} ml error {sym}: {ml_result['error']}")
            except Exception as exc:
                if dbg:
                    print(f"{LOG_TAG} ml exception {sym}: {exc}")
                ml_prob = None

        tname = symbol_tier(sym)
        tier_key = tname or "?"
        reduce_intents, reduce_notional = _reduce_plan(
            sym,
            signal,
            tf,
            positions_by_symbol.get(sym_key, {}),
            price,
            nav,
            min_notional,
        )
        if reduce_intents:
            out.extend(reduce_intents)
            current_portfolio_gross = max(current_portfolio_gross - reduce_notional, 0.0)
            tier_gross[tier_key] = max(tier_gross.get(tier_key, 0.0) - reduce_notional, 0.0)
            open_positions_count = max(open_positions_count - 1, 0)
        gross_usd = requested_notional  # unlevered sizing; leverage is metadata only
        qty_est = gross_usd / price if price > 0 else 0.0
        sizing_notes = {
            "floors": {
                "symbol_min_gross": sym_floor,
                "symbol_min_notional": sym_min_notional,
                "exchange_min_notional": exch_min_notional,
                "min_qty_notional": min_qty_notional,
            },
            "nav_used": nav,
            "nav_age_s": nav_age,
            "nav_sources_ok": nav_sources_ok,
        }
        intent = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "symbol": sym,
            "timeframe": tf,
            "signal": signal,
            "reduceOnly": False,
            "price": price,
            "leverage": lev,
            "min_notional": min_notional,
            "per_trade_nav_pct": per_trade_nav_pct,
            "gross_usd": gross_usd,
            "qty": abs(qty_est),
            "nav_used": nav,
            "nav_age_s": nav_age,
            "nav_sources_ok": nav_sources_ok,
            "price_used": price,
            "symbol_caps": (rlc.get("per_symbol") or {}).get(sym_key, {}) if isinstance(rlc, Mapping) else {},
            "sizing_notes": sizing_notes,
            "per_symbol_limits": rlc.get("per_symbol") if isinstance(rlc, Mapping) else {},
        }
        if ml_prob is not None:
            intent["ml_prob"] = ml_prob
        # Add trend context to intent for downstream tracking
        intent["trend"] = trend
        intent["trend_aligned"] = (
            (signal == "BUY" and trend == "BULL") or
            (signal == "SELL" and trend == "BEAR")
        )
        if dbg:
            print(
                f"[sigdbg] {sym} tf={tf} z={round(z,3)} rsi={round(rsi,1)} trend={trend} pct={per_trade_nav_pct} lev={lev} ok"
            )
        print(f"{LOG_TAG} {sym} {tf} z={round(z, 3)} rsi={round(rsi, 1)} trend={trend}")
        decision_payload = {
            "symbol": sym,
            "tf": tf,
            "intent": intent,
            "detail": nav_detail_flag if nav_unknown else {},
        }
        print(f'[decision] {json.dumps(decision_payload)}')
        out.append(intent)
        current_portfolio_gross += gross_usd
        tier_gross[tier_key] = tier_gross.get(tier_key, 0.0) + gross_usd
        open_positions_count += 1
        positions_by_symbol.setdefault(sym_key, {})[
            "LONG" if signal == "BUY" else "SHORT"
        ] = {
            "qty": abs(qty_est),
            "mark": price,
            "entry": price,  # New position entry = current price
            "notional": gross_usd,
        }
    print(f"{LOG_TAG} attempted={attempted} emitted={len(out)}")
    return IntentBatch(out, attempted)


class IntentBatch(list):
    """List-like container that also carries screener attempt metadata."""

    def __init__(self, intents: Iterable[Mapping[str, Any]], attempted: int) -> None:
        super().__init__(list(intents))
        self.attempted = attempted
        self.emitted = len(self)


def generate_intents(
    now: float | None = None,
    universe: Sequence[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
    unified: bool = True,
) -> IntentBatch:
    """
    Canonical entry point used by the executor.

    config/runtime.yaml (signal source) -> generate_intents() -> screener gates -> intents list.
    """
    return generate_signals_from_config()


def run_once(now: float | None = None) -> dict:
    """
    Compute trading intents for the current universe and return a summary.

    Returns:
        {
          "attempted": int,
          "emitted": int,
          "intents": list[dict],
        }
    """
    try:
        strategies = _load_strategy_list()
        attempted_count = len(strategies)
    except Exception:
        attempted_count = 0

    try:
        intents = list(generate_signals_from_config())
    except Exception:
        intents = []

    attempted_attr = getattr(intents, "attempted", None)
    emitted_attr = getattr(intents, "emitted", None)
    normalized: List[Dict[str, Any]] = []
    for raw in intents:
        try:
            norm = generator_normalize_intent(raw)
        except Exception:
            continue
        summary: Dict[str, Any] = {
            "symbol": norm.get("symbol"),
            "side": norm.get("signal"),
            "gross_usd": norm.get("gross_usd"),
            "notional": norm.get("per_trade_nav_pct"),
            "strategy": norm.get("strategy") or norm.get("source"),
            "timeframe": norm.get("timeframe"),
            "reduce_only": norm.get("reduceOnly"),
            "raw": norm,
        }
        if "qty" in norm:
            summary["qty"] = norm.get("qty")
        normalized.append(summary)

    return {
        "attempted": attempted_count or attempted_attr or len(intents),
        "emitted": emitted_attr if emitted_attr is not None else len(normalized),
        "intents": normalized,
    }
