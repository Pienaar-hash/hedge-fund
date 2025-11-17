#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple, Mapping, Optional

import os
from .exchange_utils import get_klines, get_price, get_symbol_filters
from .orderbook_features import evaluate_entry_gate
from .signal_generator import (
    normalize_intent as generator_normalize_intent,
    allow_trade,
    size_for,
)
from .universe_resolver import (
    resolve_allowed_symbols,
    symbol_tier,
    is_listed_on_futures,
    symbol_min_gross,
    symbol_min_notional,
    symbol_target_leverage,
)
from .risk_limits import (
    check_order,
    RiskState,
    RiskGate,
    symbol_notional_guard,
    symbol_dd_guard,
)
from .risk_engine_v6 import OrderIntent, RiskEngineV6
from .nav import PortfolioSnapshot

try:
    from .ml.predict import score_symbol as _score_symbol
except Exception:  # pragma: no cover - optional dependency
    _score_symbol = None

LOG_TAG = "[screener]"
_DEDUP_CACHE: "OrderedDict[Tuple[str, str, str, str], float]" = OrderedDict()
_DEDUP_MAX_SIZE = 2048
_ENTRY_GATE_NAME = "orderbook"
ORDERBOOK_ALIGNMENT_THRESHOLD = 0.20


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
            notional = abs_qty * abs(mark)
            out.setdefault(sym, {})[side] = {
                "qty": abs_qty,
                "mark": abs(mark),
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
        "capital_per_trade": notional,
        "leverage": 1.0,
        "gross_usd": notional,
        "source": "auto_reduce",
    }
    return [intent], notional


_SCREENER_RISK_STATE = RiskState()
_SCREENER_GATE = RiskGate({"sizing": {}, "risk": {}})
RISK_ENGINE_V6_ENABLED = os.getenv("RISK_ENGINE_V6_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
_RISK_ENGINE_V6: Optional[RiskEngineV6] = None
_RISK_ENGINE_V6_CFG_DIGEST: Optional[str] = None


def _strategy_params(entry: Mapping[str, Any]) -> Dict[str, Any]:
    params = entry.get("params")
    if isinstance(params, Mapping):
        return dict(params)
    return {}


def _update_screener_risk_state(
    snapshot: PortfolioSnapshot,
    open_notional: float,
    open_positions: int,
) -> None:
    _SCREENER_GATE.nav_provider = snapshot
    try:
        loss_pct = float(_SCREENER_GATE._daily_loss_pct())
    except Exception:
        loss_pct = 0.0
    _SCREENER_RISK_STATE.daily_pnl_pct = -loss_pct
    _SCREENER_RISK_STATE.open_notional = float(open_notional)
    _SCREENER_RISK_STATE.open_positions = int(open_positions)


def _risk_cfg_path() -> str:
    return os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")


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
    try:
        cfg = json.load(open(_risk_cfg_path()))
    except Exception:
        cfg = {}
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


def _pairs_cfg_path() -> str:
    return os.getenv("PAIRS_UNIVERSE_CONFIG", "config/pairs_universe.json")


def _load_pairs_cfg() -> Dict[str, Any]:
    try:
        payload = json.load(open(_pairs_cfg_path()))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _get_risk_engine_v6(risk_cfg: Mapping[str, Any]) -> Optional[RiskEngineV6]:
    global _RISK_ENGINE_V6, _RISK_ENGINE_V6_CFG_DIGEST
    digest = json.dumps(risk_cfg, sort_keys=True, default=str)
    if _RISK_ENGINE_V6 is None or _RISK_ENGINE_V6_CFG_DIGEST != digest:
        pairs_cfg = _load_pairs_cfg()
        _RISK_ENGINE_V6 = RiskEngineV6.from_configs(risk_cfg, pairs_cfg)
        _RISK_ENGINE_V6_CFG_DIGEST = digest
    return _RISK_ENGINE_V6


def would_emit(
    symbol: str,
    side: str,
    *,
    notional: float = 10.0,
    lev: float = 20.0,
    nav: float = 1000.0,
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

    cfg = _load_risk_cfg()
    trade_equity_cap, max_trade_cap = _extract_trade_nav_caps(cfg)
    tname = symbol_tier(sym)
    temp_state = RiskState()
    temp_state.daily_pnl_pct = _SCREENER_RISK_STATE.daily_pnl_pct
    temp_state.open_notional = float(current_gross_notional)
    temp_state.open_positions = int(open_positions_count)
    nav_f = float(nav) if isinstance(nav, (int, float)) else float(nav or 0.0)
    try:
        gross_request = float(notional) * float(lev)
    except Exception:
        gross_request = float(notional)
    trade_obs_pct = None
    if nav_f > 0.0:
        trade_obs_pct = (gross_request / nav_f) * 100.0
    trade_meta: Dict[str, Any] = {}
    if trade_equity_cap > 0.0:
        trade_meta["trade_equity_nav_pct"] = trade_equity_cap
    if max_trade_cap > 0.0:
        trade_meta["max_trade_nav_pct"] = max_trade_cap
    if trade_obs_pct is not None:
        trade_meta["trade_equity_nav_obs"] = trade_obs_pct
        trade_meta["max_trade_nav_obs"] = trade_obs_pct
    if trade_meta:
        extra.setdefault("trade_nav", {}).update(trade_meta)
    clamp_reasons: List[str] = []
    if trade_obs_pct is not None:
        if trade_equity_cap > 0.0 and trade_obs_pct > trade_equity_cap:
            clamp_reasons.append("trade_gt_equity_cap")
        if max_trade_cap > 0.0 and trade_obs_pct > max_trade_cap:
            clamp_reasons.append("trade_gt_max_trade_nav_pct")
    if clamp_reasons:
        for reason in clamp_reasons:
            if reason not in reasons:
                reasons.append(reason)
        return False, reasons, extra

    engine = _get_risk_engine_v6(cfg) if RISK_ENGINE_V6_ENABLED else None

    if engine is not None and RISK_ENGINE_V6_ENABLED:
        intent_payload = OrderIntent(
            symbol=sym,
            side=side,
            qty=float(notional),
            quote_notional=float(notional) * float(lev),
            leverage=float(lev),
            price=0.0,
            tier_name=tname,
            tier_gross_notional=float(current_tier_gross_notional),
            current_gross_notional=float(current_gross_notional),
            symbol_open_qty=0.0,
            nav_usd=float(nav),
            open_positions_count=int(open_positions_count),
        )
        decision = engine.check_order(intent_payload, temp_state)
        risk_veto = not decision.allowed
        details = decision.diagnostics or {}
    else:
        risk_veto, details = check_order(
            symbol=sym,
            side=side,
            requested_notional=float(notional) * float(lev),
            price=0.0,
            nav=float(nav),
            open_qty=0.0,
            now=time.time(),
            cfg=cfg,
            state=temp_state,
            current_gross_notional=float(current_gross_notional),
            lev=float(lev),
            open_positions_count=int(open_positions_count),
            tier_name=tname,
            current_tier_gross_notional=float(current_tier_gross_notional),
        )
    detail_dict = details if isinstance(details, dict) else {}
    if isinstance(details, dict):
        extra["risk"] = detail_dict
    elif details is not None:
        extra["risk"] = {"detail": details}
    rs = [str(r) for r in (detail_dict.get("reasons") or [])]
    for reason in rs:
        if reason and reason not in reasons:
            reasons.append(reason)
    overall_ok = (len(reasons) == 0) and (not risk_veto)

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


def generate_signals_from_config() -> Iterable[Dict[str, Any]]:
    try:
        strategies = _load_strategy_list()
    except Exception as e:
        print(f"{LOG_TAG} error loading config: {e}")
        return []
    try:
        base_cfg = json.load(open("config/strategy_config.json"))
    except Exception:
        base_cfg = {}
    sizing_cfg = (base_cfg.get("sizing") or {})
    min_gross_floor = 0.0
    ml_cfg = (base_cfg.get("ml") or {})
    ml_enabled = bool(ml_cfg.get("enabled")) and _score_symbol is not None
    out = []
    attempted = 0
    dbg = os.getenv("DEBUG_SIGNALS", "0").lower() in ("1","true","yes")
    # Prepare allowed set and tier map (best effort)
    try:
        allowed_syms, tier_by = resolve_allowed_symbols()
        allowed_set = {s.upper() for s in (allowed_syms or [])}
    except Exception:
        allowed_set = set()
        tier_by = {}
    # Risk cfg + portfolio snapshot (best effort)
    rlc = _load_risk_cfg()
    risk_global = (rlc.get("global") or {}) if isinstance(rlc, Mapping) else {}
    trade_equity_cap, max_trade_cap = _extract_trade_nav_caps(rlc)
    kill_switch = os.environ.get("KILL_SWITCH", "0").lower() in ("1", "true", "yes", "on")
    snapshot = PortfolioSnapshot(base_cfg)
    try:
        nav_override = float(os.getenv("SCREENER_NAV", "0") or 0.0)
    except Exception:
        nav_override = 0.0
    if nav_override > 0:
        nav = nav_override
    else:
        nav = float(snapshot.current_nav_usd())
        if nav <= 0:
            nav = 1000.0
    current_portfolio_gross = float(snapshot.current_gross_usd())
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
    _update_screener_risk_state(snapshot, current_portfolio_gross, open_positions_count)
    cap_pct = float(risk_global.get("max_gross_exposure_pct", 0.0) or 0.0)
    if os.environ.get("EVENT_GUARD", "0") == "1":
        cap_pct *= 0.8
    cap_usd = nav * (cap_pct / 100.0) if nav > 0 and cap_pct > 0 else 0.0
    ml_threshold = float(ml_cfg.get("prob_threshold", 0.0) or 0.0)
    for scfg in strategies:
        attempted += 1
        sym = scfg.get("symbol")
        sym_key = str(sym).upper()
        tf = scfg.get("timeframe", "15m")
        params = scfg.get("params") or {}
        cap_cfg = float(params.get("capital_per_trade", 0) or 0) or 10.0
        lev = float(params.get("leverage") or symbol_target_leverage(sym_key) or 1.0)
        if lev <= 0:
            lev = 1.0
        gross_cap = cap_cfg * lev
        sym_floor = max(symbol_min_gross(sym_key), 0.0)
        config_floor = max(gross_cap, min_gross_floor, sym_floor)
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
        cfg_min_notional = float(risk_global.get("min_notional_usdt", 0.0) or 0.0)
        sym_min_notional = symbol_min_notional(sym_key)
        floor_notional = max(
            config_floor,
            cfg_min_notional,
            sym_min_notional,
            exch_min_notional,
            min_qty_notional,
        )
        requested_notional = floor_notional
        cap = requested_notional / lev
        if cap_usd > 0:
            if current_portfolio_gross >= cap_usd:
                print(
                    f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["portfolio_cap_reached"]}}'
                )
                continue
            if current_portfolio_gross + requested_notional > cap_usd:
                remaining = max(cap_usd - current_portfolio_gross, 0.0)
                print(
                    f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["portfolio_cap_reached"],"remaining":{round(remaining,4)}}}'
                )
                continue
        effective_notional = requested_notional
        min_notional = max(exch_min_notional, cfg_min_notional, sym_min_notional)
        vetoes = []
        if effective_notional < min_notional:
            vetoes.append("min_notional")
        if effective_notional < min_qty_notional:
            vetoes.append("min_qty_notional")
        if vetoes and not entry_forced:
            if dbg:
                print(
                    f'[sigdbg] {sym} tf={tf} px={round(price,4)} cap={cap} lev={lev} gross={round(effective_notional,4)} min_notional={min_notional} min_qty_notional={round(min_qty_notional,4)} veto={vetoes}'
                )
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","gross":{effective_notional},"min_notional":{min_notional},"veto":{vetoes}}}'
            )
            continue

        trade_obs_pct = None
        trade_vetoes: List[str] = []
        if nav > 0.0:
            trade_obs_pct = (effective_notional / nav) * 100.0
            if trade_equity_cap > 0.0 and trade_obs_pct > trade_equity_cap:
                trade_vetoes.append("trade_gt_equity_cap")
            if max_trade_cap > 0.0 and trade_obs_pct > max_trade_cap:
                trade_vetoes.append("trade_gt_max_trade_nav_pct")
        if trade_vetoes:
            detail_payload = {
                "trade_equity_nav_pct": trade_equity_cap if trade_equity_cap > 0.0 else None,
                "max_trade_nav_pct": max_trade_cap if max_trade_cap > 0.0 else None,
                "trade_equity_nav_obs": trade_obs_pct,
                "max_trade_nav_obs": trade_obs_pct,
            }
            detail_payload = {k: v for k, v in detail_payload.items() if v is not None}
            print(
                f'[decision] {json.dumps({"symbol": sym, "tf": tf, "veto": trade_vetoes, "detail": detail_payload})}'
            )
            continue

        # Execution hardening gates
        if not allow_trade(sym_key):
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["signal_gate"]}}'
            )
            continue
        if not symbol_notional_guard(sym_key, cfg=rlc):
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["symbol_notional_cap"]}}'
            )
            continue
        if not symbol_dd_guard(sym_key, cfg=rlc):
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["symbol_drawdown_cap"]}}'
            )
            continue

        scaled_notional = size_for(sym_key, effective_notional)
        if scaled_notional <= 0:
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["sizing_zero"]}}'
            )
            continue
        requested_notional = max(float(scaled_notional), min_notional)
        effective_notional = requested_notional
        cap = requested_notional / lev if lev > 0 else requested_notional

        signal = (
            "BUY"
            if z < -0.8
            else ("SELL" if z > 0.8 else ("BUY" if entry_forced else None))
        )
        if signal is None:
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","z":{round(z, 4)},"rsi":{round(rsi, 1)},"veto":["no_cross"]}}'
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

        # Risk pre-check
        tname = symbol_tier(sym)
        tier_key = tname or "?"
        tcur = tier_gross.get(tier_key, 0.0)

        reduce_intents, reduce_notional = _reduce_plan(
            sym,
            signal,
            tf,
            positions_by_symbol.get(sym_key, {}),
            price,
        )
        adj_portfolio_gross = max(current_portfolio_gross - reduce_notional, 0.0)
        adj_tier_gross = max(tcur - reduce_notional, 0.0)
        adj_open_positions = open_positions_count
        if reduce_notional > 0.0 and adj_open_positions > 0:
            adj_open_positions -= 1

        risk_veto, details = check_order(
            symbol=sym,
            side=signal,
            requested_notional=requested_notional,
            price=price,
            nav=nav,
            open_qty=0.0,
            now=time.time(),
            cfg=rlc,
            state=None,  # type: ignore[arg-type]
            current_gross_notional=adj_portfolio_gross,
            lev=lev,
            open_positions_count=adj_open_positions,
            tier_name=tname,
            current_tier_gross_notional=adj_tier_gross,
        )
        if risk_veto:
            rs = details.get("reasons", []) if isinstance(details, dict) else []
            if reduce_intents:
                for ri in reduce_intents:
                    out.append(ri)
                    pos_side = ri.get("positionSide")
                    if pos_side:
                        positions_by_symbol.setdefault(sym_key, {}).pop(pos_side, None)
                current_portfolio_gross = adj_portfolio_gross
                tier_gross[tier_key] = adj_tier_gross
                open_positions_count = adj_open_positions
                _update_screener_risk_state(
                    snapshot,
                    current_portfolio_gross,
                    open_positions_count,
                )
            veto_payload = {
                "symbol": sym,
                "tf": tf,
                "veto": rs,
                "detail": details if isinstance(details, dict) else {"detail": details},
            }
            print(f'[decision] {json.dumps(veto_payload)}')
            continue
        if reduce_intents:
            for ri in reduce_intents:
                out.append(ri)
                pos_side = ri.get("positionSide")
                if pos_side:
                    positions_by_symbol.setdefault(sym_key, {}).pop(pos_side, None)
            current_portfolio_gross = adj_portfolio_gross
            tier_gross[tier_key] = adj_tier_gross
            open_positions_count = adj_open_positions
            _update_screener_risk_state(
                snapshot,
                current_portfolio_gross,
                open_positions_count,
            )
        intent = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "symbol": sym,
            "timeframe": tf,
            "signal": signal,
            "reduceOnly": False,
            "price": price,
            "capital_per_trade": cap,
            "leverage": lev,
            "cap_usd": cap,
            "gross_usd": requested_notional,
            "min_notional": min_notional,
        }
        if ml_prob is not None:
            intent["ml_prob"] = ml_prob
        if dbg:
            print(
                f"[sigdbg] {sym} tf={tf} z={round(z,3)} rsi={round(rsi,1)} cap={cap} lev={lev} ok"
            )
        print(f"{LOG_TAG} {sym} {tf} z={round(z, 3)} rsi={round(rsi, 1)}")
        decision_payload = {
            "symbol": sym,
            "tf": tf,
            "intent": intent,
            "detail": details if isinstance(details, dict) else {},
        }
        print(f'[decision] {json.dumps(decision_payload)}')
        out.append(intent)
        current_portfolio_gross += effective_notional
        tier_gross[tier_key] = adj_tier_gross + effective_notional
        open_positions_count += 1
        qty_est = requested_notional / price if price > 0 else 0.0
        positions_by_symbol.setdefault(sym_key, {})[
            "LONG" if signal == "BUY" else "SHORT"
        ] = {
            "qty": abs(qty_est),
            "mark": price,
            "notional": requested_notional,
        }
        _update_screener_risk_state(
            snapshot,
            current_portfolio_gross,
            open_positions_count,
        )
    print(f"{LOG_TAG} attempted={attempted} emitted={len(out)}")
    return out


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

    normalized: List[Dict[str, Any]] = []
    for raw in intents:
        try:
            norm = generator_normalize_intent(raw)
        except Exception:
            continue
        summary: Dict[str, Any] = {
            "symbol": norm.get("symbol"),
            "side": norm.get("signal"),
            "notional": norm.get("gross_usd") or norm.get("capital_per_trade"),
            "strategy": norm.get("strategy") or norm.get("source"),
            "timeframe": norm.get("timeframe"),
            "reduce_only": norm.get("reduceOnly"),
            "raw": norm,
        }
        normalized_meta = norm.get("normalized")
        if isinstance(normalized_meta, dict) and "qty" in normalized_meta:
            summary["qty"] = normalized_meta.get("qty")
        normalized.append(summary)

    return {
        "attempted": attempted_count or len(intents),
        "emitted": len(normalized),
        "intents": normalized,
    }
