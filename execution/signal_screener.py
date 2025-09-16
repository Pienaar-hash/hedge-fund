#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List

import os
from .exchange_utils import get_klines, get_price
from .orderbook_features import evaluate_entry_gate

try:
    from .exchange_utils import get_symbol_filters  # prefers public alias
except Exception:
    from .exchange_utils import _symbol_filters as get_symbol_filters  # fallback

LOG_TAG = "[screener]"


def _load_strategy_list() -> List[Dict[str, Any]]:
    scfg = json.load(open("config/strategy_config.json"))
    raw = scfg.get("strategies", scfg)
    lst = (
        raw
        if isinstance(raw, list)
        else (list(raw.values()) if isinstance(raw, dict) else [])
    )

    # Authoritative whitelist from risk_limits.json (fallback to strategy_config.json if needed)
    wl: List[str] = []
    try:
        rlc = json.load(open("config/risk_limits.json"))
        wl = (rlc.get("global") or {}).get("whitelist") or []
    except Exception:
        pass
    if not wl:
        wl = (scfg.get("whitelist") or []) if isinstance(scfg, dict) else []
    if isinstance(wl, list) and wl:
        wl_set = {str(x).upper() for x in wl}
        lst = [
            s
            for s in lst
            if isinstance(s, dict) and str(s.get("symbol", "")).upper() in wl_set
        ]
    # Optional universe from pairs_universe.json (secondary filter if present)
    try:
        uni = json.load(open("config/pairs_universe.json")).get("symbols", [])
        if isinstance(uni, list) and uni:
            uni_set = {str(x).upper() for x in uni}
            lst = [
                s
                for s in lst
                if isinstance(s, dict) and str(s.get("symbol", "")).upper() in uni_set
            ]
    except Exception:
        pass
    return [s for s in lst if isinstance(s, dict) and s.get("enabled")]


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
    out = []
    attempted = 0
    dbg = os.getenv("DEBUG_SIGNALS", "0").lower() in ("1","true","yes")
    for scfg in strategies:
        attempted += 1
        sym = scfg.get("symbol")
        tf = scfg.get("timeframe", "15m")
        cap = float(scfg.get("capital_per_trade", 0) or 0)
        lev = float(scfg.get("leverage", 1) or 1)
        entry_forced = (scfg.get("entry", {}) or {}).get("type") == "always_on"
        try:
            kl = get_klines(sym, tf, limit=150)
            closes = [c for _, c in kl]
            price = get_price(sym)
            rsi = _rsi(closes, 14)
            z = _zscore(closes, 20)
            filters = get_symbol_filters(sym)
            exch_min_notional = float(
                (filters.get("MIN_NOTIONAL", {}) or {}).get("notional", 0) or 0.0
            )
            lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
            step = float(lot.get("stepSize", 1.0))
            min_qty = float(lot.get("minQty", 1.0))
            # Exchange min qty notional is price*minQty; compare against gross cap (no leverage division)
            min_qty_notional = price * max(min_qty, step)
        except Exception as e:
            print(f"{LOG_TAG} {sym} {tf} error: {e}")
            continue
        # Use global min_notional_usdt from risk_limits.json (fallback to strategy_config.json if present there)
        try:
            rlc = json.load(open("config/risk_limits.json"))
            cfg_min_notional = float(
                (rlc.get("global") or {}).get("min_notional_usdt", 0) or 0
            )
        except Exception:
            try:
                cfg_min_notional = float(
                    (json.load(open("config/strategy_config.json"))).get(
                        "min_notional_usdt", 0
                    )
                    or 0
                )
            except Exception:
                cfg_min_notional = 0.0
        min_notional = max(exch_min_notional, cfg_min_notional)
        vetoes = []
        if cap < min_notional:
            vetoes.append("min_notional")
        if cap < min_qty_notional:
            vetoes.append("min_qty_notional")
        if vetoes and not entry_forced:
            if dbg:
                print(
                    f'[sigdbg] {sym} tf={tf} px={round(price,4)} cap={cap} lev={lev} min_notional={min_notional} min_qty_notional={round(min_qty_notional,4)} veto={vetoes}'
                )
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","notional":{cap},"min_notional":{min_notional},"veto":{vetoes}}}'
            )
            continue
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
        feat = scfg.get("features", {}) if isinstance(scfg, dict) else {}
        ob_enabled = bool(feat.get("orderbook_gate"))
        veto, info = evaluate_entry_gate(sym, signal, enabled=ob_enabled)
        if veto:
            if dbg:
                print(f"[sigdbg] {sym} tf={tf} ob_imbalance={float(info.get('metric',0.0)):.3f} veto")
            print(
                f'[decision] {{"symbol":"{sym}","tf":"{tf}","veto":["ob_imbalance"],"metric":{round(float(info.get("metric",0.0)),3)}}}'
            )
            continue
        intent = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "symbol": sym,
            "timeframe": tf,
            "signal": signal,
            "reduceOnly": False,
            "price": price,
            "capital_per_trade": cap,
            "leverage": lev,
            "min_notional": min_notional,
        }
        if dbg:
            print(
                f"[sigdbg] {sym} tf={tf} z={round(z,3)} rsi={round(rsi,1)} cap={cap} lev={lev} ok"
            )
        print(f"{LOG_TAG} {sym} {tf} z={round(z, 3)} rsi={round(rsi, 1)}")
        print(
            f'[decision] {{"symbol":"{sym}","tf":"{tf}","z":{round(z, 4)},"rsi":{round(rsi, 1)},"cap":{cap},"lev":{lev}}}'
        )
        out.append(intent)
    print(f"{LOG_TAG} attempted={attempted} emitted={len(out)}")
    return out
