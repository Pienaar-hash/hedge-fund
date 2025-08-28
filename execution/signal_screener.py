#!/usr/bin/env python3
from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterable, List

from .exchange_utils import get_klines, get_price
try:
    from .exchange_utils import get_symbol_filters  # prefers public alias
except Exception:
    from .exchange_utils import _symbol_filters as get_symbol_filters  # fallback

LOG_TAG = "[screener]"

def _load_strategy_list() -> List[Dict[str, Any]]:
    cfg = json.load(open("config/strategy_config.json"))
    raw = cfg.get("strategies", cfg)
    lst = raw if isinstance(raw, list) else (list(raw.values()) if isinstance(raw, dict) else [])
    try:
        uni = json.load(open("config/pairs_universe.json")).get("symbols", [])
        if isinstance(uni, list) and uni:
            lst = [s for s in lst if isinstance(s, dict) and s.get("symbol") in uni]
    except Exception:
        pass
    return [s for s in lst if isinstance(s, dict) and s.get("enabled")]

def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) <= period: return 50.0
    gains, losses = [], []
    for i in range(1, period+1):
        d = closes[-i] - closes[-i-1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    ag = sum(gains)/period; al = sum(losses)/period
    if al == 0: return 100.0
    rs = ag/al; return 100.0 - (100.0/(1.0+rs))

def _zscore(closes: List[float], lookback: int = 20) -> float:
    if len(closes) < lookback: return 0.0
    seg = closes[-lookback:]; mean = sum(seg)/lookback
    var = sum((x-mean)**2 for x in seg)/lookback
    std = (var ** 0.5) or 1.0
    return (closes[-1] - mean)/std

def generate_signals_from_config() -> Iterable[Dict[str, Any]]:
    try:
        strategies = _load_strategy_list()
    except Exception as e:
        print(f"{LOG_TAG} error loading config: {e}")
        return []
    out = []; attempted = 0
    for scfg in strategies:
        attempted += 1
        sym = scfg.get("symbol"); tf = scfg.get("timeframe", "15m")
        cap = float(scfg.get("capital_per_trade", 0) or 0)
        lev = float(scfg.get("leverage", 1) or 1)
        entry_forced = (scfg.get("entry", {}) or {}).get("type") == "always_on"
        try:
            kl = get_klines(sym, tf, limit=150); closes = [c for _,c in kl]
            price = get_price(sym); rsi = _rsi(closes, 14); z = _zscore(closes, 20)
            min_notional = float(get_symbol_filters(sym).get("MIN_NOTIONAL", {}).get("notional", 0) or 0)
            lot = get_symbol_filters(sym).get('MARKET_LOT_SIZE') or get_symbol_filters(sym).get('LOT_SIZE', {})
            step = float(lot.get('stepSize', 1.0)); min_qty = float(lot.get('minQty', 1.0))
            min_qty_notional = (price * max(min_qty, step)) / max(lev, 1e-9)
        except Exception as e:
            print(f"{LOG_TAG} {sym} {tf} error: {e}"); continue
        if (cap < min_notional or cap < min_qty_notional) and not entry_forced:
            print(f"[decision] {{\"symbol\":\"{sym}\",\"tf\":\"{tf}\",\"notional\":{cap},\"min_notional\":{min_notional},\"veto\":[\"min_notional\"]}}")
            continue
        signal = "BUY" if z < -0.8 else ("SELL" if z > 0.8 else ("BUY" if entry_forced else None))
        if signal is None:
            print(f"[decision] {{\"symbol\":\"{sym}\",\"tf\":\"{tf}\",\"z\":{round(z,4)},\"rsi\":{round(rsi,1)},\"veto\":[\"no_cross\"]}}")
            continue
        intent = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "symbol": sym, "timeframe": tf, "signal": signal, "reduceOnly": False,
            "price": price, "capital_per_trade": cap, "leverage": lev, "min_notional": min_notional,
        }
        print(f"{LOG_TAG} {sym} {tf} z={round(z,3)} rsi={round(rsi,1)}")
        print(f"[decision] {{\"symbol\":\"{sym}\",\"tf\":\"{tf}\",\"z\":{round(z,4)},\"rsi\":{round(rsi,1)},\"cap\":{cap},\"lev\":{lev}}}")
        out.append(intent)
    print(f"{LOG_TAG} attempted={attempted} emitted={len(out)}")
    return out
