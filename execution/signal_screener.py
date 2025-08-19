
"""
execution/signal_screener.py — Phase-4.1 "Stability & Signals"

Purpose
-------
Emit actionable BUY/SELL intents from a simple, robust screener that:
- Loads strategy_config.json (path override via STRATEGY_CFG env)
- Uses explicit strategies when present; falls back to a 10-symbol whitelist when sparse
- Maintains a tiny rolling state per symbol/timeframe on disk (screener_state.json)
- Computes lightweight indicators (z-score of returns, RSI on deltas, ATR proxy)
- Emits exits (SELL) for basic TP/timeout/ATR-proxy when configured
- Never raises; returns an iterator of dict intents

Intent shape
------------
{
  "symbol": "BTCUSDT",
  "signal": "BUY" | "SELL",
  "price": 60000.0,                 # current price snapshot
  "qty": 0.001,                     # optional; otherwise executor derives from capital_per_trade
  "capital_per_trade": 25.0,        # optional passthrough knobs
  "leverage": 3,
  "kelly_fraction": 0.25,
  "min_notional": 5.0,
  "positionSide": "LONG",           # optional hedge flags passthrough
  "reduceOnly": true
}

Environment
-----------
STRATEGY_CFG = "config/strategy_config.json"  (default)
SCREENER_STATE = "screener_state.json"        (rolling price cache)
WHITELIST = "ADAUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,DOTUSDT,ETHUSDT,LTCUSDT,SOLUSDT,TONUSDT,XRPUSDT"

Notes
-----
- Indicators are approximate and file-backed so they work without a candles API.
- If you already have a proper candles/indicators stack, you can plug it into
  `_fetch_indicators_from_your_stack()` and bypass the rolling cache logic below.
"""

from __future__ import annotations

import os
import json
import math
import time
from typing import Dict, Any, Iterable, List, Tuple

from execution.exchange_utils import get_price  # we only need spot price snapshot

# ------------------------ config & constants ------------------------

DEFAULT_WHITELIST = [
    "ADAUSDT","BNBUSDT","BTCUSDT","DOGEUSDT","DOTUSDT",
    "ETHUSDT","LTCUSDT","SOLUSDT","TONUSDT","XRPUSDT"
]

CFG_PATH = os.getenv("STRATEGY_CFG", "config/strategy_config.json")
STATE_PATH = os.getenv("SCREENER_STATE", "screener_state.json")
ENV_WHITELIST = os.getenv("WHITELIST", ",".join(DEFAULT_WHITELIST))

# state size per symbol/timeframe
MAX_POINTS = int(os.getenv("SCREENER_MAX_POINTS", "200"))

# ------------------------ utils ------------------------

def _safe_load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass  # fail silent; screener must never crash

def _now_ts() -> float:
    return time.time()

def _symbols_from_cfg(cfg: dict) -> List[str]:
    syms = []
    for _, scfg in (cfg.get("strategies") or {}).items():
        s = scfg.get("symbol")
        if s:
            syms.append(s)
    if syms:
        return syms
    # fallback: env whitelist or default
    wl = [s.strip().upper() for s in (ENV_WHITELIST.split(",") if ENV_WHITELIST else []) if s.strip()]
    return wl or DEFAULT_WHITELIST

def _load_cfg() -> dict:
    cfg = _safe_load_json(CFG_PATH, {})
    if "whitelist" not in cfg:
        cfg["whitelist"] = _symbols_from_cfg(cfg)
    return cfg

# ------------------------ rolling state (file-backed) ------------------------

def _load_state() -> dict:
    st = _safe_load_json(STATE_PATH, {})
    if not isinstance(st, dict):
        st = {}
    return st

def _save_state(st: dict) -> None:
    _safe_write_json(STATE_PATH, st)

def _push_price_point(st: dict, symbol: str, timeframe: str, ts: float, price: float) -> None:
    sym = st.setdefault(symbol, {})
    tf = sym.setdefault(timeframe, [])
    tf.append({"t": float(ts), "p": float(price)})
    if len(tf) > MAX_POINTS:
        del tf[:len(tf) - MAX_POINTS]

def _series_for(st: dict, symbol: str, timeframe: str) -> List[Tuple[float,float]]:
    arr = ((st.get(symbol, {}) or {}).get(timeframe, [])) or []
    # return list of (ts, price)
    return [(float(x.get("t", 0.0)), float(x.get("p", 0.0))) for x in arr]

# ------------------------ indicators (approximate) ------------------------

def _pct_returns(prices: List[float]) -> List[float]:
    out = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i-1], prices[i]
        if p0 > 0:
            out.append((p1 - p0) / p0)
    return out

def _zscore(x: List[float]) -> float:
    if not x:
        return 0.0
    m = sum(x) / len(x)
    v = sum((xi - m) ** 2 for xi in x) / max(1, len(x) - 1)
    s = math.sqrt(v) if v > 0 else 0.0
    return (x[-1] - m) / s if s > 0 else 0.0

def _rsi(deltas: List[float], period: int = 14) -> float:
    """RSI on deltas; returns 0..100; 50 when flat/no data."""
    if len(deltas) < 1:
        return 50.0
    gains = [max(d, 0.0) for d in deltas[-period:]]
    losses = [max(-d, 0.0) for d in deltas[-period:]]
    avg_gain = sum(gains) / max(1, len(gains))
    avg_loss = sum(losses) / max(1, len(losses))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _atr_proxy(prices: List[float], period: int = 14) -> float:
    """ATR proxy using absolute percent changes (no OHLC)."""
    rets = [abs(x) for x in _pct_returns(prices)]
    if not rets:
        return 0.0
    return sum(rets[-period:]) / max(1, min(period, len(rets)))

def _compute_indicators_from_series(prices: List[float], scfg: dict) -> Dict[str, float]:
    # Inputs
    rsi_period = int((scfg.get("exit", {}) or {}).get("rsi_period", 14))
    atr_period = int((scfg.get("exit", {}) or {}).get("atr_period", 14))
    # Derived
    rets = _pct_returns(prices)
    z = _zscore(rets[-50:])  # use last 50 rets if available
    rsi = _rsi(rets, period=rsi_period)
    atrp = _atr_proxy(prices, period=atr_period)  # fraction, e.g., 0.01 = 1%
    return {"z": z, "rsi": rsi, "atr_proxy": atrp}

# Placeholder for teams with a proper indicators stack
def _fetch_indicators_from_your_stack(symbol: str, timeframe: str, scfg: dict) -> Dict[str, float] | None:
    return None

# ------------------------ entry / exit logic ------------------------

def _entry_ok(z: float, rsi: float, entry_cfg: dict) -> bool:
    zmin = float(entry_cfg.get("zscore_min", 0.8))
    rsi_band = entry_cfg.get("rsi_band", [30, 70])
    rlo, rhi = float(rsi_band[0]), float(rsi_band[1])
    return (z >= zmin) and (rlo <= rsi <= rhi)

def _exit_ok(prices: List[float], inds: Dict[str,float], exit_cfg: dict) -> bool:
    """
    Basic exits: tp_pct vs last N bars, ATR-proxy trail, or max_bars timeout.
    Since we don't track per-trade entry price here, treat tp_pct as
    price drift over last window.
    """
    if not prices or len(prices) < 5:
        return False
    tp_pct = float(exit_cfg.get("tp_pct", 0.0))  # 0.003 = 0.3%
    max_bars = int(exit_cfg.get("max_bars", 0))
    atr_mult = float(exit_cfg.get("atr_multiple", 0.0))
    # Take-profit proxy: last price vs median of last 10
    p_now = prices[-1]
    base = sum(prices[-10:]) / max(1, min(10, len(prices)))
    if tp_pct > 0 and base > 0 and (p_now - base) / base >= tp_pct:
        return True
    # ATR trail proxy (if volatility contraction after spike)
    if atr_mult > 0 and inds.get("atr_proxy", 0.0) <= (atr_mult * 0.001):  # heuristic
        return True
    # Timeout
    if max_bars > 0 and len(prices) >= max_bars:
        return True
    return False

def _make_intent(symbol, side, price=None, cfg=None, qty=None, extra=None):
    out = {"symbol": symbol, "signal": side}
    if price is not None:
        try: out["price"] = float(price)
        except Exception: pass
    if qty is not None:
        try: out["qty"] = float(qty)
        except Exception: pass
    if cfg:
        for k in ("capital_per_trade","leverage","kelly_fraction","min_notional","positionSide","reduceOnly"):
            if k in cfg:
                out[k] = cfg[k]
    if extra:
        out.update(extra)
    return out

# ------------------------ main generator ------------------------

def generate_signals_from_config() -> Iterable[Dict[str, Any]]:
    """
    Yields dicts for executor. Never raises.
    """
    cfg = _load_cfg()
    strategies = cfg.get("strategies") or {}
    whitelist = _symbols_from_cfg(cfg)

    st = _load_state()
    emitted = 0
    attempted = 0

    # 1) Walk explicit strategies
    for name, scfg in strategies.items():
        symbol = scfg.get("symbol")
        if not symbol:
            continue
        timeframe = scfg.get("timeframe", "1h")
        attempted += 1
        try:
            px = float(scfg.get("price") or get_price(symbol))
            ts = _now_ts()
            # update rolling series
            _push_price_point(st, symbol, timeframe, ts, px)
            series = _series_for(st, symbol, timeframe)
            prices = [p for _, p in series]
            # indicators (prefer external stack; else local)
            inds = _fetch_indicators_from_your_stack(symbol, timeframe, scfg) or _compute_indicators_from_series(prices, scfg)

            # entry
            if _entry_ok(inds["z"], inds["rsi"], scfg.get("entry", {})):
                yield _make_intent(symbol, "BUY", price=px, cfg=scfg)
                emitted += 1
                continue

            # exit
            if _exit_ok(prices, inds, scfg.get("exit", {})):
                # reduceOnly recommended for exits on futures hedge mode
                yield _make_intent(symbol, "SELL", price=px, cfg=scfg, extra={"reduceOnly": True})
                emitted += 1
        except Exception:
            # skip but never crash
            continue

    # 2) Fallback scan when no strategies present
    if not strategies:
        for symbol in whitelist:
            attempted += 1
            try:
                px = float(get_price(symbol))
                ts = _now_ts()
                timeframe = "1h"
                _push_price_point(st, symbol, timeframe, ts, px)
                # usually no signal on fallback; it's a presence/health check
            except Exception:
                continue

    # persist state
    _save_state(st)

    # lightweight console trace (optional)
    try:
        print(f"[screener] attempted={attempted} emitted={emitted}", flush=True)
    except Exception:
        pass

# CLI for ad-hoc testing: `python -m execution.signal_screener`
if __name__ == "__main__":
    for sig in generate_signals_from_config() or []:
        try:
            print(sig)
        except Exception:
            pass
