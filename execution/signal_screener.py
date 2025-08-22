from __future__ import annotations
import json, math, os, time
from typing import Any, Dict, Iterable, List, Tuple, Optional
from execution.exchange_utils import get_price, get_positions, get_klines  # needs get_klines in exchange_utils

CFG_PATH   = os.getenv("STRATEGY_CFG", "config/strategy_config.json")
STATE_PATH = os.getenv("SCREENER_STATE", "screener_state.json")
TF_SPACING_SEC = {"15m": 60, "30m": 120, "1h": 180}
PRICE_EPS = 0.0002
MAX_POINTS = 240

def _safe_load_json(path: str, default):
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception: return default

def _safe_write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f: json.dump(data, f)
        os.replace(tmp, path)
    except Exception: pass

def _now_ts() -> float: return time.time()

def _load_state() -> Dict[str, Any]:
    st = _safe_load_json(STATE_PATH, {}) or {}
    st.setdefault("prices", {}); st.setdefault("meta", {})
    return st

def _save_state(st: Dict[str, Any]) -> None: _safe_write_json(STATE_PATH, st)

def _push_price_point(st, symbol, timeframe, ts, price) -> None:
    prices = st["prices"].setdefault(symbol, {}).setdefault(timeframe, [])
    keep = True
    if prices:
        last_t = float(prices[-1].get("t", 0.0))
        last_p = float(prices[-1].get("p", price))
        spacing = TF_SPACING_SEC.get(timeframe, 60)
        moved = abs((price - last_p) / last_p) if last_p > 0 else 1.0
        if (ts - last_t) < spacing and moved < PRICE_EPS: keep = False
    if keep:
        prices.append({"t": float(ts), "p": float(price)})
        if len(prices) > MAX_POINTS: del prices[: len(prices) - MAX_POINTS]

def _series_for(st, symbol, timeframe) -> List[Tuple[float, float]]:
    arr = ((st.get("prices", {}) or {}).get(symbol, {}) or {}).get(timeframe, []) or []
    return [(float(x.get("t", 0.0)), float(x.get("p", 0.0))) for x in arr]

def _load_meta(st, symbol, timeframe) -> Dict[str, Any]:
    return st["meta"].setdefault(symbol, {}).setdefault(timeframe, {"in_trade": False, "side": None, "prev_z": 0.0})

def _save_meta(st, symbol, timeframe, **kv) -> None:
    m = _load_meta(st, symbol, timeframe); m.update(kv)

def _position_side(symbol: str) -> Optional[str]:
    try:
        for p in (get_positions(symbol) or []):
            qty = float(p.get("qty", 0.0))
            if abs(qty) > 1e-12: return "LONG" if qty > 0 else "SHORT"
    except Exception: pass
    return None

def _pct_returns(prices: List[float]) -> List[float]:
    out=[]
    for i in range(1, len(prices)):
        p0, p1 = prices[i-1], prices[i]
        if p0 > 0: out.append((p1 - p0) / p0)
    return out

def _zscore(x: List[float]) -> float:
    if not x: return 0.0
    m = sum(x)/len(x); v = sum((xi - m)**2 for xi in x) / max(1, len(x)-1)
    s = math.sqrt(v) if v>0 else 0.0
    return (x[-1] - m) / s if s>0 else 0.0

def _rsi(deltas: List[float], period: int = 14) -> float:
    if len(deltas) < 1: return 50.0
    gains=[max(d,0.0) for d in deltas[-period:]]; losses=[max(-d,0.0) for d in deltas[-period:]]
    avg_gain = sum(gains)/max(1,len(gains)); avg_loss = sum(losses)/max(1,len(losses))
    if avg_loss == 0: return 100.0 if avg_gain>0 else 50.0
    rs = avg_gain/avg_loss; return 100.0 - (100.0/(1.0+rs))

def _atr_proxy(prices: List[float], period: int = 14) -> float:
    rets = [abs(x) for x in _pct_returns(prices)]
    if not rets: return 0.0
    return sum(rets[-period:]) / max(1, min(period, len(rets)))

def _compute_inds(prices: List[float], scfg: Dict[str, Any]) -> Dict[str, float]:
    rsi_period = int((scfg.get("exit", {}) or {}).get("rsi_period", 14))
    atr_period = int((scfg.get("exit", {}) or {}).get("atr_period", 14))
    rets = _pct_returns(prices)
    z = _zscore(rets[-50:])
    rsi = _rsi(rets, period=rsi_period)
    atrp = _atr_proxy(prices, period=atr_period)
    return {"z": z, "rsi": rsi, "atr_proxy": atrp}

def _make_intent(symbol, timeframe, side, price=None, scfg=None, reduce=False):
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "symbol": symbol, "timeframe": timeframe,
           "signal": "SELL" if (side in ("EXIT_LONG","ENTER_SHORT")) else "BUY"}
    if reduce: out["reduceOnly"] = True
    if price is not None:
        try: out["price"] = float(price)
        except Exception: pass
    if scfg:
        for k in ("capital_per_trade","leverage","min_notional"):
            if k in scfg: out[k] = scfg[k]
    return out

def _seed_series_with_klines(st, symbol: str, timeframe: str, need: int = 80):
    cur = st["prices"].setdefault(symbol, {}).setdefault(timeframe, [])
    if len(cur) >= need: return
    tf_map = {"15m":"15m","30m":"30m","1h":"1h"}
    k = get_klines(symbol, tf_map.get(timeframe,"15m"), limit=max(need, 120)) or []
    if not k: return
    cur[:] = [{"t": float(ts/1000.0), "p": float(px)} for ts, px in k][-need:]

def _exit_ok(prices: List[float], inds: Dict[str, float], exit_cfg: Dict[str, Any]) -> bool:
    if len(prices) < 5: return False
    tp_pct   = float(exit_cfg.get("tp_pct", 0.0))
    max_bars = int(exit_cfg.get("max_bars", 0))
    atr_mult = float(exit_cfg.get("atr_multiple", 0.0))
    p_now = prices[-1]; base = sum(prices[-10:]) / max(1, min(10, len(prices)))
    if tp_pct>0 and base>0 and (p_now-base)/base >= tp_pct: return True
    if atr_mult>0 and inds.get("atr_proxy",0.0) <= (atr_mult * 0.001): return True
    if max_bars>0 and len(prices) >= max_bars: return True
    return False

def generate_signals_from_config() -> Iterable[Dict[str, Any]]:
    cfg = _safe_load_json(CFG_PATH, {}) or {}; strategies = cfg.get("strategies") or {}
    st = _load_state(); attempted=0; emitted=0

    for name, scfg in strategies.items():
        symbol = scfg.get("symbol"); tf = scfg.get("timeframe","15m")
        if not symbol: continue
        attempted += 1
        try:
            px = float(get_price(symbol) or 0.0)
            if not (math.isfinite(px) and px>0):
                m = _load_meta(st, symbol, tf); _save_meta(st, symbol, tf, prev_z=float(m.get("prev_z",0.0))); continue

            ts = _now_ts()
            _push_price_point(st, symbol, tf, ts, px)
            _seed_series_with_klines(st, symbol, tf, need=80)
            series = _series_for(st, symbol, tf); prices = [p for _,p in series]
            if len(prices) < 10: _save_meta(st, symbol, tf, prev_z=0.0); continue

            inds = _compute_inds(prices, scfg)
            z=float(inds["z"]); rsi=float(inds["rsi"]); atrp=float(inds["atr_proxy"])
            meta = _load_meta(st, symbol, tf); prev_z=float(meta.get("prev_z",0.0))
            in_trade = bool(meta.get("in_trade", False)); side_meta = meta.get("side")

            entry_cfg = scfg.get("entry", {}) or {}
            mode=str(entry_cfg.get("mode","momentum")).lower()
            zmin=float(entry_cfg.get("zscore_min",0.8))
            rlo,rhi = [float(x) for x in entry_cfg.get("rsi_band",[30,70])]
            atr_min=float(entry_cfg.get("atr_min",0.0))
            cross_up = (prev_z <  zmin and z >=  zmin); cross_down = (prev_z > -zmin and z <= -zmin)
            if mode=="breakout":
                lb=int(entry_cfg.get("breakout_lookback",20))
                long_ok=short_ok=False
                if len(prices) >= max(lb+2,3):
                    prev_p = prices[-2]; window = prices[-(lb+1):-1]
                    res=max(window); sup=min(window)
                    long_ok=(prev_p<=res and prices[-1]>res); short_ok=(prev_p>=sup and prices[-1]<sup)
                cross_up, cross_down = long_ok, short_ok

            rsi_ok=(rlo<=rsi<=rhi); atr_ok=(atrp>=atr_min)

            pos_side = _position_side(symbol); pos_exists = pos_side is not None
            if pos_exists and not in_trade: in_trade,side_meta=True,pos_side; _save_meta(st,symbol,tf,in_trade=True,side=side_meta)
            if (not pos_exists) and in_trade: in_trade,side_meta=False,None; _save_meta(st,symbol,tf,in_trade=False,side=None)

            veto=[]
            if not (cross_up or cross_down): veto.append("no_cross")
            if not rsi_ok: veto.append("rsi_veto")
            if not atr_ok: veto.append("atr_floor")
            if in_trade: veto.append("already_in_trade")

            cap=float(scfg.get("capital_per_trade",0.0) or 0.0)
            lev=float(scfg.get("leverage",1) or 1)
            min_notional=float(scfg.get("min_notional",0.0) or 0.0)
            notional=cap*lev
            if min_notional>0 and notional<min_notional: veto.append("below_min_notional")

            print(f"[screener] {symbol} {tf} z={z:+.3f} rsi={rsi:.1f} atr={atrp:.5f} cross_up={bool(cross_up)} cross_down={bool(cross_down)} in_trade={in_trade} side={(side_meta or '-')}", flush=True)
            print("[decision] " + json.dumps({"symbol":symbol,"tf":tf,"z":round(z,4),"prev_z":round(prev_z,4),"zmin":zmin,
                                             "cross_up":bool(cross_up),"cross_down":bool(cross_down),
                                             "rsi":rsi,"band":[rlo,rhi],"rsi_ok":rsi_ok,
                                             "atr":atrp,"atr_min":atr_min,"atr_ok":atr_ok,
                                             "in_trade":in_trade,"pos_side":pos_side,
                                             "cap":cap,"lev":lev,"min_notional":min_notional,"notional":notional,
                                             "veto":veto}, sort_keys=True), flush=True)

            if cross_up and not in_trade and rsi_ok and atr_ok and "below_min_notional" not in veto:
                intent=_make_intent(symbol,tf,"ENTER_LONG",price=px,scfg=scfg,reduce=False)
                print("[screener->executor] " + json.dumps(intent, sort_keys=True), flush=True)
                _save_meta(st,symbol,tf,in_trade=True,side="LONG",prev_z=z); emitted+=1; yield intent; continue
            if cross_down and not in_trade and rsi_ok and atr_ok and "below_min_notional" not in veto:
                intent=_make_intent(symbol,tf,"ENTER_SHORT",price=px,scfg=scfg,reduce=False)
                print("[screener->executor] " + json.dumps(intent, sort_keys=True), flush=True)
                _save_meta(st,symbol,tf,in_trade=True,side="SHORT",prev_z=z); emitted+=1; yield intent; continue

            exit_cfg=scfg.get("exit",{}) or {}
            if (in_trade or pos_exists) and _exit_ok(prices, inds, exit_cfg):
                eff_side = side_meta or pos_side
                intent=_make_intent(symbol,tf,"EXIT_SHORT" if eff_side=="SHORT" else "EXIT_LONG",price=px,scfg=scfg,reduce=True)
                print("[screener->executor] " + json.dumps(intent, sort_keys=True), flush=True)
                _save_meta(st,symbol,tf,in_trade=False,side=None,prev_z=z); emitted+=1; yield intent
            else:
                _save_meta(st,symbol,tf,prev_z=z)
        except Exception:
            continue

    _save_state(st); print(f"[screener] attempted={attempted} emitted={emitted}", flush=True)

if __name__=="__main__":
    for sig in (generate_signals_from_config() or []):
        try: print(sig)
        except Exception: pass
