#!/usr/bin/env python3
import os
import json
import time
import hmac
import hashlib
from typing import Any, Dict
import requests

# --- Load screener pieces
import execution.signal_screener as sc

def _now_ms(): return int(time.time()*1000)

def _dual_side():
    if os.getenv("USE_FUTURES","") not in ("1","true","True"):
        return False, "spot-mode"
    key = os.environ.get("BINANCE_API_KEY","")
    sec = os.environ.get("BINANCE_API_SECRET","").encode()
    base = "https://testnet.binancefuture.com" if os.getenv("BINANCE_TESTNET","") in ("1","true","True") else "https://fapi.binance.com"
    q = f"timestamp={_now_ms()}"
    sig = hmac.new(sec, q.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"{base}/fapi/v1/positionSide/dual?{q}&signature={sig}", headers={"X-MBX-APIKEY":key}, timeout=10)
    j = r.json()
    return bool(j.get("dualSidePosition", False)), j

def _price(sym:str):
    try:
        from execution.exchange_utils import get_price
        p = get_price(sym)
        return None if (p is None or (isinstance(p,(int,float)) and p<=0)) else float(p)
    except Exception:
        return None

def _series(st:dict, sym:str, tf:str):
    arr = sc._series_for(st, sym, tf)
    return [p for _,p in arr]

def _meta(st:dict, sym:str, tf:str):
    return ((st.get(sym, {}) or {}).get(f"{tf}__meta", {}) or {})

def main():
    cfg = sc._load_cfg()
    st  = sc._safe_load_json(sc.STATE_PATH, {})
    out = {"env":{}, "strategies":[]}

    # env snapshot
    out["env"]["dualSide"], out["env"]["dualSide_raw"] = _dual_side()
    out["env"]["telegram_enabled"] = os.getenv("TELEGRAM_ENABLED","") in ("1","true","True")
    out["env"]["heartbeat_minutes"] = os.getenv("HEARTBEAT_MINUTES","")
    out["env"]["dry_run"] = os.getenv("DRY_RUN","")
    out["env"]["testnet"] = os.getenv("BINANCE_TESTNET","")
    out["env"]["use_futures"] = os.getenv("USE_FUTURES","")

    for name, scfg in (cfg.get("strategies") or {}).items():
        sym = scfg["symbol"]
        tf  = scfg.get("timeframe","30m")
        pxs = _series(st, sym, tf)
        meta= _meta(st, sym, tf)
        prev_z = meta.get("prev_z", None)
        in_trade = bool(meta.get("in_trade", False))
        side = meta.get("side")
        px_live = _price(sym)

        inds: Dict[str, Any] = {}
        if pxs:
            inds = sc._fetch_indicators_from_your_stack(sym, tf, scfg) or sc._compute_indicators_from_series(pxs, scfg)
        z   = float(inds.get("z", 0.0))
        rsi = float(inds.get("rsi", 50.0))
        atr = float(inds.get("atr_proxy", 0.0))

        entry = scfg.get("entry",{})
        zmin  = float(entry.get("zscore_min", 0.8))
        rband = entry.get("rsi_band", [0,100])
        rlo, rhi = float(rband[0]), float(rband[1])
        atr_min = float(entry.get("atr_min", 0.0))
        rsi_ok  = (rlo <= rsi <= rhi)
        atr_ok  = atr >= atr_min

        if prev_z is None:
            cross_up = cross_down = False
            cross_reason = "prev_z not seeded"
        else:
            cross_up   = (prev_z <  zmin) and (z >=  zmin)
            cross_down = (prev_z > -zmin) and (z <= -zmin)
            cross_reason = "cross ok" if (cross_up or cross_down) else "no cross"

        blocked_by = []
        if px_live is None:
            blocked_by.append("price_unavailable")
        if not (cross_up or cross_down):
            blocked_by.append("no_cross")
        if not rsi_ok:
            blocked_by.append("rsi_veto")
        if not atr_ok:
            blocked_by.append("atr_floor")
        if in_trade:
            blocked_by.append("already_in_trade")
        out["strategies"].append({
            "name": name, "symbol": sym, "tf": tf,
            "series_len": len(pxs),
            "live_px": px_live,
            "z": z, "prev_z": prev_z,
            "cross_up": cross_up, "cross_down": cross_down, "cross_reason": cross_reason,
            "rsi": rsi, "rsi_band": [rlo, rhi], "rsi_ok": rsi_ok,
            "atr": atr, "atr_min": atr_min, "atr_ok": atr_ok,
            "in_trade": in_trade, "side": side,
            "blocked_by": blocked_by
        })

    print(json.dumps(out, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()
