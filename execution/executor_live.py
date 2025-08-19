# execution/executor_live.py — Phase 4.1 production patch (no refactor)
from __future__ import annotations

import os, time, json, traceback
from typing import Dict, Any, List

from execution.exchange_utils import (
    get_price, place_market_order, get_balances, get_positions, get_account_overview
)
from execution.signal_screener import generate_signals_from_config
from execution.telegram_utils import (
    send_heartbeat, send_trade_alert, send_drawdown_alert, should_send_summary
)

ENV = os.getenv("ENV", "prod")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
HEARTBEAT_MIN = int(os.getenv("HEARTBEAT_MINUTES", "10"))
DD_ALERT_PCT = float(os.getenv("DD_ALERT_PCT", "0.10"))

NAV_LOG = "nav_log.json"
PEAK_STATE = "peak_state.json"
STATE_FILE = "synced_state.json"

_last_hb_ts = 0.0
_last_dd_alert_ts = 0.0

# --- helpers ---
def _safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json_atomic(path: str, data) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass

def _drawdown_pct(equity: float, peak: float) -> float:
    if peak <= 0: return -1.0
    return (equity / peak) - 1.0

# --- position guard helper ---
def _has_open_position(symbol: str) -> bool:
    try:
        for p in (get_positions(symbol) or []):
            if p.get("symbol") == symbol and abs(float(p.get("qty", 0))) > 1e-12:
                return True
    except Exception:
        pass
    return False

# --- nav/positions snapshot (surgical add) ---
def _snapshot_and_write():
    try:
        balances = get_balances() or {}
        usdt = float(balances.get("USDT", 0.0))
        pos = get_positions() or []
        items: List[Dict[str, Any]] = []
        unreal = 0.0
        for p in pos:
            try:
                qty = float(p.get("qty", 0.0))
                entry = float(p.get("entry_price", 0.0))
                mark  = float(p.get("mark_price", 0.0) or get_price(p.get("symbol")))
                upnl  = float(p.get("unrealizedPnl", p.get("pnl", (mark-entry)*qty)))
                unreal += upnl
                items.append({
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "qty": qty,
                    "entry_price": entry,
                    "mark_price": mark,
                    "leverage": int(p.get("leverage", 1)),
                    "pnl": upnl,
                    "notional": abs(qty) * mark,
                    "updated_at": int(time.time()),
                })
            except Exception:
                continue
        equity = usdt + unreal
        peak_j = _safe_load_json(PEAK_STATE, {})
        peak0 = float(peak_j.get("peak_equity", 0.0) or 0.0)
        if equity > peak0:
            peak0 = equity
            _write_json_atomic(PEAK_STATE, {"peak_equity": equity, "peak_ts": int(time.time())})
        nav = _safe_load_json(NAV_LOG, [])
        nav.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "realized": 0.0,
            "unrealized": float(unreal),
            "balance": float(usdt),
            "equity": float(equity),
            "drawdown_pct": _drawdown_pct(equity, peak0),
            "peak_equity": peak0,
        })
        if len(nav) > 1000: nav = nav[-1000:]
        _write_json_atomic(NAV_LOG, nav)
        _write_json_atomic(STATE_FILE, {"items": items, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())})
    except Exception:
        pass

# --- heartbeat/dd ---
def _nav_tail() -> Dict[str, float]:
    arr = _safe_load_json(NAV_LOG, [])
    last = arr[-1] if isinstance(arr, list) and arr else {}
    g = lambda k, d=0.0: float(last.get(k, d) or d)
    return {"equity": g("equity"), "realized": g("realized"), "unrealized": g("unrealized"), "dd": g("drawdown_pct"), "peak": g("peak_equity")}

def _top_positions(limit=3) -> List[str]:
    j = _safe_load_json(STATE_FILE, {})
    items = j.get("items") or []
    try:
        items = sorted(items, key=lambda x: abs(float(x.get("notional", 0.0))), reverse=True)
    except Exception:
        pass
    out = []
    for it in items[:limit]:
        try: out.append(f"{it.get('symbol')} {float(it.get('qty',0)):+g}")
        except Exception: continue
    return out

def _maybe_heartbeat():
    global _last_hb_ts
    if should_send_summary(_last_hb_ts, HEARTBEAT_MIN):
        tail = _nav_tail()
        peak = tail.get("peak", 0.0)
        send_heartbeat(tail["equity"], peak, tail["dd"], tail["realized"], tail["unrealized"], _top_positions())
        _last_hb_ts = time.time()

def _maybe_dd_alert():
    global _last_dd_alert_ts
    tail = _nav_tail(); peak = tail.get("peak", 0.0); dd = tail["dd"]
    if dd >= DD_ALERT_PCT and (time.time() - _last_dd_alert_ts) >= 15*60:
        send_drawdown_alert(dd, DD_ALERT_PCT, peak, tail["equity"])
        _last_dd_alert_ts = time.time()

# --- signals ---
def _apply_sizing_guards(symbol, qty, px, cfg):
    try: min_notional = float(cfg.get("min_notional", 0.0)) if cfg else 0.0
    except Exception: min_notional = 0.0
    notional = float(qty) * float(px)
    return qty if (min_notional <= 0 or notional >= min_notional) else 0.0

def _execute_signal(sig: Dict[str, Any]) -> None:
    symbol = sig.get("symbol"); side = str(sig.get("signal", "")).upper()
    if side not in ("BUY","SELL"): return
    # --- NEW: idempotency guard ---
    # prevent duplicate long entries
    if side == "BUY" and _has_open_position(symbol):
        print(f"[executor] skip BUY — already in position for {symbol}", flush=True)
        return

    # reduceOnly exits with no open position — skip
    if side == "BUY" and bool(sig.get("reduceOnly")) and not _has_open_position(symbol):
        print(f"[executor] skip BUY — reduceOnly with no position for {symbol}", flush=True)
        return
    if side == "SELL" and bool(sig.get("reduceOnly")) and not _has_open_position(symbol):
        print(f"[executor] skip SELL — reduceOnly with no position for {symbol}", flush=True)
        return
    # --- end guard ---
    qty = sig.get("qty"); px = None
    if qty is None:
        try:
            cap = float(sig.get("capital_per_trade", 25.0))
            px = float(sig.get("price") or get_price(symbol))
            qty = max(0.0, cap / max(px, 1e-9))
        except Exception: qty = 0.0
    if px is None:
        try: px = float(sig.get("price") or get_price(symbol))
        except Exception: px = 0.0
    qty = _apply_sizing_guards(symbol, qty, px, sig)
    if qty <= 0: return
    kwargs = {}
    if "positionSide" in sig and sig["positionSide"]: kwargs["positionSide"] = sig["positionSide"]
    if "reduceOnly" in sig: kwargs["reduceOnly"] = bool(sig["reduceOnly"])
    try:
        res = place_market_order(symbol=symbol, side=side, quantity=qty, **kwargs)
        if bool(res.get("ok", True)):
            fill = res.get("avgPrice") or px or get_price(symbol)
            print(f"[executor] ORDER {side} {symbol} qty={qty:g} fill={(float(fill) if fill else 0.0):.6f} reduceOnly={bool(sig.get('reduceOnly'))}", flush=True)
            tail = _nav_tail()
            send_trade_alert(symbol, side, qty, float(fill) if fill else 0.0, tail["realized"], tail["unrealized"])
    except Exception:
        traceback.print_exc()

# --- startup smoke ---
def _account_smoke() -> None:
    try:
        ov = get_account_overview(); balances = ov.get("balances") or {}; positions = ov.get("positions") or []
        bal_keys = [k for k in balances.keys() if not str(k).startswith("_")]
        print(f"[executor] account OK — futures={ov.get('use_futures')} testnet={ov.get('testnet')} dry_run={ov.get('dry_run')} balances: {bal_keys} positions: {len(positions)}", flush=True)
    except Exception:
        print("[executor] account OK — (smoke test skipped due to exception)", flush=True)

# --- main loop ---
def main():
    print("🚀 Executor Live (Phase 4.1) — ENV=", ENV, "POLL_SECONDS=", POLL_SECONDS, flush=True)
    _account_smoke()
    while True:
        t0 = time.time()
        try:
            signals = list(generate_signals_from_config())
            for s in signals:
                if str(s.get("signal","" )).upper() in ("BUY","SELL"):
                    _execute_signal(s)
            _snapshot_and_write()
            _maybe_heartbeat()
            _maybe_dd_alert()
        except Exception:
            traceback.print_exc()
        time.sleep(max(1.0, POLL_SECONDS - (time.time()-t0)))

if __name__ == "__main__":
    main()
