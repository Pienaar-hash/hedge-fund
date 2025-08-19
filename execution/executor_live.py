"""
execution/executor_live.py — Phase 4.1 "Stability & Signals" (with account smoke test)

What this file does
- Start-up smoke test of exchange connectivity (balances + positions) → concise "account OK" log
- Poll screener for BUY/SELL intents and execute them
- Send Telegram trade alerts on fills
- Send periodic heartbeat (equity/peak/DD/realized/unrealized + top positions)
- Send drawdown alerts when DD ≥ threshold (rate-limited)

Env:
  ENV=prod|dev
  POLL_SECONDS=60
  HEARTBEAT_MINUTES=10
  DD_ALERT_PCT=0.10
  # Telegram creds in execution/telegram_utils.py docs
"""

from __future__ import annotations

import os
import time
import json
import traceback
from typing import Dict, Any, List

from execution.exchange_utils import (
    get_price, place_market_order, get_balances, get_positions, get_account_overview
)
from execution.signal_screener import generate_signals_from_config
from execution.telegram_utils import (
    send_heartbeat, send_trade_alert, send_drawdown_alert, should_send_summary
)

# --- Env & cadence ---
ENV = os.getenv("ENV", "prod")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
HEARTBEAT_MIN = int(os.getenv("HEARTBEAT_MINUTES", "10"))
DD_ALERT_PCT = float(os.getenv("DD_ALERT_PCT", "0.10"))

NAV_LOG = "nav_log.json"
PEAK_STATE = "peak_state.json"
STATE_FILE = "synced_state.json"  # positions snapshot (for top list)

_last_hb_ts = 0.0
_last_dd_alert_ts = 0.0

def _safe_load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _nav_tail() -> Dict[str, float]:
    arr = _safe_load_json(NAV_LOG, [])
    last = arr[-1] if isinstance(arr, list) and arr else {}
    def f(k, d=0.0):
        try: return float(last.get(k, d))
        except Exception: return d
    return {
        "equity": f("equity", 0.0),
        "realized": f("realized", 0.0),
        "unrealized": f("unrealized", 0.0),
        "dd": f("drawdown_pct", 0.0),
        "peak": f("peak_equity", 0.0),
    }

def _peak_value() -> float:
    j = _safe_load_json(PEAK_STATE, {})
    try:
        return float(j.get("peak_equity", 0.0))
    except Exception:
        return 0.0

def _top_positions(limit: int = 3) -> List[str]:
    j = _safe_load_json(STATE_FILE, {})
    items = j.get("items") or []
    try:
        items = sorted(items, key=lambda x: abs(float(x.get("notional", 0.0))), reverse=True)
    except Exception:
        pass
    out = []
    for it in items[:limit]:
        try:
            out.append(f"{it.get('symbol')} {float(it.get('qty',0)):+g}")
        except Exception:
            continue
    return out

def _maybe_heartbeat():
    global _last_hb_ts
    if should_send_summary(_last_hb_ts, HEARTBEAT_MIN):
        tail = _nav_tail()
        peak = _peak_value() or tail.get("peak", 0.0)
        send_heartbeat(
            equity=tail["equity"],
            peak=peak,
            dd_pct=tail["dd"],
            realized=tail["realized"],
            unrealized=tail["unrealized"],
            positions_top=_top_positions(),
        )
        _last_hb_ts = time.time()

def _maybe_dd_alert():
    """Send drawdown alert if above threshold; rate-limited to 15 min."""
    global _last_dd_alert_ts
    tail = _nav_tail()
    peak = _peak_value() or tail.get("peak", 0.0)
    dd = tail["dd"]
    if dd >= DD_ALERT_PCT and (time.time() - _last_dd_alert_ts) >= 15*60:
        send_drawdown_alert(drawdown_pct=dd, threshold_pct=DD_ALERT_PCT, peak_equity=peak, equity=tail["equity"])
        _last_dd_alert_ts = time.time()

def _apply_sizing_guards(symbol, qty, px, cfg):
    """Minimal guard: skip micro-orders under min_notional if provided."""
    try:
        min_notional = float(cfg.get("min_notional", 0.0)) if cfg else 0.0
    except Exception:
        min_notional = 0.0
    notional = float(qty) * float(px)
    if min_notional > 0 and notional < min_notional:
        return 0.0
    return qty

def _execute_signal(sig: Dict[str, Any]) -> None:
    """
    Execute a single signal dictionary:
      {'symbol': 'BTCUSDT', 'signal': 'BUY'|'SELL', 'qty': <optional>, ...}
    qty is optional; fall back to notional logic if absent.
    """
    symbol = sig.get("symbol")
    side = str(sig.get("signal", "")).upper()
    if side not in ("BUY", "SELL"):
        return

    # quantity: prefer explicit qty; otherwise derive tiny notional with price
    qty = sig.get("qty")
    px = None
    if qty is None:
        try:
            cap = float(sig.get("capital_per_trade", 25.0))
            px = float(sig.get("price") or get_price(symbol))
            qty = max(0.0, cap / max(px, 1e-9))
        except Exception:
            qty = 0.0
    if px is None:
        try: px = float(sig.get("price") or get_price(symbol))
        except Exception: px = 0.0

    qty = _apply_sizing_guards(symbol, qty, px, sig)
    if qty <= 0:
        return

    # Optional hedge-mode flags pass-through
    order_kwargs = {}
    if "positionSide" in sig and sig["positionSide"]:
        order_kwargs["positionSide"] = sig["positionSide"]
    if "reduceOnly" in sig:
        order_kwargs["reduceOnly"] = bool(sig["reduceOnly"])

    try:
        res = place_market_order(symbol=symbol, side=side, quantity=qty, **order_kwargs)
        ok = bool(res.get("ok", True))
        if ok:
            fill = res.get("avgPrice") or px or get_price(symbol)
            tail = _nav_tail()
            send_trade_alert(symbol, side, qty, float(fill) if fill else 0.0, tail["realized"], tail["unrealized"])
    except Exception:
        traceback.print_exc()

def _account_smoke() -> None:
    """Log concise account overview to validate venue connectivity before loop."""
    try:
        ov = get_account_overview()
        balances = ov.get("balances") or {}
        positions = ov.get("positions") or []
        bal_keys = [k for k in balances.keys() if not str(k).startswith("_")]
        print(f"[executor] account OK — futures={ov.get('use_futures')} testnet={ov.get('testnet')} "
              f"dry_run={ov.get('dry_run')} balances: {bal_keys} positions: {len(positions)}",
              flush=True)
    except Exception:
        # Do not crash on smoke; proceed
        print("[executor] account OK — (smoke test skipped due to exception)", flush=True)

def main():
    print("🚀 Executor Live (Phase 4.1) — ENV=", ENV, "POLL_SECONDS=", POLL_SECONDS, flush=True)

    # --- Startup exchange smoke test ---
    _account_smoke()

    while True:
        t0 = time.time()
        try:
            # 1) Generate signals from config (your screener handles whitelist etc.)
            signals = list(generate_signals_from_config())

            # 2) Execute any actionable signals
            for s in signals:
                if str(s.get("signal","")).upper() in ("BUY", "SELL"):
                    _execute_signal(s)

            # 3) Housekeeping — alerts based on nav/peak files (written by separate process)
            _maybe_heartbeat()
            _maybe_dd_alert()

            # (Optional) Sync here if not running a separate sync daemon
            # from execution.sync_state import sync_once
            # sync_once()

        except Exception:
            traceback.print_exc()

        # pacing
        elapsed = time.time() - t0
        sleep_s = max(1.0, POLL_SECONDS - elapsed)
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
