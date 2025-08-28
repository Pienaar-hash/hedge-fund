#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import os
import time
import logging
import json
from typing import Any, Dict, Iterable, List, Optional, Callable

# Optional .env so Supervisor doesn't need to export everything
try:
    from dotenv import load_dotenv
    load_dotenv() or load_dotenv("/root/hedge-fund/.env")
except Exception:
    pass

LOG = logging.getLogger("exutil")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [exutil] %(message)s")

# ---- Exchange utils (binance) ----
from execution.exchange_utils import (
    is_testnet, get_balances, get_positions, _is_dual_side, place_market_order_sized,
)
from execution.risk_limits import RiskState, check_order

# ---- Screener (best effort) ----
generate_signals_from_config: Optional[Callable[[], Iterable[Dict[str, Any]]]] = None
try:
    from execution.signal_screener import generate_signals_from_config as _gen
    generate_signals_from_config = _gen
except Exception:
    generate_signals_from_config = None

# ---- Firestore publisher handle (revisions differ) ----
_PUB: Any
try:
    from execution.state_publish import StatePublisher, publish_intent_audit, publish_order_audit, publish_close_audit
    _PUB = StatePublisher(interval_s=int(os.getenv("FS_PUB_INTERVAL", "60")))
except Exception:
    class _Publisher:
        def __init__(self, interval_s: int = 60):
            self.interval_s = interval_s
    def publish_intent_audit(intent: Dict[Any, Any]) -> None:
        pass
    def publish_order_audit(symbol: str, event: Dict[Any, Any]) -> None:
        pass
    def publish_close_audit(symbol: str, position_side: str, event: Dict[Any, Any]) -> None:
        pass

    _PUB = _Publisher(interval_s=int(os.getenv("FS_PUB_INTERVAL", "60")))

# ---- risk limits config ----
_RISK_CFG_PATH = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")
_RISK_CFG: Dict[str, Any] = {}
try:
    with open(_RISK_CFG_PATH, "r") as fh:
        _RISK_CFG = json.load(fh) or {}
except Exception as e:
    logging.getLogger("exutil").warning("[risk] config load failed (%s): %s", _RISK_CFG_PATH, e)

_RISK_STATE = RiskState()

# ---- knobs ----
SLEEP       = int(os.getenv("LOOP_SLEEP", "60"))
MAX_LOOPS   = int(os.getenv("MAX_LOOPS", "0") or 0)
DRY_RUN     = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
INTENT_TEST = os.getenv("INTENT_TEST", "0").lower() in ("1","true","yes")

# ------------- helpers -------------

def _account_snapshot() -> None:
    try:
        bals = get_balances() or []
        assets = sorted({b.get("asset","?") for b in bals})
        pos = [p for p in get_positions() if float(p.get("qty", p.get("positionAmt", 0)) or 0) != 0]
        LOG.info("[executor] account OK — futures=%s testnet=%s dry_run=%s balances: %s positions: %d",
                 True, is_testnet(), DRY_RUN, assets, len(pos))
    except Exception as e:
        LOG.exception("[executor] preflight_error: %s", e)


def _compute_nav() -> float:
    nav_val = 0.0
    try:
        from execution.state_publish import compute_nav
        nav_val = float(compute_nav())
        return nav_val
    except Exception:
        pass
    try:
        from execution.exchange_utils import get_account
        acc = get_account()
        nav_val = float(
            acc.get("totalMarginBalance") or
            (float(acc.get("totalWalletBalance", 0) or 0) + float(acc.get("totalUnrealizedProfit", 0) or 0))
        )
    except Exception as e:
        LOG.error("[executor] account NAV error: %s", e)
    return float(nav_val or 0.0)


def _gross_and_open_qty(symbol: str, pos_side: str, positions: Iterable[Dict[str, Any]]) -> tuple[float, float]:
    gross = 0.0
    open_qty = 0.0
    for p in positions or []:
        try:
            qty = float(p.get("qty", p.get("positionAmt", 0)) or 0.0)
            entry = float(p.get("entryPrice") or 0.0)
            gross += abs(qty) * abs(entry)
            if str(p.get("symbol")) == symbol:
                ps = p.get("positionSide", "BOTH")
                if ps == pos_side or (ps == "BOTH" and pos_side in ("LONG", "SHORT")):
                    open_qty = max(open_qty, abs(qty))
        except Exception:
            continue
    return gross, open_qty

def _send_order(intent: Dict[str, Any]) -> None:
    symbol = intent["symbol"]
    sig = str(intent.get("signal","")).upper()
    side = "BUY" if sig == "BUY" else "SELL"
    pos_side = intent.get("positionSide") or ("LONG" if side=="BUY" else "SHORT")
    cap = float(intent.get("capital_per_trade", 0) or 0)
    lev = float(intent.get("leverage", 1) or 1)
    notional = cap * lev
    reduce_only = bool(intent.get("reduceOnly", False))

    LOG.info("[executor] INTENT symbol=%s side=%s ps=%s cap=%.4f lev=%.2f reduceOnly=%s",
             symbol, side, pos_side, cap, lev, reduce_only)

    # Risk checks
    try:
        positions = list(get_positions() or [])
    except Exception:
        positions = []
    nav = _compute_nav()
    current_gross, sym_open_qty = _gross_and_open_qty(symbol, pos_side, positions)

    # Note attempt for burst limiting
    try:
        _RISK_STATE.note_attempt(time.time())
    except Exception:
        pass

    ok, details = check_order(
        symbol=symbol,
        side=side,
        requested_notional=notional,
        price=0.0,
        nav=nav,
        open_qty=sym_open_qty,
        now=time.time(),
        cfg=_RISK_CFG,
        state=_RISK_STATE,
        current_gross_notional=current_gross,
        lev=lev,
    )
    reasons = details.get("reasons", []) if isinstance(details, dict) else []
    if not ok:
        reason = reasons[0] if reasons else "blocked"
        price = float(intent.get("price", 0.0) or 0.0)
        block_info = {
            "symbol": symbol,
            "side": side,
            "reason": reason,
            "notional": notional,
            "price": price,
        }
        LOG.warning("[risk] block %s", block_info)
        # Best-effort Telegram alert
        try:
            from execution.telegram_utils import send_telegram
            send_telegram(f"Risk‑block {symbol} {side}: {reason}\nnotional={notional:.2f} price={price:.2f}", silent=True)
        except Exception:
            pass
        try:
            audit = {
                "phase": "blocked",
                "side": side,
                "positionSide": pos_side,
                "reason": reason,
                "reasons": reasons,
                "notional": notional,
                "nav": nav,
                "open_qty": sym_open_qty,
                "gross": current_gross,
            }
            if isinstance(details, dict) and "cooldown_until" in details:
                audit["cooldown_until"] = details.get("cooldown_until")
            publish_order_audit(symbol, audit)
        except Exception:
            pass
        return

    # Audit intent snapshot
    try:
        publish_intent_audit({
            **intent,
            "t": time.time(),
            "side": side,
            "positionSide": pos_side,
            "notional": notional,
            "reduceOnly": reduce_only,
        })
    except Exception:
        pass

    if DRY_RUN:
        LOG.info("[executor] DRY_RUN — skipping SEND_ORDER")
        try:
            publish_order_audit(symbol, {"phase":"dry_run", "side": side, "positionSide": pos_side, "notional": notional})
        except Exception:
            pass
        return

    LOG.info("[executor] SEND_ORDER %s %s", symbol, side)
    try:
        publish_order_audit(symbol, {"phase":"request", "side": side, "positionSide": pos_side, "notional": notional, "reduceOnly": reduce_only})
    except Exception:
        pass
    try:
        resp = place_market_order_sized(
            symbol=symbol, side=side, notional=notional, leverage=lev,
            position_side=pos_side, reduce_only=reduce_only
        )
    except Exception as e:
        try:
            _RISK_STATE.note_error(time.time())
        except Exception:
            pass
        try:
            publish_order_audit(symbol, {"phase":"error", "side": side, "positionSide": pos_side, "error": str(e)})
        except Exception:
            pass
        raise
    oid = resp.get("orderId")
    avg = resp.get("avgPrice","0.00")
    qty = resp.get("executedQty", resp.get("origQty","0"))
    st  = resp.get("status")
    LOG.info("[executor] ORDER_REQ 200 id=%s avgPrice=%s qty=%s", oid, avg, qty)
    try:
        publish_order_audit(symbol, {"phase":"response", "side": side, "positionSide": pos_side, "status": st, "orderId": oid, "avgPrice": avg, "qty": qty})
    except Exception:
        pass
    if st == "FILLED":
        LOG.info("[executor] ORDER_FILL id=%s status=%s avgPrice=%s qty=%s", oid, st, avg, qty)
        try:
            _RISK_STATE.note_fill(symbol, time.time())
        except Exception:
            pass
        # Close audit (best-effort): mark reduceOnly fills as close events
        if reduce_only or (side == "SELL" and pos_side == "LONG") or (side == "BUY" and pos_side == "SHORT"):
            try:
                publish_close_audit(symbol, pos_side, {"orderId": oid, "avgPrice": avg, "qty": qty, "status": st})
            except Exception:
                pass

def _pub_tick() -> None:
    """
    Try StatePublisher.{tick,run_once,step,publish,update}; if no such method,
    publish a minimal NAV+positions snapshot inline so the dashboard stays live.
    """
    # Try common method names on the object first
    for meth in ("tick","run_once","step","publish","update"):
        m = getattr(_PUB, meth, None)
        if callable(m):
            try:
                m()
                return
            except Exception as e:
                LOG.error("[executor] publisher.%s error: %s", meth, e)

    # Inline minimal publish (NAV + positions) — Firestore-first dashboard compatible
    try:
        nav_val = None
        try:
            from execution.state_publish import compute_nav
            nav_val = float(compute_nav())
        except Exception as e:
            LOG.error("[executor] compute_nav not available: %s", e)
            try:
                from execution.exchange_utils import get_account
                acc = get_account()
                nav_val = float(
                    acc.get("totalMarginBalance") or
                    (float(acc.get("totalWalletBalance",0) or 0) +
                     float(acc.get("totalUnrealizedProfit",0) or 0))
                )
            except Exception as ee:
                LOG.error("[executor] account NAV error: %s", ee)

        # Positions -> normalized rows
        try:
            raw: Iterable[Dict[str, Any]] = get_positions()
        except Exception as e:
            LOG.error("[executor] get_positions error: %s", e)
            raw = []
        rows: List[Dict[str, Any]] = []
        for p in raw:
            try:
                rows.append({
                    "symbol": p.get("symbol"),
                    "positionSide": p.get("positionSide","BOTH"),
                    "qty": float(p.get("qty", p.get("positionAmt", 0)) or 0.0),
                    "entryPrice": float(p.get("entryPrice") or 0.0),
                    "unrealized": float(p.get("unRealizedProfit", p.get("unrealized", 0)) or 0.0),
                    "leverage": float(p.get("leverage") or 0.0),
                })
            except Exception:
                continue

        # Firestore write
        try:
            from google.cloud import firestore
            ENV = os.getenv("ENV","prod")
            db = firestore.Client()

            if nav_val is not None:
                doc_nav = db.document(f"hedge/{ENV}/state/nav")
                snap = doc_nav.get()
                series: List[Dict[str, Any]] = []
                if snap.exists:
                    d = snap.to_dict() or {}
                    series = d.get("series") or d.get("rows") or []
                series.append({"t": time.time(), "nav": float(nav_val)})
                series = series[-20000:]
                doc_nav.set({"series": series}, merge=False)

            db.document(f"hedge/{ENV}/state/positions").set({"rows": rows}, merge=False)
        except Exception as e:
            LOG.error("[executor] Firestore publish error: %s", e)
    except Exception as e:
        LOG.error("[executor] publisher fallback error: %s", e)

def _loop_once(i: int) -> None:
    _account_snapshot()

    if INTENT_TEST:
        intent = {
            "symbol":"BTCUSDT","signal":"BUY","capital_per_trade":120.0,
            "leverage":1, "positionSide":"LONG", "reduceOnly":False
        }
        LOG.info("[screener->executor] %s", intent)
        _send_order(intent)
    else:
        if callable(generate_signals_from_config):
            try:
                for intent in (generate_signals_from_config() or []):
                    LOG.info("[screener->executor] %s", intent)
                    _send_order(intent)
            except Exception as e:
                LOG.error("[screener] error: %s", e)
        else:
            LOG.error("[screener] missing signal generator")

    _pub_tick()

def main() -> None:
    try:
        if not _is_dual_side():
            LOG.warning("[executor] WARNING — account not in hedge (dualSide) mode")
    except Exception as e:
        LOG.error("[executor] dualSide check failed: %s", e)

    i = 0
    while True:
        _loop_once(i)
        i += 1
        if MAX_LOOPS and i >= MAX_LOOPS:
            LOG.info("[executor] MAX_LOOPS reached — exiting.")
            break
        time.sleep(SLEEP)

if __name__ == "__main__":
    main()
