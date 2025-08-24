#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, logging
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from execution.exchange_utils import is_testnet, get_balances, get_positions, _is_dual_side, place_market_order_sized, get_order
try:
    from execution.state_publish import StatePublisher, normalize_positions, publish_nav_value, compute_nav
    _PUB = StatePublisher(interval_s=int(os.getenv('FS_PUB_INTERVAL','60')))
except Exception:
    _PUB = None

# Screener (optional; if absent, we just idle)
try:
    from execution.signal_screener import generate_signals_from_config
except Exception:
    generate_signals_from_config = None

LOG = logging.getLogger("executor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRY_RUN     = os.getenv("DRY_RUN", "0") in ("1","true","True")
POLL_SEC    = int(os.getenv("POLL_SEC", "60"))
INTENT_TEST = os.getenv("INTENT_TEST", "0") in ("1","true","True")
MAX_LOOPS   = int(os.getenv("MAX_LOOPS", "0"))  # 0 = infinite

def _account_banner():
    try:
        bals = get_balances(); assets = sorted({b.get("asset") for b in bals})
    except Exception:
        assets = []
    try:
        pos = get_positions(); open_cnt = sum(1 for p in pos if abs(float(p.get("qty", p.get("positionAmt",0)) or 0))>0)
    except Exception:
        open_cnt = 0
    LOG.info("[executor] account OK — futures=%s testnet=%s dry_run=%s balances: %s positions: %s",
             True, is_testnet(), DRY_RUN, assets or [], open_cnt)

def _send_intent(intent: Dict[str, Any]) -> None:
    sym = intent["symbol"]
    side = str(intent.get("signal","BUY")).upper()
    ps   = intent.get("positionSide") or ("LONG" if side=="BUY" else "SHORT")
    usd  = float(intent.get("capital_per_trade", 100.0))
    lev  = float(intent.get("leverage", 1))
    reduce = bool(intent.get("reduceOnly", False))
    LOG.info("[executor] INTENT symbol=%s side=%s ps=%s cap=%.4f lev=%.2f reduceOnly=%s", sym, side, ps, usd, lev, reduce)
    if DRY_RUN:
        LOG.info("[executor] DRY_RUN — skipping SEND_ORDER")
        return
    LOG.info("[executor] SEND_ORDER %s %s", sym, side)
    try:
        resp = place_market_order_sized(symbol=sym, side=side, usd_capital=usd, leverage=lev, position_side=ps, reduce_only=reduce)
        LOG.info("[executor] ORDER_REQ 200 id=%s avgPrice=%s qty=%s", resp.get("orderId"), resp.get("avgPrice"), resp.get("executedQty"))
        try:
            oid = resp.get("orderId"); sym = resp.get("symbol") or sym
            for _ in range(2):
                time.sleep(0.6)
                st = get_order(sym, oid)
                if float(st.get("executedQty", "0") or 0) > 0:
                    LOG.info("[executor] ORDER_FILL id=%s status=%s avgPrice=%s qty=%s", st.get("orderId"), st.get("status"), st.get("avgPrice"), st.get("executedQty"))
                    break
        except Exception as e:
            LOG.info("[executor] ORDER_FETCH_WARN %s", e)
    except Exception as e:
        LOG.exception("[executor] ORDER_ERR %s", e)

def main():
    if not _is_dual_side():
        LOG.warning("[executor] WARNING: dualSide is False; enable Hedge Mode before live trading.")
    _account_banner()

    loops=0; sent_test=False
    while True:
        signals: List[Dict[str, Any]] = []
        if generate_signals_from_config:
            try:
                for s in generate_signals_from_config() or []:
                    signals.append(s)
            except Exception as e:
                LOG.exception("[screener] error: %s", e)

        if not signals and INTENT_TEST and not sent_test:
            ti = {"symbol":"BTCUSDT","signal":"BUY","capital_per_trade":120.0,"leverage":1,"positionSide":"LONG"}
            LOG.info("[screener->executor] %s", ti)
            _send_intent(ti); sent_test=True

        for intent in signals:
            LOG.info("[screener->executor] %s", intent)
            _send_intent(intent)

        loops += 1
        if MAX_LOOPS and loops >= MAX_LOOPS:
            LOG.info("[executor] MAX_LOOPS reached — exiting.")
            break
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()