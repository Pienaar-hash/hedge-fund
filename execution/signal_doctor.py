#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from .risk_limits import RiskState, check_order

try:
    from .exchange_utils import get_price, get_symbol_filters
except Exception:
    # Allow running without network: stub minimal behavior
    def get_price(symbol: str) -> float:
        return 0.0

    def get_symbol_filters(symbol: str) -> Dict[str, Any]:
        raise RuntimeError("exchange access unavailable")


WL_DEFAULT = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "LTCUSDT",
    "WIFUSDT",
    "LINKUSDT",
]


def _load_cfg() -> Dict[str, Any]:
    try:
        return json.load(open("config/risk_limits.json"))
    except Exception:
        return {"global": {}, "per_symbol": {}}


def _nav_snapshot_fallback() -> float:
    try:
        # Best effort from executor's account snapshot if available
        from .executor_live import _compute_nav

        return float(_compute_nav())
    except Exception:
        return 1000.0  # safe placeholder


def diagnose_symbols(env: str, testnet: bool, symbols: List[str]) -> int:
    cfg = _load_cfg()
    g = cfg.get("global") or {}
    wl = g.get("whitelist") or WL_DEFAULT
    wl_set = {str(x).upper() for x in wl}
    now = time.time()
    nav = _nav_snapshot_fallback()
    st = RiskState()

    print(f"[doctor] ENV={env} testnet={int(testnet)} nav={nav:.2f}")
    exit_code = 0
    for sym in symbols:
        s = str(sym).upper()
        if s not in wl_set:
            print(
                json.dumps(
                    {
                        "symbol": s,
                        "would_emit": False,
                        "would_block": True,
                        "reasons": ["not_whitelisted"],
                    }
                )
            )
            continue
        # Defaults
        exch_min_notional = 0.0
        price = 0.0
        listed = True
        try:
            price = float(get_price(s))
            f = get_symbol_filters(s)
            exch_min_notional = float(
                (f.get("MIN_NOTIONAL", {}) or {}).get("notional", 0) or 0.0
            )
        except Exception:
            listed = False

        # Strategy leverage and capital from strategy_config.json (fallbacks)
        cap = 10.0
        lev = float(g.get("max_leverage", 20) or 20)
        try:
            scfg = json.load(open("config/strategy_config.json"))
            arr = scfg.get("strategies", []) if isinstance(scfg, dict) else []
            for row in arr:
                if isinstance(row, dict) and str(row.get("symbol", "")).upper() == s:
                    cap = float(row.get("capital_per_trade", cap) or cap)
                    lev = float(row.get("leverage", lev) or lev)
                    break
        except Exception:
            pass

        vetoes: List[str] = []
        if not listed:
            vetoes.append("not_listed")

        # Risk check (gross notional = cap)
        ok, details = check_order(
            symbol=s,
            side="BUY",
            requested_notional=cap,
            price=price,
            nav=nav,
            open_qty=0.0,
            now=now,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
            lev=lev,
        )
        reasons = list(details.get("reasons", [])) if isinstance(details, dict) else []
        vetoes.extend([r for r in reasons if r not in vetoes])
        would_emit = listed and ok and (cap >= max(exch_min_notional, float(g.get("min_notional_usdt", 0) or 0)))
        out = {
            "symbol": s,
            "price": price,
            "cap": cap,
            "exch_min_notional": exch_min_notional,
            "lev": lev,
            "veto": vetoes,
            "would_emit": bool(would_emit),
            "would_block": not bool(would_emit),
        }
        print(json.dumps(out))
        if not would_emit:
            exit_code = 0  # doctor itself should not hard-fail; keep 0
    return exit_code


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Signal doctor: diagnostics for whitelist symbols")
    p.add_argument("--env", default=os.getenv("ENV", "prod"))
    p.add_argument("--testnet", type=int, default=int(os.getenv("BINANCE_TESTNET", "0")))
    p.add_argument("--symbols", type=str, default=",".join(WL_DEFAULT))
    p.add_argument("--once", action="store_true", help="Run once and exit")
    args = p.parse_args(argv)
    syms = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    rc = diagnose_symbols(args.env, bool(args.testnet), syms)
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
