#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def _load_risk_cfg() -> Dict[str, Any]:
    path = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def probe_symbols(symbols: List[str], notional: float, lev: float, nav: float) -> List[Dict[str, Any]]:
    from execution.signal_screener import would_emit
    from execution.universe_resolver import symbol_tier

    out: List[Dict[str, Any]] = []
    cfg = _load_risk_cfg()
    max_conc = int(float(((cfg.get("global") or {}).get("max_concurrent_positions", 0)) or 0))

    # For a simple probe, assume zero current exposure/positions, focus on gating logic.
    for sym in symbols:
        ok, reasons, extra = would_emit(
            sym,
            side="BUY",
            notional=notional,
            lev=lev,
            nav=nav,
            open_positions_count=0,
            current_gross_notional=0.0,
            current_tier_gross_notional=0.0,
            orderbook_gate=True,
        )
        out.append(
            {
                "symbol": sym,
                "tier": symbol_tier(sym),
                "would_emit": bool(ok),
                "reasons": reasons,
                "ob": extra.get("flag") if isinstance(extra, dict) else None,
                "max_concurrent_positions": max_conc,
            }
        )
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Screener probe: print would_emit and veto stack per symbol")
    ap.add_argument("symbols", nargs="*", help="Symbols to probe (e.g., BTCUSDT ETHUSDT)")
    ap.add_argument("--notional", type=float, default=float(os.getenv("PROBE_NOTIONAL", "10")))
    ap.add_argument("--lev", type=float, default=float(os.getenv("PROBE_LEV", "20")))
    ap.add_argument("--nav", type=float, default=float(os.getenv("PROBE_NAV", "1000")))
    args = ap.parse_args(argv)

    syms = [s.upper() for s in (args.symbols or [])]
    if not syms:
        # default to CORE tier leader candidates if tiers config present
        try:
            tiers = json.load(open(os.getenv("SYMBOL_TIERS_CONFIG", "config/symbol_tiers.json")))
            syms = list((tiers.get("CORE") or [])[:5])
        except Exception:
            syms = ["BTCUSDT"]

    rows = probe_symbols(syms, notional=args.notional, lev=args.lev, nav=args.nav)
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

