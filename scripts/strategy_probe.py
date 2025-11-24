#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Mapping

from execution.signal_screener import generate_intents
from execution.risk_limits import check_order, RiskState
from execution.nav import PortfolioSnapshot
from execution.utils import load_json
from execution.universe_resolver import resolve_allowed_symbols


def _load_nav() -> float:
    cfg = load_json("config/strategy_config.json") or {}
    snap = PortfolioSnapshot(cfg)
    try:
        return float(snap.current_nav_usd())
    except Exception:
        return 0.0


def _intent_summary(intent: Mapping[str, Any]) -> str:
    symbol = intent.get("symbol")
    tf = intent.get("timeframe") or intent.get("tf")
    side = intent.get("signal")
    gross = intent.get("gross_usd") or intent.get("capital_per_trade")
    reasons = intent.get("veto") or []
    return f"- {symbol} {tf} side={side} gross={round(float(gross or 0.0), 4)} reasons={list(reasons)}"


def main() -> None:
    nav = _load_nav()
    allowed, _ = resolve_allowed_symbols()
    intents = list(generate_intents())
    print("=== STRATEGY PROBE (testnet) ===")
    print(f"NAV: {round(nav, 4)}")
    print(f"Symbols: {', '.join(sorted(allowed or []))}")
    print("Intents:")
    eligible_for_risk = 0
    for intent in intents:
        print(_intent_summary(intent))
        allowed_flag, detail = check_order(
            symbol=intent.get("symbol", ""),
            side=intent.get("signal", ""),
            requested_notional=float(intent.get("gross_usd") or 0.0),
            price=float(intent.get("price") or 0.0),
            nav=nav,
            open_qty=0.0,
            now=0.0,
            cfg=None,
            state=RiskState(),
            current_gross_notional=0.0,
            lev=float(intent.get("leverage") or 1.0),
            open_positions_count=0,
            tier_name=None,
            current_tier_gross_notional=0.0,
        )
        if not allowed_flag:
            vetoes = detail.get("reasons") if isinstance(detail, Mapping) else None
            print(f"  risk: veto reasons={vetoes}")
        else:
            eligible_for_risk += 1
    print(f"Summary: attempted={getattr(intents, 'attempted', len(intents))} emitted={getattr(intents, 'emitted', len(intents))} eligible_for_risk={eligible_for_risk}")


if __name__ == "__main__":
    main()
