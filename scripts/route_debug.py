#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.intel.router_policy import router_policy
from execution.order_router import (
    MAX_SPREAD_FOR_MAKER_BPS,
    WIDE_SPREAD_OFFSET_CLAMP_BPS,
)


def evaluate_route(symbol: str, side: str, notional: float, price: float, spread_bps: float | None, reduce_only: bool) -> Dict[str, Any]:
    policy = router_policy(symbol)
    reasons = []
    maker_allowed = True
    maker_qty = notional / price if price > 0 else None
    maker_price = price
    taker_bias = str(policy.taker_bias or "").lower()
    try:
        offset_bps = float(policy.offset_bps) if policy.offset_bps is not None else float(suggest_maker_offset_bps(symbol))
    except Exception:
        offset_bps = 2.0
    if not policy.maker_first:
        maker_allowed = False
        reasons.append("policy_maker_disabled")
    if policy.quality != "good":
        maker_allowed = False
        reasons.append("policy_quality_not_good")
    if taker_bias == "prefer_taker":
        maker_allowed = False
        reasons.append("policy_bias_prefers_taker")
    if reduce_only:
        maker_allowed = False
        reasons.append("reduce_only")
    if maker_qty is None or maker_qty <= 0:
        maker_allowed = False
        reasons.append("missing_maker_qty")
    if maker_price is None or maker_price <= 0:
        maker_allowed = False
        reasons.append("missing_maker_price")
    if spread_bps is not None and spread_bps > MAX_SPREAD_FOR_MAKER_BPS:
        maker_allowed = False
        reasons.append("spread_too_wide")
    spread_clamped = False
    if maker_allowed and spread_bps is not None and spread_bps > WIDE_SPREAD_OFFSET_CLAMP_BPS:
        offset_bps = min(offset_bps, WIDE_SPREAD_OFFSET_CLAMP_BPS)
        spread_clamped = True
        reasons.append("wide_spread_clamped")
    route = "maker" if maker_allowed else "taker"
    return {
        "symbol": symbol,
        "side": side,
        "notional": notional,
        "price": price,
        "route": route,
        "maker_allowed": maker_allowed,
        "spread_bps": spread_bps,
        "spread_clamped": spread_clamped,
        "reasons": reasons,
        "policy": {
            "maker_first": policy.maker_first,
            "taker_bias": policy.taker_bias,
            "quality": policy.quality,
            "reason": policy.reason,
            "offset_bps": offset_bps,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Router policy dry-run debugger (no orders sent)")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g., BTCUSDT")
    parser.add_argument("--side", choices=["BUY", "SELL"], required=True, help="Side")
    parser.add_argument("--notional", type=float, required=True, help="Quote notional (USDT)")
    parser.add_argument("--price", type=float, required=False, help="Price hint used for sizing; defaults to notional-based unit")
    parser.add_argument("--spread-bps", type=float, default=None, help="Order book spread in bps (optional)")
    parser.add_argument("--reduce-only", action="store_true", help="Simulate reduce-only request")
    parser.add_argument("--json", action="store_true", help="Print raw JSON decision")
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    side = args.side.upper()
    notional = float(args.notional or 0.0)
    price = float(args.price or 1.0)
    if notional <= 0.0 or price <= 0.0:
        sys.stderr.write("[route_debug] notional and price must be positive\n")
        return 1

    decision = evaluate_route(symbol, side, notional, price, args.spread_bps, args.reduce_only)
    print("== Route Decision (dry-run) ==")
    print(f"symbol={symbol} side={side} route={decision['route']} maker_allowed={decision['maker_allowed']}")
    print(f"policy quality={decision['policy']['quality']} maker_first={decision['policy']['maker_first']} bias={decision['policy']['taker_bias']}")
    print(f"offset_bps={decision['policy']['offset_bps']} spread_bps={decision.get('spread_bps')} reasons={decision['reasons']}")
    if args.json:
        print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
