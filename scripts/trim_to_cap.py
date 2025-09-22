#!/usr/bin/env python3
import json
import os
from decimal import Decimal, ROUND_DOWN
from typing import List, Dict, Any

from execution.utils import load_json
from execution.nav import PortfolioSnapshot
from execution.exchange_utils import (
    get_positions,
    get_price,
    get_symbol_filters,
    place_market_order,
)


def _quantize(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    d_step = Decimal(str(step))
    return float((Decimal(str(qty)) / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


def main() -> None:
    cfg = load_json("config/strategy_config.json") or {}
    snapshot = PortfolioSnapshot(cfg)
    nav_usd = float(snapshot.current_nav_usd())
    gross_usd = float(snapshot.current_gross_usd())

    sizing = (cfg.get("sizing") or {})
    cap_pct = float(sizing.get("max_gross_exposure_pct", 0.0) or 0.0)
    if os.environ.get("EVENT_GUARD", "0") == "1":
        cap_pct *= 0.8
    cap_usd = nav_usd * (cap_pct / 100.0) if nav_usd > 0 and cap_pct > 0 else 0.0

    if cap_usd <= 0:
        print(json.dumps({"status": "no_cap", "nav_usd": nav_usd, "cap_usd": cap_usd}, indent=2))
        return

    excess = gross_usd - cap_usd
    if excess <= 0:
        print(
            json.dumps(
                {
                    "status": "within_cap",
                    "nav_usd": nav_usd,
                    "gross_usd": gross_usd,
                    "cap_usd": cap_usd,
                },
                indent=2,
            )
        )
        return

    dry_run = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes", "on")
    remaining = excess
    actions: List[Dict[str, Any]] = []

    positions = list(get_positions() or [])
    # Sort by largest notional first
    enriched: List[Dict[str, Any]] = []
    for p in positions:
        qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
        if qty == 0.0:
            continue
        symbol = str(p.get("symbol", "")).upper()
        mark = float(p.get("markPrice") or 0.0)
        if mark <= 0:
            try:
                mark = float(get_price(symbol))
            except Exception:
                mark = abs(float(p.get("entryPrice") or 0.0))
        notional = abs(qty) * abs(mark)
        enriched.append(
            {
                "symbol": symbol,
                "qty": qty,
                "abs_qty": abs(qty),
                "mark": mark,
                "notional": notional,
                "positionSide": p.get("positionSide", "BOTH"),
            }
        )

    enriched.sort(key=lambda x: x["notional"], reverse=True)

    total_trimmed = 0.0
    for pos in enriched:
        if remaining <= 0:
            break
        symbol = pos["symbol"]
        qty = pos["qty"]
        abs_qty = pos["abs_qty"]
        mark = max(pos["mark"], 0.0)
        if mark <= 0:
            continue
        filters = get_symbol_filters(symbol)
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
        step = float(lot.get("stepSize", 0.0) or 0.0)

        max_reducible_usd = abs_qty * mark
        target_usd = min(remaining, max_reducible_usd)
        if target_usd <= 0:
            continue

        qty_target = min(abs_qty, target_usd / mark)
        qty_rounded = _quantize(qty_target, step)
        if qty_rounded <= 0:
            qty_rounded = _quantize(abs_qty, step)
        if qty_rounded <= 0:
            continue

        order_side = "SELL" if qty > 0 else "BUY"
        position_side = pos.get("positionSide") or ("LONG" if qty > 0 else "SHORT")
        notional_sent = qty_rounded * mark

        action = {
            "symbol": symbol,
            "side": order_side,
            "positionSide": position_side,
            "qty": qty_rounded,
            "approx_notional": notional_sent,
            "reduceOnly": True,
        }

        if dry_run:
            action["status"] = "dry_run"
        else:
            try:
                resp = place_market_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=qty_rounded,
                    position_side=position_side,
                    reduce_only=True,
                )
                action["status"] = "ok"
                action["response"] = resp
            except Exception as exc:
                action["status"] = "error"
                action["error"] = str(exc)
                actions.append(action)
                # If order failed we do not adjust remaining
                continue

        remaining -= notional_sent
        total_trimmed += notional_sent
        actions.append(action)

    summary = {
        "nav_usd": nav_usd,
        "gross_usd_before": gross_usd,
        "cap_usd": cap_usd,
        "target_trim_usd": excess,
        "executed_trim_usd": total_trimmed,
        "gross_usd_after_estimate": max(gross_usd - total_trimmed, 0.0),
        "actions": actions,
        "dry_run": dry_run,
        "remaining_excess_usd": max(remaining, 0.0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
