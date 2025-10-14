#!/usr/bin/env python3
"""Read exchangeInfo and propose additions to config/pairs_universe.json.

Default is read-only: prints symbols (USD-M, quote=USDT, status=TRADING).
Use --write to merge them into the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import requests

B = (
    "https://testnet.binancefuture.com"
    if str(os.getenv("BINANCE_TESTNET", "1")).lower() in ("1", "true", "yes", "on")
    else "https://fapi.binance.com"
)
ROOT = "/root/hedge-fund"
CONF = f"{ROOT}/config/pairs_universe.json"
AK = os.getenv("BINANCE_API_KEY", "")
H = {"X-MBX-APIKEY": AK} if AK else {}


def fetch_symbols() -> List[str]:
    response = requests.get(B + "/fapi/v1/exchangeInfo", headers=H, timeout=15)
    response.raise_for_status()
    data = response.json()
    symbols: List[str] = []
    for entry in data.get("symbols", []) or []:
        if entry.get("contractType") not in (
            "PERPETUAL",
            "CURRENT_QUARTER",
            "NEXT_QUARTER",
        ):
            continue
        if entry.get("quoteAsset") != "USDT":
            continue
        if entry.get("status") != "TRADING":
            continue
        symbols.append(entry["symbol"])
    return sorted(symbols)


def load_universe() -> Dict[str, Dict[str, float]]:
    try:
        with open(CONF, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {"symbols": [], "overrides": {}}


def save_universe(obj: Dict[str, Dict[str, float]]) -> None:
    tmp = CONF + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
    os.replace(tmp, CONF)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist merged universe to config/pairs_universe.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Only show first N additions (for preview)",
    )
    args = parser.parse_args()

    live = set(load_universe().get("symbols", []))
    exch = set(fetch_symbols())
    additions = sorted(exch - live)

    if not additions:
        print("Universe up-to-date. No additions.")
        return

    print(f"Candidates to add (first {args.limit}):")
    for symbol in additions[: args.limit]:
        print("-", symbol)

    if not args.write:
        return

    universe = load_universe()
    universe_symbols = sorted(set(universe.get("symbols", [])) | exch)
    universe["symbols"] = universe_symbols
    overrides = universe.setdefault("overrides", {})
    for symbol in additions:
        overrides.setdefault(symbol, {"target_leverage": 3, "min_notional": 5.0})
    save_universe(universe)
    print("config/pairs_universe.json updated.")


if __name__ == "__main__":
    main()
