#!/usr/bin/env python3
"""
Read exchangeInfo and propose additions to config/pairs_universe.json.
Default is read-only: prints symbols (USD-M, quote=USDT, status=TRADING).
Use --write to merge them into the JSON.
"""
from __future__ import annotations
import os, json, requests

B = "https://testnet.binancefuture.com" if str(os.getenv("BINANCE_TESTNET","1")).lower() in ("1","true","yes","on") else "https://fapi.binance.com"
ROOT = "/root/hedge-fund"
CONF = f"{ROOT}/config/pairs_universe.json"
AK = os.getenv("BINANCE_API_KEY","")
H = {"X-MBX-APIKEY": AK} if AK else {}

def fetch_symbols():
    r = requests.get(B+"/fapi/v1/exchangeInfo", headers=H, timeout=15); r.raise_for_status()
    data = r.json()
    syms = []
    for s in data.get("symbols", []):
        if s.get("contractType") in ("PERPETUAL","CURRENT_QUARTER","NEXT_QUARTER") and s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING":
            syms.append(s["symbol"])
    return sorted(syms)

def load_universe():
    try:
        with open(CONF, "r") as f: return json.load(f)
    except Exception:
        return {"symbols": [], "overrides": {}}

def save_universe(obj):
    tmp = CONF + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, CONF)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Persist merged universe to config/pairs_universe.json")
    ap.add_argument("--limit", type=int, default=50, help="Only show first N additions (for preview)")
    a = ap.parse_args()

    live = set(load_universe().get("symbols", []))
    exch = set(fetch_symbols())
    add  = sorted(exch - live)

    if not add:
        print("Universe up-to-date. No additions.")
    else:
        print("Candidates to add (first {}):".format(a.limit))
        for s in add[:a.limit]:
            print("-", s)

        if a.write:
            u = load_universe()
            u_syms = sorted(set(u.get("symbols", [])) | exch)
            u["symbols"] = u_syms
            for s in add:
                u["overrides"].setdefault(s, {"target_leverage": 3, "min_notional": 5.0})
            save_universe(u)
            print("config/pairs_universe.json updated.")
