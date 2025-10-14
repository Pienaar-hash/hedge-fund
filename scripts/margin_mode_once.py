#!/usr/bin/env python3
"""
Set USD-M futures marginType=CROSSED once per symbol (idempotent).
Skips with code -4046 "No need to change margin type."
"""
import os
import time
import hmac
import hashlib
import requests
import argparse

B = (
    "https://testnet.binancefuture.com"
    if str(os.getenv("BINANCE_TESTNET", "1")).lower() in ("1", "true", "yes", "on")
    else "https://fapi.binance.com"
)
AK = os.environ["BINANCE_API_KEY"]
SK = os.environ["BINANCE_API_SECRET"]
H = {"X-MBX-APIKEY": AK}


def sig(query: str) -> str:
    return hmac.new(SK.encode(), query.encode(), hashlib.sha256).hexdigest()

def set_cross(sym: str):
    ts = str(int(time.time() * 1000))
    query = f"symbol={sym}&marginType=CROSSED&timestamp={ts}&recvWindow=5000"
    response = requests.post(
        B + "/fapi/v1/marginType?" + query + "&signature=" + sig(query),
        headers=H,
        timeout=10,
    )
    if response.status_code == 200:
        print(sym, "OK", response.text)
        return
    text = response.text
    if '"code":-4046' in text:
        print(sym, "SKIP already CROSSED")
        return
    print(sym, "ERR", text)
    response.raise_for_status()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+", help="Symbols to set marginType=CROSSED")
    a = ap.parse_args()
    for s in a.symbols:
        set_cross(s)
