#!/usr/bin/env python3
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


def set_leverage(sym: str, lev: int):
    ts = str(int(time.time() * 1000))
    payload = f"symbol={sym}&leverage={lev}&timestamp={ts}&recvWindow=5000"
    response = requests.post(
        B + "/fapi/v1/leverage?" + payload + "&signature=" + sig(payload),
        headers=H,
        timeout=10,
    )
    print(sym, response.status_code, response.text)
    response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lev", type=int, required=True)
    parser.add_argument("symbols", nargs="+")
    args = parser.parse_args()
    for symbol in args.symbols:
        set_leverage(symbol, args.lev)
