#!/usr/bin/env python3
import os
import time
import hmac
import hashlib
import requests
import argparse
B="https://testnet.binancefuture.com" if str(os.getenv("BINANCE_TESTNET","1")).lower() in ("1","true","yes","on") else "https://fapi.binance.com"
AK=os.environ["BINANCE_API_KEY"]; SK=os.environ["BINANCE_API_SECRET"]; H={"X-MBX-APIKEY":AK}
def sig(q): return hmac.new(SK.encode(), q.encode(), hashlib.sha256).hexdigest()
def set_leverage(sym: str, lev: int):
    ts=str(int(time.time()*1000)); q=f"symbol={sym}&leverage={lev}&timestamp={ts}&recvWindow=5000"
    r=requests.post(B+"/fapi/v1/leverage?"+q+"&signature="+sig(q), headers=H, timeout=10)
    print(sym, r.status_code, r.text); r.raise_for_status()
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--lev", type=int, required=True); ap.add_argument("symbols", nargs="+")
    a=ap.parse_args(); [set_leverage(s, a.lev) for s in a.symbols]
