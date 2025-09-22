import hashlib
import hmac
import json
import math
import os
import time

import requests

from execution.signal_screener import generate_signals_from_config

BASE = "https://testnet.binancefuture.com"
AK = os.environ["BINANCE_API_KEY"]
SK = os.environ["BINANCE_API_SECRET"]


def sig(q):
    return hmac.new(SK.encode(), q.encode(), hashlib.sha256).hexdigest()


def get(path, **params):
    r = requests.get(
        BASE + path, params=params, headers={"X-MBX-APIKEY": AK}, timeout=10
    )
    r.raise_for_status()
    return r.json()


def post(path, params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    r = requests.post(
        BASE + path + "?" + q + "&signature=" + sig(q),
        headers={"X-MBX-APIKEY": AK},
        timeout=10,
    )
    print("POST", path, r.status_code, r.text)
    r.raise_for_status()
    return r.json()


def round_step(q, step):
    return math.floor(q / step) * step


def place_from_intent(intent):
    sym = intent["symbol"]
    px = float(get("/fapi/v1/ticker/price", symbol=sym)["price"])
    info = get("/fapi/v1/exchangeInfo")
    fx = {
        x["filterType"]: x
        for x in next(s for s in info["symbols"] if s["symbol"] == sym)["filters"]
    }
    step = float(fx["LOT_SIZE"]["stepSize"])
    min_q = float(fx["LOT_SIZE"]["minQty"])
    cap = float(intent.get("capital_per_trade", 0.0))
    lev = float(intent.get("leverage", 1.0))
    notional = max(1.0, cap * lev)
    qty = round_step(notional / px, step)
    if qty < min_q:
        qty = min_q

    ts = int(time.time() * 1000)
    side = "BUY" if intent["signal"] == "BUY" else "SELL"
    reduce = bool(intent.get("reduceOnly", False))

    # Hedge Mode: set positionSide based on intent + reduceOnly
    posSide = None
    if reduce:
        posSide = "LONG" if side == "SELL" else "SHORT"
    else:
        posSide = "LONG" if side == "BUY" else "SHORT"

    params = {
        "symbol": sym,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "timestamp": ts,
        "positionSide": posSide,
    }
    print("INTENT:", json.dumps(intent, sort_keys=True))
    print("ORDER_PARAMS:", params, "px", px, "notional", notional)
    return post("/fapi/v1/order", params)


def main():
    placed = 0
    for intent in generate_signals_from_config() or []:
        # Only act on the first eligible intent to keep this probe safe/controlled.
        try:
            place_from_intent(intent)
            placed += 1
            break
        except Exception as e:
            print("ERR placing:", e)
    print("placed:", placed)


if __name__ == "__main__":
    main()
