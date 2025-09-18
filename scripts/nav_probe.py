#!/usr/bin/env python
import os, hmac, time, json, hashlib
from decimal import Decimal
from pathlib import Path
from typing import Dict

BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
SPOT_BASE = "https://api.binance.com"
FUT_BASE  = "https://fapi.binance.com"

INCLUDE_SPOT_IN_TRADING = os.environ.get("INCLUDE_SPOT_IN_TRADING_NAV", "0") == "1"

TOKENS = ["USDT","USDC","FDUSD","BTC","BNB","XAUT"]

def _hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _qs(params: Dict[str, str]) -> str:
    return "&".join(f"{k}={params[k]}" for k in params)

def _spot_signed_get(path: str, params: Dict[str,str]) -> dict:
    import requests
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("Spot signed GET requires BINANCE_API_KEY/SECRET in env")
    params = dict(params)
    params["timestamp"] = str(int(time.time() * 1000))
    params.setdefault("recvWindow", "60000")
    query = _qs(params)
    sig = _hmac_sha256(BINANCE_API_SECRET, query)
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    url = f"{SPOT_BASE}{path}?{query}&signature={sig}"
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def get_spot_balances() -> Dict[str, float]:
    a = _spot_signed_get("/api/v3/account", {})
    bals = {x["asset"]: Decimal(x["free"]) + Decimal(x["locked"]) for x in a["balances"]}
    out = {k: float(bals.get(k, Decimal("0"))) for k in TOKENS}
    # include any other non-zero assets we hold
    for k,v in bals.items():
        if float(v) and k not in out:
            out[k] = float(v)
    return out

def _public_price(symbol: str) -> float:
    # fallback to public futures price if get_price import fails
    import requests
    url = f"{FUT_BASE}/fapi/v1/ticker/price"
    r = requests.get(url, params={"symbol": f"{symbol}USDT"}, timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])

def px_usdt(symbol: str) -> float:
    if symbol == "USDT":
        return 1.0
    try:
        from execution.exchange_utils import get_price
        return float(get_price(f"{symbol}USDT"))
    except Exception:
        return _public_price(symbol)

def futures_nav_usd() -> float:
    try:
        from execution.exchange_utils import get_balances
        b = get_balances()  # futures wallet NAV
        return float(b.get("futures_nav_usd") or b.get("nav_usd") or 0.0)
    except Exception:
        return 0.0

def value_map(bal_map: Dict[str,float]):
    prices = {k: px_usdt(k) for k in bal_map.keys()}
    usd = {k: float(bal_map[k]) * float(prices.get(k, 0.0)) for k in bal_map.keys()}
    return prices, usd

def load_reserves():
    p = Path("config/off_exchange_reserves.json")
    if not p.exists():
        return {"reserves": [], "fee_wallets": []}
    return json.loads(p.read_text())

def reserves_usd(res_json):
    out, fees = [], []
    total = fee_total = 0.0
    for r in res_json.get("reserves", []):
        asset = r["asset"].upper(); amt = float(r["amount"]); px = px_usdt(asset); usd = amt*px
        total += usd; out.append({**r, "px_usdt": px, "usd": usd})
    for r in res_json.get("fee_wallets", []):
        asset = r["asset"].upper(); amt = float(r["amount"]); px = px_usdt(asset); usd = amt*px
        fee_total += usd; fees.append({**r, "px_usdt": px, "usd": usd})
    return out, total, fees, fee_total

def main():
    fut_nav = futures_nav_usd()
    try:
        spot = get_spot_balances()
    except Exception:
        spot = {}
    spot_px, spot_usd = value_map(spot)
    spot_total = sum(spot_usd.values())

    reserves_cfg = load_reserves()
    reserves_list, reserves_total, fee_list, fees_total = reserves_usd(reserves_cfg)

    trading_nav = fut_nav + (spot_total if INCLUDE_SPOT_IN_TRADING else 0.0)
    portfolio_nav = fut_nav + spot_total + reserves_total + fees_total

    out = {
        "ts": int(time.time()),
        "trading_nav_usd": trading_nav,
        "futures_nav_usd": fut_nav,
        "spot_total_usd": spot_total,
        "spot_balances": spot,
        "spot_prices": spot_px,
        "spot_usd": spot_usd,
        "reserves_total_usd": reserves_total,
        "reserves": reserves_list,
        "fee_wallets_total_usd": fees_total,
        "fee_wallets": fee_list,
        "portfolio_nav_usd": portfolio_nav,
        "include_spot_in_trading_nav": INCLUDE_SPOT_IN_TRADING
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
