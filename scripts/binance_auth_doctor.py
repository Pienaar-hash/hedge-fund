import hashlib
import hmac
import os
import sys
import time

import requests

BASE = "https://fapi.binance.com"  # USD-M mainnet
KEY = os.environ.get("BINANCE_API_KEY", "").strip()
SEC = os.environ.get("BINANCE_API_SECRET", "").strip().encode()


def _signed_get(path: str, params=None, recv_window: int = 5000) -> requests.Response:
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = recv_window
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(SEC, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{path}?{qs}&signature={sig}"
    return requests.get(url, headers={"X-MBX-APIKEY": KEY}, timeout=10)


def main() -> None:
    print("[doctor] base:", BASE, "key_len:", len(KEY), "secret_len:", len(SEC))
    if len(KEY) == 0 or len(SEC) < 32:
        print(
            "AUTH_DOCTOR_FAIL: Missing/short Binance API credentials in environment (.env not loaded?).",
            file=sys.stderr,
        )
        print(
            "Action: set BINANCE_API_KEY / BINANCE_API_SECRET in .env and 'set -a; source ./.env; set +a'",
            file=sys.stderr,
        )
        sys.exit(2)
    r_pub = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=10)
    print("[doctor] exchangeInfo:", r_pub.status_code)

    # Use v2 account endpoint (v1 may 404 HTML under proxies)
    r1 = _signed_get("/fapi/v2/account")
    print("[doctor] /fapi/v2/account:", r1.status_code)

    r2 = _signed_get("/fapi/v2/positionRisk")
    print("[doctor] /fapi/v2/positionRisk:", r2.status_code)

    bad = (r1.status_code in (401, 403)) or (r2.status_code in (401, 403))
    if bad:
        print("\n[doctor] HINTS:")
        print("- Enable USD-M Futures permission on this API key")
        print("- If IP-restricted, add the server public IPv4 to the whitelist")
        print("- If permissions/whitelist were changed, REGENERATE the secret and update .env")
        print("- Ensure key belongs to the correct (sub)account USD-M wallet")
        print("- Clock skew unlikely if NTP is synchronized")


if __name__ == "__main__":
    main()
