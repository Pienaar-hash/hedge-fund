#!/usr/bin/env python
import json, time
from execution.exchange_utils import _client, _futures_base_url  # adjust if wrapped
from execution.exchange_utils import get_balances  # your high-level
def main():
    c = _client()
    # Raw futures: /fapi/v2/balance and /fapi/v2/account
    rb = c.futures_account_balance()
    ra = c.futures_account()  # positions and wallet
    print("[raw futures balance]")
    print(json.dumps(rb, indent=2)[:2000])
    print("[raw futures account keys]", list(ra.keys()))
    print("[computed get_balances()]")
    b = get_balances()
    print(json.dumps(b, indent=2))
if __name__ == "__main__":
    main()
