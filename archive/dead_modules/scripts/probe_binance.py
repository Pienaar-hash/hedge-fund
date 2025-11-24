#!/usr/bin/env python3
from __future__ import annotations

import sys

import httpx


def main() -> None:
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ping", timeout=3)
        if resp.status_code == 200:
            print("spot ping OK")
        else:
            print(f"spot ping fail {resp.status_code}")
    except Exception as exc:  # pragma: no cover - probe script
        print(f"probe error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
