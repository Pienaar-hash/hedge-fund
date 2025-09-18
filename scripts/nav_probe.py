#!/usr/bin/env python
"""Lightweight NAV probe that reuses exchange_utils for balances & positions."""
import argparse
import json
import os
import time
from typing import Any, Dict, Iterable

from execution.exchange_utils import get_balances, get_positions, get_price

try:  # Firestore optional
    from utils.firestore_client import get_db  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when Firestore deps missing
    get_db = None

FUTURES = os.environ.get("USE_FUTURES", "1") == "1"
OUT_PATH = os.environ.get("NAV_PROBE_OUT", "nav_probe.json")
QUOTE = os.environ.get("NAV_QUOTE", "USDT").upper()
RESERVES = [asset.strip().upper() for asset in os.environ.get("NAV_RESERVES", "BTC,ETH").split(",") if asset.strip()]
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_NAV_COL", "nav_snapshots")


def _call_get_balances(futures: bool):
    try:
        return get_balances(futures=futures)  # type: ignore[call-arg]
    except TypeError:  # exchange_utils signature without futures kw
        return get_balances()


def _call_get_positions(futures: bool):
    try:
        return get_positions(futures=futures)  # type: ignore[call-arg]
    except TypeError:
        return get_positions()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _balances_to_dict(raw: Any) -> Dict[str, float]:
    if isinstance(raw, dict):
        return {str(k).upper(): _to_float(v) for k, v in raw.items()}
    balances: Dict[str, float] = {}
    if isinstance(raw, Iterable):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            asset = entry.get("asset")
            if not asset:
                continue
            amount = (
                entry.get("balance")
                or entry.get("walletBalance")
                or entry.get("availableBalance")
                or entry.get("free")
            )
            balances[str(asset).upper()] = _to_float(amount)
    return balances


def _equity(balances: Dict[str, float]) -> float:
    total = _to_float(balances.get(QUOTE))
    for asset in RESERVES:
        if not asset or asset == QUOTE:
            continue
        amount = _to_float(balances.get(asset))
        if amount <= 0:
            continue
        symbol = f"{asset}{QUOTE}"
        try:
            price = _to_float(get_price(symbol))
        except Exception:
            price = 0.0
        total += amount * price
    return total


def _publish_to_firestore(snapshot: Dict[str, Any]) -> None:
    if os.environ.get("FIRESTORE_ENABLED", "0") != "1" or get_db is None:
        return
    try:
        db = get_db()
        db.collection(FIRESTORE_COLLECTION).document("latest").set(snapshot)
    except Exception as exc:  # pragma: no cover - safe logging for ops visibility
        print(json.dumps({"publish": "error", "err": str(exc)}))


def main(publish: bool = False) -> None:
    balances_raw = _call_get_balances(futures=FUTURES) or {}
    balances = _balances_to_dict(balances_raw)
    positions = _call_get_positions(futures=FUTURES) or []
    equity = _equity(balances)

    snapshot: Dict[str, Any] = {
        "ts": int(time.time()),
        "futures": FUTURES,
        "quote": QUOTE,
        "balances": balances,
        "positions": positions,
        "equity": equity,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False)

    print(json.dumps({"ok": True, "equity": equity, "out": OUT_PATH}))

    if publish:
        _publish_to_firestore(snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe current NAV state")
    parser.add_argument("--publish", action="store_true", help="Write snapshot to Firestore")
    args = parser.parse_args()
    main(publish=args.publish)
