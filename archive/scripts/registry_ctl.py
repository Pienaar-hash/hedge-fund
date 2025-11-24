#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

REGISTRY_PATH = Path("config/strategy_registry.json")


def load_registry() -> Dict[str, Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return {
                    str(k): (dict(v) if isinstance(v, dict) else {})
                    for k, v in payload.items()
                }
        except Exception as exc:
            raise SystemExit(f"Failed to read registry: {exc}")
    return {}


def save_registry(data: Dict[str, Dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toggle strategy flags in config/strategy_registry.json",
    )
    parser.add_argument("strategy", help="Strategy key in the registry (e.g., momentum)")
    group_enable = parser.add_mutually_exclusive_group()
    group_enable.add_argument("--enable", action="store_true", help="Enable the strategy")
    group_enable.add_argument("--disable", action="store_true", help="Disable the strategy")
    group_sandbox = parser.add_mutually_exclusive_group()
    group_sandbox.add_argument("--sandbox", action="store_true", help="Mark strategy as sandbox (skipped)")
    group_sandbox.add_argument("--unsandbox", action="store_true", help="Clear sandbox flag")
    parser.add_argument("--confidence", type=float, help="Override confidence (0-1.5)")
    parser.add_argument("--max-concurrent", type=int, help="Override max concurrent intents")
    parser.add_argument("--capacity-usd", type=float, help="Override capacity in USD")
    return parser.parse_args()


def clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.5, float(value)))


def main() -> None:
    args = parse_args()
    registry = load_registry()
    entry = registry.setdefault(args.strategy, {})

    if args.enable:
        entry["enabled"] = True
    if args.disable:
        entry["enabled"] = False
    if args.sandbox:
        entry["sandbox"] = True
    if args.unsandbox:
        entry["sandbox"] = False
    if args.confidence is not None:
        entry["confidence"] = clamp_confidence(args.confidence)
    if args.max_concurrent is not None:
        entry["max_concurrent"] = max(0, int(args.max_concurrent))
    if args.capacity_usd is not None:
        entry["capacity_usd"] = float(args.capacity_usd)

    save_registry(registry)
    print(json.dumps({args.strategy: entry}, indent=2))


if __name__ == "__main__":
    main()
