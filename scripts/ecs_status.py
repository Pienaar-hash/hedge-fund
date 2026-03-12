#!/usr/bin/env python3
"""ECS soak-status diagnostic — ``make ecs-status``.

Reads both shadow and soak telemetry JSONL files and prints a compact
summary suitable for monitoring the Phase 4 ECS migration soak.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

SHADOW_PATH = Path("logs/execution/ecs_shadow_events.jsonl")
SOAK_PATH = Path("logs/execution/ecs_soak_events.jsonl")
EXECUTOR_LOG = Path("/var/log/hedge-executor.out.log")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _flag_status() -> str:
    val = (os.getenv("USE_ECS_SELECTOR") or "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return "ON"
    # Check supervisor config as fallback
    sup_conf = Path("/etc/supervisor/conf.d/hedge.conf")
    if sup_conf.exists():
        text = sup_conf.read_text()
        if "USE_ECS_SELECTOR" in text:
            for part in text.split(","):
                if "USE_ECS_SELECTOR" in part:
                    if '"1"' in part or "'1'" in part:
                        return "ON (supervisor)"
                    return "OFF (supervisor)"
    return "OFF"


def _ecs_error_count() -> int:
    """Count [ecs_selector] fail-open lines in executor log."""
    if not EXECUTOR_LOG.exists():
        return -1
    count = 0
    try:
        with open(EXECUTOR_LOG) as f:
            for line in f:
                if "[ecs_selector] fail-open" in line:
                    count += 1
    except Exception:
        return -1
    return count


def _print_section(title: str, events: list[dict], schema_label: str) -> None:
    if not events:
        print(f"\n  {title}: no events")
        return

    cycles = set(e.get("cycle", 0) for e in events)
    agree_key = "agreement" if "agreement" in events[0] else "agreement"
    agree = sum(1 for e in events if e.get(agree_key))
    disagree = len(events) - agree
    rate = 100.0 * agree / len(events) if events else 0.0

    print(f"\n  {title}  ({schema_label})")
    print(f"    events:      {len(events)}")
    print(f"    cycles:      {len(cycles)}")
    print(f"    agreement:   {agree}/{len(events)}  ({rate:.1f}%)")
    print(f"    divergences: {disagree}")

    # Winner distribution by symbol
    winners: Counter[tuple[str, str]] = Counter()
    for e in events:
        sym = e.get("symbol", "?")
        # Handle both shadow (executor_winner) and soak (ecs_winner)
        winner = e.get("executor_winner") or e.get("ecs_winner") or "?"
        winners[(sym, winner)] += 1

    print("    winners:")
    syms = sorted(set(s for s, _ in winners))
    for sym in syms:
        parts = []
        for (s, w), c in sorted(winners.items()):
            if s == sym:
                parts.append(f"{w}={c}")
        total = sum(c for (s, _), c in winners.items() if s == sym)
        print(f"      {sym:10s}  {', '.join(parts)}  (n={total})")

    # Conflicts (2+ candidates)
    conflicts = sum(1 for e in events if (e.get("candidates_count") or 0) >= 2)
    conflict_pct = 100.0 * conflicts / len(events) if events else 0.0
    print(f"    conflicts:   {conflicts}/{len(events)}  ({conflict_pct:.1f}%)")


def main() -> None:
    print("=" * 60)
    print("  ECS Migration Status")
    print("=" * 60)

    print(f"\n  USE_ECS_SELECTOR:  {_flag_status()}")

    shadow = _load_jsonl(SHADOW_PATH)
    soak = _load_jsonl(SOAK_PATH)

    _print_section("Shadow Telemetry (pre-cutover)", shadow, "ecs_shadow_v1")
    _print_section("Soak Telemetry (post-cutover)", soak, "ecs_soak_v1")

    errors = _ecs_error_count()
    print(f"\n  ECS fail-open errors: {errors if errors >= 0 else 'n/a (no log)'}")

    # SOL legacy preservation check
    sol_shadow = [e for e in shadow if e.get("symbol") == "SOLUSDT"]
    sol_soak = [e for e in soak if e.get("symbol") == "SOLUSDT"]

    if sol_shadow:
        sol_legacy = sum(1 for e in sol_shadow if e.get("executor_winner") == "legacy")
        sol_pct = 100.0 * sol_legacy / len(sol_shadow)
        print(f"\n  SOL legacy rate (shadow):  {sol_legacy}/{len(sol_shadow)}  ({sol_pct:.1f}%)")

    if sol_soak:
        sol_legacy_soak = sum(1 for e in sol_soak if e.get("ecs_winner") == "legacy")
        sol_soak_pct = 100.0 * sol_legacy_soak / len(sol_soak)
        print(f"  SOL legacy rate (soak):    {sol_legacy_soak}/{len(sol_soak)}  ({sol_soak_pct:.1f}%)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
