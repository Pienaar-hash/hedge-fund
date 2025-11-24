#!/usr/bin/env python3
"""CLI probe for router auto-tune v6 suggestions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.intel import expectancy_v6, router_autotune_v6
from execution.state_publish import write_router_policy_suggestions_state
from execution.utils import load_json


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router auto-tune v6 probe")
    parser.add_argument("--state-dir", default="logs/state", help="Directory containing router health + expectancy state snaps")
    parser.add_argument("--risk-config", default="config/risk_limits.json", help="Path to risk_limits.json for bounds")
    parser.add_argument("--lookback-days", type=float, default=7.0, help="Lookback metadata to annotate suggestions with")
    parser.add_argument("--limit", type=int, default=20, help="Number of symbols to print")
    return parser.parse_args()


def maker_first_str(flag: bool | None) -> str:
    if flag is None:
        return "unknown"
    return "maker" if flag else "taker"


def _policy_str(policy: dict) -> str:
    return f"{maker_first_str(policy.get('maker_first'))} / {float(policy.get('taker_bias') or 0.0):.2f} / {float(policy.get('offset_bps') or 0.0):.1f}"


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir)
    expectancy_snapshot = expectancy_v6.load_expectancy(state_dir / "expectancy_v6.json")
    symbol_scores_snapshot = _load_state(state_dir / "symbol_scores_v6.json")
    router_health_snapshot = _load_state(state_dir / "router_health.json")
    risk_cfg = load_json(args.risk_config)

    suggestions = router_autotune_v6.build_suggestions(
        expectancy_snapshot=expectancy_snapshot,
        symbol_scores_snapshot=symbol_scores_snapshot,
        router_health_snapshot=router_health_snapshot,
        risk_config=risk_cfg,
        state_dir=state_dir,
        risk_config_path=args.risk_config,
        lookback_days=args.lookback_days,
    )
    write_router_policy_suggestions_state(suggestions)

    rows = suggestions.get("symbols", [])
    if not rows:
        print("No router policy suggestions generated (missing router_health snapshot?).")
        return

    limit = args.limit if args.limit > 0 else len(rows)
    header = "SYMBOL | REGIME          | QUALITY | CURRENT (maker/bias/offset) | PROPOSED (maker/bias/offset)"
    print(header)
    print("-" * len(header))
    for entry in rows[:limit]:
        current = entry.get("current_policy") or {}
        proposed = entry.get("proposed_policy") or {}
        print(
            f"{entry.get('symbol', ''):<6} | {entry.get('regime', ''):<15} | {entry.get('quality', ''):<7} | "
            f"{_policy_str(current):<27} | {_policy_str(proposed)}"
        )


if __name__ == "__main__":
    main()
