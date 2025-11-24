#!/usr/bin/env python3
"""CLI to inspect Symbol Scores v6."""

from __future__ import annotations

import argparse
from pathlib import Path

from execution.intel import symbol_score_v6 as scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbol score probe")
    parser.add_argument("--state-dir", default="logs/state", help="Directory containing expectancy/router_health state files")
    parser.add_argument("--limit", type=int, default=20, help="Number of symbols to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir)
    expectancy = scores.load_expectancy_snapshot(state_dir / "expectancy_v6.json")
    router = scores.load_router_health_snapshot(state_dir / "router_health.json")
    snapshot = scores.score_universe(expectancy, router)
    rows = snapshot.get("symbols", [])
    for row in rows[: args.limit]:
        components = row.get("components", {})
        print(
            f"{row['symbol']:<10} score={row['score']:.3f} "
            f"exp={components.get('expectancy'):.2f} router={components.get('router'):.2f} "
            f"slip_pen={components.get('slippage_penalty'):.2f} fee_pen={components.get('fee_drag_penalty'):.2f}"
        )


if __name__ == "__main__":
    main()
