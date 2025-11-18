#!/usr/bin/env python3
"""CLI probe for Feedback Allocator v6 suggestions."""

from __future__ import annotations

import argparse
from pathlib import Path

from execution.intel import feedback_allocator_v6
from execution.state_publish import write_risk_allocation_suggestions_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback allocator v6 probe")
    parser.add_argument("--state-dir", default="logs/state", help="Directory containing intel state files")
    parser.add_argument("--risk-config", default="config/risk_limits.json", help="Path to risk_limits.json")
    parser.add_argument("--pairs-universe", default="config/pairs_universe.json", help="Path to pairs_universe.json")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to display")
    return parser.parse_args()


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def _fmt_weight(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def main() -> None:
    args = parse_args()
    suggestions = feedback_allocator_v6.build_suggestions(
        state_dir=Path(args.state_dir),
        risk_config_path=Path(args.risk_config),
        pairs_universe_path=Path(args.pairs_universe),
    )
    write_risk_allocation_suggestions_state(suggestions)

    rows = suggestions.get("symbols", [])
    risk_mode = (suggestions.get("global") or {}).get("risk_mode")
    if not rows:
        print(f"No feedback allocator suggestions generated (risk_mode={risk_mode}).")
        return

    limit = args.limit if args.limit > 0 else len(rows)
    header = (
        "SYMBOL | SCORE | EXPECTANCY | RISK_MODE | CURRENT max_nav_pct | SUGGESTED max_nav_pct | "
        "SUGGESTED weight"
    )
    print(header)
    print("-" * len(header))
    for entry in rows[:limit]:
        score = entry.get("score")
        exp = entry.get("expectancy", {}).get("expectancy")
        caps = entry.get("caps", {})
        suggested_caps = entry.get("suggested_caps", {})
        print(
            f"{entry.get('symbol',''):<6} | "
            f"{(f'{score:.2f}' if isinstance(score, (int, float)) else '-'):<5} | "
            f"{(f'{exp:.2f}' if isinstance(exp, (int, float)) else '-'):<10} | "
            f"{risk_mode or '-':<9} | "
            f"{_fmt_pct(caps.get('current_max_nav_pct')):<22} | "
            f"{_fmt_pct(suggested_caps.get('max_nav_pct')):<24} | "
            f"{_fmt_weight(entry.get('suggested_weight'))}"
        )


if __name__ == "__main__":
    main()
