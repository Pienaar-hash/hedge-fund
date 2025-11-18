#!/usr/bin/env python3
"""CLI to build v6 expectancy snapshots from logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from execution.intel import expectancy_v6
from execution.state_publish import write_expectancy_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v6 expectancy snapshot")
    parser.add_argument("--log-dir", default="logs", help="Root logs directory")
    parser.add_argument("--lookback-days", type=float, default=2.0, help="Lookback window in days")
    parser.add_argument(
        "--output",
        help="Optional explicit output path; defaults to logs/state/expectancy_v6.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = expectancy_v6.load_inputs(Path(args.log_dir), args.lookback_days)
    snapshot = expectancy_v6.build_expectancy(inputs)
    if args.output:
        expectancy_v6.save_expectancy(args.output, snapshot)
    else:
        write_expectancy_state(snapshot)


if __name__ == "__main__":
    main()
