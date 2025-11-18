#!/usr/bin/env python3
"""CLI probe for pipeline v6 comparison."""

from __future__ import annotations

import argparse

from execution.intel.pipeline_v6_compare import compare_pipeline_v6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v5 live vs v6 shadow pipeline")
    parser.add_argument("--shadow-limit", type=int, default=500, help="Number of shadow events to process")
    parser.add_argument("--orders", default="logs/execution/orders_executed.jsonl", help="Path to live orders log")
    parser.add_argument("--metrics", default="logs/execution/order_metrics.jsonl", help="Path to router metrics log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_pipeline_v6(
        shadow_limit=args.shadow_limit,
        orders_path=args.orders,
        metrics_path=args.metrics,
    )
    print("Pipeline v6 comparison summary:")
    print(summary)


if __name__ == "__main__":
    main()
