#!/usr/bin/env python3
"""P6B.5 Replay Runner — execute the historical replay pipeline."""
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

sys.path.insert(0, ".")

from execution.p6_replay import (
    load_kline_cache,
    run_full_replay_pipeline,
)

def main():
    t0 = time.time()

    # Try to load persisted kline cache for reproducibility
    cache = load_kline_cache()
    if cache:
        print(f"Using persisted kline cache ({sum(len(v) for v in cache.values())} bars)")
    else:
        print("No kline cache found — will fetch from Binance FAPI")

    result = run_full_replay_pipeline(dry_run=False, kline_cache=cache)

    elapsed = time.time() - t0
    outcome = result["outcome"]
    n_records = len(result["records"])

    print(f"\n{'='*60}")
    print(f"P6B.5 REPLAY COMPLETE in {elapsed:.1f}s")
    print(f"  Records: {n_records}")
    print(f"  Outcome: {outcome}")
    print(f"{'='*60}")

    # Quick per-candidate summary
    for cid, stats in sorted(result["summary"].get("per_candidate", {}).items()):
        ff = result["fast_fail"].get(cid, {})
        print(f"\n  {cid}:")
        print(f"    signals={stats['n_signals']}, pass_rate={stats['pass_rate']:.4f}")
        print(f"    mean_expected_edge={stats['mean_expected_edge_pct']:.8f}")
        print(f"    mean_realized_edge={stats['mean_realized_edge_pct']:.8f}")
        print(f"    spearman_rho={stats.get('spearman_rho')}")
        print(f"    fast_fail: passed={ff.get('passed', False)}, fails={ff.get('fails', [])}")

    return result

if __name__ == "__main__":
    main()
