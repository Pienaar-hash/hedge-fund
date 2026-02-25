#!/usr/bin/env python3
"""IDCP Synchronization Audit v2 — streaming, memory-efficient.

For each dislocation sample, validates that the two legs (UP/DOWN) had
synchronized timestamps when the snapshot was taken.  Reports skew
between legs and whether dislocation survives.
"""
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT = Path(__file__).resolve().parent.parent
CLOB_LOG = PROJECT / "logs" / "prediction" / "clob_market.jsonl"
ROUNDS_LOG = PROJECT / "logs" / "prediction" / "binary_rounds.jsonl"


def polymarket_taker_fee(price: float) -> float:
    p = max(0.0, min(1.0, price))
    return 0.02 * min(p, 1.0 - p)


def load_rounds() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(ROUNDS_LOG) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass
    return out


def identify_asset_pair(events: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    aids = sorted(set(e.get("asset_id", "") for e in events if e.get("asset_id")))
    return (aids[0], aids[1]) if len(aids) >= 2 else (None, None)


def audit_sample(events: List[Dict], asset_a: str, asset_b: str,
                 sample_ts_ms: int) -> Dict[str, Any]:
    """Find last UP/DOWN leg events before sample_ts, compute skew."""
    last_a: Optional[Dict] = None
    last_b: Optional[Dict] = None

    for ev in events:
        ev_ts = ev.get("ts_arrival_ms", 0)
        if ev_ts > sample_ts_ms:
            continue
        aid = ev.get("asset_id", "")
        if aid == asset_a:
            if last_a is None or ev_ts > last_a.get("ts_arrival_ms", 0):
                last_a = ev
        elif aid == asset_b:
            if last_b is None or ev_ts > last_b.get("ts_arrival_ms", 0):
                last_b = ev

    if last_a is None or last_b is None:
        return {"error": "missing_leg", "has_a": last_a is not None, "has_b": last_b is not None}

    ts_a = last_a["ts_arrival_ms"]
    ts_b = last_b["ts_arrival_ms"]
    ask_a = last_a["best_ask"]
    ask_b = last_b["best_ask"]
    bundle = round(ask_a + ask_b, 6)
    fee_a = polymarket_taker_fee(ask_a)
    fee_b = polymarket_taker_fee(ask_b)
    fee_adj = round(bundle + fee_a + fee_b, 6)

    return {
        "ts_a_ms": ts_a, "ts_b_ms": ts_b,
        "skew_ms": abs(ts_a - ts_b),
        "ask_a": ask_a, "ask_b": ask_b,
        "bundle": bundle,
        "fee_a": round(fee_a, 6), "fee_b": round(fee_b, 6),
        "fee_adj": fee_adj,
        "fee_adj_sub_1": fee_adj < 1.0,
        "evt_type_a": last_a.get("event_type"),
        "evt_type_b": last_b.get("event_type"),
    }


def main():
    print("=" * 70)
    print("IDCP SYNCHRONIZATION AUDIT v2")
    print("=" * 70)

    rounds = load_rounds()
    idcp = [r for r in rounds
            if r.get("intraround_stats", {}).get("dislocation_fee_adjusted_count", 0) > 0]
    idcp.sort(key=lambda r: r["intraround_stats"].get("min_fee_adjusted_bundle", 2.0))

    print(f"Total rounds: {len(rounds)}")
    print(f"Rounds with fee-adjusted dislocations: {len(idcp)}")

    # Collect samples
    samples: List[Tuple[int, Dict, Dict]] = []
    for ri, rnd in enumerate(idcp[:10]):
        for s in rnd.get("intraround_samples", []):
            fab = s.get("fee_adjusted_bundle")
            if fab is not None and fab < 1.0:
                samples.append((ri, rnd, s))

    print(f"Dislocation samples to audit: {len(samples)}")
    print()

    # Load events — use tail for memory efficiency
    print("Loading recent CLOB events (tail -n300000)...")
    proc = subprocess.run(
        ["tail", "-n", "300000", str(CLOB_LOG)],
        capture_output=True, text=True, timeout=60,
    )

    all_events: List[Dict] = []
    for line in proc.stdout.split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("best_bid") is not None and rec.get("best_ask") is not None:
                all_events.append(rec)
        except (json.JSONDecodeError, ValueError):
            pass
    del proc  # free stdout memory

    all_events.sort(key=lambda e: e.get("ts_arrival_ms", 0))
    print(f"Events with bid/ask: {len(all_events)}")
    if all_events:
        t0 = datetime.fromtimestamp(all_events[0]["ts_arrival_ms"] / 1000, tz=timezone.utc)
        t1 = datetime.fromtimestamp(all_events[-1]["ts_arrival_ms"] / 1000, tz=timezone.utc)
        print(f"Time range: {t0.isoformat()[:19]} → {t1.isoformat()[:19]}")
    print()

    asset_a, asset_b = identify_asset_pair(all_events)
    if not asset_a or not asset_b:
        print("ERROR: Cannot identify asset pair")
        return
    print(f"Asset A: ...{asset_a[-12:]}")
    print(f"Asset B: ...{asset_b[-12:]}")
    print()

    # Audit
    total = 0; surviving = 0; artifact = 0; errors = 0
    skews: List[int] = []
    current_ri = -1

    for ri, rnd, s in samples:
        if ri != current_ri:
            current_ri = ri
            st = rnd["intraround_stats"]
            print(f"[Round {ri+1}] {rnd['logged_at'][:19]}  "
                  f"min_fee_adj={st.get('min_fee_adjusted_bundle')}  "
                  f"max_window={st.get('max_dislocation_window_s')}s")

        fab = s["fee_adjusted_bundle"]
        sample_ts_ms = int(datetime.fromisoformat(s["ts"]).timestamp() * 1000)

        result = audit_sample(all_events, asset_a, asset_b, sample_ts_ms)
        total += 1

        if "error" in result:
            errors += 1
            print(f"  {s['ts'][:19]}: ERROR {result['error']}")
            continue

        skews.append(result["skew_ms"])
        if result["fee_adj_sub_1"]:
            surviving += 1; tag = "SURVIVES"
        else:
            artifact += 1; tag = "ARTIFACT"

        print(f"  {s['ts'][:19]} elapsed={s['elapsed_s']}s")
        print(f"    reported:  ask_up={s['ask_up']}  ask_down={s['ask_down']}  fee_adj={fab}")
        print(f"    audited:   ask_a={result['ask_a']}  ask_b={result['ask_b']}  fee_adj={result['fee_adj']}")
        print(f"    skew={result['skew_ms']}ms  types=({result['evt_type_a']}, {result['evt_type_b']})  → {tag}")
    print()

    # Summary
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Samples audited:     {total}")
    print(f"Errors (no data):    {errors}")
    print(f"Surviving (real):    {surviving}")
    print(f"Artifact (skew):     {artifact}")
    if skews:
        print()
        print("TIMESTAMP SKEW DISTRIBUTION:")
        print(f"  min:    {min(skews):>8,}ms")
        print(f"  median: {int(statistics.median(skews)):>8,}ms")
        print(f"  mean:   {int(statistics.mean(skews)):>8,}ms")
        print(f"  max:    {max(skews):>8,}ms")
        print(f"  >  1s:  {sum(1 for s in skews if s > 1_000)}/{len(skews)}")
        print(f"  >  5s:  {sum(1 for s in skews if s > 5_000)}/{len(skews)}")
        print(f"  > 30s:  {sum(1 for s in skews if s > 30_000)}/{len(skews)}")
        print(f"  > 60s:  {sum(1 for s in skews if s > 60_000)}/{len(skews)}")

    print()
    print("VERDICT:")
    auditable = surviving + artifact
    if auditable == 0:
        print("  No samples auditable.")
    elif artifact == 0:
        print(f"  ALL {surviving} dislocations survive sync check.")
        print("  Timestamps aligned — proceed to depth validation.")
    elif surviving == 0:
        print(f"  ALL {artifact} dislocations are timestamp artifacts.")
        print("  Edge is NOT real.")
    else:
        pct = 100 * surviving / auditable
        print(f"  {surviving} survive ({pct:.0f}%), {artifact} artifacts ({100-pct:.0f}%)")
        if any(s > 30_000 for s in skews):
            print("  WARNING: Large skews (>30s) — structural desync likely.")
        if pct < 50:
            print("  Majority artifacts — treat metrics with high skepticism.")


if __name__ == "__main__":
    main()
