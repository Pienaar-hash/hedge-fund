#!/usr/bin/env python3
"""
Backfill nav_log.json from historical orders_attempted data.

Reconstructs NAV history by extracting nav_snapshot.nav_usd from
orders_attempted JSONL files, building daily closing NAVs, interpolating
gaps, and prepending to the live nav_log.json.

Usage:
    python scripts/backfill_nav_log.py [--dry-run] [--start-date YYYY-MM-DD]

Default start date: 2025-12-16 (doctrine start)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ORDERS_FILES = [
    Path("logs/execution/orders_attempted.2.jsonl"),
    Path("logs/execution/orders_attempted.1.jsonl"),
    Path("logs/execution/orders_attempted.jsonl"),
]
NAV_LOG_PATH = Path("logs/nav_log.json")
BACKUP_PATH = Path("logs/nav_log.pre_backfill.json")

# Reasonable NAV bounds for filtering outliers
NAV_FLOOR = 8_000.0
NAV_CEIL = 12_000.0

# Entries per day in backfill (every 6 hours)
ENTRIES_PER_DAY = 4
INTERVAL_S = 86400 / ENTRIES_PER_DAY  # 21600 = 6h


def extract_nav_from_orders() -> list[tuple[float, float]]:
    """Extract (unix_ts, nav_usd) from all orders_attempted files."""
    navs: list[tuple[float, float]] = []
    for fpath in ORDERS_FILES:
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    snap = e.get("nav_snapshot", {})
                    if not isinstance(snap, dict):
                        continue
                    nav_val = snap.get("nav_usd")
                    ts = e.get("local_ts")
                    if nav_val and ts:
                        nav_f = float(nav_val)
                        if NAV_FLOOR <= nav_f <= NAV_CEIL:
                            navs.append((float(ts), nav_f))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    navs.sort()
    return navs


def build_daily_close(
    navs: list[tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Build daily closing NAV: date_str -> (closing_ts, closing_nav)."""
    daily: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ts, nav in navs:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_str = dt.strftime("%Y-%m-%d")
        daily[day_str].append((ts, nav))

    closing: dict[str, tuple[float, float]] = {}
    for day_str, pts in daily.items():
        pts.sort()
        # Use last reading of the day as closing NAV
        closing[day_str] = pts[-1]
    return closing


def interpolate_gaps(
    daily_close: dict[str, tuple[float, float]],
    start_date: str,
    end_ts: float,
) -> list[tuple[float, float]]:
    """Produce a smooth time series with interpolated gap-filling.

    Returns list of (unix_ts, nav) entries at ENTRIES_PER_DAY resolution.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Collect all date keys that exist in range
    sorted_days = sorted(daily_close.keys())
    if not sorted_days:
        return []

    # Build anchor points: known (ts, nav) values
    anchors: list[tuple[float, float]] = []
    current = start_dt
    while current <= end_dt:
        day_str = current.strftime("%Y-%m-%d")
        if day_str in daily_close:
            anchors.append(daily_close[day_str])
        current += timedelta(days=1)

    if len(anchors) < 2:
        return [(a[0], a[1]) for a in anchors]

    # Generate interpolated entries between anchors
    result: list[tuple[float, float]] = []

    for i in range(len(anchors) - 1):
        ts_start, nav_start = anchors[i]
        ts_end, nav_end = anchors[i + 1]

        # How many entries in this segment
        span_s = ts_end - ts_start
        n_entries = max(1, int(span_s / INTERVAL_S))

        for j in range(n_entries):
            frac = j / n_entries
            t = ts_start + frac * span_s
            nav = nav_start + frac * (nav_end - nav_start)
            result.append((t, nav))

    # Always include last anchor
    result.append(anchors[-1])

    return result


def load_current_nav_log() -> list[dict]:
    """Load current nav_log.json."""
    if not NAV_LOG_PATH.exists():
        return []
    try:
        data = json.loads(NAV_LOG_PATH.read_text())
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def run_backfill(start_date: str, dry_run: bool = False) -> None:
    """Execute the NAV backfill."""
    print(f"[backfill] Extracting NAV from orders_attempted files...")
    raw_navs = extract_nav_from_orders()
    print(f"[backfill] Extracted {len(raw_navs)} clean NAV points")

    if not raw_navs:
        print("[backfill] ERROR: No NAV data found. Aborting.")
        sys.exit(1)

    daily_close = build_daily_close(raw_navs)
    print(f"[backfill] Daily closing NAVs: {len(daily_close)} days")
    for day in sorted(daily_close.keys()):
        ts, nav = daily_close[day]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        print(f"  {day}: ${nav:,.2f} ({dt.strftime('%H:%M')} UTC)")

    # Load current nav_log
    current_log = load_current_nav_log()
    print(f"[backfill] Current nav_log: {len(current_log)} entries")

    # Determine end of backfill: start of current log (or now)
    if current_log:
        end_ts = float(current_log[0]["t"])
    else:
        end_ts = datetime.now(tz=timezone.utc).timestamp()

    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    print(f"[backfill] Backfill range: {start_date} → {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")

    # Build interpolated series
    backfill_entries = interpolate_gaps(daily_close, start_date, end_ts)
    print(f"[backfill] Generated {len(backfill_entries)} backfill entries")

    if not backfill_entries:
        print("[backfill] WARNING: No entries generated. Check date range.")
        return

    # Convert to nav_log format
    backfill_log = [
        {"nav": round(nav, 8), "t": round(ts, 6), "unrealized_pnl": 0}
        for ts, nav in backfill_entries
        if ts < end_ts  # Don't overlap with existing log
    ]
    print(f"[backfill] Entries before current log start: {len(backfill_log)}")

    # Merge: backfill + current
    merged = backfill_log + current_log
    print(f"[backfill] Final merged log: {len(merged)} entries")

    if dry_run:
        print("[backfill] DRY RUN — not writing to disk")
        # Show first/last few entries
        for i, e in enumerate(merged[:3]):
            dt = datetime.fromtimestamp(e["t"], tz=timezone.utc)
            print(f"  [{i}] {dt.strftime('%Y-%m-%d %H:%M')} ${e['nav']:,.2f}")
        print("  ...")
        for i, e in enumerate(merged[-3:], start=len(merged) - 3):
            dt = datetime.fromtimestamp(e["t"], tz=timezone.utc)
            print(f"  [{i}] {dt.strftime('%Y-%m-%d %H:%M')} ${e['nav']:,.2f}")
        return

    # Backup current nav_log
    if NAV_LOG_PATH.exists():
        shutil.copy2(NAV_LOG_PATH, BACKUP_PATH)
        print(f"[backfill] Backed up current nav_log → {BACKUP_PATH}")

    # Write merged log
    NAV_LOG_PATH.write_text(json.dumps(merged, indent=None))
    print(f"[backfill] ✓ Wrote {len(merged)} entries to {NAV_LOG_PATH}")

    # Summary
    first = merged[0]
    last = merged[-1]
    dt_first = datetime.fromtimestamp(first["t"], tz=timezone.utc)
    dt_last = datetime.fromtimestamp(last["t"], tz=timezone.utc)
    span_days = (dt_last - dt_first).total_seconds() / 86400
    delta = last["nav"] - first["nav"]
    pct = (delta / first["nav"] * 100) if first["nav"] > 0 else 0
    sign = "+" if delta >= 0 else ""
    print(f"\n[backfill] ═══ Summary ═══")
    print(f"  Span:  {dt_first.strftime('%b %d')} → {dt_last.strftime('%b %d')} ({span_days:.1f} days)")
    print(f"  Start: ${first['nav']:,.2f}")
    print(f"  End:   ${last['nav']:,.2f}")
    print(f"  Δ NAV: {sign}${delta:,.2f} ({sign}{pct:.2f}%)")
    print(f"  Points: {len(merged)}")


def main():
    parser = argparse.ArgumentParser(description="Backfill nav_log.json from historical data")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--start-date",
        default="2025-12-16",
        help="Start date for backfill (YYYY-MM-DD). Default: 2025-12-16 (doctrine start)",
    )
    args = parser.parse_args()
    run_backfill(start_date=args.start_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
