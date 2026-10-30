"""One exploratory PDH/PDL candle screen. No trade ingestion or execution imports.

Run: python -m research.liquidity.candle_screen --output /srv/policyops/data/...
Fixed 2022--2024 sample; 2025 cannot be requested. Definition in docs/research.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import resource
import shutil
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

MINUTE = 60_000
DAY = 1440 * MINUTE
START = 1640995200000  # 2022-01-01 UTC
END = 1735689600000    # 2025-01-01 UTC, exclusive and never read
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFINITION = "docs/research/PDH_PDL_CANDLE_SCREEN.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def complete(rows):
    return (len(rows) == 1440 and rows[0][0] % DAY == 0
            and all(b[0] - a[0] == MINUTE for a, b in zip(rows, rows[1:])))


def screen_day(symbol, rows, levels, following=()):
    """Rows: (UTC open ms, open, high, low, close). Levels known before rows[0]."""
    rejected = Counter()
    if not complete(rows):
        return [], Counter(incomplete_crossing_day=1)
    if levels is None:
        return [], Counter(unavailable_previous_day=1)
    high, low = levels
    upper = next((i for i, row in enumerate(rows) if row[2] > high), None)
    lower = next((i for i, row in enumerate(rows) if row[3] < low), None)
    if upper is not None and upper == lower:
        return [], Counter(ambiguous_double_penetration=2)
    tape = rows + list(following[:20])
    events = []
    for side, i, level, sign in (("PDH", upper, high, -1), ("PDL", lower, low, 1)):
        if i is None:
            rejected["no_penetration"] += 1
            continue
        if i + 20 >= len(tape) or tape[i + 20][0] >= END:
            rejected["unavailable_exit_context"] += 1
            continue
        if any(tape[j][0] != tape[i][0] + (j - i) * MINUTE for j in range(i, i + 21)):
            rejected["gap_in_event_window"] += 1
            continue
        decision_close = tape[i + 4][4]
        if decision_close == level:
            rejected["equality_deadband"] += 1
            continue
        state = "rejection" if sign * (decision_close - level) > 0 else "acceptance"
        entry, exit_row = tape[i + 5], tape[i + 20]
        gross = sign * 10_000 * math.log(exit_row[1] / entry[1])
        events.append({"symbol": symbol, "day": datetime.fromtimestamp(rows[0][0] / 1000, timezone.utc).date().isoformat(),
                       "year": datetime.fromtimestamp(rows[0][0] / 1000, timezone.utc).year,
                       "side": side, "level": level, "cross_bar_ms": tape[i][0],
                       "classification_ms": tape[i][0] + 5 * MINUTE,
                       "classification_close": decision_close, "state": state,
                       "entry_ms": entry[0], "entry_open": entry[1],
                       "exit_ms": exit_row[0], "exit_open": exit_row[1],
                       "gross_bps": gross, "net_bps": gross - 15})
    return events, rejected


def summarize(events):
    def cell(selected):
        result = {}
        for state in ("rejection", "acceptance"):
            gross = [e["gross_bps"] for e in selected if e["state"] == state]
            result[state] = {"n": len(gross),
                             "mean_gross_bps": float(np.mean(gross)) if gross else None,
                             "median_gross_bps": float(np.median(gross)) if gross else None,
                             "mean_net_bps": float(np.mean(gross)) - 15 if gross else None,
                             "median_net_bps": float(np.median(gross)) - 15 if gross else None}
        r, a = result["rejection"], result["acceptance"]
        for measure in ("mean", "median"):
            result[f"{measure}_difference_bps"] = (r[f"{measure}_gross_bps"] - a[f"{measure}_gross_bps"]
                                                    if r["n"] and a["n"] else None)
        return result

    return {"pooled": cell(events),
            "symbols": {s: cell([e for e in events if e["symbol"] == s]) for s in SYMBOLS},
            "years": {str(y): cell([e for e in events if e["year"] == y]) for y in (2022, 2023, 2024)},
            "symbol_years": {f"{s}-{y}": cell([e for e in events if e["symbol"] == s and e["year"] == y])
                             for s in SYMBOLS for y in (2022, 2023, 2024)}}


def decide(summary):
    def positive(cell):
        return (cell["rejection"]["median_net_bps"] is not None
                and cell["rejection"]["median_net_bps"] > 0
                and cell["median_difference_bps"] is not None and cell["median_difference_bps"] > 0)
    pooled = summary["pooled"]
    gates = {
        "at_least_50_each_class_each_symbol_year": all(
            c[state]["n"] >= 50 for c in summary["symbol_years"].values()
            for state in ("rejection", "acceptance")),
        "pooled_rejection_median_net_at_least_5bp": (
            pooled["rejection"]["median_net_bps"] is not None
            and pooled["rejection"]["median_net_bps"] >= 5),
        "pooled_median_difference_at_least_10bp": (
            pooled["median_difference_bps"] is not None and pooled["median_difference_bps"] >= 10),
        "positive_rejection_net_and_difference_all_symbols": all(positive(c) for c in summary["symbols"].values()),
        "positive_rejection_net_and_difference_two_of_three_years": sum(
            positive(c) for c in summary["years"].values()) >= 2}
    return {"decision": "JUSTIFY_MORE_PRECISE_VALIDATION" if all(gates.values()) else "REJECT_SCREEN_MECHANISM",
            "gates": gates}


def acquire(symbol, period, daily, directory, manifest):
    kind = "daily" if daily else "monthly"
    name = f"{symbol}-1m-{period}.zip"
    url = f"https://data.binance.vision/data/futures/um/{kind}/klines/{symbol}/1m/{name}"
    path = directory / name
    # Only fixed study periods and one previous-day dependency can be requested.
    assert (daily and period == "2021-12-31") or (not daily and 2022 <= int(period[:4]) <= 2024)
    entry = {"symbol": symbol, "period": period, "url": url, "filename": name}
    manifest.append(entry)
    try:
        with requests.get(url + ".CHECKSUM", timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            body = response.raw.read(4097)
        if len(body) > 4096:
            raise ValueError("oversized checksum response")
        tokens = body.decode("ascii").split()
        if len(tokens) != 2 or tokens[1].lstrip("*") != name or len(tokens[0]) != 64:
            raise ValueError("invalid source checksum identity")
        path.with_suffix(".zip.CHECKSUM").write_bytes(body)
        with requests.get(url, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            size = 0
            with path.open("xb") as handle:
                for block in response.iter_content(1024**2):
                    size += len(block)
                    if size > 16 * 1024**2 or shutil.disk_usage(directory).free < 3 * 1024**3:
                        raise ValueError("archive/disk limit")
                    handle.write(block)
        actual = sha256(path)
        if actual != tokens[0].lower():
            raise ValueError("source checksum mismatch")
        entry.update({"bytes": size, "sha256": actual, "checksum_verified": True})
        return path, entry
    except requests.HTTPError as exc:
        if exc.response.status_code != 404:
            raise
        entry["status"] = "MISSING_404"
        return None, entry


def candle_rows(symbol, directory, manifest, gaps):
    last_t = None
    periods = [("2021-12-31", True)] + [(f"{y}-{m:02d}", False) for y in (2022, 2023, 2024) for m in range(1, 13)]
    for period, daily in periods:
        path, entry = acquire(symbol, period, daily, directory, manifest)
        if path is None:
            continue
        first = int(datetime.fromisoformat(period + ("" if daily else "-01")).replace(tzinfo=timezone.utc).timestamp() * 1000)
        if daily:
            end = first + DAY
        else:
            y, m = map(int, period.split("-"))
            end = int(datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        count = 0
        with zipfile.ZipFile(path) as zf:
            members = zf.infolist()
            if len(members) != 1 or members[0].file_size > 64 * 1024**2:
                raise ValueError("unexpected kline ZIP shape/size")
            with zf.open(members[0]) as raw:
                line_no = 0
                while True:
                    line = raw.readline(4097)
                    if not line:
                        break
                    line_no += 1
                    if len(line) > 4096:
                        raise ValueError("oversized candle row")
                    fields = line.decode("ascii").strip().split(",")
                    if line_no == 1 and fields[0] == "open_time":
                        continue
                    if len(fields) != 12:
                        raise ValueError("invalid candle schema")
                    t = int(fields[0])
                    o, high, low, close = map(float, fields[1:5])
                    if (not first <= t < end or t % MINUTE or int(fields[6]) != t + MINUTE - 1
                            or not all(math.isfinite(v) and v > 0 for v in (o, high, low, close))
                            or not low <= min(o, close) <= max(o, close) <= high):
                        raise ValueError(f"invalid candle {symbol} {period} line {line_no}")
                    if last_t is not None:
                        if t <= last_t:
                            raise ValueError("duplicate/unordered candle timestamp")
                        if t - last_t != MINUTE:
                            gaps.append({"symbol": symbol, "after_ms": last_t, "before_ms": t,
                                         "missing_minutes": (t - last_t) // MINUTE - 1})
                    last_t = t
                    count += 1
                    yield (t, o, high, low, close)
        entry.update({"rows": count, "expected_rows": (end - first) // MINUTE,
                      "missing_minutes": (end - first) // MINUTE - count, "zip_crc": "VERIFIED"})
        print(f"{symbol} {period}: {count} candles", flush=True)


def run(output):
    output.mkdir(parents=True)  # new directory only, no accidental reruns/overwrites
    archives = output / "archives"
    archives.mkdir()
    root = Path(__file__).resolve().parents[2]
    metadata = {"screen": "PDH_PDL_CANDLE_SCREEN_001", "held_out_2025": "NOT_DOWNLOADED_OR_INSPECTED",
                "definition_sha256": sha256(root / DEFINITION), "code_sha256": sha256(Path(__file__)),
                "hypothesis_relationship": "NOT_H_LIQ_001_PRIMARY_TEST"}
    save(output / "definition-identity.json", metadata)
    manifest, gaps, events, coverage = [], [], [], {}
    started = time.monotonic()
    try:
        for symbol in SYMBOLS:
            pending = None
            levels = None
            reasons = Counter()
            seen_days = complete_days = 0
            for day_id, grouped in itertools.groupby(candle_rows(symbol, archives, manifest, gaps), lambda row: row[0] // DAY):
                rows = list(grouped)
                if pending is not None:
                    prior_id, prior_rows, prior_levels = pending
                    if START <= prior_id * DAY < END:
                        found, skipped = screen_day(symbol, prior_rows, prior_levels, rows[:20])
                        events.extend(found)
                        reasons.update(skipped)
                    levels = ((max(r[2] for r in prior_rows), min(r[3] for r in prior_rows))
                              if day_id == prior_id + 1 and complete(prior_rows) else None)
                if START <= day_id * DAY < END:
                    seen_days += 1
                    complete_days += int(complete(rows))
                pending = day_id, rows, levels
            if pending is not None and START <= pending[0] * DAY < END:
                found, skipped = screen_day(symbol, pending[1], pending[2])
                events.extend(found)
                reasons.update(skipped)
            coverage[symbol] = {"expected_screen_days": (END - START) // DAY, "days_observed": seen_days,
                                "complete_days": complete_days, "missing_whole_days": (END - START) // DAY - seen_days,
                                "exclusions_or_absent_candidates": dict(reasons)}
        events.sort(key=lambda e: (e["symbol"], e["cross_bar_ms"], e["side"]))
        with (output / "events.csv").open("w", newline="") as handle:
            if events:
                writer = csv.DictWriter(handle, fieldnames=list(events[0]))
                writer.writeheader()
                writer.writerows(events)
        summary = summarize(events)
        save(output / "results.json", {**metadata, **summary, **decide(summary), "coverage": coverage})
    finally:
        save(output / "sources.json", manifest)
        save(output / "gaps.json", gaps)
        save(output / "runtime.json", {"elapsed_seconds": time.monotonic() - started,
                                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                       "remaining_disk_bytes": shutil.disk_usage(output).free})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
