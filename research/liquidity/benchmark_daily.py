"""Exactly one preselected infrastructure benchmark, never an event study.

Run under a systemd user scope with MemoryMax=2G, MemorySwapMax=0,
CPUQuota=100%, numerical threads=1 and an external 900-second timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from research.liquidity.daily_ingestion import (
    Budget, Limits, file_hash, host_health, ingest_day, partition_archive, write_json,
)

DAY = date(2022, 1, 2)
SOURCES = [(date(2022, 1, 1), "klines_1m"), (DAY, "klines_1m"), (DAY, "aggTrades")]
FROZEN = "16c7ad13079c0f8c95743a362520b143decaa3007bdb0321fd79eac5b5e59e63"
BASE_SHA = "cf25dd1727804ee8e014eb5ed96bcf0884f510ab"


def cgroup_snapshot(group: Path) -> dict:
    return {name: (group / name).read_text().strip() for name in
            ("memory.max", "memory.swap.max", "memory.peak", "memory.events", "cpu.max")}


def run(output: Path) -> int:
    repo = Path(__file__).resolve().parents[2]
    data_root = Path("/srv/policyops/data/H-LIQ-001").resolve()
    if not output.resolve().is_relative_to(data_root) or output.is_symlink():
        raise ValueError("output must be below /srv/policyops/data/H-LIQ-001")
    output.mkdir()  # exclusive; no reusing previous benchmark outputs
    started = time.monotonic()
    result = {"benchmark_verdict": "FAIL", "hypothesis_status": "UNTESTED",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "symbol": "BTCUSDT", "day": DAY.isoformat(), "workers": 1,
              "sources": [], "outcomes_computed": False}
    group = None
    budget = Budget(output, Limits())
    try:
        group = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1].lstrip("/")
        before = cgroup_snapshot(group)
        result["cgroup_before"] = before
        if before["memory.max"] != str(2 * 1024**3) or before["memory.swap.max"] != "0":
            raise RuntimeError("required enforceable 2 GiB/no-swap cgroup absent")
        quota, period = before["cpu.max"].split()
        if quota == "max" or int(quota) > int(period):
            raise RuntimeError("CPU quota must be at most one CPU")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if os.environ.get(name) != "1":
                raise RuntimeError(f"{name} must be 1")
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        if branch != "research/liquidity-sweep-reclaim-001":
            raise RuntimeError("unexpected research branch")
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=repo, check=True)
        result["repository_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        result["branch"] = branch
        if file_hash(repo / "docs/research/H_LIQ_001_PRE_REGISTRATION.md") != FROZEN:
            raise RuntimeError("frozen pre-registration modified")
        paths = ["research/liquidity/daily_ingestion.py", "research/liquidity/benchmark_daily.py",
                 "tests/research/liquidity/test_daily_ingestion.py",
                 "docs/research/H_LIQ_001_DAILY_INGESTION_ADDENDUM.md"]
        result["research_file_sha256"] = {name: file_hash(repo / name) for name in paths}
        result["original_pre_registration_sha256"] = FROZEN
        result["limits"] = asdict(budget.limits)
        result["host_before"] = host_health(output)
        result["host_ram_outside_cgroup_cap_bytes"] = result["host_before"]["MemTotal_bytes"] - 2 * 1024**3
        if result["host_ram_outside_cgroup_cap_bytes"] < 1024**3:
            raise RuntimeError("less than 1 GiB physical RAM outside cap")
        if result["host_before"]["MemAvailable_bytes"] < 3 * 1024**3:
            raise RuntimeError("initial host available RAM below 3 GiB admission floor")
        budget.check()
        write_json(output / "selection-and-boundary.json", result, budget)
        for day, kind in SOURCES:
            print(f"Ingesting BTCUSDT {day} {kind}", flush=True)
            entry = ingest_day("BTCUSDT", day, kind, output, budget)
            result["sources"].append(entry)
            print(f"Validated {entry['rows']} rows; {entry['byte_size']} source bytes", flush=True)
        acquisition_seconds = time.monotonic() - started
        result["acquire_validate_partition_seconds"] = acquisition_seconds
        # Re-read the SAME local archives to prove deterministic partition bytes.
        # No further downloads, symbols or dates; total resource budget still applies.
        replay_started = time.monotonic()
        result["deterministic_replay"] = []
        for (day, kind), entry in zip(SOURCES, result["sources"]):
            source = output / f"BTCUSDT-{day.isoformat()}-{kind}"
            replay = partition_archive(source / entry["filename"], day, kind, source / "replay", budget)
            if replay["partitions"] != entry["partitions"] or replay["gaps"] != entry["gaps"]:
                raise RuntimeError("partition replay differs")
            result["deterministic_replay"].append({"day": day.isoformat(), "kind": kind,
                                                    "rows": replay["rows"], "identical": True})
        result["replay_seconds"] = time.monotonic() - replay_started
        # Canonical provenance excludes measurements, paths and wall-clock timing.
        manifest = {"schema_version": 1, "sources": [
            {key: value for key, value in entry.items() if key != "elapsed_seconds"}
            for entry in result["sources"]]}
        write_json(output / "source-manifest.json", manifest, budget)
        result["source_manifest_sha256"] = file_hash(output / "source-manifest.json")
        budget.check()
        after = cgroup_snapshot(group)
        old_events = dict(line.split() for line in before["memory.events"].splitlines())
        new_events = dict(line.split() for line in after["memory.events"].splitlines())
        if any(int(new_events[key]) != int(old_events[key]) for key in ("max", "oom", "oom_kill")):
            raise RuntimeError("cgroup memory-limit event")
        result["benchmark_verdict"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"], flush=True)
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["host_after"] = host_health(output)
        result["budget_observations"] = {
            "written_bytes": budget.written, "health_checks": budget.health_checks,
            "min_available_ram_seen_bytes": budget.min_available_ram_seen,
            "min_available_disk_seen_bytes": budget.min_free_disk_seen,
            "max_memory_full_avg10": budget.max_memory_full_avg10}
        if group is not None:
            result["cgroup_after"] = cgroup_snapshot(group)
        # Publish failure diagnostics even when normal resource admission refuses writes.
        (output / "benchmark-result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"Infrastructure benchmark {result['benchmark_verdict']}; H-LIQ-001 UNTESTED", flush=True)
    return 0 if result["benchmark_verdict"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
