#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from execution.preflight import (
    assert_version_alignment,
    run_step,
    state_health_report,
    version_report,
)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_commands(commands: List[Tuple[str, Sequence[str]]]) -> bool:
    ok = True
    for label, cmd in commands:
        print(f"[preflight] {label}: {' '.join(cmd)}")
        result = run_step(cmd, env=os.environ)
        if result.returncode != 0:
            ok = False
            print(f"[preflight][fail] {label} (exit {result.returncode})")
    return ok


def _dry_run_executor(max_loops: int) -> bool:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "BINANCE_TESTNET": "1",
            "MAX_LOOPS": str(max_loops),
            "LOOP_SLEEP": "1",
        }
    )
    cmd = [sys.executable, "execution/executor_live.py"]
    print(f"[preflight] dry-run executor for {max_loops} loops")
    result = run_step(cmd, env=env)
    if result.returncode != 0:
        print(f"[preflight][fail] dry-run executor exit {result.returncode}")
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v7.6 preflight harness")
    parser.add_argument(
        "--state-dir",
        default="logs/state",
        help="State directory to validate",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest/test-fast execution",
    )
    parser.add_argument(
        "--dry-run-loops",
        type=int,
        default=0,
        help="Optional executor dry-run loop count (0 to skip)",
    )
    parser.add_argument(
        "--allowable-lag-seconds",
        type=float,
        default=900.0,
        help="Stale threshold for state surface validation",
    )
    args = parser.parse_args(argv)

    # Version alignment
    _print_header("Version Check")
    vr = version_report(expected="v7.6")
    print(f"engine_version={vr['engine_version']} docs_version={vr['docs_version']} expected={vr['expected']}")
    try:
        assert_version_alignment(expected="v7.6")
    except Exception as exc:  # pragma: no cover - handled in tests
        print(f"[preflight][fail] VERSION mismatch: {exc}")
        return 1

    # Tests
    if not args.skip_tests:
        _print_header("Test Suite")
        commands: List[Tuple[str, Sequence[str]]] = [
            ("make test-fast", ["make", "test-fast"]),
            (
                "integration state contract subset",
                [
                    "pytest",
                    "-q",
                    "tests/integration/test_manifest_state_contract.py",
                    "tests/integration/test_state_surface_health.py",
                    "tests/integration/test_state_publish_diagnostics.py",
                    "tests/integration/test_state_publish_factor_diagnostics.py",
                    "tests/integration/test_risk_snapshot_contract.py",
                    "tests/integration/test_state_positions_ledger_contract.py",
                ],
            ),
        ]
        if not _run_commands(commands):
            return 1

    # Optional dry-run executor loop
    if args.dry_run_loops > 0:
        _print_header("Executor Dry-Run")
        if not _dry_run_executor(args.dry_run_loops):
            return 1

    # State health
    _print_header("State Surface Health")
    health = state_health_report(
        state_dir=Path(args.state_dir),
        allowable_lag_seconds=float(args.allowable_lag_seconds),
    )
    issues = [v for vals in health.values() for v in vals]
    print(f"missing={health.get('missing_files')} stale={health.get('stale_files')} schema={health.get('schema_violations')} cross={health.get('cross_surface_violations')}")
    if issues:
        print("[preflight][warn] state surfaces need attention before tagging.")

    print("\n[preflight] v7.6 checks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
