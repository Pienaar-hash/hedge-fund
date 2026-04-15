#!/usr/bin/env python3
"""
Manifest Audit — v7 state file registry integrity checker.

Modes:
  ci      — exit 0 if all required files exist and no untracked files found.
            Phantoms that are ``optional: true`` are allowed.
  enforce — same checks as CI, but returns structured JSON for executor preflight.

Usage:
  python scripts/manifest_audit.py ci          # CI gate
  python scripts/manifest_audit.py enforce     # executor preflight (returns JSON)
  python scripts/manifest_audit.py             # defaults to 'ci'
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "v7_manifest.json"

# Sections that contain path-bearing entries
_SECTIONS_WITH_PATHS = ("state_files", "execution_logs", "prediction_layer")

# Rotated log pattern: *.1.jsonl, *.2.jsonl etc — covered by base entry
_ROTATION_SUFFIXES = tuple(f".{i}.jsonl" for i in range(1, 20))


def _load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _extract_manifest_paths(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {path: {key, optional, lifecycle, section}} for every manifest entry."""
    result: Dict[str, Dict[str, Any]] = {}
    for section in _SECTIONS_WITH_PATHS:
        if section not in manifest:
            continue
        for key, entry in manifest[section].items():
            if isinstance(entry, dict) and "path" in entry:
                result[entry["path"]] = {
                    "key": key,
                    "optional": entry.get("optional", False),
                    "lifecycle": entry.get("lifecycle", "static"),
                    "section": section,
                }
    return result


def _scan_disk() -> Set[str]:
    """Return all relevant files on disk (state JSON, execution/prediction JSONL)."""
    files: Set[str] = set()

    # State JSON files
    state_dir = REPO_ROOT / "logs" / "state"
    if state_dir.is_dir():
        for f in state_dir.iterdir():
            if f.suffix == ".json":
                files.add(f"logs/state/{f.name}")

    # Execution and prediction JSONL files (skip rotated)
    for subdir in ("logs/execution", "logs/prediction"):
        d = REPO_ROOT / subdir
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".jsonl" and not f.name.endswith(_ROTATION_SUFFIXES):
                    files.add(f"{subdir}/{f.name}")

    # Prediction JSON (non-log)
    pred_dir = REPO_ROOT / "prediction"
    if pred_dir.is_dir():
        for f in pred_dir.iterdir():
            if f.suffix == ".json":
                files.add(f"prediction/{f.name}")

    return files


def audit() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Run the manifest audit with 3-class lifecycle model (AUDIT-1.2).

    Lifecycle classes:
      - ``static``    — REQUIRED_STATIC: must exist in repo / clean clone.
      - ``bootstrap`` — REQUIRED_BOOTSTRAP: must exist after first executor run.
                         Absent in clean clone is OK.
      - ``optional``  — feature-dependent, never required.

    Returns:
        (missing_static, missing_bootstrap, phantoms_optional, untracked)
    """
    manifest = _load_manifest()
    manifest_paths = _extract_manifest_paths(manifest)
    disk_files = _scan_disk()

    manifest_set = set(manifest_paths.keys())

    # Phantoms: in manifest but not on disk
    phantoms = manifest_set - disk_files
    missing_static: List[str] = []
    missing_bootstrap: List[str] = []
    phantoms_optional: List[str] = []
    for p in sorted(phantoms):
        info = manifest_paths[p]
        if info["optional"]:
            phantoms_optional.append(p)
        elif info["lifecycle"] == "bootstrap":
            missing_bootstrap.append(p)
        else:
            missing_static.append(p)

    # Untracked: on disk but not in manifest
    untracked = sorted(disk_files - manifest_set)

    return missing_static, missing_bootstrap, phantoms_optional, untracked


def run_ci() -> int:
    """CI mode: exit 0 on clean, exit 1 on any violation.

    AUDIT-1.2: CI checks REQUIRED_STATIC and untracked only.
    REQUIRED_BOOTSTRAP files are expected to be absent in clean clone.
    """
    missing_static, missing_bootstrap, optional_phantoms, untracked = audit()

    ok = True

    if missing_static:
        print(f"FAIL: {len(missing_static)} REQUIRED_STATIC entries missing on disk:")
        for p in missing_static:
            print(f"  - {p}")
        ok = False

    if untracked:
        print(f"FAIL: {len(untracked)} untracked files (not in manifest):")
        for p in untracked:
            print(f"  - {p}")
        ok = False

    if ok:
        bootstrap_note = f", {len(missing_bootstrap)} bootstrap pending" if missing_bootstrap else ""
        print(f"MANIFEST_OK — 0 violations ({len(optional_phantoms)} optional phantoms{bootstrap_note})")
        return 0
    else:
        return 1


def run_enforce() -> Dict[str, Any]:
    """Enforce mode: return structured JSON for executor preflight.

    AUDIT-1.2: Executor preflight checks BOTH static AND bootstrap files.
    """
    missing_static, missing_bootstrap, optional_phantoms, untracked = audit()

    has_violations = bool(missing_static or untracked)
    has_bootstrap_gaps = bool(missing_bootstrap)

    result = {
        "status": "MANIFEST_OK" if (not has_violations and not has_bootstrap_gaps) else (
            "MANIFEST_DRIFT" if has_violations else "BOOTSTRAP_INCOMPLETE"
        ),
        "missing_static": missing_static,
        "missing_bootstrap": missing_bootstrap,
        "missing_required": missing_static,  # backward compat
        "phantoms_optional": optional_phantoms,
        "untracked": untracked,
        "violations": len(missing_static) + len(untracked),
    }

    return result


def preflight_check() -> str:
    """
    Executor-callable preflight. Returns 'MANIFEST_OK' or raises RuntimeError.

    Usage from executor:
        from scripts.manifest_audit import preflight_check
        status = preflight_check()  # logs one line
    """
    result = run_enforce()
    if result["status"] == "MANIFEST_OK":
        return "MANIFEST_OK"
    else:
        raise RuntimeError(
            f"MANIFEST_DRIFT: {result['violations']} violations — "
            f"missing_required={result['missing_required']}, "
            f"untracked={result['untracked']}"
        )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ci"

    if mode == "ci":
        sys.exit(run_ci())
    elif mode == "enforce":
        result = run_enforce()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "MANIFEST_OK" else 1)
    else:
        print(f"Unknown mode: {mode}. Use 'ci' or 'enforce'.")
        sys.exit(2)


if __name__ == "__main__":
    main()
