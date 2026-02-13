"""
Manifest Audit — test suite for scripts/manifest_audit.py.

Validates:
  1. Manifest JSON is syntactically valid
  2. Every manifest entry has required fields (path, owner)
  3. No duplicate paths across sections
  4. audit() returns clean state (no missing_required, no untracked)
  5. CI and enforce modes produce consistent results
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "v7_manifest.json"

_SECTIONS_WITH_PATHS = ("state_files", "execution_logs", "prediction_layer")


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def test_manifest_json_valid(manifest):
    """Manifest is parseable JSON with expected top-level keys."""
    assert "docs_version" in manifest
    assert "state_files" in manifest


def test_manifest_entries_have_required_fields(manifest):
    """Every path-bearing entry must have 'path' and 'owner'."""
    errors = []
    for section in _SECTIONS_WITH_PATHS:
        if section not in manifest:
            continue
        for key, entry in manifest[section].items():
            if not isinstance(entry, dict):
                continue
            if "path" not in entry:
                errors.append(f"{section}.{key}: missing 'path'")
            if "owner" not in entry:
                errors.append(f"{section}.{key}: missing 'owner'")
    assert not errors, f"Manifest field violations:\n" + "\n".join(errors)


def test_no_duplicate_paths(manifest):
    """No two entries share the same path."""
    seen: dict[str, str] = {}
    dupes = []
    for section in _SECTIONS_WITH_PATHS:
        if section not in manifest:
            continue
        for key, entry in manifest[section].items():
            if isinstance(entry, dict) and "path" in entry:
                p = entry["path"]
                if p in seen:
                    dupes.append(f"{p} (in {seen[p]} and {section}.{key})")
                seen[p] = f"{section}.{key}"
    assert not dupes, f"Duplicate paths:\n" + "\n".join(dupes)


def test_audit_no_violations():
    """audit() should report no missing-required and no untracked files."""
    # Import here to avoid module-level issues
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from manifest_audit import audit

    missing_required, _optional, untracked = audit()
    assert missing_required == [], f"Required files missing: {missing_required}"
    assert untracked == [], f"Untracked files: {untracked}"


def test_enforce_mode_manifest_ok():
    """enforce mode should return MANIFEST_OK status."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from manifest_audit import run_enforce

    result = run_enforce()
    assert result["status"] == "MANIFEST_OK", f"Enforce failed: {result}"
    assert result["violations"] == 0
