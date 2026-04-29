"""Integration tests for intent_factor_log manifest registration and schema contract."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from execution.intent_factor_log import build_factor_log_record, REQUIRED_COMPONENTS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "v7_manifest.json"


def _all_manifest_entries(manifest: dict) -> dict:
    """Merge all dict-valued top-level sections to find any registered entry."""
    merged = {}
    for key, val in manifest.items():
        if isinstance(val, dict):
            merged.update(val)
    return merged


@pytest.mark.integration
class TestIntentFactorLogManifest:
    """Verify intent_factor_log is properly registered in v7_manifest.json."""

    @pytest.fixture(autouse=True)
    def _load_manifest(self):
        with MANIFEST_PATH.open() as f:
            self.manifest = json.load(f)
        self.all_entries = _all_manifest_entries(self.manifest)

    def test_registered_in_manifest(self):
        assert "intent_factor_log" in self.all_entries, (
            "intent_factor_log not found in v7_manifest.json"
        )

    def test_manifest_path_correct(self):
        entry = self.all_entries["intent_factor_log"]
        assert entry["path"] == "logs/execution/intent_factor_log.jsonl"

    def test_manifest_fields_match_build_output(self):
        """Schema fields declared in manifest must match actual build output keys."""
        entry = self.all_entries["intent_factor_log"]
        manifest_fields = set(entry.get("fields", {}).keys())

        # Build a sample record to get actual keys
        intent = {
            "intent_id": "ord_test",
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "hybrid_score": 0.5,
            "hybrid_components": {k: 0.5 for k in REQUIRED_COMPONENTS},
            "hybrid_weights_used": {k: 0.25 for k in REQUIRED_COMPONENTS},
            "hybrid_carry_details": {"inputs": {"funding_rate": 0.0, "basis_pct": 0.0}},
        }
        rec = build_factor_log_record(intent)
        assert rec is not None
        actual_fields = set(rec.keys())

        assert manifest_fields == actual_fields, (
            f"Manifest/code field mismatch.\n"
            f"  In manifest only: {manifest_fields - actual_fields}\n"
            f"  In code only:     {actual_fields - manifest_fields}"
        )
