"""
Schema-validation tests for funding_snapshot.json and basis_snapshot.json.

These state surfaces are published by *publish_funding_snapshot()* and
*publish_basis_snapshot()* in ``execution/state_publish.py`` and registered
in ``v7_manifest.json`` as **optional** (they only appear once the carry
pipeline has executed at least one intel cycle).

Tests verify:
    * Manifest registration (path, owner, optional flag, fields key)
    * Round-trip publish → load schema correctness
    * Per-symbol field shapes match the contract
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Manifest registration tests
# ---------------------------------------------------------------------------

MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[2] / "v7_manifest.json"


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


class TestManifestRegistration:
    """Both snapshots must be registered in v7_manifest.json."""

    def test_funding_snapshot_registered(self) -> None:
        m = _load_manifest()
        entry = m["state_files"].get("funding_snapshot")
        assert entry is not None, "funding_snapshot missing from v7_manifest.json"
        assert entry["path"] == "logs/state/funding_snapshot.json"
        assert entry["owner"] == "executor"
        assert entry.get("optional") is True
        assert "fields" in entry

    def test_basis_snapshot_registered(self) -> None:
        m = _load_manifest()
        entry = m["state_files"].get("basis_snapshot")
        assert entry is not None, "basis_snapshot missing from v7_manifest.json"
        assert entry["path"] == "logs/state/basis_snapshot.json"
        assert entry["owner"] == "executor"
        assert entry.get("optional") is True
        assert "fields" in entry


# ---------------------------------------------------------------------------
# Schema shape tests (round-trip via publish functions)
# ---------------------------------------------------------------------------

FUNDING_REQUIRED_KEYS = {"rate", "funding_rate", "next_funding_time"}
BASIS_REQUIRED_KEYS = {"basis_pct", "mark_price", "index_price"}


def _publish_and_load(
    publisher,
    sample_data: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    """Publish to a temp dir and read back the JSON."""
    with tempfile.TemporaryDirectory() as td:
        state_dir = pathlib.Path(td)
        publisher(sample_data, state_dir=state_dir)
        out = state_dir / filename
        assert out.exists(), f"{filename} not written"
        return json.loads(out.read_text())


class TestFundingSnapshotSchema:
    """Verify funding_snapshot.json shape matches the manifest contract."""

    SAMPLE = {
        "symbols": {
            "BTCUSDT": {
                "rate": 0.0001,
                "funding_rate": 0.0001,
                "next_funding_time": 1700000000000,
            },
            "ETHUSDT": {
                "rate": -0.00005,
                "funding_rate": -0.00005,
                "next_funding_time": 1700000000000,
            },
        },
    }

    def _load(self) -> Dict[str, Any]:
        from execution.state_publish import publish_funding_snapshot

        return _publish_and_load(
            publish_funding_snapshot, self.SAMPLE, "funding_snapshot.json"
        )

    def test_top_level_keys(self) -> None:
        data = self._load()
        assert "symbols" in data
        assert "updated_ts" in data

    def test_symbols_is_dict(self) -> None:
        data = self._load()
        assert isinstance(data["symbols"], dict)
        assert len(data["symbols"]) == 2

    def test_per_symbol_fields(self) -> None:
        data = self._load()
        for sym, info in data["symbols"].items():
            missing = FUNDING_REQUIRED_KEYS - set(info.keys())
            assert not missing, f"{sym} missing fields: {missing}"

    def test_rate_numeric(self) -> None:
        data = self._load()
        for sym, info in data["symbols"].items():
            assert isinstance(info["rate"], (int, float)), f"{sym}.rate not numeric"
            assert isinstance(
                info["funding_rate"], (int, float)
            ), f"{sym}.funding_rate not numeric"

    def test_empty_symbols_ok(self) -> None:
        """Empty symbols dict should still produce valid JSON."""
        from execution.state_publish import publish_funding_snapshot

        data = _publish_and_load(
            publish_funding_snapshot, {"symbols": {}}, "funding_snapshot.json"
        )
        assert data["symbols"] == {}
        assert "updated_ts" in data


class TestBasisSnapshotSchema:
    """Verify basis_snapshot.json shape matches the manifest contract."""

    SAMPLE = {
        "symbols": {
            "BTCUSDT": {
                "basis_pct": 0.0012,
                "mark_price": 40100.0,
                "index_price": 40050.0,
            },
            "ETHUSDT": {
                "basis_pct": -0.0003,
                "mark_price": 2200.0,
                "index_price": 2200.66,
            },
        },
    }

    def _load(self) -> Dict[str, Any]:
        from execution.state_publish import publish_basis_snapshot

        return _publish_and_load(
            publish_basis_snapshot, self.SAMPLE, "basis_snapshot.json"
        )

    def test_top_level_keys(self) -> None:
        data = self._load()
        assert "symbols" in data
        assert "updated_ts" in data

    def test_symbols_is_dict(self) -> None:
        data = self._load()
        assert isinstance(data["symbols"], dict)
        assert len(data["symbols"]) == 2

    def test_per_symbol_fields(self) -> None:
        data = self._load()
        for sym, info in data["symbols"].items():
            missing = BASIS_REQUIRED_KEYS - set(info.keys())
            assert not missing, f"{sym} missing fields: {missing}"

    def test_numeric_fields(self) -> None:
        data = self._load()
        for sym, info in data["symbols"].items():
            assert isinstance(
                info["basis_pct"], (int, float)
            ), f"{sym}.basis_pct not numeric"
            assert isinstance(
                info["mark_price"], (int, float)
            ), f"{sym}.mark_price not numeric"
            assert isinstance(
                info["index_price"], (int, float)
            ), f"{sym}.index_price not numeric"

    def test_empty_symbols_ok(self) -> None:
        """Empty symbols dict should still produce valid JSON."""
        from execution.state_publish import publish_basis_snapshot

        data = _publish_and_load(
            publish_basis_snapshot, {"symbols": {}}, "basis_snapshot.json"
        )
        assert data["symbols"] == {}
        assert "updated_ts" in data


# ---------------------------------------------------------------------------
# Cross-check: manifest fields description matches actual output
# ---------------------------------------------------------------------------


class TestManifestFieldsAlignment:
    """Ensure manifest 'fields' descriptions mention the actual keys."""

    def test_funding_fields_mention_symbols_and_ts(self) -> None:
        entry = _load_manifest()["state_files"]["funding_snapshot"]
        fields = entry["fields"]
        assert "updated_ts" in fields
        assert "symbols" in fields

    def test_basis_fields_mention_symbols_and_ts(self) -> None:
        entry = _load_manifest()["state_files"]["basis_snapshot"]
        fields = entry["fields"]
        assert "updated_ts" in fields
        assert "symbols" in fields
