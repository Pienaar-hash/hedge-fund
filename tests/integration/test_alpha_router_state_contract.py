"""
State contract tests for alpha_router.py (v7.8_P2)

Tests that verify the alpha router state surface follows the State Contract:
- Schema matches v7_manifest.json definition
- Single writer (executor)
- Atomic writes
- All required fields present
"""
import json
import pytest
from pathlib import Path
from typing import Any, Dict

from execution.alpha_router import (
    AlphaRouterState,
    create_neutral_state,
    compute_target_allocation,
    write_alpha_router_state,
    load_alpha_router_state,
    AlphaRouterConfig,
)


# ---------------------------------------------------------------------------
# Schema Contract Tests
# ---------------------------------------------------------------------------


class TestSchemaContract:
    """Tests for state schema compliance with manifest."""

    def test_required_fields_present(self, tmp_path: Path):
        """State must have all required fields per v7_manifest.json."""
        state_path = tmp_path / "alpha_router_state.json"
        
        cfg = AlphaRouterConfig(enabled=True, smoothing_alpha=0.0)
        state = compute_target_allocation(
            health={"health_score": 0.70},
            meta_state=None,
            router_quality=0.75,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        
        write_alpha_router_state(state, state_path)
        
        with open(state_path, "r") as f:
            data = json.load(f)
        
        # Per v7_manifest.json, required fields:
        # - updated_ts: ISO timestamp
        # - target_allocation: float ∈ [0, 1]
        # - raw_components: object with health_base, vol_penalty, etc.
        # - smoothed: bool
        # - prev_allocation: float
        
        assert "updated_ts" in data
        assert isinstance(data["updated_ts"], str)
        assert len(data["updated_ts"]) > 0  # Not empty
        
        assert "target_allocation" in data
        assert isinstance(data["target_allocation"], (int, float))
        assert 0 <= data["target_allocation"] <= 1
        
        assert "raw_components" in data
        assert isinstance(data["raw_components"], dict)
        
        assert "smoothed" in data
        assert isinstance(data["smoothed"], bool)
        
        assert "prev_allocation" in data
        assert isinstance(data["prev_allocation"], (int, float))

    def test_raw_components_has_breakdown(self, tmp_path: Path):
        """raw_components must have allocation breakdown fields."""
        state_path = tmp_path / "alpha_router_state.json"
        
        cfg = AlphaRouterConfig(enabled=True, smoothing_alpha=0.0)
        state = compute_target_allocation(
            health={"health_score": 0.70},
            meta_state=None,
            router_quality=0.75,
            vol_regime="HIGH",
            dd_state="RECOVERY",
            cfg=cfg,
        )
        
        write_alpha_router_state(state, state_path)
        
        with open(state_path, "r") as f:
            data = json.load(f)
        
        components = data["raw_components"]
        
        # Required breakdown fields
        assert "health_score" in components or "health_base" in components
        assert "vol_penalty" in components
        assert "dd_penalty" in components
        assert "router_penalty" in components
        assert "raw_allocation" in components

    def test_neutral_state_valid(self):
        """Neutral state should be valid JSON-serializable."""
        state = create_neutral_state()
        d = state.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        
        assert parsed["target_allocation"] == 1.0
        assert parsed["prev_allocation"] == 1.0
        assert parsed["smoothed"] is False


# ---------------------------------------------------------------------------
# Atomic Write Tests
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_atomic_write_creates_file(self, tmp_path: Path):
        """Atomic write should create the file."""
        state_path = tmp_path / "alpha_router_state.json"
        state = create_neutral_state()
        
        assert not state_path.exists()
        write_alpha_router_state(state, state_path)
        assert state_path.exists()

    def test_atomic_write_overwrites(self, tmp_path: Path):
        """Atomic write should overwrite existing file."""
        state_path = tmp_path / "alpha_router_state.json"
        
        # Write first state
        state1 = create_neutral_state()
        state1.target_allocation = 0.80
        write_alpha_router_state(state1, state_path)
        
        # Write second state
        state2 = create_neutral_state()
        state2.target_allocation = 0.60
        write_alpha_router_state(state2, state_path)
        
        # Should have second state
        loaded = load_alpha_router_state(state_path)
        assert loaded is not None
        assert loaded.target_allocation == 0.60

    def test_temp_file_not_left_behind(self, tmp_path: Path):
        """Atomic write should not leave .tmp file behind."""
        state_path = tmp_path / "alpha_router_state.json"
        state = create_neutral_state()
        
        write_alpha_router_state(state, state_path)
        
        # Check no .tmp file exists
        tmp_file = state_path.with_suffix(".tmp")
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Path Convention Tests
# ---------------------------------------------------------------------------


class TestPathConventions:
    """Tests for state file path conventions."""

    def test_default_path_in_logs_state(self):
        """Default path should be in logs/state/."""
        from execution.alpha_router import DEFAULT_ALPHA_ROUTER_PATH
        
        assert "logs/state" in str(DEFAULT_ALPHA_ROUTER_PATH)
        assert DEFAULT_ALPHA_ROUTER_PATH.name == "alpha_router_state.json"

    def test_creates_parent_directories(self, tmp_path: Path):
        """Write should create parent directories if needed."""
        state_path = tmp_path / "deep" / "nested" / "path" / "alpha_router_state.json"
        state = create_neutral_state()
        
        write_alpha_router_state(state, state_path)
        
        assert state_path.exists()
        assert state_path.parent.exists()


# ---------------------------------------------------------------------------
# Round-Trip Consistency Tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for data consistency through round-trips."""

    def test_float_precision_preserved(self, tmp_path: Path):
        """Float precision should be preserved to 4 decimal places."""
        state_path = tmp_path / "alpha_router_state.json"
        
        original = AlphaRouterState(
            updated_ts="2025-01-01T00:00:00+00:00",
            target_allocation=0.7234,
            raw_components={"health_base": 0.8567},
            smoothed=True,
            prev_allocation=0.7012,
        )
        
        write_alpha_router_state(original, state_path)
        loaded = load_alpha_router_state(state_path)
        
        # Should round to 4 decimal places
        assert loaded.target_allocation == 0.7234
        assert loaded.raw_components["health_base"] == 0.8567
        assert loaded.prev_allocation == 0.7012

    def test_bool_preserved(self, tmp_path: Path):
        """Boolean values should be preserved."""
        state_path = tmp_path / "alpha_router_state.json"
        
        for smoothed_val in [True, False]:
            original = AlphaRouterState(
                smoothed=smoothed_val,
            )
            
            write_alpha_router_state(original, state_path)
            loaded = load_alpha_router_state(state_path)
            
            assert loaded.smoothed == smoothed_val

    def test_complex_components_preserved(self, tmp_path: Path):
        """Complex raw_components dict should be preserved."""
        state_path = tmp_path / "alpha_router_state.json"
        
        original = AlphaRouterState(
            updated_ts="2025-01-01T00:00:00+00:00",
            target_allocation=0.65,
            raw_components={
                "health_score": 0.70,
                "health_base": 0.75,
                "vol_regime": "HIGH",
                "vol_penalty": 0.80,
                "dd_state": "RECOVERY",
                "dd_penalty": 0.85,
                "router_quality": 0.72,
                "router_penalty": 0.90,
                "meta_adjustment": -0.02,
                "raw_allocation": 0.64,
            },
            smoothed=True,
            prev_allocation=0.68,
        )
        
        write_alpha_router_state(original, state_path)
        loaded = load_alpha_router_state(state_path)
        
        assert loaded.raw_components["vol_regime"] == "HIGH"
        assert loaded.raw_components["dd_state"] == "RECOVERY"
        assert loaded.raw_components["meta_adjustment"] == -0.02


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests verifying alignment with v7_manifest.json."""

    def test_manifest_entry_exists(self):
        """Verify alpha_router_state exists in manifest."""
        manifest_path = Path("v7_manifest.json")
        
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        state_files = manifest.get("state_files", {})
        assert "alpha_router_state" in state_files
        
        entry = state_files["alpha_router_state"]
        assert entry["path"] == "logs/state/alpha_router_state.json"
        assert entry["owner"] == "executor"
        assert entry.get("optional", False) is True

    def test_manifest_fields_documented(self):
        """Verify manifest documents required fields."""
        manifest_path = Path("v7_manifest.json")
        
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        entry = manifest.get("state_files", {}).get("alpha_router_state", {})
        fields = entry.get("fields", {})
        
        # Verify key fields are documented
        assert "updated_ts" in fields
        assert "target_allocation" in fields
        assert "raw_components" in fields
