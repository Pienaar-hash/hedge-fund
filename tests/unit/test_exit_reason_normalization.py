"""
Exit Reason Normalization — Phase B.1 Test Suite

Tests:
  1. All known raw reasons map to canonical.
  2. Canonical reasons map to themselves (identity).
  3. Unmapped reason triggers controlled fallback in shadow mode.
  4. Unmapped reason raises in enforced mode.
  5. HOLD assertion fires (not an exit).
  6. YAML has no duplicate raw entries.
  7. Every canonical DLE enum value is reachable from at least one raw source.
  8. Doctrine ExitReason coverage — every value (except HOLD) is mapped.
  9. Episode ledger raw strings are all mapped.
  10. Exit scanner raw values are all mapped.
"""

from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from execution.exit_reason_normalizer import (
    CanonicalExitReason,
    CANONICAL_VALUES,
    NormalizedExitReason,
    normalize_exit_reason,
    reload_map,
    verify_doctrine_coverage,
    get_unmapped_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = REPO_ROOT / "config" / "exit_reason_map.yaml"


@pytest.fixture(autouse=True)
def _fresh_map():
    """Reload map before each test to ensure clean state."""
    reload_map()
    yield


# ---------------------------------------------------------------------------
# 1. All known raw reasons map to a canonical value
# ---------------------------------------------------------------------------

_EXPECTED_RAW_MAPPINGS = {
    # doctrine_kernel ExitReason values
    "CRISIS_OVERRIDE": "EMERGENCY_HALT",
    "REGIME_FLIP": "REGIME_CHANGE",
    "REGIME_CONFIDENCE_COLLAPSE": "REGIME_CHANGE",
    "TREND_DECAY": "THESIS_INVALIDATED",
    "CARRY_DISAPPEARED": "THESIS_INVALIDATED",
    "CROSSFIRE_RESOLVED": "THESIS_INVALIDATED",
    "EXECUTION_ALPHA_DRAG": "THESIS_INVALIDATED",
    "TIME_STOP": "DECISION_EXPIRED",
    "STOP_LOSS_SEATBELT": "STOP_LOSS",
    # episode_ledger strings
    "tp": "TAKE_PROFIT",
    "sl": "STOP_LOSS",
    "thesis": "THESIS_INVALIDATED",
    "regime_flip": "REGIME_CHANGE",
    "crisis": "EMERGENCY_HALT",
    "manual": "MANUAL_CLOSE",
    "phase_end": "PHASE_END",
    "time_stop": "DECISION_EXPIRED",
    "signal_close": "THESIS_INVALIDATED",
    "position_flip": "THESIS_INVALIDATED",
    "risk": "RISK_VETO",
    "trailing": "TRAILING_STOP",
}


@pytest.mark.parametrize("raw,expected", list(_EXPECTED_RAW_MAPPINGS.items()))
def test_raw_to_canonical_mapping(raw: str, expected: str):
    result = normalize_exit_reason(raw, source="test")
    assert result.canonical == expected, f"{raw!r} → {result.canonical!r}, expected {expected!r}"
    assert result.mapped is True
    assert result.raw == raw


# ---------------------------------------------------------------------------
# 2. Canonical reasons map to themselves (identity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", sorted(CANONICAL_VALUES))
def test_canonical_identity(canonical: str):
    result = normalize_exit_reason(canonical, source="test")
    assert result.canonical == canonical
    assert result.mapped is True


# ---------------------------------------------------------------------------
# 3. Unmapped reason → shadow fallback
# ---------------------------------------------------------------------------

def test_unmapped_shadow_fallback():
    result = normalize_exit_reason("unknown", source="test")
    assert result.canonical == "THESIS_INVALIDATED"
    assert result.mapped is False
    assert result.raw == "unknown"


def test_unmapped_novel_string_fallback():
    result = normalize_exit_reason("SOME_FUTURE_REASON", source="test")
    assert result.canonical == "THESIS_INVALIDATED"
    assert result.mapped is False


# ---------------------------------------------------------------------------
# 4. Unmapped reason → enforced mode raises
# ---------------------------------------------------------------------------

def test_unmapped_enforced_raises():
    with pytest.raises(ValueError, match="Unmapped exit reason"):
        normalize_exit_reason("unknown", source="test", mode="enforced")


def test_unmapped_novel_enforced_raises():
    with pytest.raises(ValueError, match="Unmapped exit reason"):
        normalize_exit_reason("NOT_A_REAL_REASON", source="test", mode="enforced")


# ---------------------------------------------------------------------------
# 5. HOLD assertion (retention verdict, not exit)
# ---------------------------------------------------------------------------

def test_hold_assertion():
    with pytest.raises(AssertionError, match="HOLD is a retention verdict"):
        normalize_exit_reason("HOLD", source="test")


# ---------------------------------------------------------------------------
# 6. YAML has no duplicate raw entries
# ---------------------------------------------------------------------------

def test_yaml_no_duplicate_raw_entries():
    with open(MAP_PATH) as f:
        data = yaml.safe_load(f)

    seen: dict[str, str] = {}
    dupes = []
    for canonical_key, entry in data["canonical"].items():
        for raw_val in entry.get("raw", []):
            if raw_val in seen:
                dupes.append(f"{raw_val!r} in both {seen[raw_val]} and {canonical_key}")
            seen[raw_val] = canonical_key

    assert not dupes, f"Duplicate raw mappings:\n" + "\n".join(dupes)


# ---------------------------------------------------------------------------
# 7. Every canonical DLE enum value reachable from at least one raw source
# ---------------------------------------------------------------------------

def test_all_canonical_values_reachable():
    with open(MAP_PATH) as f:
        data = yaml.safe_load(f)

    reachable = set()
    for canonical_key, entry in data["canonical"].items():
        if entry.get("raw"):
            reachable.add(canonical_key)

    unreachable = CANONICAL_VALUES - reachable
    assert not unreachable, f"Canonical values with no raw mapping: {unreachable}"


# ---------------------------------------------------------------------------
# 8. Doctrine ExitReason coverage
# ---------------------------------------------------------------------------

def test_doctrine_coverage():
    """Every doctrine_kernel ExitReason (except HOLD) must be mapped."""
    verify_doctrine_coverage()  # raises ValueError if incomplete


# ---------------------------------------------------------------------------
# 9. Episode ledger raw strings all mapped
# ---------------------------------------------------------------------------

_EPISODE_LEDGER_STRINGS = [
    "tp", "sl", "thesis", "regime_flip", "signal_close",
    "position_flip", "time_stop", "crisis", "manual", "phase_end", "risk",
]


@pytest.mark.parametrize("raw", _EPISODE_LEDGER_STRINGS)
def test_episode_ledger_strings_mapped(raw: str):
    result = normalize_exit_reason(raw, source="episode_ledger")
    assert result.mapped is True, f"episode_ledger string {raw!r} is unmapped"
    assert result.canonical in CANONICAL_VALUES


# ---------------------------------------------------------------------------
# 10. Exit scanner raw values all mapped
# ---------------------------------------------------------------------------

_EXIT_SCANNER_DOCTRINE_VALUES = [
    "CRISIS_OVERRIDE", "REGIME_FLIP", "REGIME_CONFIDENCE_COLLAPSE",
    "TREND_DECAY", "CARRY_DISAPPEARED", "CROSSFIRE_RESOLVED",
    "EXECUTION_ALPHA_DRAG", "TIME_STOP", "STOP_LOSS_SEATBELT",
]

_EXIT_SCANNER_SEATBELT_VALUES = ["tp", "sl"]


@pytest.mark.parametrize("raw", _EXIT_SCANNER_DOCTRINE_VALUES + _EXIT_SCANNER_SEATBELT_VALUES)
def test_exit_scanner_values_mapped(raw: str):
    result = normalize_exit_reason(raw, source="exit_scanner")
    assert result.mapped is True, f"exit_scanner value {raw!r} is unmapped"
    assert result.canonical in CANONICAL_VALUES


# ---------------------------------------------------------------------------
# 11. Canonical enum matches DLE_EPISODE_SCHEMA exactly
# ---------------------------------------------------------------------------

def test_canonical_enum_is_exactly_10():
    """Constitutional: canonical enum must be exactly 10 values."""
    assert len(CanonicalExitReason) == 10, (
        f"CanonicalExitReason has {len(CanonicalExitReason)} values, expected 10. "
        f"Do not expand without constitutional review."
    )


_DLE_EPISODE_SCHEMA_VALUES = {
    "TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP", "THESIS_INVALIDATED",
    "REGIME_CHANGE", "MANUAL_CLOSE", "RISK_VETO", "PHASE_END",
    "DECISION_EXPIRED", "EMERGENCY_HALT",
}


def test_canonical_enum_matches_dle_schema():
    assert CANONICAL_VALUES == _DLE_EPISODE_SCHEMA_VALUES, (
        f"Drift: {CANONICAL_VALUES.symmetric_difference(_DLE_EPISODE_SCHEMA_VALUES)}"
    )


# ---------------------------------------------------------------------------
# 12. Provenance preserved in NormalizedExitReason
# ---------------------------------------------------------------------------

def test_provenance_preserved():
    result = normalize_exit_reason("TREND_DECAY", source="doctrine_kernel")
    assert result.canonical == "THESIS_INVALIDATED"
    assert result.raw == "TREND_DECAY"
    assert result.source == "doctrine_kernel"
    assert result.mapped is True
