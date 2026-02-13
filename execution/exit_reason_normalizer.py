"""
Exit Reason Normalization — Phase B.1

Maps raw exit reason strings from any source vocabulary to the canonical
DLE_EPISODE_SCHEMA exit_reason enum (10 values, constitutional surface).

Usage:
    from execution.exit_reason_normalizer import normalize_exit_reason

    result = normalize_exit_reason("TREND_DECAY")
    # result.canonical == "THESIS_INVALIDATED"
    # result.raw == "TREND_DECAY"
    # result.mapped == True

Sources:
    - doctrine_kernel.py ExitReason enum values
    - episode_ledger.py _extract_exit_reason() strings
    - exit_scanner.py SeatbeltExitReason enum values

Authority:
    config/exit_reason_map.yaml (BINDING)

Modes:
    SHADOW   — unmapped reasons log warning, fallback to THESIS_INVALIDATED
    ENFORCED — unmapped reasons raise (future Phase B enforcement)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import yaml

LOG = logging.getLogger("exit_reason_normalizer")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAP_PATH = _REPO_ROOT / "config" / "exit_reason_map.yaml"


# ---------------------------------------------------------------------------
# Canonical enum — mirrors DLE_EPISODE_SCHEMA exactly
# ---------------------------------------------------------------------------

class CanonicalExitReason(str, Enum):
    """The 10 constitutional exit reasons. Do not expand."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    REGIME_CHANGE = "REGIME_CHANGE"
    MANUAL_CLOSE = "MANUAL_CLOSE"
    RISK_VETO = "RISK_VETO"
    PHASE_END = "PHASE_END"
    DECISION_EXPIRED = "DECISION_EXPIRED"
    EMERGENCY_HALT = "EMERGENCY_HALT"


CANONICAL_VALUES: FrozenSet[str] = frozenset(e.value for e in CanonicalExitReason)


# ---------------------------------------------------------------------------
# Normalization result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedExitReason:
    """Result of exit reason normalization."""

    canonical: str        # One of CanonicalExitReason values
    raw: str              # Original raw reason string
    mapped: bool          # True if raw was in the map; False if fallback used
    source: str = ""      # Optional: who produced the raw reason


# ---------------------------------------------------------------------------
# Map loader (once at startup)
# ---------------------------------------------------------------------------

_RAW_TO_CANONICAL: Optional[Dict[str, str]] = None
_CANONICAL_SET: Optional[FrozenSet[str]] = None


def _load_map() -> Dict[str, str]:
    """Load exit_reason_map.yaml and build raw → canonical lookup."""
    with open(_MAP_PATH) as f:
        data = yaml.safe_load(f)

    canonical_section = data.get("canonical", {})
    raw_to_canonical: Dict[str, str] = {}

    for canonical_key, entry in canonical_section.items():
        # Validate canonical key is in our enum
        if canonical_key not in CANONICAL_VALUES:
            raise ValueError(
                f"exit_reason_map.yaml contains unknown canonical key: {canonical_key!r}. "
                f"Valid: {sorted(CANONICAL_VALUES)}"
            )

        for raw_val in entry.get("raw", []):
            if raw_val in raw_to_canonical:
                raise ValueError(
                    f"exit_reason_map.yaml: duplicate raw mapping {raw_val!r} "
                    f"(already mapped to {raw_to_canonical[raw_val]!r}, "
                    f"now also {canonical_key!r})"
                )
            raw_to_canonical[raw_val] = canonical_key

    return raw_to_canonical


def _ensure_loaded() -> Dict[str, str]:
    """Lazy-load the map on first use."""
    global _RAW_TO_CANONICAL, _CANONICAL_SET
    if _RAW_TO_CANONICAL is None:
        _RAW_TO_CANONICAL = _load_map()
        _CANONICAL_SET = frozenset(_RAW_TO_CANONICAL.values())
    return _RAW_TO_CANONICAL


def reload_map() -> None:
    """Force reload (useful after config changes in tests)."""
    global _RAW_TO_CANONICAL, _CANONICAL_SET
    _RAW_TO_CANONICAL = None
    _CANONICAL_SET = None
    _ensure_loaded()


# ---------------------------------------------------------------------------
# Core normalization function
# ---------------------------------------------------------------------------

_SHADOW_FALLBACK = CanonicalExitReason.THESIS_INVALIDATED.value
_UNMAPPED_COUNTER = 0


def normalize_exit_reason(
    raw_reason: str,
    source: str = "",
    *,
    mode: str = "shadow",
) -> NormalizedExitReason:
    """
    Normalize a raw exit reason to canonical DLE vocabulary.

    Args:
        raw_reason: Raw reason from any source vocabulary.
        source: Optional label for provenance (e.g. "doctrine_kernel", "episode_ledger").
        mode: "shadow" (default) or "enforced".

    Returns:
        NormalizedExitReason with canonical value and provenance.

    Raises:
        ValueError: In "enforced" mode if raw_reason is unmapped.
        AssertionError: If raw_reason is "HOLD" (retention verdict, not an exit).
    """
    global _UNMAPPED_COUNTER

    # HOLD must never appear in exit pipeline
    assert raw_reason != "HOLD", (
        f"HOLD is a retention verdict, not an exit reason. "
        f"Source: {source!r}. This is a bug in the exit pipeline."
    )

    lookup = _ensure_loaded()

    # Direct lookup
    canonical = lookup.get(raw_reason)
    if canonical is not None:
        return NormalizedExitReason(
            canonical=canonical,
            raw=raw_reason,
            mapped=True,
            source=source,
        )

    # Canonical values map to themselves
    if raw_reason in CANONICAL_VALUES:
        return NormalizedExitReason(
            canonical=raw_reason,
            raw=raw_reason,
            mapped=True,
            source=source,
        )

    # Unmapped
    _UNMAPPED_COUNTER += 1

    if mode == "enforced":
        raise ValueError(
            f"Unmapped exit reason {raw_reason!r} (source={source!r}). "
            f"In enforced mode, all exit reasons must be in exit_reason_map.yaml."
        )

    # Shadow mode: warn and fallback
    LOG.warning(
        "exit_reason_unmapped raw=%r source=%r fallback=%s count=%d",
        raw_reason, source, _SHADOW_FALLBACK, _UNMAPPED_COUNTER,
    )

    return NormalizedExitReason(
        canonical=_SHADOW_FALLBACK,
        raw=raw_reason,
        mapped=False,
        source=source,
    )


def get_unmapped_count() -> int:
    """Return the number of unmapped exit reasons encountered (for metrics)."""
    return _UNMAPPED_COUNTER


# ---------------------------------------------------------------------------
# Startup invariant
# ---------------------------------------------------------------------------

def verify_doctrine_coverage() -> None:
    """
    Verify that every doctrine_kernel ExitReason (except HOLD) is mapped.

    Call at startup. Raises ValueError if any doctrine exit reason is missing
    from exit_reason_map.yaml.
    """
    try:
        from execution.doctrine_kernel import ExitReason as DoctrineExitReason
    except ImportError:
        LOG.warning("doctrine_kernel not available — skipping coverage check")
        return

    lookup = _ensure_loaded()
    all_mapped = set(lookup.keys()) | CANONICAL_VALUES

    missing = []
    for reason in DoctrineExitReason:
        if reason.value == "HOLD":
            continue  # HOLD is retention, not exit
        if reason.value not in all_mapped:
            missing.append(reason.value)

    if missing:
        raise ValueError(
            f"Doctrine ExitReason values missing from exit_reason_map.yaml: {missing}. "
            f"Every doctrine exit reason (except HOLD) must be mapped to a canonical value."
        )

    LOG.info("exit_reason_normalizer: doctrine coverage verified — all %d exit reasons mapped",
             len(DoctrineExitReason) - 1)  # -1 for HOLD
