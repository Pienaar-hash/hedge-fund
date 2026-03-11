"""
Executable Candidate Selector (ECS) — Phase 4 Commit 1.

Additive boundary that models the merge → conviction → select pipeline
as a single decision surface.  Does NOT modify executor behavior.

The selector replaces the current three-stage approach:
    merge_with_single_strategy_intents → conviction enrichment → fallback swap
with a unified:
    build_candidates → enrich → select

Usage (future — shadow only until flag enabled):
    candidates = build_candidates(symbol, hydra_intent, legacy_intent)
    enrich_candidates_with_conviction(candidates, strategy_config)
    result = select_executable_candidate(candidates, min_conviction_band)
    if result["selected"]:
        _send_order(result["selected"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger(__name__)

_CONVICTION_BAND_ORDER: Dict[str, int] = {
    "very_low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
}


def _intent_score(intent: Dict[str, Any]) -> float:
    """Extract a comparable score from any intent."""
    for key in ("hybrid_score", "score"):
        val = intent.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def build_candidates(
    symbol: str,
    hydra_intent: Optional[Dict[str, Any]] = None,
    legacy_intent: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build a candidate list from available engine intents for a symbol.

    Each candidate is a shallow copy of the source intent enriched with
    selector metadata: ``_selector_score``, ``_selector_source``.

    Returns:
        List of candidate dicts (0, 1, or 2 items).
    """
    candidates: List[Dict[str, Any]] = []

    if hydra_intent and isinstance(hydra_intent, dict):
        c = dict(hydra_intent)
        c["_selector_source"] = c.get("source", "hydra")
        c["_selector_score"] = _intent_score(c)
        c.setdefault("symbol", symbol)
        candidates.append(c)

    if legacy_intent and isinstance(legacy_intent, dict):
        c = dict(legacy_intent)
        c["_selector_source"] = c.get("source", "legacy")
        c["_selector_score"] = _intent_score(c)
        c.setdefault("symbol", symbol)
        candidates.append(c)

    # Sort by score descending (highest first)
    candidates.sort(key=lambda c: c["_selector_score"], reverse=True)
    return candidates


def enrich_candidates_with_conviction(
    candidates: List[Dict[str, Any]],
    strategy_config: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Enrich each candidate with conviction score/band.

    Delegates to the conviction engine.  Each candidate is enriched
    independently (no ``_fallback`` nesting required).

    Args:
        candidates: Candidate list from :func:`build_candidates`.
        strategy_config: Full strategy_config dict (for conviction block).

    Returns:
        The same list, each candidate enriched with conviction fields.
    """
    from execution.conviction_engine import enrich_intents_with_conviction

    # enrich_intents_with_conviction also enriches nested _fallback,
    # but candidates are flat — each is a top-level intent.
    return enrich_intents_with_conviction(candidates, strategy_config)


def select_executable_candidate(
    candidates: List[Dict[str, Any]],
    min_conviction_band: str = "",
) -> Dict[str, Any]:
    """Select the best candidate that meets the conviction band gate.

    Selection order:
    1. Candidates are already sorted by score (highest first).
    2. If ``min_conviction_band`` is set, skip candidates below the band.
    3. First candidate that passes the band gate wins.
    4. If no candidate passes, ``selected`` is None.

    Args:
        candidates: Enriched candidate list.
        min_conviction_band: Minimum conviction band for entry (e.g. "medium").
            Empty string disables the band gate.

    Returns:
        Dict with keys:
            selected: The winning intent dict, or None.
            candidates: Full candidate list (for telemetry).
            winner_engine: Source of selected candidate, or "none".
            loser_engine: Source of the losing candidate (if 2 candidates), or None.
            selection_reason: Why this candidate was chosen.
    """
    min_rank = _CONVICTION_BAND_ORDER.get(min_conviction_band.lower(), -1) if min_conviction_band else -1
    band_gate_active = min_rank >= 0

    selected = None
    selection_reason = "no_candidates"

    for c in candidates:
        if band_gate_active:
            band = str(c.get("conviction_band", "")).lower()
            rank = _CONVICTION_BAND_ORDER.get(band, -1)
            if rank < min_rank:
                LOG.debug(
                    "[candidate_selector] reject sym=%s source=%s band=%s(%d) min=%s(%d)",
                    c.get("symbol"), c.get("_selector_source"), band, rank,
                    min_conviction_band, min_rank,
                )
                continue

        selected = c
        selection_reason = "band_pass" if band_gate_active else "highest_score"
        break

    winner_engine = str(selected.get("_selector_source", "unknown")) if selected else "none"

    # Determine loser
    loser_engine: Optional[str] = None
    if len(candidates) >= 2:
        for c in candidates:
            src = str(c.get("_selector_source", "unknown"))
            if src != winner_engine:
                loser_engine = src
                break

    if not selected and candidates:
        selection_reason = "all_rejected_by_band_gate"

    return {
        "selected": selected,
        "candidates": candidates,
        "winner_engine": winner_engine,
        "loser_engine": loser_engine,
        "selection_reason": selection_reason,
    }
