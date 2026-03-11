"""
ECS Shadow Selector — Phase 4 Commit 2.

Runs the candidate selector in observation mode alongside the executor's
existing fallback-swap logic.  Compares results and emits telemetry.

**Shadow only** — never gates execution, never modifies intents.

Operates on **copies** of intent objects to prevent any double-mutation
of conviction fields or selector metadata.

Env:
    ECS_SHADOW_ENABLED=1   Enable shadow selector telemetry.
    ECS_SHADOW_ENABLED=0   (default) Shadow selector is a no-op.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

_LOG_PATH = Path("logs/execution/ecs_shadow_events.jsonl")


def _shadow_enabled() -> bool:
    return (os.getenv("ECS_SHADOW_ENABLED", "0") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _append_event(event: Dict[str, Any]) -> None:
    """Append a single JSONL event (fail-open)."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        LOG.debug("[ecs_shadow] log_write_failed: %s", exc)


def run_shadow_comparison(
    *,
    symbol: str,
    raw_intent: Dict[str, Any],
    executor_winner_source: str,
    executor_used_fallback: bool,
    min_conviction_band: str = "",
    cycle: int = 0,
) -> Optional[Dict[str, Any]]:
    """Run the selector on deep copies and compare with executor result.

    Args:
        symbol: The symbol being evaluated.
        raw_intent: The **original** merged intent (before executor fallback
            swap), containing ``_fallback`` if both engines competed.
        executor_winner_source: Source tag the executor ultimately picked
            (e.g. "hydra", "legacy").
        executor_used_fallback: True if executor swapped to fallback.
        min_conviction_band: Band gate string from strategy config.
        cycle: Current executor cycle number.

    Returns:
        Telemetry dict (also appended to JSONL log), or None if shadow
        is disabled or an error occurred.
    """
    if not _shadow_enabled():
        return None

    try:
        from execution.candidate_selector import (
            build_candidates,
            select_executable_candidate,
        )

        # Deep-copy to prevent any mutation of live objects
        primary_copy = copy.deepcopy(raw_intent)
        fallback_copy = primary_copy.pop("_fallback", None)

        # Determine engine sources
        hydra_intent = None
        legacy_intent = None
        primary_source = str(primary_copy.get("source", "")).lower()
        fallback_source = str((fallback_copy or {}).get("source", "")).lower() if fallback_copy else ""

        if primary_source == "hydra":
            hydra_intent = primary_copy
            if fallback_copy:
                legacy_intent = fallback_copy
        elif primary_source == "legacy":
            legacy_intent = primary_copy
            if fallback_copy:
                hydra_intent = fallback_copy
        else:
            # Unknown source — treat as hydra (current reality)
            hydra_intent = primary_copy
            if fallback_copy:
                legacy_intent = fallback_copy

        candidates = build_candidates(symbol, hydra_intent=hydra_intent, legacy_intent=legacy_intent)
        # Note: We skip enrich_candidates_with_conviction here because
        # the copies already carry conviction fields from the executor's
        # enrichment pass.  Re-running conviction would be redundant and
        # could produce slightly different results if state changed.
        result = select_executable_candidate(candidates, min_conviction_band)

        selector_winner = result["winner_engine"]
        agreement = selector_winner == executor_winner_source

        event = {
            "ts": time.time(),
            "schema": "ecs_shadow_v1",
            "symbol": symbol,
            "cycle": cycle,
            "selector_winner": selector_winner,
            "executor_winner": executor_winner_source,
            "agreement": agreement,
            "selector_reason": result["selection_reason"],
            "executor_used_fallback": executor_used_fallback,
            "candidates_count": len(candidates),
            "selector_loser": result["loser_engine"],
            "min_conviction_band": min_conviction_band or "none",
        }
        _append_event(event)

        if not agreement:
            LOG.info(
                "[ecs_shadow] DIVERGENCE sym=%s selector=%s executor=%s reason=%s",
                symbol, selector_winner, executor_winner_source, result["selection_reason"],
            )

        return event

    except Exception as exc:
        LOG.debug("[ecs_shadow] shadow_comparison_failed: %s", exc)
        return None
