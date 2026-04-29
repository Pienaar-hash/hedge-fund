"""Per-episode factor logging — records full hybrid score decomposition with intent_id linkage.

Write path:  logs/execution/intent_factor_log.jsonl  (append-only)
Linkage key: intent_id  (deterministic join to episode_ledger / orders_executed)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCORE_VERSION = "v6"
REQUIRED_COMPONENTS = frozenset({"carry", "trend", "expectancy", "router"})


def build_factor_log_record(intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a single factor-log record from a scored intent dict.

    Returns ``None`` when:
    * ``hybrid_components`` is missing or incomplete (all 4 factors required)
    * ``hybrid_weights_used`` is missing or incomplete
    * ``intent_id`` is missing or empty

    The caller should treat a ``None`` return as "nothing to log" (skip silently).
    """
    intent_id = intent.get("intent_id")
    if not intent_id:
        return None

    components = intent.get("hybrid_components")
    if not components or not REQUIRED_COMPONENTS.issubset(components):
        return None

    weights = intent.get("hybrid_weights_used")
    if not weights or not REQUIRED_COMPONENTS.issubset(weights):
        return None

    carry_details = intent.get("hybrid_carry_details") or {}
    carry_inputs = carry_details.get("inputs") or {}

    reconstructed = sum(
        float(weights.get(k, 0)) * float(components.get(k, 0))
        for k in REQUIRED_COMPONENTS
    )

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "intent_id": str(intent_id),
        "symbol": str(intent.get("symbol", "")),
        "side": str(intent.get("positionSide", intent.get("side", ""))),
        "hybrid_score": float(intent.get("hybrid_score", 0.0)),
        "hybrid_score_reconstructed": round(reconstructed, 8),
        "carry_score": float(components.get("carry", 0.0)),
        "trend_score": float(components.get("trend", 0.0)),
        "expectancy_score": float(components.get("expectancy", 0.0)),
        "router_score": float(components.get("router", 0.0)),
        "weights": {k: float(weights.get(k, 0.0)) for k in REQUIRED_COMPONENTS},
        "funding_rate": float(carry_inputs.get("funding_rate", 0.0)),
        "basis_pct": float(carry_inputs.get("basis_pct", 0.0)),
        "conviction_band": str(intent.get("conviction_band", "")),
        "confidence": float(intent.get("conviction_score", intent.get("confidence", 0.0))),
        "score_version": SCORE_VERSION,
    }
