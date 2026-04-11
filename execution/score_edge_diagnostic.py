"""P3 Score-to-Edge Diagnostic — carry-neutral shadow instrumentation.

Logs per-intent decomposition of score → edge → gate verdict,
plus carry-neutral counterfactual for structural defect classification.

P4C extension: shadow compare between the current confidence-threshold model
and the empirical expectancy bridge.  Both edge estimates are logged
side-by-side in every record.

Write path:  logs/execution/score_edge_diagnostic.jsonl  (append-only)
Linkage key: intent_id  (join to intent_factor_log / episode_ledger)

Defect classification targets:
  Class A — score defect:  carry-neutral scores improve PnL alignment
  Class B — expectancy defect:  scores fine, edge surface unwired (confidence < 0.5)
  Class C — dual defect:  both A and B

Core diagnostic fields:
  confidence_gap    = 0.5 - confidence  (positive → below threshold → zero edge)
  carry_delta       = carry_neutral_score - hybrid_score  (positive → carry was penalizing)
  expected_edge_pct = output of true_edge (0 when confidence < 0.5)

Shadow compare fields (P4C):
  bridge_expected_edge_pct = empirical E[net_edge | conviction_band]
  bridge_band_key          = which band was matched
  bridge_lookup_tier       = "band" | "global" | "cold_start"
  bridge_would_pass        = bridge edge >= fee_required_pct
  bridge_vs_threshold_delta = bridge_edge - threshold_edge
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOG = logging.getLogger("score_edge_diagnostic")

REQUIRED_COMPONENTS = frozenset({"carry", "trend", "expectancy", "router"})
CARRY_NEUTRAL = 0.5  # Shadow carry component for counterfactual

# ── Bridge shadow cache ──────────────────────────────────────────────────
# Lazily loaded on first diagnostic write.  Rebuilt on stale.
_bridge_table = None
_bridge_load_attempted = False

_regime_tables = None
_regime_load_attempted = False


def _get_bridge_table():
    """Lazily load the bridge band table (cached for process lifetime)."""
    global _bridge_table, _bridge_load_attempted
    if _bridge_load_attempted:
        return _bridge_table
    _bridge_load_attempted = True
    try:
        from execution.expectancy_bridge import load_band_table
        _bridge_table = load_band_table()
        if _bridge_table is not None:
            LOG.info("[score_edge_diag] loaded bridge table: %d bands, %d episodes",
                     len(_bridge_table.bands), _bridge_table.n_episodes_scored)
    except Exception as exc:
        LOG.warning("[score_edge_diag] bridge table load failed: %s", exc)
    return _bridge_table


def _get_regime_tables():
    """Lazily load regime-conditional bridge tables (cached for process lifetime)."""
    global _regime_tables, _regime_load_attempted
    if _regime_load_attempted:
        return _regime_tables
    _regime_load_attempted = True
    try:
        from execution.expectancy_bridge import load_regime_bridge
        _regime_tables = load_regime_bridge()
        if _regime_tables is not None:
            LOG.info("[score_edge_diag] loaded regime bridge: %s",
                     list(_regime_tables.keys()))
    except Exception as exc:
        LOG.warning("[score_edge_diag] regime bridge load failed: %s", exc)
    return _regime_tables


def build_diagnostic_record(
    intent: Dict[str, Any],
    te_result: Any,
    fg_details: Dict[str, Any],
    fg_allowed: bool,
    veto_reason: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a score→edge diagnostic record.

    Args:
        intent: Full intent dict from screener/selector.
        te_result: TrueEdgeResult from compute_true_edge().
        fg_details: Details dict from check_fee_edge_v2().
        fg_allowed: Whether fee_gate passed.
        veto_reason: Override veto reason (e.g. "min_notional").

    Returns:
        Dict for JSONL append, or None if no useful data is available.
        When hybrid_components are missing, carry analysis fields are null
        but the edge surface is still logged (sufficient for Class B detection).
    """
    components = intent.get("hybrid_components") or {}
    weights = intent.get("hybrid_weights_used") or {}
    has_full_decomposition = (
        REQUIRED_COMPONENTS.issubset(components)
        and REQUIRED_COMPONENTS.issubset(weights)
    )

    # Must have at minimum a hybrid_score or conviction_score to be useful
    hybrid_score = float(intent.get("hybrid_score", 0.0))
    confidence = float(
        intent.get("conviction_score", intent.get("confidence", 0.0))
    )
    if hybrid_score == 0.0 and confidence == 0.0:
        return None

    # ── Edge decomposition (always available) ────────────────────────────
    notional = te_result.notional_usd if te_result else 0.0
    expected_edge_pct = te_result.expected_edge_pct if te_result else 0.0
    fee_required_usd = float(fg_details.get("required_edge_usd", 0))
    fee_rt_usd = float(fg_details.get("round_trip_fee_usd", 0))
    fee_required_pct = round(fee_required_usd / notional, 8) if notional > 0 else 0.0

    # ── Structural gap measure (always available) ────────────────────────
    confidence_gap = round(0.5 - confidence, 6)

    verdict = veto_reason or ("pass" if fg_allowed else "fee_gate")

    rec: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": str(intent.get("symbol", "")),
        "side": str(intent.get("positionSide", intent.get("side", ""))),
        "intent_id": str(intent.get("intent_id", "")),
        # ── Score surface ──
        "hybrid_score": round(hybrid_score, 8),
        # ── Edge surface ──
        "confidence": round(confidence, 6),
        "confidence_gap": confidence_gap,
        "expected_edge_pct": round(expected_edge_pct, 8),
        "fee_required_pct": fee_required_pct,
        "fee_required_usd": round(fee_required_usd, 4),
        "fee_rt_usd": round(fee_rt_usd, 4),
        "notional_usd": round(notional, 2),
        "edge_source": te_result.source if te_result else "",
        "adv": round(te_result.adv, 6) if te_result else 0.0,
        "atr_pct": round(te_result.atr_pct, 8) if te_result else 0.0,
        "fallback_reason": te_result.fallback_reason if te_result else "",
        # ── Verdict ──
        "veto_reason": verdict,
        "has_decomposition": has_full_decomposition,
    }

    if has_full_decomposition:
        # ── Score decomposition ──────────────────────────────────────
        reconstructed = sum(
            float(weights.get(k, 0)) * float(components.get(k, 0))
            for k in REQUIRED_COMPONENTS
        )
        post_modifier_delta = round(hybrid_score - reconstructed, 8)

        # ── Carry-neutral shadow ─────────────────────────────────────
        carry_neutral_reconstructed = sum(
            float(weights.get(k, 0)) * (
                CARRY_NEUTRAL if k == "carry" else float(components.get(k, 0))
            )
            for k in REQUIRED_COMPONENTS
        )
        carry_neutral_score = round(carry_neutral_reconstructed + post_modifier_delta, 8)
        carry_delta = round(carry_neutral_score - hybrid_score, 8)

        rec.update({
            "hybrid_score_reconstructed": round(reconstructed, 8),
            "post_modifier_delta": post_modifier_delta,
            "carry_score": round(float(components.get("carry", 0)), 6),
            "trend_score": round(float(components.get("trend", 0)), 6),
            "expectancy_score": round(float(components.get("expectancy", 0)), 6),
            "router_score": round(float(components.get("router", 0)), 6),
            "weights": {k: round(float(weights.get(k, 0)), 4) for k in sorted(REQUIRED_COMPONENTS)},
            "carry_neutral_score": carry_neutral_score,
            "carry_neutral_reconstructed": round(carry_neutral_reconstructed, 8),
            "carry_delta": carry_delta,
        })
    else:
        # Null placeholders for carry analysis — edge surface still logged
        rec.update({
            "hybrid_score_reconstructed": None,
            "post_modifier_delta": None,
            "carry_score": None,
            "trend_score": None,
            "expectancy_score": None,
            "router_score": None,
            "weights": None,
            "carry_neutral_score": None,
            "carry_neutral_reconstructed": None,
            "carry_delta": None,
        })

    # ── P4C: Bridge shadow compare ───────────────────────────────────
    bridge_table = _get_bridge_table()
    if bridge_table is not None and confidence > 0:
        try:
            from execution.expectancy_bridge import lookup_expected_edge
            bridge_result = lookup_expected_edge(confidence, bridge_table)
            bridge_edge = bridge_result.expected_edge_pct
            bridge_would_pass = (
                bridge_edge > 0 and bridge_edge >= fee_required_pct
            ) if fee_required_pct > 0 else bridge_edge > 0

            rec.update({
                "bridge_expected_edge_pct": round(bridge_edge, 8),
                "bridge_band_key": bridge_result.band_key,
                "bridge_n_episodes": bridge_result.n_episodes,
                "bridge_sufficient": bridge_result.sufficient,
                "bridge_lookup_tier": bridge_result.lookup_tier,
                "bridge_win_rate": round(bridge_result.win_rate, 4),
                "bridge_would_pass": bridge_would_pass,
                "bridge_vs_threshold_delta": round(
                    bridge_edge - expected_edge_pct, 8
                ),
            })
        except Exception as bridge_exc:
            LOG.warning("[score_edge_diag] bridge lookup failed: %s", bridge_exc)
            rec.update(_bridge_null_fields())
    else:
        rec.update(_bridge_null_fields())

    # ── P5B: Regime bridge shadow compare ────────────────────────────
    regime_tables = _get_regime_tables()
    regime_raw = str(intent.get("regime", intent.get("regime_at_entry", "")))
    if regime_tables is not None and confidence > 0 and regime_raw:
        try:
            from execution.expectancy_bridge import lookup_expected_edge_conditional
            regime_result = lookup_expected_edge_conditional(
                conviction=confidence,
                regime=regime_raw,
                regime_tables=regime_tables,
                pooled_table=bridge_table,
            )
            rec.update(regime_result.to_dict())
        except Exception as regime_exc:
            LOG.warning("[score_edge_diag] regime bridge lookup failed: %s", regime_exc)
            rec.update(_regime_bridge_null_fields())
    else:
        rec.update(_regime_bridge_null_fields())

    return rec


def _bridge_null_fields() -> Dict[str, Any]:
    """Null placeholders for bridge shadow fields."""
    return {
        "bridge_expected_edge_pct": None,
        "bridge_band_key": None,
        "bridge_n_episodes": None,
        "bridge_sufficient": None,
        "bridge_lookup_tier": None,
        "bridge_win_rate": None,
        "bridge_would_pass": None,
        "bridge_vs_threshold_delta": None,
    }


def _regime_bridge_null_fields() -> Dict[str, Any]:
    """Null placeholders for regime bridge shadow fields."""
    return {
        "bridge_regime_expected_edge_pct": None,
        "bridge_regime_lookup_tier": None,
        "bridge_regime_band_key": None,
        "bridge_regime_key": None,
        "bridge_regime_sample_n": None,
        "bridge_regime_is_sufficient": None,
        "bridge_regime_fallback_depth": None,
    }


def classify_defect(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify defect type from a batch of diagnostic records.

    Requires >= 3 records to produce a verdict.

    Returns:
        {"verdict": "CLASS_A"|"CLASS_B"|"CLASS_C"|"INSUFFICIENT_DATA",
         "evidence": {...}}
    """
    if len(records) < 3:
        return {"verdict": "INSUFFICIENT_DATA", "n": len(records), "evidence": {}}

    n = len(records)
    zero_edge_count = sum(1 for r in records if r.get("expected_edge_pct", 0) == 0)
    confidence_below_count = sum(1 for r in records if r.get("confidence_gap", 0) > 0)

    # Carry impact: how often does carry_delta have magnitude > 0.01?
    carry_material = sum(1 for r in records if abs(r.get("carry_delta") or 0) > 0.01)
    # Carry inversion: how often does carry push score DOWN (positive delta)?
    carry_penalizing = sum(1 for r in records if (r.get("carry_delta") or 0) > 0.005)

    zero_edge_pct = zero_edge_count / n
    confidence_below_pct = confidence_below_count / n
    carry_material_pct = carry_material / n

    avg_confidence_gap = sum(r.get("confidence_gap") or 0 for r in records) / n
    avg_carry_delta = sum(r.get("carry_delta") or 0 for r in records) / n

    evidence = {
        "n_records": n,
        "zero_edge_pct": round(zero_edge_pct, 4),
        "confidence_below_threshold_pct": round(confidence_below_pct, 4),
        "carry_material_pct": round(carry_material_pct, 4),
        "carry_penalizing_pct": round(carry_penalizing / n, 4) if n > 0 else 0,
        "avg_confidence_gap": round(avg_confidence_gap, 6),
        "avg_carry_delta": round(avg_carry_delta, 6),
    }

    # ── Classification logic ─────────────────────────────────────────────
    is_class_b = confidence_below_pct >= 0.80  # 80%+ of records have confidence < 0.5
    is_class_a = carry_material_pct >= 0.30    # 30%+ of records show material carry impact

    if is_class_a and is_class_b:
        verdict = "CLASS_C"
    elif is_class_a:
        verdict = "CLASS_A"
    elif is_class_b:
        verdict = "CLASS_B"
    else:
        verdict = "INCONCLUSIVE"

    return {"verdict": verdict, "evidence": evidence}
