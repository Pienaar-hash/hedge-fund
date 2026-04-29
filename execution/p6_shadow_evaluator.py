"""
P6 Shadow Evaluator (v7.9-P6B)

Pipes P6Signal objects through the regime-conditional expectancy bridge and
logs the result to a shadow JSONL file with full provenance.

This module is OBSERVATION-ONLY. It never gates execution, never modifies
positions, and never interacts with doctrine or risk.

Log schema: 17+ fields per record (see SHADOW_FIELDS below).
Append-only: logs/execution/p6_shadow_signals.jsonl
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.expectancy_bridge import (
    BandTable,
    BridgeConfig,
    RegimeBridgeLookupResult,
    load_band_table,
    load_regime_bridge,
    lookup_expected_edge_conditional,
)
from execution.log_utils import append_jsonl
from execution.p6_simple_rules import P6Signal

LOG = logging.getLogger("p6_shadow_evaluator")

# ── Paths ────────────────────────────────────────────────────────────────

SHADOW_LOG_PATH = Path("logs/execution/p6_shadow_signals.jsonl")

# Fee threshold (same as existing fee_gate).
# 0.12% roundtrip friction — signals must clear this to be considered viable.
DEFAULT_FEE_THRESHOLD_PCT = 0.12


# ── Bridge evaluation ────────────────────────────────────────────────────

def evaluate_signal_against_bridge(
    signal: P6Signal,
    regime_tables: Optional[Dict[str, BandTable]],
    pooled_table: Optional[BandTable],
    bridge_config: Optional[BridgeConfig] = None,
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
) -> Dict[str, Any]:
    """Evaluate a single P6Signal through the expectancy bridge.

    Uses the signal's proxy conviction as primary lookup, and also runs
    a control lookup at conviction=0.5 for A/B comparison.

    Returns a dict with full provenance fields for shadow logging.
    This function is pure — no side effects — for testability.
    """
    # Primary: use the signal's derived conviction proxy
    conviction = signal.conviction
    # Control: fixed 0.5 baseline
    control_conviction = 0.5

    # Map regime to bridge regime key
    regime = signal.regime
    bridge_regime = "MEAN_REVERT" if regime == "MEAN_REVERT" else "OTHER"

    # Run primary bridge lookup
    bridge_result: Optional[RegimeBridgeLookupResult] = None
    control_result: Optional[RegimeBridgeLookupResult] = None
    if regime_tables is not None:
        bridge_result = lookup_expected_edge_conditional(
            conviction=conviction,
            regime=bridge_regime,
            regime_tables=regime_tables,
            pooled_table=pooled_table,
            config=bridge_config,
        )
        # A/B control at fixed 0.5
        control_result = lookup_expected_edge_conditional(
            conviction=control_conviction,
            regime=bridge_regime,
            regime_tables=regime_tables,
            pooled_table=pooled_table,
            config=bridge_config,
        )

    # Compute fee gate verdict
    edge_pct = bridge_result.expected_edge_pct if bridge_result else 0.0
    bridge_would_pass = edge_pct > fee_threshold_pct

    control_edge_pct = control_result.expected_edge_pct if control_result else 0.0
    control_would_pass = control_edge_pct > fee_threshold_pct

    # Build provenance record
    ts_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    record: Dict[str, Any] = {
        # ── Core identity ──
        "ts": ts_iso,
        "signal_ts": signal.ts,
        "candidate_id": signal.candidate_id,
        "candidate_family": signal.candidate_family,
        "rule_name": signal.rule_name,
        "symbol": signal.symbol,
        "side": signal.side,
        "polarity": signal.polarity,
        "regime": signal.regime,
        "region": signal.region,
        "selected_for_eval": signal.selected_for_eval,
        # ── Primary bridge provenance (proxy conviction) ──
        "conviction": round(conviction, 6),
        "bridge_regime_key": bridge_regime,
        "bridge_expected_edge_pct": round(edge_pct, 8),
        "bridge_lookup_tier": bridge_result.lookup_tier if bridge_result else "no_bridge",
        "bridge_band_key": bridge_result.band_key if bridge_result else "",
        "bridge_sample_n": bridge_result.sample_n if bridge_result else 0,
        "bridge_is_sufficient": bridge_result.is_sufficient if bridge_result else False,
        "bridge_fallback_depth": bridge_result.fallback_depth if bridge_result else -1,
        "bridge_cold_start_reason": bridge_result.cold_start_reason if bridge_result else "",
        "bridge_would_pass": bridge_would_pass,
        # ── Fee gate ──
        "fee_required_pct": fee_threshold_pct,
        "fee_pass": bridge_would_pass,
        # ── A/B control (conviction=0.5 baseline) ──
        "control_conviction": control_conviction,
        "control_expected_edge_pct": round(control_edge_pct, 8),
        "control_band_key": control_result.band_key if control_result else "",
        "control_would_pass": control_would_pass,
        # ── Feature snapshot (full replay provenance) ──
        "feature_snapshot": signal.feature_snapshot,
        # ── Suppressed alternatives (logged for audit) ──
        "suppressed_alternatives": [],  # Filled by caller if applicable
    }

    return record


# ── Batch evaluation ─────────────────────────────────────────────────────

def evaluate_and_log_signals(
    signals: List[P6Signal],
    suppressed: Optional[List[P6Signal]] = None,
    regime_tables: Optional[Dict[str, BandTable]] = None,
    pooled_table: Optional[BandTable] = None,
    bridge_config: Optional[BridgeConfig] = None,
    fee_threshold_pct: float = DEFAULT_FEE_THRESHOLD_PCT,
    log_path: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate a batch of P6 signals and log to shadow JSONL.

    Args:
        signals: Selected signals to evaluate.
        suppressed: Suppressed alternatives (attached to records for audit).
        regime_tables: From load_regime_bridge(). Auto-loaded if None.
        pooled_table: From load_bridge(). Auto-loaded if None.
        bridge_config: Optional override.
        fee_threshold_pct: Fee clearing threshold.
        log_path: Shadow log file. Uses SHADOW_LOG_PATH if None.
        dry_run: If True, returns records but does not write to disk.

    Returns:
        List of provenance records (one per signal).
    """
    if log_path is None:
        log_path = SHADOW_LOG_PATH

    # Auto-load bridge tables if not provided
    if regime_tables is None:
        regime_tables = load_regime_bridge() or {}
    if pooled_table is None:
        pooled_table = load_band_table()

    # Index suppressed by (symbol, candidate_id)
    suppressed_index: Dict[str, List[Dict[str, Any]]] = {}
    if suppressed:
        for s in suppressed:
            key = f"{s.symbol}:{s.candidate_id}"
            suppressed_index.setdefault(key, []).append({
                "rule_name": s.rule_name,
                "side": s.side,
                "polarity": s.polarity,
            })

    records: List[Dict[str, Any]] = []

    for sig in signals:
        rec = evaluate_signal_against_bridge(
            signal=sig,
            regime_tables=regime_tables,
            pooled_table=pooled_table,
            bridge_config=bridge_config,
            fee_threshold_pct=fee_threshold_pct,
        )

        # Attach suppressed alternatives if any
        key = f"{sig.symbol}:{sig.candidate_id}"
        if key in suppressed_index:
            rec["suppressed_alternatives"] = suppressed_index[key]

        records.append(rec)

        if not dry_run:
            try:
                append_jsonl(log_path, rec)
            except Exception:
                LOG.exception("Failed to write P6 shadow record for %s", sig.symbol)

    LOG.info(
        "P6 shadow: evaluated %d signals, %d fee_pass",
        len(records),
        sum(1 for r in records if r.get("fee_pass")),
    )

    return records


# ── Summary stats (for state file) ──────────────────────────────────────

def compute_shadow_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate shadow evaluation records into a summary dict.

    Written to logs/state/p6_shadow_summary.json for dashboard consumption.
    """
    if not records:
        return {
            "n_signals": 0,
            "n_fee_pass": 0,
            "by_candidate": {},
            "by_symbol": {},
            "by_regime": {},
            "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }

    by_candidate: Dict[str, Dict[str, int]] = {}
    by_symbol: Dict[str, Dict[str, int]] = {}
    by_regime: Dict[str, Dict[str, int]] = {}

    for rec in records:
        cid = rec.get("candidate_id", "unknown")
        sym = rec.get("symbol", "unknown")
        reg = rec.get("regime", "unknown")
        fp = rec.get("fee_pass", False)

        for group, key in [(by_candidate, cid), (by_symbol, sym), (by_regime, reg)]:
            if key not in group:
                group[key] = {"n_signals": 0, "n_fee_pass": 0}
            group[key]["n_signals"] += 1
            if fp:
                group[key]["n_fee_pass"] += 1

    return {
        "n_signals": len(records),
        "n_fee_pass": sum(1 for r in records if r.get("fee_pass")),
        "by_candidate": by_candidate,
        "by_symbol": by_symbol,
        "by_regime": by_regime,
        "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }
