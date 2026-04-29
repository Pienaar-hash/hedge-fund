"""
FPS v1 Class Evaluation Scorecard — fixed rubric for hypothesis-driven setup classes.

Usage:
    PYTHONPATH=. python research/fps_class_scorecard.py [--log PATH] [--min-samples N]

Reads the shadow JSONL and evaluates each setup class against a fixed rubric.
Emits a per-class verdict: ADVANCE | REFINE | KILL | INSUFFICIENT_DATA.

This is a read-only analysis tool. It does not gate execution, modify state,
or influence any trading decision. It exists to kill weak hypotheses fast.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Verdicts (scorecard-level, not FPS pipeline verdicts)
# ---------------------------------------------------------------------------
ADVANCE = "ADVANCE"                # passes all criteria, sufficient data
REFINE = "REFINE"                  # borderline on 1+ criteria
KILL = "KILL"                      # fails hard kill criterion
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # below minimum sample size

# ---------------------------------------------------------------------------
# Rubric thresholds — fixed from day one, not tunable per class
# ---------------------------------------------------------------------------
_MIN_SAMPLES = 30                  # minimum permits before any verdict is meaningful
_PERMIT_RATE_FLOOR = 0.01         # <1% → class is dead / never fires
_PERMIT_RATE_CEIL = 0.15          # >15% → class is too loose / drifting
_FEE_PASS_RATE_FLOOR = 0.55      # <55% → class structurally loses after friction
_DIRECTION_MATCH_FLOOR = 0.80    # <80% → class fires with wrong direction too often
_SYMBOL_CONCENTRATION_CEIL = 0.80  # >80% on one symbol → not generalizable
_BURST_RATE_CEIL = 0.30           # >30% permits in 1h window → temporal clustering
_ADVANCE_FEE_PASS = 0.70         # need ≥70% fee-pass for ADVANCE (vs 50% kill)
_ADVANCE_DIRECTION_MATCH = 0.90  # need ≥90% direction match for ADVANCE

# ---------------------------------------------------------------------------
# Hypothesis-driven classes under evaluation
# (regime-event classes are excluded — they have their own track record)
# ---------------------------------------------------------------------------
_HYPOTHESIS_CLASSES = frozenset({
    "VOL_EXPANSION_BREAKOUT",
    "EXHAUSTION_REVERSAL",
    "TREND_PULLBACK",
})


# ---------------------------------------------------------------------------
# Per-class scorecard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassScore:
    """Evaluation metrics for a single setup class."""

    setup_class: str
    verdict: str                    # ADVANCE | REFINE | KILL | INSUFFICIENT_DATA

    # Volume
    total_evals: int                # total shadow evals where class was in config
    permit_count: int               # times this class fired PERMIT_CANDIDATE
    permit_rate: float              # permit_count / total_evals

    # Fee feasibility
    fee_pass_count: int             # permits that cleared fee bridge
    fee_deny_count: int             # permits denied by fee bridge
    fee_pass_rate: float            # fee_pass_count / permit_count (0.0 if no permits)

    # Direction coherence
    direction_match_count: int      # permits where direction was validated
    direction_mismatch_count: int   # permits where direction failed gate 2
    direction_match_rate: float     # match / (match + mismatch)

    # Concentration
    by_symbol: Dict[str, int]       # symbol → permit count
    max_symbol_share: float         # highest single-symbol share of permits
    top_symbol: str                 # most frequent symbol

    # Regime distribution
    by_regime: Dict[str, int]       # regime → permit count

    # Temporal
    burst_1h_max_rate: float        # peak 1h permit density for this class
    max_consecutive: int            # longest consecutive streak

    # Post-fee expectancy (populated when realized_return present in shadow)
    mean_return_per_permit: Optional[float]  # mean realized return per permit (None if unavailable)
    return_sample_count: int        # how many permits had realized_return data

    # Kill reasons (empty = no kills)
    kill_reasons: tuple             # tuple of strings explaining kill
    refine_reasons: tuple           # tuple of strings explaining borderline


@dataclass(frozen=True)
class ScorecardReport:
    """Full scorecard across all hypothesis classes."""

    total_records: int
    class_scores: Dict[str, ClassScore]
    summary_verdict: str            # worst verdict across all classes
    alerts: List[str]


# ---------------------------------------------------------------------------
# JSONL loader (shared pattern with compute_sparsity)
# ---------------------------------------------------------------------------
def _load_shadow_records(log_path: Path) -> List[Dict[str, Any]]:
    """Load and parse shadow JSONL. Returns empty list if file missing."""
    records: List[Dict[str, Any]] = []
    if not log_path.exists():
        return records
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ---------------------------------------------------------------------------
# Per-class metric extraction
# ---------------------------------------------------------------------------
def _extract_class_metrics(
    setup_class: str,
    records: List[Dict[str, Any]],
    *,
    min_samples: int = _MIN_SAMPLES,
) -> ClassScore:
    """Compute all scorecard metrics for a single setup class."""

    total_evals = len(records)

    # --- Permits for this class ---
    permits = [r for r in records
               if r.get("setup_class") == setup_class
               and r.get("verdict") == "PERMIT_CANDIDATE"]
    permit_count = len(permits)
    permit_rate = permit_count / total_evals if total_evals > 0 else 0.0

    # --- Fee bridge ---
    # PERMIT_CANDIDATE already passed fee bridge (gate 3).
    # DENY_STRUCTURAL with this class means fee bridge failed.
    fee_denied = [r for r in records
                  if r.get("setup_class") == setup_class
                  and r.get("verdict") == "DENY_STRUCTURAL"
                  and r.get("deny_reason") == "fee_bridge_insufficient"]
    fee_pass_count = permit_count
    fee_deny_count = len(fee_denied)
    fee_total = fee_pass_count + fee_deny_count
    fee_pass_rate = fee_pass_count / fee_total if fee_total > 0 else 0.0

    # --- Direction coherence ---
    # Permits have validated direction. ABSTAINs with this class in gate_trace
    # but "g2:direction_mismatch" indicate direction failure after gate 1 matched.
    dir_mismatches = [r for r in records
                      if r.get("setup_class") == setup_class
                      and r.get("verdict") == "ABSTAIN"
                      and "g2:direction_mismatch" in (r.get("gate_trace") or [])]
    direction_match_count = permit_count + fee_deny_count  # passed gate 2
    direction_mismatch_count = len(dir_mismatches)
    dir_total = direction_match_count + direction_mismatch_count
    direction_match_rate = direction_match_count / dir_total if dir_total > 0 else 0.0

    # --- Symbol concentration ---
    by_symbol: Dict[str, int] = {}
    for r in permits:
        s = r.get("symbol", "?")
        by_symbol[s] = by_symbol.get(s, 0) + 1
    if by_symbol:
        top_symbol = max(by_symbol, key=by_symbol.get)  # type: ignore[arg-type]
        max_symbol_share = by_symbol[top_symbol] / permit_count
    else:
        top_symbol = "N/A"
        max_symbol_share = 0.0

    # --- Regime distribution ---
    by_regime: Dict[str, int] = {}
    for r in permits:
        rg = r.get("regime_current", "?")
        by_regime[rg] = by_regime.get(rg, 0) + 1

    # --- Temporal: burst + consecutive ---
    permit_ts = sorted(r.get("ts", 0.0) for r in permits)
    burst_1h_max = _compute_max_window_rate(permit_ts, records, 3600.0)
    max_consec = _compute_max_consecutive(records, setup_class)

    # --- Post-fee expectancy (when realized_return is backfilled) ---
    returns = [r["realized_return"] for r in permits
               if r.get("realized_return") is not None]
    return_sample_count = len(returns)
    mean_return = sum(returns) / return_sample_count if return_sample_count > 0 else None

    # --- Verdict ---
    kill_reasons: List[str] = []
    refine_reasons: List[str] = []

    if permit_count < min_samples:
        return ClassScore(
            setup_class=setup_class,
            verdict=INSUFFICIENT_DATA,
            total_evals=total_evals,
            permit_count=permit_count,
            permit_rate=permit_rate,
            fee_pass_count=fee_pass_count,
            fee_deny_count=fee_deny_count,
            fee_pass_rate=fee_pass_rate,
            direction_match_count=direction_match_count,
            direction_mismatch_count=direction_mismatch_count,
            direction_match_rate=direction_match_rate,
            by_symbol=by_symbol,
            max_symbol_share=max_symbol_share,
            top_symbol=top_symbol,
            by_regime=by_regime,
            burst_1h_max_rate=burst_1h_max,
            max_consecutive=max_consec,
            mean_return_per_permit=mean_return,
            return_sample_count=return_sample_count,
            kill_reasons=(),
            refine_reasons=(f"only {permit_count}/{min_samples} samples",),
        )

    # Kill criteria (hard)
    if permit_rate < _PERMIT_RATE_FLOOR:
        kill_reasons.append(f"permit_rate {permit_rate:.1%} < {_PERMIT_RATE_FLOOR:.0%} floor (dead)")
    if permit_rate > _PERMIT_RATE_CEIL:
        kill_reasons.append(f"permit_rate {permit_rate:.1%} > {_PERMIT_RATE_CEIL:.0%} ceil (drift)")
    if fee_pass_rate < _FEE_PASS_RATE_FLOOR:
        kill_reasons.append(f"fee_pass_rate {fee_pass_rate:.1%} < {_FEE_PASS_RATE_FLOOR:.0%} (friction kills it)")
    if direction_match_rate < _DIRECTION_MATCH_FLOOR:
        kill_reasons.append(f"direction_match {direction_match_rate:.1%} < {_DIRECTION_MATCH_FLOOR:.0%} (class fires misaligned)")
    if max_symbol_share > _SYMBOL_CONCENTRATION_CEIL:
        kill_reasons.append(f"symbol concentration {max_symbol_share:.1%} on {top_symbol} > {_SYMBOL_CONCENTRATION_CEIL:.0%}")

    if kill_reasons:
        verdict = KILL
    else:
        # Refine criteria (soft — borderline zones)
        if fee_pass_rate < _ADVANCE_FEE_PASS:
            refine_reasons.append(f"fee_pass_rate {fee_pass_rate:.1%} < {_ADVANCE_FEE_PASS:.0%} advance threshold")
        if direction_match_rate < _ADVANCE_DIRECTION_MATCH:
            refine_reasons.append(f"direction_match {direction_match_rate:.1%} < {_ADVANCE_DIRECTION_MATCH:.0%} advance threshold")
        if burst_1h_max > _BURST_RATE_CEIL:
            refine_reasons.append(f"burst_1h {burst_1h_max:.1%} > {_BURST_RATE_CEIL:.0%} (temporal clustering)")
        if len(by_symbol) < 2:
            refine_reasons.append("permits on single symbol only — generalizability unproven")

        verdict = REFINE if refine_reasons else ADVANCE

    return ClassScore(
        setup_class=setup_class,
        verdict=verdict,
        total_evals=total_evals,
        permit_count=permit_count,
        permit_rate=permit_rate,
        fee_pass_count=fee_pass_count,
        fee_deny_count=fee_deny_count,
        fee_pass_rate=fee_pass_rate,
        direction_match_count=direction_match_count,
        direction_mismatch_count=direction_mismatch_count,
        direction_match_rate=direction_match_rate,
        by_symbol=by_symbol,
        max_symbol_share=max_symbol_share,
        top_symbol=top_symbol,
        by_regime=by_regime,
        burst_1h_max_rate=burst_1h_max,
        max_consecutive=max_consec,
        mean_return_per_permit=mean_return,
        return_sample_count=return_sample_count,
        kill_reasons=tuple(kill_reasons),
        refine_reasons=tuple(refine_reasons),
    )


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------
def _compute_max_window_rate(
    permit_ts: List[float],
    all_records: List[Dict[str, Any]],
    window_s: float,
) -> float:
    """Peak permit density within rolling windows of `window_s` seconds."""
    if not permit_ts or not all_records:
        return 0.0
    all_ts = sorted(r.get("ts", 0.0) for r in all_records)
    if not all_ts:
        return 0.0
    max_rate = 0.0
    for start_ts in permit_ts:
        end_ts = start_ts + window_s
        window_total = sum(1 for t in all_ts if start_ts <= t < end_ts)
        window_permits = sum(1 for t in permit_ts if start_ts <= t < end_ts)
        if window_total > 0:
            rate = window_permits / window_total
            if rate > max_rate:
                max_rate = rate
    return max_rate


def _compute_max_consecutive(
    records: List[Dict[str, Any]],
    setup_class: str,
) -> int:
    """Longest run of consecutive permits for this class in timestamp order."""
    sorted_recs = sorted(records, key=lambda r: r.get("ts", 0.0))
    max_run = 0
    current_run = 0
    for r in sorted_recs:
        if (r.get("setup_class") == setup_class
                and r.get("verdict") == "PERMIT_CANDIDATE"):
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Main scorecard computation
# ---------------------------------------------------------------------------
def compute_scorecard(
    log_path: Optional[Path] = None,
    *,
    min_samples: int = _MIN_SAMPLES,
    classes: Optional[frozenset] = None,
) -> ScorecardReport:
    """
    Evaluate all hypothesis classes against the fixed rubric.

    Pure read-only. Does not modify any file or state.
    """
    path = log_path or Path("logs/execution/futures_permit_surface_shadow.jsonl")
    target_classes = classes or _HYPOTHESIS_CLASSES
    records = _load_shadow_records(path)

    if not records:
        return ScorecardReport(
            total_records=0,
            class_scores={},
            summary_verdict=INSUFFICIENT_DATA,
            alerts=["no_data: shadow log empty or missing"],
        )

    scores: Dict[str, ClassScore] = {}
    for cls in sorted(target_classes):
        scores[cls] = _extract_class_metrics(
            cls, records, min_samples=min_samples,
        )

    # Summary verdict = worst across all classes
    verdict_priority = {KILL: 0, REFINE: 1, INSUFFICIENT_DATA: 2, ADVANCE: 3}
    worst = max(scores.values(), key=lambda s: -verdict_priority.get(s.verdict, 99))
    summary = worst.verdict

    alerts: List[str] = []
    for cls, score in scores.items():
        if score.verdict == KILL:
            alerts.append(f"KILL {cls}: {', '.join(score.kill_reasons)}")
        elif score.verdict == REFINE:
            alerts.append(f"REFINE {cls}: {', '.join(score.refine_reasons)}")
        elif score.verdict == INSUFFICIENT_DATA:
            alerts.append(f"INSUFFICIENT_DATA {cls}: {score.permit_count} permits")

    return ScorecardReport(
        total_records=len(records),
        class_scores=scores,
        summary_verdict=summary,
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_scorecard(report: ScorecardReport) -> None:
    """Print human-readable scorecard to stdout."""
    print("=" * 72)
    print("FPS v1 CLASS EVALUATION SCORECARD")
    print(f"Total shadow records: {report.total_records}")
    print(f"Summary verdict: {report.summary_verdict}")
    print("=" * 72)

    if not report.class_scores:
        print("\nNo data. Shadow log is empty or missing.")
        return

    for cls in sorted(report.class_scores):
        s = report.class_scores[cls]
        print(f"\n{'─' * 72}")
        print(f"  {s.setup_class}    [{s.verdict}]")
        print(f"{'─' * 72}")
        print(f"  Permits:           {s.permit_count:>6d} / {s.total_evals} evals  ({s.permit_rate:.2%})")
        print(f"  Fee pass rate:     {s.fee_pass_count:>6d} / {s.fee_pass_count + s.fee_deny_count}          ({s.fee_pass_rate:.2%})")
        print(f"  Direction match:   {s.direction_match_count:>6d} / {s.direction_match_count + s.direction_mismatch_count}          ({s.direction_match_rate:.2%})")
        print(f"  Symbol concentration: {s.max_symbol_share:.1%} on {s.top_symbol}")
        print(f"  Burst (1h peak):   {s.burst_1h_max_rate:.1%}")
        print(f"  Max consecutive:   {s.max_consecutive}")
        if s.mean_return_per_permit is not None:
            print(f"  Mean return/permit: {s.mean_return_per_permit:+.4%}  (n={s.return_sample_count})")
        else:
            print("  Mean return/permit: N/A  (no realized_return data yet)")

        if s.by_symbol:
            print(f"  By symbol:         {dict(sorted(s.by_symbol.items(), key=lambda x: -x[1]))}")
        if s.by_regime:
            print(f"  By regime:         {dict(sorted(s.by_regime.items(), key=lambda x: -x[1]))}")

        if s.kill_reasons:
            print("  KILL reasons:")
            for r in s.kill_reasons:
                print(f"    ✗ {r}")
        if s.refine_reasons:
            print("  REFINE reasons:")
            for r in s.refine_reasons:
                print(f"    ⚠ {r}")

    if report.alerts:
        print(f"\n{'=' * 72}")
        print("ALERTS:")
        for a in report.alerts:
            print(f"  → {a}")

    print(f"\n{'=' * 72}")
    print("Rubric thresholds (fixed, not tunable per class):")
    print(f"  min_samples:            {_MIN_SAMPLES}")
    print(f"  permit_rate:            [{_PERMIT_RATE_FLOOR:.0%}, {_PERMIT_RATE_CEIL:.0%}]")
    print(f"  fee_pass (kill/adv):    {_FEE_PASS_RATE_FLOOR:.0%} / {_ADVANCE_FEE_PASS:.0%}")
    print(f"  direction (kill/adv):   {_DIRECTION_MATCH_FLOOR:.0%} / {_ADVANCE_DIRECTION_MATCH:.0%}")
    print(f"  symbol_concentration:   < {_SYMBOL_CONCENTRATION_CEIL:.0%}")
    print(f"  burst_1h:               < {_BURST_RATE_CEIL:.0%}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FPS v1 class evaluation scorecard")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path to shadow JSONL (default: logs/execution/futures_permit_surface_shadow.jsonl)")
    parser.add_argument("--min-samples", type=int, default=_MIN_SAMPLES,
                        help=f"Minimum permits for meaningful verdict (default: {_MIN_SAMPLES})")
    args = parser.parse_args()

    report = compute_scorecard(log_path=args.log, min_samples=args.min_samples)
    print_scorecard(report)

    # Exit code: 0=all advance, 1=refine/insufficient, 2=any kill
    if report.summary_verdict == KILL:
        sys.exit(2)
    elif report.summary_verdict in (REFINE, INSUFFICIENT_DATA):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
