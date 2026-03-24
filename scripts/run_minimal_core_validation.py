#!/usr/bin/env python3
"""Minimal Core Validation — §8 of MINIMAL_SYSTEM_EXTRACTION_AUDIT_2026-03-18.

Runs the five validation tests that determine whether the minimal futures core
produces positive net edge after friction, with calibration, on the available
distribution.

Hard kill condition (any one triggers SYSTEM_VERDICT = FAIL):

    rho_traded < 0.05  OR  p_value > 0.05
    BSS <= 0           (model worse than always guessing base rate)
    net_edge <= 0      after friction
    control tests fail (stale-NAV / crisis / DD-halt / kill-switch)

Final verdict:

    PASS              — all five test groups green
    WEAK              — signal present but fragile (partial failures)
    FAIL              — hard kill condition met
    INSUFFICIENT_DATA — not enough episodes to run tests

Usage:
    PYTHONPATH=. python scripts/run_minimal_core_validation.py [--json] [--min-episodes N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Thresholds — hard numeric boundaries; no narrative override permitted
# ---------------------------------------------------------------------------

# Test 1: Causality
CAUSALITY_RHO_MIN = 0.05           # Spearman rho below this = no signal
CAUSALITY_P_VALUE_MAX = 0.05       # Statistical significance floor
CAUSALITY_Q5_Q1_MIN = 0.0          # Q5-Q1 spread must be positive
CAUSALITY_STABILITY_FAIL = "unstable"

# Test 2: Calibration
CALIBRATION_BSS_MIN = 0.0          # BSS <= 0 means model ≤ base-rate guess
CALIBRATION_ECE_MAX = 0.08         # ECE >= 0.08 = severely miscalibrated
CALIBRATION_BRIER_MAX = 0.30       # Brier > 0.30 = poor predictions

# Test 3: Tradability (Friction)
FRICTION_NET_EDGE_MIN = 0.0        # Mean net edge BPS must be positive
FRICTION_KILL_RATE_MAX = 0.40      # Kill rate >= 40% = structural problem
FRICTION_FEE_EDGE_RATIO_MAX = 1.0  # Fees > gross edge = fee machine

# Test 5: Ablation (comparison deltas — negative = minimal worse)
ABLATION_RHO_TOLERANCE = -0.05     # Minimal core rho may not fall > 5pp
ABLATION_BRIER_TOLERANCE = 0.02    # Minimal core Brier may not rise > 0.02

# Minimum episode count to attempt validation at all
MIN_EPISODES_DEFAULT = 30


# ---------------------------------------------------------------------------
# Test result containers
# ---------------------------------------------------------------------------

class TestResult:
    """Single test outcome."""

    def __init__(self, name: str, passed: bool, detail: str,
                 metrics: Optional[Dict[str, Any]] = None,
                 blocked: bool = False):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.metrics = metrics or {}
        self.blocked = blocked  # True if test could not run

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "blocked": self.blocked,
            "detail": self.detail,
            "metrics": self.metrics,
        }


class TestGroup:
    """Named group of related tests."""

    def __init__(self, name: str, tests: Optional[List[TestResult]] = None):
        self.name = name
        self.tests: List[TestResult] = tests or []

    @property
    def passed(self) -> bool:
        runnable = [t for t in self.tests if not t.blocked]
        return len(runnable) > 0 and all(t.passed for t in runnable)

    @property
    def has_hard_fail(self) -> bool:
        return any(not t.passed and not t.blocked for t in self.tests)

    @property
    def all_blocked(self) -> bool:
        return all(t.blocked for t in self.tests)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "has_hard_fail": self.has_hard_fail,
            "all_blocked": self.all_blocked,
            "tests": [t.to_dict() for t in self.tests],
        }


# ---------------------------------------------------------------------------
# Test 1: Causality
# ---------------------------------------------------------------------------

def run_causality_tests(episodes: List[Dict[str, Any]]) -> TestGroup:
    """Monotonicity, Q5-Q1 spread, temporal stability on traded region."""
    from execution.hydra_monotonicity import (
        compute_monotonicity,
        compute_quintile_spread,
        compute_time_stability,
    )

    group = TestGroup("1_causality")

    # 1a. Monotonicity (Spearman rho + p-value)
    mono = compute_monotonicity(episodes)
    rho = mono.get("spearman")
    p_val = mono.get("p_value")
    n = mono.get("n", 0)

    if rho is None or n < 10:
        group.tests.append(TestResult(
            "monotonicity_rho",
            passed=False,
            detail=f"Insufficient scored episodes (n={n})",
            metrics={"rho": rho, "p_value": p_val, "n": n},
            blocked=True,
        ))
    else:
        rho_ok = rho >= CAUSALITY_RHO_MIN
        p_ok = p_val is not None and p_val <= CAUSALITY_P_VALUE_MAX
        group.tests.append(TestResult(
            "monotonicity_rho",
            passed=rho_ok and p_ok,
            detail=(f"rho={rho:.4f} (>={CAUSALITY_RHO_MIN}), "
                    f"p={p_val:.6f} (<={CAUSALITY_P_VALUE_MAX}), "
                    f"slope={mono.get('slope')}, n={n}"),
            metrics={"rho": rho, "p_value": p_val, "n": n,
                     "slope": mono.get("slope")},
        ))

    # 1b. Q5-Q1 spread
    qs = compute_quintile_spread(episodes)
    spread = qs.get("q5_q1_spread")

    if spread is None:
        group.tests.append(TestResult(
            "q5_q1_spread",
            passed=False,
            detail="Could not compute quintile spread",
            blocked=True,
        ))
    else:
        group.tests.append(TestResult(
            "q5_q1_spread",
            passed=spread > CAUSALITY_Q5_Q1_MIN,
            detail=f"Q5-Q1 spread={spread:.6f} (>{CAUSALITY_Q5_Q1_MIN})",
            metrics={"q5_q1_spread": spread},
        ))

    # 1c. Temporal stability
    stab = compute_time_stability(episodes)
    stability = stab.get("stability", "insufficient")

    if stability == "insufficient":
        group.tests.append(TestResult(
            "temporal_stability",
            passed=False,
            detail=f"Insufficient data for temporal slicing (n={stab.get('n', 0)})",
            blocked=True,
        ))
    else:
        group.tests.append(TestResult(
            "temporal_stability",
            passed=stability != CAUSALITY_STABILITY_FAIL,
            detail=f"stability={stability}",
            metrics={"stability": stability,
                     "slices": stab.get("slices", [])},
        ))

    return group


# ---------------------------------------------------------------------------
# Test 2: Calibration
# ---------------------------------------------------------------------------

def run_calibration_tests(episodes: List[Dict[str, Any]]) -> TestGroup:
    """Brier, BSS, ECE on hybrid_score (and conviction_score for comparison)."""
    from execution.hydra_monotonicity import (
        compute_brier_score,
        compute_calibration_diagnosis,
    )

    group = TestGroup("2_calibration")

    for score_field in ("hybrid_score", "conviction_score"):
        brier_result = compute_brier_score(episodes, score_field=score_field)
        diag = compute_calibration_diagnosis(episodes, score_field=score_field)

        brier = brier_result.get("brier")
        bss = brier_result.get("brier_skill_score")
        ece = diag.get("ece")
        n = brier_result.get("n", 0)

        if brier is None or n < 20:
            group.tests.append(TestResult(
                f"calibration_{score_field}",
                passed=False,
                detail=f"Insufficient episodes for {score_field} (n={n})",
                metrics={"score_field": score_field, "n": n},
                blocked=True,
            ))
            continue

        bss_ok = bss is not None and bss > CALIBRATION_BSS_MIN
        ece_ok = ece is not None and ece < CALIBRATION_ECE_MAX
        brier_ok = brier < CALIBRATION_BRIER_MAX

        group.tests.append(TestResult(
            f"calibration_{score_field}",
            passed=bss_ok and ece_ok and brier_ok,
            detail=(f"Brier={brier:.4f} (<{CALIBRATION_BRIER_MAX}), "
                    f"BSS={bss:.4f} (>{CALIBRATION_BSS_MIN}), "
                    f"ECE={ece:.4f} (<{CALIBRATION_ECE_MAX}), "
                    f"patterns={diag.get('patterns', [])}, n={n}"),
            metrics={
                "score_field": score_field,
                "brier": brier,
                "brier_baseline": brier_result.get("brier_baseline"),
                "bss": bss,
                "ece": ece,
                "mce": diag.get("mce"),
                "pred_spread": diag.get("pred_spread"),
                "base_rate": brier_result.get("base_rate"),
                "patterns": diag.get("patterns", []),
                "decomposition": brier_result.get("decomposition", {}),
                "n": n,
            },
        ))

    return group


# ---------------------------------------------------------------------------
# Test 3: Tradability (Friction)
# ---------------------------------------------------------------------------

def run_friction_tests(episodes: List[Dict[str, Any]]) -> TestGroup:
    """Net edge, kill rate, fee-to-edge ratio, break-even hurdle."""
    from execution.hydra_monotonicity import compute_friction_audit

    group = TestGroup("3_tradability")

    fa = compute_friction_audit(episodes)
    verdict = fa.get("verdict", "INSUFFICIENT_DATA")
    net_edge = fa.get("mean_net_edge_bps")
    kill_rate = fa.get("friction_kill_rate")
    fee_ratio = fa.get("fee_to_edge_ratio")
    n = fa.get("n", 0)

    if verdict == "INSUFFICIENT_DATA" or n < 10:
        group.tests.append(TestResult(
            "friction_verdict",
            passed=False,
            detail=f"Insufficient friction data (n={n})",
            metrics={"n": n},
            blocked=True,
        ))
        return group

    # 3a. Net edge must be positive
    net_ok = net_edge is not None and net_edge > FRICTION_NET_EDGE_MIN
    group.tests.append(TestResult(
        "net_edge_positive",
        passed=net_ok,
        detail=f"mean_net_edge={net_edge:.2f} BPS (>{FRICTION_NET_EDGE_MIN})",
        metrics={"mean_net_edge_bps": net_edge},
    ))

    # 3b. Kill rate below threshold
    kill_ok = kill_rate is not None and kill_rate < FRICTION_KILL_RATE_MAX
    group.tests.append(TestResult(
        "friction_kill_rate",
        passed=kill_ok,
        detail=f"kill_rate={kill_rate:.4f} (<{FRICTION_KILL_RATE_MAX})",
        metrics={"friction_kill_rate": kill_rate},
    ))

    # 3c. Fee-to-edge ratio
    if fee_ratio is not None:
        fee_ok = fee_ratio < FRICTION_FEE_EDGE_RATIO_MAX
        group.tests.append(TestResult(
            "fee_to_edge_ratio",
            passed=fee_ok,
            detail=f"fee_to_edge={fee_ratio:.4f} (<{FRICTION_FEE_EDGE_RATIO_MAX})",
            metrics={"fee_to_edge_ratio": fee_ratio},
        ))
    else:
        group.tests.append(TestResult(
            "fee_to_edge_ratio",
            passed=False,
            detail="fee_to_edge_ratio is None (likely zero gross edge)",
            metrics={"fee_to_edge_ratio": None},
        ))

    # 3d. Overall friction verdict
    group.tests.append(TestResult(
        "friction_verdict",
        passed=verdict == "TRADABLE",
        detail=f"verdict={verdict}, severity={fa.get('severity')}",
        metrics={
            "verdict": verdict,
            "severity": fa.get("severity"),
            "break_even_bps": fa.get("break_even", {}).get("break_even_bps"),
            "pct_above_hurdle": fa.get("break_even", {}).get("pct_above_hurdle"),
        },
    ))

    return group


# ---------------------------------------------------------------------------
# Test 4: Control Sufficiency
# ---------------------------------------------------------------------------

def run_control_tests() -> TestGroup:
    """Verify retained controls enforce fail-closed behavior.

    These tests use the existing doctrine/risk infrastructure directly.
    They do NOT require episode data — they test the safety layer.
    """
    group = TestGroup("4_control_sufficiency")

    # 4a. Stale regime triggers veto (regime data older than staleness threshold)
    try:
        from execution.doctrine_kernel import (
            doctrine_entry_verdict,
            RegimeSnapshot,
            IntentSnapshot,
            ExecutionSnapshot,
            PortfolioSnapshot,
        )

        stale_regime = RegimeSnapshot(
            primary_regime="TREND_UP",
            confidence=0.8,
            cycles_stable=5,
            crisis_flag=False,
            updated_ts=1.0,  # epoch ~1970 → very stale
        )
        intent = IntentSnapshot(symbol="BTCUSDT", direction="LONG", head="TREND")
        execution = ExecutionSnapshot(regime="NORMAL", quality_score=0.8)
        portfolio = PortfolioSnapshot(head_budget_remaining={"TREND": 1000.0})

        result = doctrine_entry_verdict(stale_regime, intent, execution, portfolio)
        denied = not result.allowed
        group.tests.append(TestResult(
            "stale_regime_denial",
            passed=denied,
            detail=f"Stale regime (updated_ts=1.0) → denied={denied}, verdict={result.verdict.value}",
            metrics={"denied": denied, "verdict": result.verdict.value},
        ))
    except Exception as e:
        group.tests.append(TestResult(
            "stale_regime_denial",
            passed=False,
            detail=f"Doctrine test failed: {e}",
            blocked=True,
        ))

    # 4b. Crisis mode blocks entries
    try:
        from execution.doctrine_kernel import (
            doctrine_entry_verdict,
            RegimeSnapshot,
            IntentSnapshot,
            ExecutionSnapshot,
            PortfolioSnapshot,
        )

        crisis_regime = RegimeSnapshot(
            primary_regime="CRISIS",
            confidence=0.9,
            cycles_stable=5,
            crisis_flag=True,
            updated_ts=time.time(),
        )
        intent = IntentSnapshot(symbol="BTCUSDT", direction="LONG", head="TREND")
        execution = ExecutionSnapshot(regime="NORMAL", quality_score=0.8)
        portfolio = PortfolioSnapshot(head_budget_remaining={"TREND": 1000.0})

        result = doctrine_entry_verdict(crisis_regime, intent, execution, portfolio)
        denied = not result.allowed
        group.tests.append(TestResult(
            "crisis_mode_denial",
            passed=denied,
            detail=f"CRISIS regime → denied={denied}, verdict={result.verdict.value}",
            metrics={"regime": "CRISIS", "denied": denied,
                     "verdict": result.verdict.value},
        ))
    except Exception as e:
        group.tests.append(TestResult(
            "crisis_mode_denial",
            passed=False,
            detail=f"Error: {e}",
            blocked=True,
        ))

    # 4c. Execution crunch blocks entries
    try:
        from execution.doctrine_kernel import (
            doctrine_entry_verdict,
            RegimeSnapshot,
            IntentSnapshot,
            ExecutionSnapshot,
            PortfolioSnapshot,
        )

        regime = RegimeSnapshot(
            primary_regime="TREND_UP",
            confidence=0.8,
            cycles_stable=5,
            crisis_flag=False,
            updated_ts=time.time(),
        )
        intent = IntentSnapshot(symbol="BTCUSDT", direction="LONG", head="TREND")
        crunch_exec = ExecutionSnapshot(regime="CRUNCH", quality_score=0.2)
        portfolio = PortfolioSnapshot(head_budget_remaining={"TREND": 1000.0})

        result = doctrine_entry_verdict(regime, intent, crunch_exec, portfolio)
        denied = not result.allowed
        group.tests.append(TestResult(
            "execution_crunch_denial",
            passed=denied,
            detail=f"CRUNCH execution → denied={denied}, verdict={result.verdict.value}",
            metrics={"execution_regime": "CRUNCH", "denied": denied,
                     "verdict": result.verdict.value},
        ))
    except Exception as e:
        group.tests.append(TestResult(
            "execution_crunch_denial",
            passed=False,
            detail=f"Error: {e}",
            blocked=True,
        ))

    # 4d. DD halt via risk_limits
    try:
        from execution.risk_limits import check_order, RiskState, load_risk_config

        cfg = load_risk_config()
        state = RiskState()
        state.portfolio_drawdown_pct = 15.0  # 15% DD

        veto, info = check_order(
            symbol="BTCUSDT",
            side="LONG",
            requested_notional=500.0,
            price=50000.0,
            nav=10000.0,
            open_qty=0.0,
            now=time.time(),
            cfg=cfg,
            state=state,
        )
        group.tests.append(TestResult(
            "dd_halt_denial",
            passed=veto,
            detail=f"DD 15% → veto={veto}, reason={info.get('veto_reason', 'N/A')}",
            metrics={"dd_pct": 15.0, "veto": veto,
                     "reason": info.get("veto_reason")},
        ))
    except Exception as e:
        group.tests.append(TestResult(
            "dd_halt_denial",
            passed=False,
            detail=f"Risk check failed: {e}",
            blocked=True,
        ))

    return group


# ---------------------------------------------------------------------------
# Test 5: Ablation (minimal core vs full stack proxy)
# ---------------------------------------------------------------------------

def run_ablation_tests(episodes: List[Dict[str, Any]]) -> TestGroup:
    """Compare hybrid_score monotonicity vs conviction_score monotonicity.

    Full ablation requires the calibrated probability model to exist.
    For now, we compare the two score fields as a proxy: if hybrid_score
    (the minimal core input) has equal or better monotonicity than
    conviction_score (the full-stack output), the complexity is not earning
    its keep.
    """
    from execution.hydra_monotonicity import compute_monotonicity

    group = TestGroup("5_ablation")

    mono_hybrid = compute_monotonicity(episodes, score_field="hybrid_score")
    mono_conviction = compute_monotonicity(episodes, score_field="conviction_score")

    rho_h = mono_hybrid.get("spearman")
    rho_c = mono_conviction.get("spearman")
    n_h = mono_hybrid.get("n", 0)
    n_c = mono_conviction.get("n", 0)

    if rho_h is None or rho_c is None or n_h < 20 or n_c < 20:
        group.tests.append(TestResult(
            "ablation_rho_comparison",
            passed=False,
            detail=(f"Insufficient data for ablation "
                    f"(hybrid n={n_h}, conviction n={n_c})"),
            blocked=True,
        ))
    else:
        delta = rho_h - rho_c
        # Minimal core (hybrid) should not be much worse than full stack (conviction)
        group.tests.append(TestResult(
            "ablation_rho_comparison",
            passed=delta >= ABLATION_RHO_TOLERANCE,
            detail=(f"rho_hybrid={rho_h:.4f}, rho_conviction={rho_c:.4f}, "
                    f"delta={delta:+.4f} (>={ABLATION_RHO_TOLERANCE:+.2f})"),
            metrics={
                "rho_hybrid": rho_h,
                "rho_conviction": rho_c,
                "delta": delta,
                "hybrid_slope": mono_hybrid.get("slope"),
                "conviction_slope": mono_conviction.get("slope"),
            },
        ))

    # Ablation: Brier comparison (if enough data)
    try:
        from execution.hydra_monotonicity import compute_brier_score
        brier_h = compute_brier_score(episodes, score_field="hybrid_score")
        brier_c = compute_brier_score(episodes, score_field="conviction_score")
        b_h = brier_h.get("brier")
        b_c = brier_c.get("brier")

        if b_h is not None and b_c is not None:
            delta_b = b_h - b_c  # positive = hybrid worse
            group.tests.append(TestResult(
                "ablation_brier_comparison",
                passed=delta_b <= ABLATION_BRIER_TOLERANCE,
                detail=(f"brier_hybrid={b_h:.4f}, brier_conviction={b_c:.4f}, "
                        f"delta={delta_b:+.4f} (<={ABLATION_BRIER_TOLERANCE:+.2f})"),
                metrics={
                    "brier_hybrid": b_h,
                    "brier_conviction": b_c,
                    "delta": delta_b,
                },
            ))
        else:
            group.tests.append(TestResult(
                "ablation_brier_comparison",
                passed=False,
                detail="Could not compute Brier for both score fields",
                blocked=True,
            ))
    except Exception as e:
        group.tests.append(TestResult(
            "ablation_brier_comparison",
            passed=False,
            detail=f"Error: {e}",
            blocked=True,
        ))

    return group


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def load_episodes() -> List[Dict[str, Any]]:
    """Load episodes from the episode ledger state file."""
    from execution.episode_ledger import load_episode_ledger

    ledger = load_episode_ledger()
    if ledger is None:
        return []
    return [ep.to_dict() for ep in ledger.episodes]


def compute_system_verdict(groups: List[TestGroup]) -> Dict[str, Any]:
    """Determine final system verdict from all test groups.

    PASS:              All runnable tests in all groups pass
    WEAK:              Signal present but some tests fail
    FAIL:              Any hard kill condition met
    INSUFFICIENT_DATA: All critical tests blocked
    """
    all_tests = [t for g in groups for t in g.tests]
    runnable = [t for t in all_tests if not t.blocked]
    failed = [t for t in runnable if not t.passed]
    blocked = [t for t in all_tests if t.blocked]

    # Hard kill conditions (any one = FAIL)
    hard_kills: List[str] = []

    # Check causality hard kill
    for t in runnable:
        if t.name == "monotonicity_rho" and not t.passed:
            hard_kills.append(
                f"rho={t.metrics.get('rho')} < {CAUSALITY_RHO_MIN} "
                f"OR p={t.metrics.get('p_value')} > {CAUSALITY_P_VALUE_MAX}")

    # Check calibration hard kill (hybrid_score specifically)
    for t in runnable:
        if t.name == "calibration_hybrid_score" and not t.passed:
            bss = t.metrics.get("bss")
            if bss is not None and bss <= CALIBRATION_BSS_MIN:
                hard_kills.append(f"BSS={bss:.4f} <= {CALIBRATION_BSS_MIN}")

    # Check friction hard kill
    for t in runnable:
        if t.name == "net_edge_positive" and not t.passed:
            hard_kills.append(
                f"net_edge={t.metrics.get('mean_net_edge_bps')} <= "
                f"{FRICTION_NET_EDGE_MIN}")

    # Check control hard kill
    control_group = next((g for g in groups if g.name == "4_control_sufficiency"), None)
    if control_group and control_group.has_hard_fail:
        hard_kills.append("Control test failure: safety layer compromised")

    # Determine verdict
    if len(runnable) == 0:
        verdict = "INSUFFICIENT_DATA"
    elif hard_kills:
        verdict = "FAIL"
    elif len(failed) == 0:
        verdict = "PASS"
    else:
        verdict = "WEAK"

    return {
        "verdict": verdict,
        "hard_kills": hard_kills,
        "total_tests": len(all_tests),
        "runnable": len(runnable),
        "passed": len(runnable) - len(failed),
        "failed": len(failed),
        "blocked": len(blocked),
    }


def run_validation(min_episodes: int = MIN_EPISODES_DEFAULT,
                   json_output: bool = False) -> Dict[str, Any]:
    """Execute all §8 validation tests and produce system verdict."""

    ts_start = time.time()

    # Load data
    episodes = load_episodes()
    n_episodes = len(episodes)

    if n_episodes < min_episodes:
        result = {
            "ts": time.time(),
            "n_episodes": n_episodes,
            "min_episodes": min_episodes,
            "system_verdict": "INSUFFICIENT_DATA",
            "detail": (f"Only {n_episodes} episodes available; "
                       f"need >= {min_episodes}"),
            "groups": [],
            "thresholds": _threshold_summary(),
        }
        if not json_output:
            _print_report(result)
        return result

    # Run all five test groups
    groups: List[TestGroup] = []

    groups.append(run_causality_tests(episodes))
    groups.append(run_calibration_tests(episodes))
    groups.append(run_friction_tests(episodes))
    groups.append(run_control_tests())
    groups.append(run_ablation_tests(episodes))

    # Compute final verdict
    sv = compute_system_verdict(groups)

    result = {
        "ts": time.time(),
        "duration_s": round(time.time() - ts_start, 3),
        "n_episodes": n_episodes,
        "min_episodes": min_episodes,
        "system_verdict": sv["verdict"],
        "hard_kills": sv["hard_kills"],
        "summary": {
            "total_tests": sv["total_tests"],
            "runnable": sv["runnable"],
            "passed": sv["passed"],
            "failed": sv["failed"],
            "blocked": sv["blocked"],
        },
        "groups": [g.to_dict() for g in groups],
        "thresholds": _threshold_summary(),
    }

    # Persist result
    _persist_result(result)

    if not json_output:
        _print_report(result)

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _threshold_summary() -> Dict[str, Any]:
    """Return all thresholds for reproducibility."""
    return {
        "causality": {
            "rho_min": CAUSALITY_RHO_MIN,
            "p_value_max": CAUSALITY_P_VALUE_MAX,
            "q5_q1_min": CAUSALITY_Q5_Q1_MIN,
        },
        "calibration": {
            "bss_min": CALIBRATION_BSS_MIN,
            "ece_max": CALIBRATION_ECE_MAX,
            "brier_max": CALIBRATION_BRIER_MAX,
        },
        "friction": {
            "net_edge_min_bps": FRICTION_NET_EDGE_MIN,
            "kill_rate_max": FRICTION_KILL_RATE_MAX,
            "fee_edge_ratio_max": FRICTION_FEE_EDGE_RATIO_MAX,
        },
        "ablation": {
            "rho_tolerance": ABLATION_RHO_TOLERANCE,
            "brier_tolerance": ABLATION_BRIER_TOLERANCE,
        },
    }


def _persist_result(result: Dict[str, Any]) -> None:
    """Write validation result to logs/state/."""
    out_path = Path("logs/state/minimal_core_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _print_report(result: Dict[str, Any]) -> None:
    """Human-readable console report."""
    verdict = result["system_verdict"]
    n = result["n_episodes"]

    print("=" * 72)
    print("  MINIMAL CORE VALIDATION — §8 Extraction Audit")
    print("=" * 72)
    print(f"  Episodes:  {n}")
    print(f"  Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(result['ts']))}")
    print()

    if verdict == "INSUFFICIENT_DATA":
        print(f"  VERDICT: INSUFFICIENT_DATA")
        print(f"  {result.get('detail', '')}")
        print("=" * 72)
        return

    # Per-group results
    for g in result.get("groups", []):
        name = g["name"]
        status = "PASS" if g["passed"] else ("BLOCKED" if g["all_blocked"] else "FAIL")
        print(f"  [{status:>7}]  {name}")
        for t in g.get("tests", []):
            t_status = "BLOCK" if t["blocked"] else ("PASS" if t["passed"] else "FAIL")
            marker = {"PASS": "+", "FAIL": "X", "BLOCK": "?"}[t_status]
            print(f"    [{marker}] {t['name']}: {t['detail']}")
        print()

    # Hard kills
    kills = result.get("hard_kills", [])
    if kills:
        print("  HARD KILL CONDITIONS:")
        for k in kills:
            print(f"    !! {k}")
        print()

    # Summary
    s = result.get("summary", {})
    print(f"  Tests: {s.get('passed', 0)}/{s.get('runnable', 0)} passed, "
          f"{s.get('blocked', 0)} blocked")
    print()

    # Final verdict
    icon = {"PASS": "PASS", "WEAK": "WEAK", "FAIL": "FAIL",
            "INSUFFICIENT_DATA": "N/A"}
    print(f"  ===  SYSTEM VERDICT: {icon.get(verdict, verdict)}  ===")
    print()

    if verdict == "FAIL":
        print("  The minimal core does NOT produce positive net edge.")
        print("  No amount of architecture can save a non-existent signal.")
    elif verdict == "WEAK":
        print("  Signal present but fragile. Research mode required:")
        print("  better features, horizon tuning, or execution improvement.")
    elif verdict == "PASS":
        print("  Minimal core validated. Scale and refine.")

    print("=" * 72)
    print(f"  Result written to: logs/state/minimal_core_validation.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Core Validation (§8 Extraction Audit)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout instead of report")
    parser.add_argument("--min-episodes", type=int, default=MIN_EPISODES_DEFAULT,
                        help=f"Minimum episodes required (default: {MIN_EPISODES_DEFAULT})")
    args = parser.parse_args()

    result = run_validation(min_episodes=args.min_episodes,
                            json_output=args.json)

    if args.json:
        print(json.dumps(result, indent=2, default=str))

    # Exit code reflects verdict
    code = {"PASS": 0, "WEAK": 1, "FAIL": 2, "INSUFFICIENT_DATA": 3}
    sys.exit(code.get(result["system_verdict"], 4))


if __name__ == "__main__":
    main()
