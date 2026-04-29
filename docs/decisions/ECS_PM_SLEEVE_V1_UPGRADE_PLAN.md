# ECS → PM Sleeve v1 Unified Upgrade Plan

**Decision ID:** `DEC_ECS_PM_SLEEVE_UPGRADE_V1`
**Date:** 2026-04-07
**Status:** PROPOSED
**Authority:** Orchestrator synthesis of Research, Modeling, Execution, and Audit agents.
**Dependencies:** `DEC_PM_SLEEVE_V1`, `DEC_ECS_PROFIT_MASK_V2`

---

## Executive Summary

This document unifies four agent outputs into a single dependency-ordered upgrade plan for deploying Candidate D to live shadow, concurrent with PM Sleeve v1 activation. It concludes with a go/no-go decision.

---

## 1. Research Agent — Structural Failures Identified

### 1.1 Failure Inventory

| ID | Failure | Severity | Evidence |
|----|---------|----------|----------|
| **F-1** | `_selector_d()` does not explicitly guard zero-score inputs | MEDIUM | `hydra_score=0.0` silently evaluates via `in_profit_region()` → correct result (False) but no telemetry tracking, violating ZERO_SCORE_POLICY |
| **F-2** | `in_profit_region()` uses V1 mask (`PNL_POSITIVE_REGIONS`) while V2 mask (`PNL_POSITIVE_REGIONS_V2`) is declared but unused | HIGH | ETH mask `(0.4393, 0.511)` remains active in V1; V2 spec discards it (unstable, overlap < 0.70) |
| **F-3** | Shadow log schema mismatch: on-disk events use `selector_v2_shadow_v1`, code emits `v2`, spec requires `v3` | HIGH | `logs/execution/selector_v2_shadow.jsonl` contains only v1 events. `SELECTOR_V2_SHADOW` env flag likely unset in production → Candidate D never evaluated in live soak |
| **F-4** | No test coverage for zero-score routing policy | MEDIUM | `tests/unit/test_shadow_selector_v2.py` has `TestSelectorD` but no `test_zero_score_*` cases |
| **F-5** | Schema v3 fields missing from `evaluate_v2_shadow()`: `d_mask_boundaries`, `d_regime_boundary`, `d_zero_score` | MEDIUM | Code emits v2 schema without lineage metadata needed for offline mask drift detection |
| **F-6** | No monotonicity guard in `_selector_d()` for mask/boundary contradiction (INV-5: mask_hi ≤ regime_boundary) | LOW | Currently correct by construction (0.4953 < 0.5369), but no runtime assertion prevents future regression |

### 1.2 Testable Hypotheses

| # | Hypothesis | Test | Pass Criterion |
|---|-----------|------|-----------------|
| H-1 | Zero-score episodes are routed to Legacy, never to Hydra | `_selector_d("BTCUSDT", 0.0)` returns `abstain=True` | ✅ (already true via `in_profit_region` returning False, but untested + unlogged) |
| H-2 | V2 mask discards ETH/SOL → all ETH/SOL scores produce ABSTAIN | `in_profit_region("ETHUSDT", x)` returns False ∀ x | ❌ (currently True for x ∈ [0.4393, 0.511] under V1) |
| H-3 | Shadow events include Candidate D fields when `SELECTOR_V2_SHADOW=1` | Parse shadow log → all events have `d_choice`, `d_rule` | ❌ (on-disk log has only v1 events) |
| H-4 | BTC mask subsample overlap remains ≥ 0.80 on current data | `ecs_profit_mask.py --salvageability` | Untested on latest data |
| H-5 | Spearman ρ > 0.15 for BTC score-PnL relationship | `compute_monotonicity(btc_episodes)` | Untested on latest data |

---

## 2. Modeling Agent — Boundary & Mask Rebuild

### 2.1 Mask State Matrix

| Symbol | V1 Mask | V2 Mask | Action | Justification |
|--------|---------|---------|--------|---------------|
| BTCUSDT | (0.4197, 0.4953) | (0.4197, 0.4953) | **RETAIN** | Stable (overlap 0.889 ≥ 0.70) |
| ETHUSDT | (0.4393, 0.5110) | [] (empty) | **DISCARD** | Unstable (overlap < 0.70); subsample boundaries shift significantly |
| SOLUSDT | [] (empty) | [] (empty) | **NO MASK** | No profitable region found; Legacy-dominant (298 cases, 0% Hydra wins) |

### 2.2 Boundary State Matrix

| Symbol | V1 Boundary | V2 Boundary | Status |
|--------|-------------|-------------|--------|
| BTCUSDT | [0.5236] | [0.4197, 0.5369] | Ready (monotonicity-constrained) |
| ETHUSDT | [0.4291, 0.4883] | [0.4291, 0.4883] | Unchanged (4-boundary deferred pending monotonicity) |
| SOLUSDT | [] | [] | Unchanged (pending monotonicity) |

### 2.3 ZERO_SCORE Rules (Deterministic)

```
IF score == 0 OR score IS NULL:
    → All candidates ABSTAIN
    → Route to Legacy (ECS fallback)
    → Log: d_zero_score=True, rule="D_zero_score"
    → Do NOT re-score, do NOT route to Hydra
```

Root causes: missing Hydra intent (common), score computation failure (rare), cold start (rare).

### 2.4 Required Constant Changes

```python
# shadow_selector_v2.py — switch active routing to V2

# CHANGE 1: in_profit_region() must use V2 masks
# Before: PNL_POSITIVE_REGIONS (ETH has mask)
# After:  PNL_POSITIVE_REGIONS_V2 (ETH empty)

# CHANGE 2: classify_regime() V2 variant for BTC 3-regime model
# Use REGIME_BOUNDARIES_V2 + REGIME_LABELS_V2

# CHANGE 3: PROFIT_MASK_VERSION bump
# "2026-03-17_v1" → "2026-04-07_v2"
```

---

## 3. Execution Agent — Gating & Confidence Rules

### 3.1 Code Changes (Dependency-Ordered)

#### Change 1: Switch `in_profit_region()` to V2 masks
**File:** `execution/shadow_selector_v2.py`
**Lines:** ~function at line 119
**Risk:** LOW (ETH/SOL become more conservative = ABSTAIN more)

```python
# BEFORE
def in_profit_region(symbol: str, hydra_score: float) -> bool:
    bands = PNL_POSITIVE_REGIONS.get(symbol, [])
    return any(lo <= hydra_score <= hi for lo, hi in bands)

# AFTER
def in_profit_region(symbol: str, hydra_score: float) -> bool:
    bands = PNL_POSITIVE_REGIONS_V2.get(symbol, [])
    return any(lo <= hydra_score <= hi for lo, hi in bands)
```

#### Change 2: Add explicit zero-score guard to `_selector_d()`
**File:** `execution/shadow_selector_v2.py`
**Lines:** ~172-189
**Risk:** LOW (zero-score already produces ABSTAIN via region check; this adds explicit tracking)

```python
def _selector_d(symbol: str, hydra_score: float) -> Dict[str, Any]:
    # ZERO_SCORE deterministic policy
    if hydra_score is None or hydra_score <= 0:
        return {
            "v2_choice": "none",
            "v2_abstain": True,
            "rule": "D_zero_score",
            "d_zero_score": True,
        }
    if in_profit_region(symbol, hydra_score):
        return {
            "v2_choice": "hydra",
            "v2_abstain": False,
            "rule": "D_profit_region",
            "d_zero_score": False,
        }
    return {
        "v2_choice": "none",
        "v2_abstain": True,
        "rule": "D_abstain",
        "d_zero_score": False,
    }
```

#### Change 3: Upgrade shadow event schema to v3
**File:** `execution/shadow_selector_v2.py`
**Lines:** ~240-268 (`evaluate_v2_shadow()`)
**Risk:** LOW (additive fields only; no downstream consumers parse programmatically)

Add to event dict:
```python
"schema": "selector_v2_shadow_v3",
# New fields:
"d_zero_score": result_d.get("d_zero_score", False),
"d_mask_boundaries": list(PNL_POSITIVE_REGIONS_V2.get(symbol, [])),
"d_regime_boundary": REGIME_BOUNDARIES_V2.get(symbol, [None])[-1],
"profit_mask_version": PROFIT_MASK_VERSION_V2,  # replaces V1 reference
```

#### Change 4: Add zero-score tests
**File:** `tests/unit/test_shadow_selector_v2.py`
**Risk:** NONE (additive test coverage)

```python
class TestZeroScore:
    def test_zero_score_abstains(self):
        r = _selector_d("BTCUSDT", 0.0)
        assert r["v2_abstain"] is True
        assert r["rule"] == "D_zero_score"
        assert r["d_zero_score"] is True

    def test_none_score_abstains(self):
        r = _selector_d("BTCUSDT", None)
        assert r["v2_abstain"] is True
        assert r["rule"] == "D_zero_score"

    def test_negative_score_abstains(self):
        r = _selector_d("BTCUSDT", -0.1)
        assert r["v2_abstain"] is True
        assert r["rule"] == "D_zero_score"
```

#### Change 5: Add INV-5 runtime assertion
**File:** `execution/shadow_selector_v2.py`
**Lines:** module-level, after constant definitions
**Risk:** NONE (static assertion at import time)

```python
# INV-5: mask upper bound must not exceed regime boundary
for _sym, _bands in PNL_POSITIVE_REGIONS_V2.items():
    _regime_bounds = REGIME_BOUNDARIES_V2.get(_sym, [])
    if _bands and _regime_bounds:
        _mask_hi = max(hi for _, hi in _bands)
        _regime_hi = _regime_bounds[-1]
        assert _mask_hi <= _regime_hi, (
            f"INV-5 violation: {_sym} mask_hi={_mask_hi} > regime_hi={_regime_hi}"
        )
```

### 3.2 PM Sleeve v1 — No Code Changes Required

The PM Sleeve signal extraction (`extract_pm_sleeve_signal()`, `check_pm_sleeve_eligibility()`) and shadow wiring are **fully implemented and tested**. The PM Sleeve v1 spec is already binding and the config is enabled (`pm_sleeve_v1.enabled: true`).

**Activation dependency:** Binary Lab S2 state must transition out of `DISABLED` (requires config hash fix or fresh activation window).

### 3.3 Dependency Graph

```
Change 1 (V2 masks)  ──────────────────────────┐
                                                 │
Change 2 (zero-score guard) ────────────────────┤
                                                 ├→ Change 3 (schema v3)
Change 5 (INV-5 assertion) ─────────────────────┘         │
                                                            ↓
                                              Change 4 (tests) ← validates all above
                                                            ↓
                                              Deploy: SELECTOR_V2_SHADOW=1
```

All changes are **additive and observation-only**. No execution gating is modified.

---

## 4. Audit Agent — Reproducibility & Drift

### 4.1 Reproducibility Verification

| Check | Status | Evidence |
|-------|--------|----------|
| `_selector_d()` is deterministic | ✅ PASS | Pure function of (symbol, hydra_score); no randomness, no external state |
| `in_profit_region()` is deterministic | ✅ PASS | Lookup against frozen constant; no mutation |
| `classify_regime()` is deterministic | ✅ PASS | Threshold comparison against frozen boundaries |
| Shadow events are append-only | ✅ PASS | `_append_v2_event()` opens file in `"a"` mode |
| Test matrix covers all boundary conditions | ⚠️ GAP | Zero-score, None score, negative score untested |
| PM Sleeve signal extraction is reproducible | ✅ PASS | `BinaryLabS2Signal` is frozen dataclass; all gates are threshold comparisons |
| Config freeze discipline maintained | ✅ PASS | `binary_lab_limits_s2.json` has `freeze_rules` block; no mid-experiment changes |

### 4.2 Drift Detection Requirements

| Drift Type | Detection Method | Frequency | Alert Threshold |
|-----------|-----------------|-----------|-----------------|
| **Mask drift** | BTC subsample overlap via `ecs_profit_mask.py` | Weekly | < 0.70 (UNSTABLE) |
| **Monotonicity drift** | BTC Spearman ρ via `compute_monotonicity()` | Weekly | < 0.15 (FLAT) or < 0 (INVERTED) |
| **Score distribution shift** | D_profit_region verdict count/day | Daily | < 5/day for 3 consecutive days |
| **ZERO_SCORE spike** | `d_zero_score=True` rate in shadow log | Daily | > 15% of total decisions |
| **Schema version staleness** | `profit_mask_version` in shadow events | Weekly | Version older than 14 days |
| **PM Sleeve region distribution** | Trade distribution by price region | Rolling 50 trades | < 60% in {extreme_low, low, mid_low} |

### 4.3 Known Drift Vectors

1. **BTC regime structure change** — If volatility regime shifts the score distribution, the profit mask may no longer overlap with actual trading scores. Detected by verdict count drop.
2. **Hydra head rebalancing** — Cerberus multiplier changes could shift the score centroid. Detected by mask drift monitor.
3. **Market microstructure change (PM)** — If Polymarket spreads widen or low-price opportunities disappear, PM Sleeve throughput drops. Detected by region distribution KPI.

---

## 5. Dependency-Ordered Change List

| Order | Change | File(s) | Type | Blocks |
|-------|--------|---------|------|--------|
| **1** | Switch `in_profit_region()` to V2 masks | `shadow_selector_v2.py` | Behavior | 3 |
| **2** | Add zero-score guard to `_selector_d()` | `shadow_selector_v2.py` | Behavior | 3 |
| **3** | Upgrade shadow event schema to v3 | `shadow_selector_v2.py` | Observability | 4 |
| **4** | Add zero-score + V2 mask tests | `test_shadow_selector_v2.py` | Test | Deploy |
| **5** | Add INV-5 static assertion | `shadow_selector_v2.py` | Safety | None |
| **6** | Set `SELECTOR_V2_SHADOW=1` in production env | Supervisor config | Deployment | — |
| **7** | Verify shadow log emits v3 events with D fields | Manual / CI | Validation | — |

**PM Sleeve v1 activation** (separate track, no code changes):

| Order | Step | Dependency |
|-------|------|------------|
| **A** | Fix config hash (fresh activation window) | None |
| **B** | Set `BINARY_LAB_LIMITS_HASH` env var | A |
| **C** | Set `PREDICTION_PHASE=P2_PRODUCTION` | A |
| **D** | Verify PM Sleeve signal extraction in shadow | B, C |
| **E** | Monitor region distribution baseline (30-day freeze) | D |

---

## 6. Go/No-Go Decision: Candidate D Live Shadow

### Decision Matrix

| Criterion | Status | Verdict |
|-----------|--------|---------|
| **Code completeness** | 5 changes identified, all additive and observation-only | ✅ GO |
| **Test coverage** | Existing tests pass; 4 new tests needed (Change 4) | ✅ GO (after Change 4) |
| **Risk to execution** | Zero — Candidate D is shadow-only (`SELECTOR_V2_SHADOW=1`), never gates execution | ✅ GO |
| **Risk to PM Sleeve** | Zero — PM Sleeve v1 is independent (different code path, different config block) | ✅ GO |
| **Mask stability (BTC)** | Overlap 0.889 ≥ 0.80 threshold | ✅ GO |
| **ETH/SOL mask** | Correctly discarded (V2 empty) | ✅ GO |
| **ZERO_SCORE policy** | Defined in spec; needs code implementation (Change 2) | ✅ GO (after Change 2) |
| **Schema lineage** | v3 schema defined; needs implementation (Change 3) | ✅ GO (after Change 3) |
| **Shadow log baseline** | Current log is v1; will be superseded by v3 after deploy | ✅ GO |
| **Promotion criteria defined** | 7 criteria in `DEC_ECS_PROFIT_MASK_V2` §5 | ✅ GO |
| **Rollback path** | Set `SELECTOR_V2_SHADOW=0` → instant disable, zero execution impact | ✅ GO |

### Verdict

> **GO — Deploy Candidate D to live shadow.**
>
> **Conditions:**
> 1. All 5 code changes (§5, items 1-5) merged and tested.
> 2. `pytest -q` green (full suite).
> 3. `SELECTOR_V2_SHADOW=1` set in production supervisor.
> 4. First v3 shadow event confirmed in log within 1 hour of deploy.
>
> **Candidate D remains observation-only.** Promotion to live routing requires meeting ALL 7 criteria in `DEC_ECS_PROFIT_MASK_V2` §5 (≥200 verdicts, favorable counterfactual PnL, mask stability, monotonicity, Q5-Q1 spread, zero-score rate < 15%, no ETH/SOL regression).

### Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Code change breaks existing tests | LOW | MEDIUM | Run full `pytest -q` before merge |
| Shadow logging fails silently | LOW | LOW | Verify v3 event in log post-deploy |
| Mask drifts before promotion decision | MEDIUM | LOW | Weekly drift monitoring (§4.2) |
| PM Sleeve activation blocked by hash | HIGH | NONE (to Candidate D) | Separate activation track |

---

**Signed:** Orchestrator
**Effective:** Upon merge of Changes 1-5
**Review cycle:** Weekly (drift checks) + 200-verdict checkpoint (promotion gate)
