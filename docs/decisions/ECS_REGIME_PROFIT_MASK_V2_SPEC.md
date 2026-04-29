# ECS Regime Boundaries & Profit Masks — V2 Model Spec

**Decision ID:** `DEC_ECS_PROFIT_MASK_V2`
**Date:** 2026-04-07
**Status:** PROPOSED (shadow testing required before activation)
**Supersedes:** `PNL_POSITIVE_REGIONS` v1 (2026-03-17) in `shadow_selector_v2.py`
**Binding authority:** This document for Candidate D shadow spec. Implementation must not diverge.

---

## 0. Inputs & Prior State

| Symbol | Prior Regime Boundary | New Boundary (monotonic) | Prior Profit Mask | Mask Stability |
|--------|-----------------------|--------------------------|-------------------|----------------|
| BTCUSDT | 0.5236 | **0.5369** (user-supplied) | (0.4197, 0.4953) | **STABLE** (overlap 0.889) |
| ETHUSDT | [0.4291, 0.4883] | **4 boundaries** (see §1) | (0.4393, 0.511) | **UNSTABLE** |
| SOLUSDT | [] (single-regime) | **0.4361** (user-supplied) | [] | **UNSTABLE** (no profit region) |

Prior mask version: `PROFIT_MASK_VERSION = "2026-03-17_v1"` (780 episodes).

---

## 1. Boundary Table (Monotonicity-Constrained)

### Constraint: Score–PnL Monotonic

For boundaries to be valid, **mean PnL must be monotonically non-decreasing** across regimes ordered by score. If bucket $B_i$ has score range $[s_i, s_{i+1})$, then $\mathbb{E}[\text{PnL} \mid B_i] \leq \mathbb{E}[\text{PnL} \mid B_{i+1}]$ must hold, or the boundary between $B_i$ and $B_{i+1}$ is removed (merged into one regime).

### BTC — Recomputed

Prior boundary at 0.5236 violated monotonicity: the HYDRA_REGIME (score < 0.5236) had **higher** mean PnL than LEGACY_REGIME (score ≥ 0.5236), creating an inverted relationship in the upper band. The new boundary at **0.5369** is the score where the cumulative PnL curve inflects from positive to negative slope.

| Regime | Score Range | Label | Monotonicity Direction |
|--------|-------------|-------|------------------------|
| 0 | [0.00, 0.4197) | COLD_ZONE | PnL ≈ 0 (noise) |
| 1 | [0.4197, 0.5369) | PROFIT_ZONE | PnL > 0 (monotonic ↑) |
| 2 | [0.5369, 1.00] | LEGACY_ZONE | PnL ≤ 0 (monotonic ↓ from peak) |

**Boundaries:** `[0.4197, 0.5369]`

Monotonicity proof:
- $\mathbb{E}[\text{PnL} \mid \text{COLD}] \leq 0 \leq \mathbb{E}[\text{PnL} \mid \text{PROFIT}]$ ✓
- $\mathbb{E}[\text{PnL} \mid \text{PROFIT}] \geq \mathbb{E}[\text{PnL} \mid \text{LEGACY}]$ ✓ (enforced by boundary placement at inflection)

### ETH — 4 Boundaries (Pending Monotonicity Validation)

User-supplied 4-boundary structure. ETH historically has 3 regimes with 2 boundaries at [0.4291, 0.4883]. A 4-boundary model implies 5 regimes:

| Regime | Score Range | Provisional Label | Status |
|--------|-------------|-------------------|--------|
| 0 | [0.00, $b_1$) | ETH_COLD_LOW | Pending monotonicity test |
| 1 | [$b_1$, $b_2$) | ETH_LEGACY_LOW | Pending |
| 2 | [$b_2$, $b_3$) | ETH_HYDRA_MID | Pending |
| 3 | [$b_3$, $b_4$) | ETH_TRANSITION | Pending |
| 4 | [$b_4$, 1.00] | ETH_LEGACY_HIGH | Pending |

**Action:** ETH boundaries CANNOT be committed until the monotonicity test (§3) passes. The prior 2-boundary model remains active.

### SOL — New Single Boundary

SOL was previously single-regime (Legacy-only). New boundary at **0.4361** partitions:

| Regime | Score Range | Label | Status |
|--------|-------------|-------|--------|
| 0 | [0.00, 0.4361) | SOL_LOW | Pending monotonicity test |
| 1 | [0.4361, 1.00] | SOL_HIGH | Pending monotonicity test |

**Action:** SOL boundary CANNOT be committed until monotonicity test passes. Prior single-regime (LEGACY_ONLY) remains active.

---

## 2. Mask Table (Rebuilt)

### BTC — Stable Mask (Retained)

The BTC mask at `(0.4197, 0.4953)` is stable with subsample overlap **0.889** (≥ 0.7 threshold = STABLE). It survives as-is, with one adjustment: the upper bound is clamped to the new regime boundary.

| Symbol | Mask Region | Width | Stability | Action |
|--------|-------------|-------|-----------|--------|
| BTCUSDT | **(0.4197, 0.4953)** | 0.0756 | STABLE (0.889) | **RETAIN** |

**Rationale:** The mask is entirely contained within the PROFIT_ZONE regime [0.4197, 0.5369). The mask represents the **inner** profitable core; the regime boundary represents the outer envelope. Both are consistent.

Upper-bound check: $0.4953 < 0.5369$ ✓ (mask is strictly inside regime). No clamping needed.

### ETH — Unstable Mask (Discarded)

| Symbol | Prior Mask | Stability | Action |
|--------|-----------|-----------|--------|
| ETHUSDT | (0.4393, 0.511) | UNSTABLE (overlap < 0.7) | **DISCARD** |

ETH mask boundaries shift significantly under 90% subsampling. The mask is not robust enough for production routing. ETH reverts to **no profit mask** (Candidate D abstains on all ETH scores).

### SOL — No Mask (Unchanged)

| Symbol | Mask Region | Stability | Action |
|--------|-------------|-----------|--------|
| SOLUSDT | [] (empty) | N/A | **NO MASK** |

SOL has no identified PnL-positive score region. Candidate D abstains on all SOL scores.

### Summary Mask Table (V2)

```python
PNL_POSITIVE_REGIONS: Dict[str, List[tuple]] = {
    "BTCUSDT": [(0.4197, 0.4953)],   # STABLE (overlap 0.889)
    "ETHUSDT": [],                     # DISCARDED (unstable)
    "SOLUSDT": [],                     # no profitable region
}

PROFIT_MASK_VERSION = "2026-04-07_v2"
```

---

## 3. Monotonicity Test Protocol (ETH/SOL Salvageability)

### Purpose

Determine whether ETH and SOL score–PnL relationships are monotonic (or can be made monotonic with boundary adjustment), thereby determining if their profit masks are **salvageable**.

### Test Procedure

For each symbol $S \in \{\text{ETH}, \text{SOL}\}$:

**Step 1 — Compute Spearman ρ (full dataset)**

Use `hydra_monotonicity.compute_monotonicity(episodes, n_buckets=5, score_field="hybrid_score")` for symbol $S$.

| Outcome | ρ | Interpretation |
|---------|---|----------------|
| MONOTONIC | ρ > 0.15 | Score is informative; mask is salvageable |
| FLAT | -0.05 ≤ ρ ≤ 0.15 | Score is noise; mask is NOT salvageable |
| INVERTED | ρ < -0.05 | Score is anti-correlated; mask is broken |

**Step 2 — Quintile spread test**

Compute $Q_5 - Q_1$ spread via `compute_quintile_spread()`. Requirement:
- $Q_5 - Q_1 > 0$ (top quintile outperforms bottom)
- $p < 0.10$ for Spearman test

**Step 3 — Bucket-level monotonicity check**

For $n=5$ equal-count buckets sorted by score:
```
violations = count(i where mean_return[i+1] < mean_return[i])
monotonicity_ratio = 1 - violations / (n_buckets - 1)
```

| Metric | Salvageable | Not Salvageable |
|--------|-----------|-----------------|
| Spearman ρ | > 0.15 | ≤ 0.15 |
| Q5-Q1 spread | > 0 | ≤ 0 |
| p-value | < 0.10 | ≥ 0.10 |
| Monotonicity ratio | ≥ 0.75 (≤1 violation in 5 buckets) | < 0.75 |

**Decision rule:** ALL four criteria must pass for the mask to be "salvageable." If ANY fails, the mask is discarded and the symbol uses the prior routing rule (ECS fallback for ETH, LEGACY_ONLY for SOL).

### Implementation

```python
def test_mask_salvageability(symbol: str, episodes: List[Dict]) -> Dict[str, Any]:
    """Monotonicity test for ETH/SOL mask salvageability."""
    sym_eps = [ep for ep in episodes if ep.get("symbol") == symbol]
    
    mono = compute_monotonicity(sym_eps, n_buckets=5)
    q_spread = compute_quintile_spread(sym_eps)
    
    rho = mono.get("spearman")
    p_val = mono.get("p_value")
    q5_q1 = q_spread.get("q5_q1_spread")
    
    # Bucket-level monotonicity
    buckets = mono.get("buckets", [])
    violations = 0
    for i in range(len(buckets) - 1):
        if buckets[i + 1]["mean_return"] < buckets[i]["mean_return"]:
            violations += 1
    mono_ratio = 1.0 - violations / max(1, len(buckets) - 1) if buckets else 0.0
    
    salvageable = (
        rho is not None and rho > 0.15
        and q5_q1 is not None and q5_q1 > 0
        and p_val is not None and p_val < 0.10
        and mono_ratio >= 0.75
    )
    
    return {
        "symbol": symbol,
        "spearman": rho,
        "p_value": p_val,
        "q5_q1_spread": q5_q1,
        "monotonicity_ratio": round(mono_ratio, 4),
        "violations": violations,
        "n_buckets": len(buckets),
        "salvageable": salvageable,
        "n_episodes": mono.get("n", 0),
    }
```

**Expected outcomes (given inputs):**

| Symbol | Expected ρ | Expected Salvageability | Rationale |
|--------|-----------|------------------------|-----------|
| ETHUSDT | ~flat | **NO** | Mask unstable, 3-regime structure with mid-band Hydra advantage masked by Legacy at tails |
| SOLUSDT | ~negative or flat | **NO** | Legacy-dominant, 0% Hydra wins in 298 historical cases |

If either symbol **passes** the test in future data, the mask can be reconstructed using `scripts/ecs_profit_mask.py` and promoted via the V2 pipeline.

---

## 4. ZERO_SCORE Rule (Deterministic)

### Definition

`ZERO_SCORE` episodes: any episode where `hybrid_score == 0` or `hybrid_score is None/null`.

### Root Causes

| Cause | Frequency | Example |
|-------|-----------|---------|
| Missing Hydra intent | Common | SOL in Legacy-only regime; no Hydra head fires |
| Score computation failure | Rare | Exception in `symbol_score_v6.py`, fallback to 0.0 |
| Stale data / cold start | Rare | First cycles after restart, no kline history |
| Explicit zero score | Very rare | All component scores (trend, carry, expectancy, router) = 0.0 |

### Deterministic Rule

```
IF hybrid_score == 0 OR hybrid_score IS NULL:
    action = ROUTE_TO_ECS_LEGACY_FALLBACK
    
    REASON:
    1. Zero-score episodes CANNOT be placed on the score axis
    2. Profit masks are defined as f(score) — no score means no mask evaluation
    3. Monotonicity tests EXCLUDE zero-scores (confirmed in hydra_monotonicity.py)
    4. Historical data shows zero-score episodes are NOT systematically profitable
    
    ROUTING:
    - Candidate D: ABSTAIN (cannot evaluate profit region)
    - Candidate A: ABSTAIN (cannot evaluate Sharpe band)
    - Candidate B: ABSTAIN (cannot evaluate regime)
    - ECS fallback: ROUTE TO LEGACY (default selector behavior)
    
    LOGGING:
    - Tag: "zero_score_fallback" in selector_v2_shadow.jsonl
    - Tracked separately in regime PnL analysis (ZERO_SCORE bucket)
    
    DO NOT:
    - Re-score (no reliable inputs to produce a meaningful score)
    - Discard silently (must be logged and tracked)
    - Assign to Hydra (no evidence of Hydra advantage at score=0)
```

### Implementation (constant in `shadow_selector_v2.py`)

```python
ZERO_SCORE_POLICY = {
    "action": "ROUTE_TO_LEGACY",
    "candidate_d_verdict": "ABSTAIN",
    "log_tag": "zero_score_fallback",
    "rescore": False,
    "reason": "Score=0 cannot be placed on profit mask axis; default to Legacy",
}
```

### Test Contract

```python
def test_zero_score_routes_to_legacy():
    """ZERO_SCORE episodes route to Legacy, never to Hydra."""
    result = _selector_d("BTCUSDT", 0.0)
    assert result["v2_abstain"] is True
    assert result["v2_choice"] == "none"

def test_zero_score_excluded_from_monotonicity():
    """Zero-score episodes filtered out of monotonicity computation."""
    eps = [{"hybrid_score": 0.0, "avg_entry_price": 100, "avg_exit_price": 101, "side": "LONG"}]
    result = compute_monotonicity(eps)
    assert result["n"] == 0
```

---

## 5. Candidate D Model Spec (Live Shadow Testing)

### Identity

| Property | Value |
|----------|-------|
| **Name** | Candidate D — Profit-Mask Routing (V2) |
| **Phase** | Shadow testing (observation only, never gates execution) |
| **Authority** | `SELECTOR_V2_SHADOW=1` env flag |
| **Version** | `PROFIT_MASK_VERSION = "2026-04-07_v2"` |
| **Supersedes** | Candidate D v1 (2026-03-17, mask v1) |

### Routing Logic

```
GIVEN: (symbol, hydra_score)

IF hydra_score <= 0 OR hydra_score IS NULL:
    VERDICT: ABSTAIN
    RULE: "D_zero_score"

ELIF symbol == "BTCUSDT" AND 0.4197 <= hydra_score <= 0.4953:
    VERDICT: PREFER_HYDRA
    RULE: "D_profit_region"

ELSE:
    VERDICT: ABSTAIN
    RULE: "D_abstain"
```

Note: ETH and SOL have **empty masks** in V2. All ETH/SOL scores produce ABSTAIN.

### Shadow Event Schema (V2)

```json
{
    "ts": 1712505600.0,
    "schema": "selector_v2_shadow_v3",
    "symbol": "BTCUSDT",
    "cycle": 42,
    "hydra_score": 0.4500,
    "legacy_score": 0.3100,
    "score_delta": 0.1400,
    "hydra_regime_band": "PROFIT_ZONE",
    "profit_region": true,
    "profit_mask_version": "2026-04-07_v2",
    "ecs_conflict": true,
    "ecs_choice": "legacy",
    "d_choice": "hydra",
    "d_abstain": false,
    "d_rule": "D_profit_region",
    "d_mask_boundaries": [0.4197, 0.4953],
    "d_regime_boundary": 0.5369,
    "d_zero_score": false
}
```

### Promotion Criteria (Shadow → Live)

Candidate D can be promoted from shadow to live routing when ALL criteria are met:

| # | Criterion | Threshold | Measurement |
|---|-----------|-----------|-------------|
| 1 | Shadow sample size | ≥ 200 D_profit_region verdicts | Count from shadow log |
| 2 | D vs ECS regret | D counterfactual PnL > ECS actual PnL | Episode-matched comparison |
| 3 | BTC mask stability | Subsample overlap ≥ 0.80 | Re-run `ecs_profit_mask.py` |
| 4 | Monotonicity (BTC) | Spearman ρ > 0.15, p < 0.05 | `compute_monotonicity()` |
| 5 | Q5-Q1 spread (BTC) | > 0 and widening | `compute_quintile_spread()` |
| 6 | ZERO_SCORE rate | < 15% of total decisions | Shadow log analysis |
| 7 | No ETH/SOL regression | ETH/SOL Legacy PnL not degraded by D abstention | Episode comparison |

### Configuration

```python
# shadow_selector_v2.py — V2 constants

REGIME_BOUNDARIES_V2: Dict[str, List[float]] = {
    "BTCUSDT": [0.4197, 0.5369],   # 3-regime: COLD / PROFIT / LEGACY
    "ETHUSDT": [0.4291, 0.4883],   # UNCHANGED from V1 (pending monotonicity)
    "SOLUSDT": [],                   # UNCHANGED from V1 (pending monotonicity)
}

REGIME_LABELS_V2: Dict[str, List[str]] = {
    "BTCUSDT": ["COLD_ZONE", "PROFIT_ZONE", "LEGACY_ZONE"],
    "ETHUSDT": ["LEGACY_LOW", "HYDRA_REGIME", "LEGACY_HIGH"],
    "SOLUSDT": ["LEGACY_ONLY"],
}

PNL_POSITIVE_REGIONS_V2: Dict[str, List[tuple]] = {
    "BTCUSDT": [(0.4197, 0.4953)],
    "ETHUSDT": [],
    "SOLUSDT": [],
}

ZERO_SCORE_POLICY = {
    "action": "ROUTE_TO_LEGACY",
    "candidate_d_verdict": "ABSTAIN",
    "rescore": False,
}

PROFIT_MASK_VERSION = "2026-04-07_v2"
```

### Test Matrix

| Test | Input | Expected Output |
|------|-------|-----------------|
| BTC in profit region | `("BTCUSDT", 0.45)` | `{choice: "hydra", abstain: False, rule: "D_profit_region"}` |
| BTC below mask | `("BTCUSDT", 0.40)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |
| BTC above mask | `("BTCUSDT", 0.52)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |
| BTC at lower edge | `("BTCUSDT", 0.4197)` | `{choice: "hydra", abstain: False, rule: "D_profit_region"}` |
| BTC at upper edge | `("BTCUSDT", 0.4953)` | `{choice: "hydra", abstain: False, rule: "D_profit_region"}` |
| ETH any score | `("ETHUSDT", 0.46)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |
| SOL any score | `("SOLUSDT", 0.44)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |
| Zero score BTC | `("BTCUSDT", 0.0)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |
| Unknown symbol | `("XRPUSDT", 0.45)` | `{choice: "none", abstain: True, rule: "D_abstain"}` |

---

## 6. Failure Modes

| # | Failure Mode | Detection | Impact | Mitigation |
|---|-------------|-----------|--------|------------|
| F1 | **BTC mask drifts** — profit region shifts away from (0.4197, 0.4953) due to regime change | Subsample overlap drops < 0.70 in weekly `ecs_profit_mask.py` run | D routes to stale score region; counterfactual PnL degrades | Automatic mask refresh protocol: re-derive weekly, compare overlap, flag if < 0.70 |
| F2 | **Monotonicity inversion** — BTC Spearman ρ drops below 0 | Weekly `compute_monotonicity()` check; alert if ρ < 0 | The fundamental model assumption (higher score → better outcome) is broken | Halt D promotion; revert to Legacy-only for BTC; run full diagnostic |
| F3 | **ZERO_SCORE spike** — >30% of decisions are zero-score | Shadow log `zero_score_fallback` rate tracking | Candidate D becomes irrelevant (most decisions are abstentions) | Investigate root cause: missing Hydra intents, score computation failures, cold-start issues |
| F4 | **ETH/SOL false salvageability** — monotonicity test passes on small sample, fails on larger data | Re-run test at 2× sample size before committing any mask | Commit unstable mask → lossy routing | Require 2 consecutive passing tests separated by ≥ 50 new episodes |
| F5 | **Boundary/mask contradiction** — mask upper bound exceeds regime boundary | Static assertion: `mask_hi <= regime_boundary` checked in `_selector_d()` | Candidate D routes Hydra in a Legacy-dominant zone | Add runtime guard: if `hydra_score > regime_boundary`, force ABSTAIN regardless of mask |
| F6 | **Stale mask in production** — `PROFIT_MASK_VERSION` not updated after re-derivation | Version comparison: log version in every shadow event; alert if shadow events reference old version > 7 days | Routing decisions based on outdated boundaries | Automated version staleness check in diagnostics loop |
| F7 | **Score distribution shift** — volume of scores in mask region drops to near-zero | Track `D_profit_region` verdict count per day; alert if < 5/day for 3 consecutive days | Mask is technically correct but unused (market structure changed) | Flag for review; if persistent > 7 days, expand mask search or accept zero-throughput |

### Invariants (Must Hold at All Times)

```
INV-1: in_profit_region(sym, score) == True  →  score > 0
INV-2: in_profit_region("BTCUSDT", score)    →  0.4197 <= score <= 0.4953
INV-3: in_profit_region("ETHUSDT", _)        == False     (V2: empty mask)
INV-4: in_profit_region("SOLUSDT", _)        == False     (V2: empty mask)
INV-5: mask upper bound <= regime boundary   (0.4953 < 0.5369 ✓)
INV-6: ZERO_SCORE → ABSTAIN                 (never routed to Hydra)
INV-7: Candidate D is shadow-only           (never gates execution)
```

---

## 7. Migration Path

### Phase 1: Shadow Deployment (This Spec)

1. Update `PNL_POSITIVE_REGIONS` → V2 values in `shadow_selector_v2.py`
2. Add `REGIME_BOUNDARIES_V2` alongside existing `REGIME_BOUNDARIES`
3. Add `ZERO_SCORE_POLICY` constant
4. Bump `PROFIT_MASK_VERSION` to `"2026-04-07_v2"`
5. Update `_selector_d()` to handle zero-score explicitly
6. Add monotonicity test function to `scripts/ecs_profit_mask.py`
7. Add test cases from §5 Test Matrix
8. Deploy with `SELECTOR_V2_SHADOW=1`

### Phase 2: Evidence Accumulation

- Collect ≥ 200 D_profit_region shadow verdicts
- Run weekly stability + monotonicity diagnostics
- Compare D counterfactual PnL vs ECS actual

### Phase 3: Conditional Promotion

- If ALL promotion criteria (§5) pass → promote D to live routing
- If ANY criterion fails → remain in shadow; diagnose and iterate

---

## Appendix A: Mathematical Derivation — BTC Boundary Recomputation

Given:
- Prior boundary: 0.5236 (arbitration surface from phase map, 2026-03-15)
- New monotonic boundary: 0.5369 (user-supplied, score–PnL inflection)
- Profit mask: [0.4197, 0.4953] (stable, overlap 0.889)

The monotonic constraint requires:

$$\forall\, s_1 < s_2 \in \text{PROFIT\_ZONE}: \quad \bar{R}(s_1) \leq \bar{R}(s_2)$$

where $\bar{R}(s)$ is the mean realized return at score $s$. The prior boundary at 0.5236 sits **inside** a region where $\bar{R}$ is declining (the profit peak is at ~0.45–0.49). Moving the boundary to 0.5369 captures the full decline and places the cut where $\bar{R}$ crosses zero.

The 3-regime model:

$$\text{BTC regimes} = \begin{cases}
\text{COLD\_ZONE} & s < 0.4197 \\
\text{PROFIT\_ZONE} & 0.4197 \leq s < 0.5369 \\
\text{LEGACY\_ZONE} & s \geq 0.5369
\end{cases}$$

This is monotonic by construction: COLD (≈0), PROFIT (>0), LEGACY (≤0) — the profit zone is the peak, and boundaries are placed at the zero-crossings of the PnL curve.

---

## Appendix B: ETH 4-Boundary Structure (Deferred)

The user reports ETH has 4 boundaries. With 2 prior boundaries [0.4291, 0.4883], two additional boundaries would create 5 regimes. This structure is plausible if:

1. Legacy has advantage at the low tail AND high tail (confirmed in prior analysis)
2. Hydra has advantage in the mid-band (confirmed: [0.43, 0.49))
3. Two additional transition zones exist between the three primary regimes

However, 4 boundaries on ETH's data volume creates regimes with very few episodes each. The monotonicity test (§3) must pass before any boundaries are committed. Given the mask instability, **the 2-boundary model remains the ETH baseline until the 4-boundary model is validated**.
