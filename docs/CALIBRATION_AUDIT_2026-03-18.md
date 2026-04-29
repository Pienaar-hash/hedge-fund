# Calibration Audit — 2026-03-18

## §1 — Purpose

This audit answers: **When the model says "0.70 conviction", does the trade win 70% of the time?**

A model that produces scores ∈ [0,1] is only useful if those scores track realized
frequency. A system that outputs 0.80 conviction but wins only 50% of the time is
overconfident. A system that outputs 0.30 but wins 60% is underconfident. Both
are miscalibrated — the first is dangerous, the second is wasteful.

This audit brings **S2 discipline** (where isotonic calibration is standard) into
the futures stack, where conviction/hybrid scores have never been formally
calibrated against outcomes.

---

## §2 — Prediction Signals

The system produces two score fields per trade, both stored in episode records:

| Score | Range | Formula | Interpretation |
|-------|-------|---------|---------------|
| `conviction_score` | [0, 1] | 0.40 × hybrid + 0.25 × expectancy + 0.20 × router_quality + 0.15 × trend_strength, then × vol_regime × dd_mult × risk_mode × alpha_decay | Full-stack prediction (regime/DD/risk adjusted) |
| `hybrid_score` | [0, 1] | 0.40 × trend + 0.25 × carry + 0.20 × expectancy + 0.15 × router | Raw signal quality (no regime overlay) |

**Outcome** (binary): direction-adjusted return > 0 → **1** (win), else **0** (loss).

### Why Two Scores?

`conviction_score` includes regime penalties, drawdown multipliers, and risk mode
scaling — information about *execution context*, not just signal quality. If conviction
calibrates better than hybrid, these adjustments are earning their keep. If worse,
they're injecting noise.

---

## §3 — Calibration Curve (Reliability Diagram)

### Implementation: `compute_calibration_curve()`

Groups episodes into equal-count buckets by predicted score. For each bucket:

| Field | Definition |
|-------|-----------|
| `mean_predicted` | Average score in bucket |
| `realized_frequency` | Fraction of wins (actual P(profitable)) |
| `gap` | `mean_predicted − realized_frequency` |
| `n` | Sample count |

**Perfect calibration:** gap ≈ 0 in every bucket (diagonal line on reliability diagram).

### Reading the Table

```
B1:  pred=0.25  realized=0.30  gap=−0.05  →  slightly underconfident
B5:  pred=0.55  realized=0.55  gap= 0.00  →  perfectly calibrated
B10: pred=0.85  realized=0.60  gap=+0.25  →  severely overconfident
```

---

## §4 — Brier Score

### Implementation: `compute_brier_score()`

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2$$

where $p_i$ is the predicted score and $o_i \in \{0, 1\}$ is the binary outcome.

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Brier** | Mean squared error | Lower = better. Range [0, 1] |
| **Brier baseline** | MSE if always predict base rate | Naive reference |
| **BSS** | $1 - \text{Brier} / \text{Brier}_\text{baseline}$ | Skill score: > 0 = beats baseline |

### Murphy Decomposition (1973)

$$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

| Component | Meaning | Want |
|-----------|---------|------|
| **Reliability** | Mean squared gap between predicted and observed per bucket | Low (< 0.01) |
| **Resolution** | Spread of observed frequencies across buckets | High (model discriminates) |
| **Uncertainty** | $\bar{o}(1 - \bar{o})$ — base rate entropy | Fixed (data property) |

### Interpretation Guide

| BSS | Assessment |
|-----|-----------|
| > 0.15 | Strong skill |
| 0.05–0.15 | Moderate skill |
| 0–0.05 | Marginal skill |
| < 0 | Model worse than always guessing base rate |

---

## §5 — Model vs Naive Baseline

The **naive baseline** always predicts the base rate $\bar{o}$ (overall win rate).

If the model's Brier score ≤ baseline, the model provides **zero predictive value**
for position sizing — the conviction score carries no information beyond "the
historical win rate is X%".

**Lift** = BSS × 100 (percentage improvement over baseline).

---

## §6 — Diagnosis: Patterns of Miscalibration

### Implementation: `compute_calibration_diagnosis()`

| Metric | Definition |
|--------|-----------|
| **ECE** | Expected Calibration Error = weighted mean \|gap\| across buckets |
| **MCE** | Maximum Calibration Error = worst single bucket |
| **pred_spread** | Range of mean predicted values (max − min) |
| **avg_gap_top_third** | Mean gap in highest-score buckets |
| **avg_gap_bottom_third** | Mean gap in lowest-score buckets |

### Pattern Detection

| Pattern | Condition | Meaning |
|---------|-----------|---------|
| `overconfident_top` | avg_gap_top > +0.05 | High scores promise more than they deliver |
| `underconfident_bottom` | avg_gap_bottom < −0.05 | Low scores are better than the model thinks |
| `collapse_to_mean` | pred_spread < 0.15 | Model outputs cluster near a single value |
| `well_calibrated` | ECE < 0.03 | Reliability diagram tracks diagonal |
| `mildly_miscalibrated` | ECE 0.03–0.08 | Some drift but potentially correctable |
| `severely_miscalibrated` | ECE ≥ 0.08 | Scores should not be trusted for sizing |

**Multiple patterns can coexist.** A model can be simultaneously `overconfident_top`
and `collapse_to_mean` (outputs range 0.45–0.55 but claims high conviction).

---

## §7 — Master Verdict

### Implementation: `compute_calibration_audit()`

Runs full calibration pipeline on both `conviction_score` and `hybrid_score`, then
compares them.

| Verdict | Conditions |
|---------|-----------|
| **CALIBRATED** | ECE < 0.05 AND BSS > 0 |
| **OVERCONFIDENT** | Top-third gap > 0.08 |
| **UNDERCONFIDENT** | Bottom-third gap < −0.08 |
| **COLLAPSED** | pred_spread < 0.10 (model outputs cluster) |
| **MISCALIBRATED** | ECE ≥ 0.08, no clear directional pattern |
| **WEAKLY_CALIBRATED** | In between — not clearly good or bad |
| **INSUFFICIENT_DATA** | < 20 usable episodes |

### Cross-Score Comparison

Reports which score (`conviction_score` vs `hybrid_score`) has the lower Brier
score, and by how much. This answers: does the conviction engine's regime/risk
overlay improve or degrade calibration?

---

## §8 — Actionable Responses by Verdict

| Verdict | Action |
|---------|--------|
| CALIBRATED | Use scores directly for position sizing. Monitor stability. |
| OVERCONFIDENT | Reduce size in high-conviction trades OR apply isotonic correction |
| UNDERCONFIDENT | System is leaving money on the table — increase sizing or lower thresholds |
| COLLAPSED | Score carries little information; sizing is effectively flat. Investigate component weights |
| MISCALIBRATED | Do NOT use scores for sizing. Fall back to fixed position sizes until recalibrated |
| WEAKLY_CALIBRATED | Usable with caution. Consider Platt scaling or isotonic recalibration |

### Isotonic Recalibration Path

If verdict ≠ CALIBRATED, the path to remedy is:

1. Collect ≥ 150 (score, outcome) pairs from episodes
2. Fit isotonic regression: monotonic map from raw score → calibrated probability
3. Apply as a post-processor: `calibrated_score = isotonic(raw_score)`
4. Re-evaluate: Brier, ECE, BSS should all improve

The S2 binary lab already implements this pattern in `binary_lab_s2_model.py`.

---

## §9 — What This Audit Cannot See

### 9.1 Left-Truncation

Only traded episodes are calibrated. The passive observation log (from the
selection bias audit) would allow calibration of below-threshold scores, but
outcomes for non-traded signals are unobserved.

### 9.2 Non-Stationarity

Calibration is computed over the full episode history. If the model was
calibrated early but degraded recently (or vice versa), a single Brier
score masks the shift. The existing `compute_time_stability()` provides
some temporal insight but not per-bucket calibration drift.

### 9.3 Score-Outcome Lag

Episodes have variable holding periods. A score predicts entry quality,
but the outcome depends on exit timing. Holding-period variation adds
noise to the calibration measurement.

### 9.4 Conviction ≠ Probability

`conviction_score` was not designed as a probability estimate — it's a
composite ranking. Calibrating it as if it were P(profitable) is a useful
diagnostic but may reveal fundamental architectural mismatch: the score
may discriminate well (high resolution) but have poor reliability (gaps
between predicted and realized).

---

## §10 — Integration

### State Output

Published to `logs/state/hydra_monotonicity.json` under `calibration_audit` key:

```json
{
  "calibration_audit": {
    "conviction_score": {
      "verdict": "OVERCONFIDENT",
      "brier": {
        "brier": 0.241500,
        "brier_baseline": 0.249975,
        "brier_skill_score": 0.0339,
        "base_rate": 0.5001,
        "decomposition": { "reliability": ..., "resolution": ..., "uncertainty": ... }
      },
      "diagnosis": {
        "ece": 0.0612,
        "mce": 0.1543,
        "patterns": ["overconfident_top", "mildly_miscalibrated"],
        "buckets": [...]
      }
    },
    "hybrid_score": { ... },
    "comparison": {
      "better_calibrated": "hybrid_score",
      "brier_delta": 0.003200
    }
  }
}
```

### Code Location

All functions in `execution/hydra_monotonicity.py`:

| Function | Purpose |
|----------|---------|
| `compute_calibration_curve()` | Reliability diagram data (predicted vs realized per bucket) |
| `compute_brier_score()` | Brier + decomposition + baseline + BSS |
| `compute_calibration_diagnosis()` | ECE/MCE + pattern detection (over/under/collapse) |
| `compute_calibration_audit()` | Master orchestrator for both scores + cross-comparison |

Called from `persist_snapshot()` → `snap["calibration_audit"]`.

---

## §11 — Relationship to Other Audits

| Audit | Relationship |
|-------|-------------|
| Signal–Outcome Causality | Causality measures monotonicity (Spearman ρ); calibration measures reliability (predicted ≈ realized). A model can be monotonic but miscalibrated. |
| Friction-Aware Edge | Friction audit uses BPS returns; calibration uses binary win/loss. They measure different aspects of the same underlying quality. |
| Edge Realization Ratio | ERR calibrates expected_edge magnitude; this audit calibrates conviction_score as a probability. Complementary. |
| Selection Bias | Calibration on traded subset may overstate quality if selection is biased toward easier trades. |

---

## §12 — Honest Position

1. **Conviction score was never designed as a probability.** It's a composite
   ranking that happens to land in [0,1]. Expecting it to be perfectly
   calibrated is unreasonable — but measuring how far it deviates from
   calibration is essential for position sizing.

2. **Brier score is the single most important number.** It captures both
   calibration and discrimination in one metric. If Brier > baseline,
   the model's scores carry no usable information.

3. **Overconfidence is the dangerous failure mode.** Underconfidence wastes
   capital; overconfidence risks it. A system that sizes aggressively on
   high-conviction trades that don't actually win more often is structurally
   dangerous.

4. **Collapse-to-mean is the silent failure.** When all outputs cluster near
   0.5, the model looks "calibrated" (gap ≈ 0 everywhere) but provides
   zero discrimination. Brier resolution component catches this.

5. **The fix is cheap.** Isotonic recalibration is a solved problem (S2 already
   does it). The hard part is admitting the raw scores need correction.
