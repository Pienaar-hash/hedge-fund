# Binary Sleeve S2 vs Futures — Comparative Edge Audit

**Date:** 2026-03-19  
**Predecessor:** `MINIMAL_SYSTEM_EXTRACTION_AUDIT_2026-03-18.md` (Audit 6/6)  
**Data window:** Futures 2026-02-24 → 2026-03-19 (783 episodes, 123 scored); S2 2026-03-18 → 2026-03-19 (83 rounds, 80 resolved)  
**Status:** BINDING — same yardstick applied to both systems

---

## §1 — Question

The futures minimal-core validation returned **FAIL** on all three edge
tests (causality, calibration, tradability). Binary Sleeve S2 has been
accumulating passive observations since 2026-03-18 but has taken zero
trades (100% friction kill rate at the 3pp executable-edge gate).

**This audit asks:** Is S2 closer to a validated edge than futures, or are
both systems in the same place?

---

## §2 — Methodology

Both systems are measured on the **same five axes** from the
§8 Validation Plan of the Extraction Audit:

| Axis | Futures source | S2 source |
|------|---------------|-----------|
| Causality | `hybrid_score` → episode PnL Spearman | market mid → binary outcome Spearman |
| Calibration | Brier / BSS / ECE on `hybrid_score` | Brier / BSS / ECE on market-mid buckets |
| Tradability | mean net edge BPS, friction kill rate | executable edge pp, friction kill rate |
| Controls | doctrine + risk gate tests | 9-gate eligibility checker |
| Information | Murphy decomposition (resolution / reliability / uncertainty) | Same |

The S2 calibration data comes from `logs/state/binary_lab_s2_calibration.json`
(80 passively resolved observations). No isotonic refit has occurred yet
(`last_refit_n: null`). The model currently returns `p_model = p_baseline = market_mid`
(zero lift).

---

## §3 — Head-to-Head Results

### 3.1 Raw Comparison

| Metric | Futures | S2 Binary | Winner |
|--------|---------|-----------|--------|
| Scored observations | 123 | 80 | — |
| **Score → outcome ρ** | **−0.049** | **+0.091** | S2 |
| ρ p-value | 0.593 | 0.420 | neither (both insignificant) |
| **Brier score** | **0.238** | **0.257** | Futures |
| **BSS (vs own baseline)** | **−0.192** | **0.000** | S2 |
| **BSS (vs base-rate)** | **−0.192** | **−0.029** | S2 |
| **ECE** | **0.185** | **0.164** | S2 |
| Resolution | 0.0083 | 0.0287 | S2 (3.5×) |
| Reliability | 0.0470 | 0.0341 | S2 |
| Uncertainty | 0.200 | 0.249 | S2 (harder problem) |
| **Resolution / Uncertainty** | **0.041** | **0.115** | S2 (2.8×) |
| Net edge | −42.17 BPS | 0 (untested) | unknown |
| Friction kill rate | 0.206 | 1.000 | Futures |
| Score → outcome monotone? | NO | NO | neither |
| Controls pass? | YES (4/4) | YES (9/9) | tie |

### 3.2 What the Numbers Mean

**Futures** has a *negative* BSS (−0.192) — the scoring model is actively
*worse* than guessing the base rate. Scores anti-predict outcomes.

**S2** has BSS = 0.000 vs its own baseline because model = baseline (no
refit yet). But vs the constant base-rate predictor, BSS = −0.029 —
essentially neutral. The market mid is neither helping nor hurting, which
is exactly what an efficient market price should do.

The critical difference is **resolution**. S2's market mid resolves 11.5%
of the problem's uncertainty (resolution/uncertainty = 0.115). Futures'
`hybrid_score` resolves 4.1%. S2 separates outcomes 2.8× better by this
measure.

However — and this is the trap:

---

## §4 — Overfitting Reality Check

The S2 resolution figure comes from 80 observations split into 7 buckets.

### 4.1 Per-Bucket Significance (Binomial Test)

| Bucket | n | Predicted | Realized | 95% CI | p-value |
|--------|---|-----------|----------|--------|---------|
| 0.20–0.30 | 4 | 0.286 | 0.750 | [0.301, 0.954] | 0.074 |
| 0.30–0.40 | 20 | 0.363 | 0.250 | [0.112, 0.469] | 0.210 |
| 0.40–0.50 | 24 | 0.451 | 0.625 | [0.427, 0.788] | 0.067 |
| 0.50–0.60 | 19 | 0.544 | 0.421 | [0.231, 0.637] | 0.199 |
| 0.60–0.70 | 8 | 0.669 | 0.500 | [0.215, 0.785] | 0.254 |
| 0.70–0.80 | 4 | 0.768 | 0.750 | [0.301, 0.954] | 0.653 |
| 0.80–0.90 | 2 | 0.850 | 0.500 | [0.095, 0.905] | 0.278 |

**Zero buckets are significant at p < 0.05.** The "resolution" is
statistically indistinguishable from noise at this sample size.

### 4.2 Standard Errors

The tail buckets (n=2, n=4) have standard errors of **21–35pp**. A
realized rate of 0.75 in a bucket with n=4 has a 95% CI of
[0.30, 0.95]. This is not information — it is a coin flip plus rounding.

### 4.3 Resolution Stability (Leave-One-Bucket-Out)

| Removed bucket | Resolution | BSS |
|----------------|-----------|-----|
| 0.20–0.30 (n=4) | 0.0248 | 0.100 |
| 0.30–0.40 (n=20) | 0.0130 | 0.053 |
| 0.40–0.50 (n=24) | 0.0265 | 0.109 |
| 0.50–0.60 (n=19) | 0.0343 | 0.137 |
| 0.60–0.70 (n=8) | 0.0303 | 0.121 |
| 0.70–0.80 (n=4) | 0.0248 | 0.100 |
| 0.80–0.90 (n=2) | 0.0280 | 0.112 |
| **Full** | **0.0274** | **0.110** |

Resolution swings from 0.013 to 0.034 depending on which bucket is
removed. Removing the 0.30–0.40 bucket (n=20) cuts BSS in half. This is
not a stable statistic.

---

## §5 — Structural Advantages of S2

Despite the noise, S2 has three structural properties futures lacks:

### 5.1 The Input IS a Probability

S2's input (`p_yes_mid`) is already a market-calibrated probability
(Polymarket CLOB mid). Futures' input (`hybrid_score`) is a synthetic
composite that was never designed as a probability. S2 starts at the
probability layer of the minimal architecture; futures hasn't reached it.

### 5.2 Friction Is Known and Fixed

| | Futures | S2 |
|-|---------|-----|
| Half-spread cost | Variable (order book dependent) | 0.55pp mean |
| Fee | Maker/taker variable | 2% × min(p, 1−p) ≈ 0.78pp |
| Total friction | Unknown per-trade | **1.32pp** (measured) |
| Edge gate | None (trades at any edge) | 3.00pp minimum |

S2 can answer "does this trade survive friction?" *before entry*.
Futures cannot — it discovers friction after the fact.

### 5.3 Binary Outcome Eliminates Exit Ambiguity

Futures' "when to exit" is a separate, unsolved problem (thesis-driven
exits, regime flips, time stops, seatbelts). S2 exits are deterministic:
15 minutes expire, settlement price resolves the round. There is no exit
thesis to get wrong.

---

## §6 — Structural Disadvantages of S2

### 6.1 Zero Model Lift

The isotonic calibrator has not been fitted (`last_refit_n: null`).
`p_model = p_baseline = market_mid` for all 83 rounds. Every round shows
`executable_edge = 0.00pp`. The model adds nothing. The "resolution"
in §3.1 comes from the market mid itself, not from any model output.

### 6.2 The Market Mid Is Efficient (By Design)

The Polymarket CLOB mid for a 15-minute BTC up/down round reflects the
aggregate belief of all market participants. For the model to have edge,
it must know something the market doesn't about the next 15 minutes of
BTC price action. The Spearman ρ = 0.091 (p = 0.42) says: **right now,
it doesn't.**

### 6.3 Zero Live Trades

With a 100% friction kill rate (0/83 trades taken), S2 has no live
tradability data. The futures system at least has 783 real episodes with
real fills, real slippage, and real fees. S2's "advantage" on BSS and
ECE is theoretical — it has never been tested against actual execution.

---

## §7 — Murphy Decomposition Comparison

The Murphy decomposition splits Brier into three components:
$$\text{Brier} = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$$

| Component | Futures | S2 | Interpretation |
|-----------|---------|-----|---------------|
| **Uncertainty** | 0.200 | 0.249 | S2's environment is harder (base rate closer to 0.50) |
| **Resolution** | 0.008 | 0.029 | S2's input sorts outcomes 3.5× better |
| **Reliability** | 0.047 | 0.034 | S2's predictions are less miscalibrated |
| **Brier** | 0.238 | 0.257 | Futures is lower but only because the problem is easier |
| **BSS** | −0.192 | −0.029 | S2 is much closer to base-rate performance |

**Key insight:** Futures' lower Brier is an artifact of its lower
base rate (0.276 vs 0.475). When normalized to BSS (skill above
base-rate), futures is *worse* — it actively destroys information that
the base rate already provides.

S2 at least preserves the information content of its input. It doesn't
add anything yet, but it doesn't subtract either.

---

## §8 — Verdict

### Neither system has validated edge.

| Gate | Futures | S2 | Threshold |
|------|---------|-----|-----------|
| ρ significant at p < 0.05 | FAIL (p = 0.59) | FAIL (p = 0.42) | p < 0.05 |
| BSS > 0 | FAIL (−0.192) | FAIL (−0.029 vs base-rate) | BSS > 0.0 |
| Net edge > 0 | FAIL (−42.17 BPS) | UNTESTABLE (0 trades) | edge > 0 |
| Monotone score → outcome | FAIL | FAIL | monotone |

**Both systems fail the minimal-core validation.**

### But S2 is structurally closer.

| Dimension | Futures distance to pass | S2 distance to pass |
|-----------|------------------------|---------------------|
| ρ | Need to flip from −0.049 to +0.05 | Need p-value from 0.42 to <0.05 (ρ already positive) |
| BSS | −0.192 → needs model redesign | −0.029 → needs calibrator refit (mechanism exists) |
| Net edge | −42 BPS → structural friction problem | Gate blocks at 0pp → needs model lift to 3pp |
| Architecture | `hybrid_score` is not a probability | `p_yes_mid` is already a probability |
| Friction | Unknown per-trade, discovered post-facto | 1.32pp known, gate enforced pre-entry |
| Exit problem | Unsolved (multi-reason, thesis-driven) | Solved (deterministic 15-min expiry) |

### Distance metric

**Futures needs:** (1) a scoring function that actually predicts direction, (2) calibration
of that function into probabilities, (3) friction measurement, (4) an edge gate. It has
none of these.

**S2 needs:** (1) either a model that adds lift above market mid, or (2) enough observations
for the isotonic calibrator to detect and exploit systematic market-mid bias. It has the
architecture for both but insufficient data (80 obs, 0 refits).

S2's remaining gap is **data volume + model lift**. Futures' remaining gap is
**the entire probability pipeline**.

---

## §9 — Quantitative Summary

```
FUTURES:
  ρ = −0.049  (wrong direction)
  BSS = −0.192 (worse than guessing)
  Resolution/Uncertainty = 0.041 (4.1% of information captured)
  Net edge = −42 BPS (losing money)
  Architecture gap: score → probability → edge → gate (all missing)

S2 BINARY:
  ρ = +0.091  (right direction, not significant)
  BSS = −0.029 (nearly neutral vs base-rate)
  Resolution/Uncertainty = 0.115 (11.5% of information captured)
  Net edge = untested (gate blocked all rounds)
  Architecture gap: model lift (mechanism exists, data insufficient)

VERDICT: S2 is structurally closer to edge.
  But "closer to edge" ≠ "has edge."
  Both systems are pre-validation.
  The honest answer is: neither has earned the right to trade capital.
```

---

## §10 — Operational Implications

### 10.1 Do Not Relax the S2 Gate

The 3pp executable-edge gate is doing exactly what it should: preventing
trades when the model has no lift. Relaxing it to "see what happens"
would be the S2 equivalent of futures' current state — trading without
validated edge.

### 10.2 The S2 Freeze Is the Right Experiment

The 30-day frozen experiment (Day 2/30) is accumulating passive
observations that will (a) test whether the isotonic calibrator can find
systematic market-mid bias, and (b) build the sample size needed for
per-bucket significance. At the current rate (~40 rounds/day), Day 30
will have ~1200 observations — enough for meaningful bucket statistics.

### 10.3 For Futures: The Problem Is Upstream

The futures system doesn't need better execution, better routing, or
better risk management. It needs a scoring function that predicts
direction (ρ > 0, p < 0.05). Until that exists, everything downstream
is optimization of nothing.

### 10.4 Capital Allocation Recommendation

| System | Recommendation | Rationale |
|--------|---------------|-----------|
| Futures | Maintain observation mode | Negative edge, negative BSS |
| S2 | Continue 30-day freeze, do not deploy | Architecture sound, data insufficient |
| Capital | Keep in reserve ($9247 futures + $900 S2) | Neither has earned deployment |

---

## §11 — What Would Change This Verdict

### For S2 to earn deployment:
1. Isotonic calibrator refits after 150 observations (confident threshold)
2. Post-refit BSS > 0.0 (model beats base-rate)
3. Executable edge > 3pp for ≥10% of rounds (survival rate)
4. 50+ live trades with net edge > 0

### For Futures to earn deployment:
1. A new scoring function with ρ > 0.05, p < 0.05 (n ≥ 100)
2. Calibration of that function into [0, 1] probabilities with BSS > 0
3. Friction measurement showing net edge > 0 BPS
4. 100+ episodes under the new model confirming edge persistence

### Timeline expectation:
- S2: ~70 more observations to confidence threshold → Day ~4. Refit verdict by Day 5–6.
  If isotonic finds signal, tradability test by Day 10. Earliest possible deployment
  justification: Day 15.
- Futures: Requires model redesign (not threshold tuning). No timeline estimate is honest.

---

*This document is the 7th audit in the validation series. It is binding
and cannot be overridden by subsequent configuration changes.*
