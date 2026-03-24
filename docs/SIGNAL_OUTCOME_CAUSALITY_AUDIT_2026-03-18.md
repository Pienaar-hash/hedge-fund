# Signal → Outcome Causality Audit — 2026-03-18

## Scope

This audit evaluates whether the primary trading signal in the futures stack has a **causal relationship** with realized trade outcomes, using the strictest standard available: monotonicity, statistical significance, and temporal stability. All narrative, regime-based, and ex-post rationalizations are rejected as evidence.

---

## 1. Signal definition

### Signal S: `hybrid_score`

| Property | Value |
|----------|-------|
| **Name** | `hybrid_score` |
| **Type** | Continuous scalar |
| **Range** | $[0, 1]$ |
| **Available at** | Decision time (before entry order) |
| **Logged** | `score_decomposition.jsonl` (per intent), `episode_ledger.json` (per episode) |
| **Join key** | `intent_id` |

**Why this signal:**

`hybrid_score` is the **sole upstream composite** that gates all trade emission. Every other signal either feeds into it (trend, carry, expectancy, router quality) or consumes it downstream (conviction, doctrine). If this signal lacks causal linkage to outcomes, nothing downstream can recover edge — it would only be reshaping noise.

**Composition (for reference, not for causal argument):**

$$S = \left(\sum_i w_i \cdot f_i\right) \cdot \text{rq\_mult} \cdot \text{decay}(t)$$

where $f_i \in \{\text{trend}, \text{carry}, \text{expectancy}, \text{router\_quality}, \text{rv\_momentum}\}$ with normalized weights. The composition is irrelevant to this audit — what matters is whether the final scalar predicts the outcome monotonically.

---

## 2. Outcome definition

### Outcome O: `realized_return`

| Property | Value |
|----------|-------|
| **Name** | `realized_return` |
| **Type** | Continuous (side-adjusted fractional return) |
| **Computation** | LONG: $(P_{\text{exit}} - P_{\text{entry}}) / P_{\text{entry}}$; SHORT: $(P_{\text{entry}} - P_{\text{exit}}) / P_{\text{entry}}$ |
| **Source** | `episode_ledger.json` → `avg_entry_price`, `avg_exit_price`, `side` |
| **Includes fees** | No (gross return). Net PnL available as `net_pnl` but mixes notional sizes. |

### Binary outcome (secondary)

$$Y = \mathbf{1}\{\text{realized\_return} > 0\}$$

This converts to a hit-rate analysis per bucket.

### Horizon

The system does **not** use a fixed holding horizon. Positions exit on thesis death, TP/SL, or regime change. Therefore:

- **Primary analysis**: variable-horizon realized return (as-traded).
- **Limitation**: variable horizon introduces confounding from hold duration. Longer holds mechanically produce wider return dispersion. The audit flags this but cannot correct it without synthetic fixed-horizon marking.

---

## 3. Evaluation framework

### 3.1 Bucketed analysis

Partition all completed episodes $\{(S_i, O_i)\}_{i=1}^N$ into $K$ equal-count buckets (quintiles $K=5$, deciles $K=10$).

For each bucket $B_k$:

$$\bar{S}_k = \frac{1}{|B_k|} \sum_{i \in B_k} S_i \qquad \bar{O}_k = \frac{1}{|B_k|} \sum_{i \in B_k} O_i \qquad \text{hit}_k = \frac{1}{|B_k|} \sum_{i \in B_k} \mathbf{1}\{O_i > 0\}$$

### 3.2 Monotonicity test

**Spearman rank correlation** on the $K$ bucket means $\{(\bar{S}_k, \bar{O}_k)\}$:

$$\rho = 1 - \frac{6 \sum d_i^2}{K(K^2 - 1)}$$

Interpretation:

| $\rho$ | Verdict |
|---------|---------|
| $> 0.80$ | Strong monotonic signal |
| $0.40$ – $0.80$ | Moderate monotonic signal |
| $0.15$ – $0.40$ | Weak signal (not reliably causal) |
| $-0.05$ – $0.15$ | Flat (no detectable signal) |
| $< -0.05$ | Inverted (signal is anti-predictive) |

**Also compute on raw observations** (not just bucket means) for an unbiased estimate.

### 3.3 Statistical significance

For raw-observation Spearman $\rho$ with $N$ episodes, the test statistic under $H_0: \rho = 0$ is:

$$t = \rho \sqrt{\frac{N - 2}{1 - \rho^2}}$$

which follows $t_{N-2}$ approximately. Report:

- $\rho$, $t$-stat, $p$-value
- Reject $H_0$ at $\alpha = 0.05$ (two-tailed)

**Minimum bar**: $p < 0.05$ **AND** $|\rho| > 0.05$ (economic significance floor).

### 3.4 Stability across time slices

Split the episode set into non-overlapping temporal thirds (T1, T2, T3 by entry timestamp). Compute $\rho$ separately within each slice.

| Stability class | Condition |
|-----------------|-----------|
| **Stable** | All three slices have $\rho > 0.10$ and same sign |
| **Unstable** | Any slice has opposite sign from the others |
| **Degrading** | $\rho_{T1} > \rho_{T2} > \rho_{T3}$ with $\rho_{T3} < 0.05$ |
| **Insufficient** | Any slice has $< 20$ episodes |

### 3.5 Q5–Q1 spread

The return spread between the top and bottom quintiles:

$$\Delta_{Q5-Q1} = \bar{O}_{Q5} - \bar{O}_{Q1}$$

This is the practical "is sorting by signal worth anything?" metric. Positive spread means higher-scored trades do better. Negative or zero spread means sorting is valueless.

---

## 4. Current evidence from the codebase

### 4.1 Existing tooling

The repo already contains `execution/hydra_monotonicity.py` which performs quintile bucketing and Spearman $\rho$ on `hybrid_score` vs `realized_return`. This is the **right test** for this audit.

**What it computes:**

- 5-bucket equal-count quintiles
- Mean score and mean return per bucket
- Spearman $\rho$ (custom implementation, tie-aware)
- Slope classification: upward ($\rho > 0.15$), flat ($0 \leq \rho \leq 0.15$), inverted ($\rho < -0.05$)
- Q5–Q1 return spread
- Per-head (regime-grouped) breakdown
- Head contamination flag (any head strong while global weak)

**What it does NOT compute:**

| Missing element | Severity | Impact |
|-----------------|----------|--------|
| $p$-value for $\rho$ | **Critical** | Cannot distinguish signal from noise at any confidence level |
| Confidence interval for bucket means | High | Cannot assess whether bucket return differences are real |
| Time-slice stability | **Critical** | Cannot detect regime-conditional signal breakdown or decay |
| Decile granularity option | Moderate | Quintiles may mask non-linearity in tails |
| Hit rate per bucket | Moderate | Lose binary-outcome interpretation |
| Duration confounding check | Moderate | Variable hold horizon not controlled |
| Bootstrap significance for Q5–Q1 | High | Spread could be noise |

### 4.2 Existing results (structural assessment)

The `hydra_monotonicity.persist_snapshot()` function runs on a ~3600s throttle inside the executor main loop. Its output goes to `logs/state/hydra_monotonicity.json`. This means:

- Results **do exist** on the deployed system.
- They update hourly, using the full episode ledger.
- The dashboard reads and displays them (color-coded $\rho$).

However, **no single run in the repo constitutes a controlled causality study** because:

1. No $p$-value is computed or logged.
2. No temporal stability check is performed.
3. No counterfactual (vetoed trades) is included.
4. The episode ledger is filtered to only executed trades, so the S→O relationship is conditioned on $S > \tau_{\text{entry}}$ (truncated left tail).

### 4.3 Selection bias: left truncation

Because only trades with `hybrid_score ≥ 0.55` (long) or `≥ 0.50` (short) are emitted, the observed $(S, O)$ distribution is **left-truncated**. The bottom 1–2 quintiles in the episode data are compressed into a narrow score band just above the entry threshold.

This means:

- Monotonicity measured on executed trades understates the true signal–outcome relationship if the signal works (floor effect).
- But it also means we cannot measure signal value below the threshold — we have no outcome data for scores $< 0.50$.

**Implication**: Even if $\rho > 0$ on executed trades, the causal claim applies only to the conditional distribution $S \geq \tau$. It does NOT prove the signal is useful for the full range. Conversely, if $\rho \approx 0$ on executed trades, the signal fails in the only region where it needs to work.

---

## 5. Signal vs outcome table (template)

This is the table that must be populated from the live episode ledger. The audit defines the schema; the executor's `hydra_monotonicity.py` produces values that fill the first three columns.

### 5.1 Quintile table

| Bucket | Score range | $\bar{S}$ | $\bar{O}$ (mean return) | Hit rate ($Y=1$ %) | $N$ | Avg hold (hrs) |
|--------|-------------|-----------|-------------------------|---------------------|-----|-----------------|
| Q1 (lowest) | — | — | — | — | — | — |
| Q2 | — | — | — | — | — | — |
| Q3 | — | — | — | — | — | — |
| Q4 | — | — | — | — | — | — |
| Q5 (highest) | — | — | — | — | — | — |

### 5.2 Summary statistics

| Metric | Value | Significance |
|--------|-------|--------------|
| Spearman $\rho$ (raw) | — | — |
| Spearman $\rho$ ($p$-value) | — | — |
| Spearman $\rho$ (bucket means) | — | — |
| Q5–Q1 spread | — | — |
| Q5–Q1 bootstrap $p$ | — | — |

### 5.3 Temporal stability

| Slice | Date range | $N$ | $\rho$ | Slope |
|-------|------------|-----|--------|-------|
| T1 | — | — | — | — |
| T2 | — | — | — | — |
| T3 | — | — | — | — |

### 5.4 Confound check: duration

| Bucket | Avg hold (hrs) | Std hold (hrs) |
|--------|----------------|----------------|
| Q1 | — | — |
| Q5 | — | — |

If Q5 holds are systematically longer than Q1 holds, the return spread may partly reflect duration exposure, not signal quality.

---

## 6. Monotonicity assessment

### 6.1 What must be true for "causal"

All of the following conditions must hold simultaneously:

1. **Monotonic relationship**: $\rho > 0.15$ on raw observations (not just bucket means).
2. **Statistical significance**: $p < 0.05$ for $H_0: \rho = 0$.
3. **Economic significance**: Q5–Q1 spread $> 0$ and larger than estimated transaction costs per round trip.
4. **Temporal stability**: $\rho > 0.10$ in all three time slices, same sign.
5. **No duration confound**: Q5 and Q1 average hold durations differ by $< 2\times$.

### 6.2 What constitutes "non-causal"

Any one of:

- $\rho \leq 0$ on raw observations.
- $\rho > 0$ but $p > 0.10$.
- Q5–Q1 spread $\leq 0$.
- Temporal instability: any time slice has opposite sign.

### 6.3 What constitutes "inconclusive"

- $0 < \rho \leq 0.15$ with $p < 0.05$ (statistically significant but economically marginal).
- Monotonicity holds at bucket level but not at raw observation level (aggregation artifact).
- $N < 50$ total episodes (insufficient power).
- Left truncation is so severe (score range $< 0.15$) that monotonicity cannot be tested.

---

## 7. Selection bias: the left-truncation problem

### 7.1 The core flaw

Every analysis in §3–§6 operates on **executed trades only** — episodes where `hybrid_score >= 0.50` (short) or `>= 0.55` (long). The observed score distribution is:

$$S_{\text{observed}} \in [0.50, 1.00]$$

This means:

- We cannot observe whether low scores produce bad outcomes.
- We cannot test global monotonicity across $[0, 1]$.
- We are testing **top-half ranking quality**, not signal validity.

### 7.2 The truncated monotonicity illusion

A system can pass all tests in §6 **and have zero edge**:

| Score range | True $E[O]$ |
|-------------|-------------|
| $[0.0, 0.5]$ | $+0.20\%$ |
| $[0.5, 1.0]$ | $+0.10\%$ |

Audit result: $\rho > 0$, Q5 > Q1, stable — **verdict CAUSAL**. Reality: the system is trading the **wrong side of the curve**. Lower scores are actually better.

This is why the v1 audit framework can confirm rank-ordering within the traded region but cannot confirm that the entry threshold is correct or that the signal has global edge.

### 7.3 Passive observation (implemented)

To close this gap, a passive observation log has been added to `signal_screener.py`:

**Log path**: `logs/execution/passive_observations.jsonl`

**What it captures**: Every scored intent from the **full unfiltered universe** — including symbols below the entry threshold and those filtered by router quality or RV momentum.

**Fields per record**:

| Field | Type | Purpose |
|-------|------|---------|
| `ts` | ISO timestamp | Decision time |
| `symbol` | string | Asset |
| `direction` | string | LONG/SHORT |
| `price` | float | Price at decision time |
| `hybrid_score` | float | Signal value $\in [0, 1]$ |
| `passes_threshold` | bool | Whether the score exceeded entry threshold |
| `components` | dict | Factor breakdown (trend, carry, expectancy, router) |
| `rq_score` | float | Router quality score |
| `rv_score` | float | RV momentum score |
| `regime` | string | Current regime label |

**Counterfactual return construction**: After horizon $H$, look up the actual price move for each symbol:

$$r_{\text{counterfactual}} = \frac{P_{t+H} - P_t}{P_t} \cdot \text{side\_sign}$$

This requires a separate backfill process (reading candle data at observation timestamps), but the passive log provides the join surface.

### 7.4 Selection bias test (implemented)

`compute_selection_bias(all_observations, traded_episodes)` compares:

- $\rho_{\text{traded}}$: Spearman on executed trades only
- $\rho_{\text{full}}$: Spearman on the full scored universe (once counterfactual returns are backfilled)

| $\rho_{\text{full}}$ vs $\rho_{\text{traded}}$ | Verdict |
|-------------------------------------------------|---------|
| $|\delta| < 0.10$ | Signal is real — truncation does not inflate monotonicity |
| $\delta > 0.10$ (traded >> full) | **Selection bias suspected** — apparent signal may be illusion |
| $\delta < -0.10$ (full >> traded) | Signal is stronger in the full range — threshold is too aggressive |

### 7.5 What this audit proves (honest scope)

**If the v1 tests pass** (§6 conditions met), the audit proves:

> "Within the traded region $[0.50, 1.00]$, higher `hybrid_score` ranks outcomes better."

**It does NOT prove**:

- That the signal has edge globally.
- That the entry threshold is optimal.
- That not trading low scores is justified.

**These require** the passive observation log to accumulate data and the selection bias test to be run on backfilled counterfactual returns.

---

## 8. Extended analyses (implemented)

### 8.1 Threshold optimality sweep

`compute_threshold_sweep()` sweeps $T \in [0.05, 0.95]$ in steps of $0.05$:

For each threshold $T$, considering only episodes with $S \geq T$:

| Metric | Formula |
|--------|---------|
| Trade frequency | $N_T / N_{\text{total}}$ |
| Mean return | $\bar{O}_T$ |
| Hit rate | $P(O > 0 \mid S \geq T)$ |
| Expectancy | $\text{hit} \cdot \bar{W} + (1 - \text{hit}) \cdot \bar{L}$ |

Outputs:

- Sweep table (threshold → metrics)
- $T^*$ = threshold maximizing expectancy
- $\Delta$ = $\text{expectancy}(T^*) - \text{expectancy}(T_{\text{current}})$

**Limitation (traded-only)**: This sweep covers only $[0.50, 1.00]$ on executed episodes. It finds the best threshold *within the traded range* but cannot evaluate $T < 0.50$ without passive observation data. Once counterfactual returns are backfilled, this function should be re-run on the full universe.

### 8.2 Continuous edge curve

`compute_edge_curve()` estimates $E[O \mid S = s]$ as a continuous function via kernel smoothing:

- Rectangular kernel with adaptive bandwidth
- $n = 20$ evaluation points across the observed score range
- Curve-level Spearman $\rho$ (monotonicity of the smooth function)

**Shape classification**:

| Shape | Detected when | Implication |
|-------|---------------|-------------|
| **Linear** | $\geq 75\%$ positive slopes, $|d^2| < \epsilon$ | Signal works proportionally |
| **Convex** | Positive slopes + positive curvature | Signal accelerates at high scores |
| **Concave** | Positive slopes + negative curvature | Signal saturates at high scores |
| **Step** | One large jump > 50% of total range | Not smooth — likely a regime boundary |
| **Inverted** | $\leq 25\%$ positive slopes | Signal is anti-predictive |
| **Noise** | None of the above | No detectable structure |

Buckets hide non-linearity. This curve exposes it.

### 8.3 Direction correctness

`compute_direction_accuracy()` computes $P(\text{correct direction} \mid \text{score bucket})$:

- Direction is correct if: LONG and $P_{\text{exit}} > P_{\text{entry}}$, or SHORT and $P_{\text{entry}} > P_{\text{exit}}$
- Baseline: 50% (random)
- Lift: accuracy $- 0.50$ per bucket

This separates **directional skill** from **magnitude prediction**. A signal can produce positive returns on average (via asymmetric payoffs) without actually predicting direction.

### 8.4 Friction overlay

`compute_friction_overlay()` computes **raw vs post-fee return** per score bucket:

For each bucket $B_k$:

| Metric | Source |
|--------|--------|
| Raw return | $(P_{\text{exit}} - P_{\text{entry}}) / P_{\text{entry}}$ (side-adjusted) |
| Net return | Raw $-$ fees / notional |
| Friction drag | Raw $-$ net |
| Edge erased | $\bar{O}_{\text{raw}} > 0$ AND $\bar{O}_{\text{net}} \leq 0$ |

Reports:

- $\rho_{\text{raw}}$: monotonicity on gross returns
- $\rho_{\text{net}}$: monotonicity on net returns
- Number of buckets where friction erases edge

If low-score buckets show positive gross return but negative net return, those trades destroy value after costs.

---

## 9. Verdict framework (upgraded)

### 9.1 Tier 1 — Traded-region causality

Based on executed episodes only (left-truncated at $S \geq 0.50$).

**CAUSAL (CONDITIONAL)**:

All conditions from §6.1 are met. The signal monotonically ranks outcomes within the traded region, with statistical significance and temporal stability. **But this is conditional** — it says nothing about global edge or threshold optimality.

**NON-CAUSAL**:

Any condition from §6.2 is met. The signal fails even in the region where the system trades. **Action**: the entire downstream stack is operating on noise.

**INCONCLUSIVE**:

Conditions from §6.3 apply.

### 9.2 Tier 2 — Global signal validation (requires passive observation data)

Once counterfactual returns are backfilled:

| Test | Condition | Verdict upgrade |
|------|-----------|-----------------|
| Selection bias | $|\delta_\rho| < 0.10$ | CONDITIONAL → **FULL CAUSAL** |
| Selection bias | $\delta_\rho > 0.10$ | CONDITIONAL → **SELECTION BIAS ILLUSION** |
| Threshold optimality | $T^* \approx T_{\text{current}}$ | Threshold is justified |
| Threshold optimality | $T^* \ll T_{\text{current}}$ | Over-filtering — leaving edge on the table |
| Edge curve | Linear or convex | Signal scales with score |
| Edge curve | Noise | Signal structure is not real |
| Direction accuracy | Lift $> 0$ with monotonic increase | True directional skill |
| Friction overlay | Net $\rho > 0$ | Edge survives costs |

### 9.3 Full verdict matrix

| Tier 1 result | Tier 2 result | System verdict |
|---------------|---------------|----------------|
| CAUSAL (COND) | Signal real + threshold OK | **PROVEN EDGE** — proceed to probability-first |
| CAUSAL (COND) | Selection bias illusion | **ILLUSORY** — edge is an artifact of filtering |
| CAUSAL (COND) | Signal real + threshold wrong | **MISTUNED** — edge exists but threshold destroys it |
| NON-CAUSAL | Any | **NO EDGE** — signal does not predict outcomes |
| INCONCLUSIVE | Any | **DEFER** — accumulate more data |

---

## 10. Relationship to other audits

This audit is the **prerequisite** for both:

1. **Probability-First Mapping Audit** (2026-03-18): If `hybrid_score` is non-causal, the probability-first rewrite has no signal to calibrate. The mapping audit assumes an upstream signal exists — this audit verifies that assumption.

2. **Over-Architecture Detection Audit** (2026-03-18): That audit classified regime, conviction, and veto layers as TRANSFORM/CONTROL rather than SIGNAL. This audit tests whether the signal they transform is itself real. If the signal is non-causal, those layers are transforming nothing.

The dependency chain is:

```
Signal Causality Audit (this document)
    ↓ verdict
    ├── PROVEN EDGE → Probability-First Audit actionable
    ├── ILLUSORY → Over-Architecture finding confirmed at deeper level
    ├── MISTUNED → Fix threshold, then re-run probability-first
    ├── NO EDGE → Signal must be replaced or rebuilt
    └── DEFER → Accumulate passive observations, re-run
```

---

## 11. Implementation status

### Completed

| Component | File | Status |
|-----------|------|--------|
| Spearman $\rho$ with $p$-value | `hydra_monotonicity.py` | `_spearman_p_value()` — pure Python via regularized beta |
| Hit rate per bucket | `hydra_monotonicity.py` | In `compute_monotonicity()` |
| Time-slice stability | `hydra_monotonicity.py` | `compute_time_stability()` |
| Duration confound | `hydra_monotonicity.py` | `compute_duration_by_bucket()` |
| Bootstrap Q5–Q1 | `hydra_monotonicity.py` | `compute_bootstrap_q5_q1()` |
| Threshold optimality sweep | `hydra_monotonicity.py` | `compute_threshold_sweep()` |
| Continuous edge curve | `hydra_monotonicity.py` | `compute_edge_curve()` |
| Direction correctness | `hydra_monotonicity.py` | `compute_direction_accuracy()` |
| Friction overlay | `hydra_monotonicity.py` | `compute_friction_overlay()` |
| Selection bias comparison | `hydra_monotonicity.py` | `compute_selection_bias()` |
| Passive observation log | `signal_screener.py` | Full universe logged to `passive_observations.jsonl` |
| State file integration | `hydra_monotonicity.py` | `persist_snapshot()` writes all to `hydra_monotonicity.json` |

### Pending (requires accumulated data)

| Component | Dependency | Action |
|-----------|------------|--------|
| Counterfactual return backfill | Passive observation records + candle data | Build offline script to join observation timestamps with actual OHLCV |
| Full-universe threshold sweep | Backfilled counterfactual returns | Re-run `compute_threshold_sweep()` on full data |
| Selection bias delta | Backfilled counterfactual returns | Run `compute_selection_bias()` |
| Tier 2 verdict | All above | Issue final verdict per §9.3 |

---

## 12. The honest position

The futures engine is currently a **decision-first system**: it filters before observing. The passive observation patch makes it an **observation-first system** — the first step toward truth extraction.

The uncomfortable question this audit is designed to answer:

> Is the architecture built around a signal that has never been globally validated?

The Tier 1 test answers: "Does the signal rank outcomes in the region where it trades?"

The Tier 2 test — when data accumulates — answers the harder question: "Should it be trading that region at all?"

Until Tier 2 data exists, any verdict issued from this audit is provisional.
