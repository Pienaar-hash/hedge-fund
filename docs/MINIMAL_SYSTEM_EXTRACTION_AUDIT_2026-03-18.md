# Minimal System Extraction Audit — 2026-03-18

**Status:** Binding
**Audit:** 6 of 6 (final synthesis)
**Depends on:** All five predecessor audits (2026-03-18 series)
**Authority:** DLE Doctrine, DLE Constitution v1, DLE Gate Invariants v1

---

## §1 — Purpose

This audit reduces the futures engine to its **minimal falsifiable core** while preserving:

* explicit decision authority
* append-only observability
* measurable signal → outcome linkage
* friction-aware tradability
* calibration discipline

Anything that does not improve prediction, improve net edge after friction, or enforce a hard risk / authority boundary is removable. This follows the DLE constitutional rule that **features are liabilities until proven otherwise** (Constitution §12.1).

This audit consumes only the outputs of the five completed predecessor audits:

| # | Audit | File | Key output consumed |
|---|-------|------|---------------------|
| 1 | Signal → Outcome Causality | `SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md` | `hybrid_score` is the sole upstream composite worth testing; monotonicity and $p$-value framework defined; selection-bias problem identified |
| 2 | Probability-First Mapping | `PROBABILITY_FIRST_MAPPING_AUDIT_2026-03-18.md` | Target architecture: `x_t → p_long, p_short, q_exec, q_risk → threshold rule → size`; mapping table for every current component |
| 3 | Over-Architecture Detection | `OVER_ARCHITECTURE_AUDIT_2026-03-18.md` | Classification of regime model (TRANSFORM), conviction scoring (TRANSFORM), veto logic (CONTROL), routing (CONTROL); no component clears the bar for proven SIGNAL |
| 4 | Friction-Aware Edge | `FRICTION_AWARE_EDGE_AUDIT_2026-03-18.md` | Friction decomposition framework; kill rate, fee-to-edge ratio, break-even hurdle defined; verdict classes TRADABLE / MARGINAL / NOT_TRADABLE |
| 5 | Calibration | `CALIBRATION_AUDIT_2026-03-18.md` | Brier score, BSS, ECE, Murphy decomposition defined; verdict classes CALIBRATED / OVERCONFIDENT / COLLAPSED / MISCALIBRATED; conviction score was never designed as a probability |

No new codebase analysis is performed. This audit is a **synthesis and extraction**, not a discovery.

---

## §2 — Binding Extraction Rules

### Rule 1: Three component classes only

Every component must be classified as one of:

| Class | Survives extraction? | Criterion |
|-------|---------------------|-----------|
| **SIGNAL** | Yes | Measurably improves calibration, monotonicity, or post-friction edge |
| **CONTROL** | Yes (if indispensable) | Enforces a hard authority, state, or catastrophic-risk boundary |
| **DEAD / TRANSFORM** | No | Reshapes existing information without proven incremental value |

Only SIGNAL and indispensable CONTROL may survive. TRANSFORM layers are removable by default. This follows the Over-Architecture audit directly: no reviewed component currently clears the bar for proven SIGNAL, but the numeric inputs to `hybrid_score` remain candidate features.

### Rule 2: No narrative authority

No component may survive because it "captures market structure," "reflects conviction," or "improves discipline" unless that claim is tied to one of:

* lower Brier score or better BSS
* stronger monotonicity ($\rho > 0.15$ with $p < 0.05$)
* better post-friction expectancy (net edge > 0, kill rate < 25%)
* necessary fail-closed safety behavior

Anything else is meaning inflation (DLE Doctrine §4.3) and must be excluded.

### Rule 3: Probability-first precedence

The extracted system must be expressible in the form:

$$\text{features} \;\to\; \text{calibrated probability} \;\to\; \text{edge after cost} \;\to\; \text{decision}$$

If a component cannot be mapped into that chain, it must justify itself as CONTROL only. This is the irreducible production target from the Probability-First audit (§7).

### Rule 4: Hard authority boundaries remain

Minimal does not mean anarchic. The extracted system must still obey:

* explicit decision / permit logic as the governing model (DLE Constitution §1)
* fail-closed behavior on ambiguity (Gate Invariant #2)
* denial logging as first-class artifact (Gate Invariant #10)
* append-only episode observability (DLE Doctrine §5.1)

Those are constitutional, not optional.

---

## §3 — Predecessor Audit Findings Summary

| Audit | Central finding | Extraction implication |
|-------|----------------|----------------------|
| **Signal Causality** | `hybrid_score` is the sole upstream composite with instrumented monotonicity testing. No global validation exists yet — traded-region analysis is selection-biased. The audit framework (Spearman $\rho$, $p$-value, Q5–Q1 spread, temporal stability) is defined but requires passive observation data to resolve. | The minimal core must retain `hybrid_score` or its numeric components as the prediction input. All downstream transforms of this score must justify themselves independently. |
| **Probability-First** | The irreducible system is `x_t → p_long, p_short, q_exec, q_risk → threshold rule → size`. Every current component must map into this chain as either a feature feeding calibrated probability, or a numeric threshold on a control probability. Components that cannot do either job should be removed. | The extraction target architecture is defined. Conviction bands, regime labels as authorization primitives, and head priority logic have no role in this chain. |
| **Over-Architecture** | No component clears the bar for proven SIGNAL. Regime model and conviction scoring are TRANSFORM. Veto logic and routing are CONTROL. The edge lives upstream of these layers — in the numeric features, not the architectural scaffolding. | TRANSFORM layers are removable by default. CONTROL layers must be evaluated for essentiality. |
| **Friction-Aware Edge** | Fee drag is the only cleanly measurable friction component. Kill rate > 40% on any symbol is a structural warning. Fee-to-edge ratio > 0.5 is a system-level warning. Duration may be a hidden friction lever — shorter holds have higher friction drag per unit of edge. Maker routing quality may matter more than signal quality. | The minimal system must include friction measurement and execution-quality tracking. No decision should be made without a cost estimate. |
| **Calibration** | Conviction score was never designed as a probability. Brier score is the single most important number. Overconfidence is the dangerous failure mode. Collapse-to-mean is the silent failure. The fix is cheap — isotonic recalibration following the `binary_lab_s2_model` pattern. | The minimal system must produce a calibrated probability, not a conviction band. The calibration state must be exposed and testable. |

---

## §4 — Component Classification

### Master classification table

| Component | Current role | Classification | Verdict | Justification |
|-----------|-------------|----------------|---------|---------------|
| `hybrid_score` numeric inputs | Upstream composite: trend, carry, expectancy, router quality, rv momentum | SIGNAL candidate | **KEEP_AS_SIGNAL** | Sole upstream composite with instrumented monotonicity testing (Causality §4.1). Numeric components are the minimal feature set. |
| `signal_screener` — score computation | Computes `hybrid_score` from numeric factors | SIGNAL candidate | **KEEP_AS_SIGNAL** | Produces the only prediction input with a defined signal–outcome test path. |
| `signal_screener` — conviction overlay | Adds conviction bands, conviction multipliers | TRANSFORM | **REMOVE** | Conviction engine is a downstream transform (Over-Architecture §2). No incremental calibration or monotonicity proof. |
| `conviction_engine` | Deterministic sizing transform: score → band → multiplier | TRANSFORM | **REMOVE** | Over-Architecture audit: TRANSFORM. Calibration audit: conviction score was never designed as a probability. No Brier/BSS evaluation exists. Replaced by continuous `size(p)` function. |
| `Sentinel-X` — regime probabilities | Continuous probability vector over 6 regime states | SIGNAL candidate (conditional) | **DEMOTE_TO_FEATURE** | Probabilities are valid numeric features for the calibrated model, but only if they improve prediction. No Brier/log-loss evaluation exists for regime probabilities themselves. Kept as optional model input; removed from authorization authority. |
| `Sentinel-X` — hard regime labels | Categorical label used to gate entry direction | TRANSFORM → authorization | **REMOVE** | Over-Architecture audit: TRANSFORM. Probability-First audit: replace with `regime_probs[j]` as numeric features; remove from direct authorization. No ablation shows label-gating improves net edge. |
| `Sentinel-X` — cycles stable / stickiness | Integer counter enforcing regime stability | TRANSFORM | **REMOVE** | Include as numeric feature if independently predictive; otherwise remove. No isolated evaluation exists. |
| `doctrine_kernel` — catastrophic vetoes | Stale NAV, crisis mode, execution crunch, portfolio DD halt, kill switch enforcement | CONTROL | **KEEP_AS_CONTROL** | Constitutional fail-closed safety. Stale state denials, catastrophe protection, and kill switch logic are irreducible. |
| `doctrine_kernel` — regime-direction gates | TREND_UP → long only, TREND_DOWN → short only, CHOPPY/CRISIS → blocked | TRANSFORM → authorization | **REMOVE** | Narrative authority. No monotonicity proof for direction gating. The Probability-First audit requires direction to derive from $\arg\max(p_\text{long}, p_\text{short})$, not from regime labels. |
| `doctrine_kernel` — confidence floor gates | Regime confidence below threshold → veto | TRANSFORM → authorization | **REMOVE** | Convert to numeric threshold on data-quality probability $q_\text{live} \geq \tau_\text{live}$ if retained; otherwise remove. Hard label confidence has no calibration evaluation. |
| `doctrine_kernel` — alpha survival gate | Entry blocked if alpha signal has decayed | TRANSFORM | **REMOVE** | Subsume into signal staleness: if features at decision time are stale, `q_\text{live}` handles it. No independent proof of edge contribution. |
| `risk_limits` — DD / stale / catastrophe caps | Per-symbol caps, portfolio DD halt, correlation limits | CONTROL | **KEEP_AS_CONTROL** | Secondary veto on hard risk boundaries. Constitutional safety. Maps to $q_\text{risk}$ thresholds. |
| `order_router` | Maker-first POST\_ONLY with taker fallback, TWAP, child aggregation | CONTROL | **KEEP_AS_CONTROL** | Friction audit: execution efficiency directly affects net edge. Controls friction, does not produce signal. |
| `Hydra` — head scores | Per-head numeric scores (TREND, MEAN\_REVERT, RELATIVE\_VALUE, CATEGORY, VOL\_HARVEST, EMERGENT\_ALPHA) | SIGNAL candidate (conditional) | **DEMOTE_TO_FEATURE** | Numeric scores are valid features for the calibrated model. The multi-head *authority* (priority, arbitration, budget enforcement, intent generation) is removed; the *numbers* survive as optional ensemble inputs. |
| `Hydra` — priority / arbitration / budgets | Head priority ordering, conflict resolution, per-head NAV budgets | TRANSFORM → orchestration | **REMOVE** | No measured probability lift from arbitration logic. Budget enforcement is a sizing concern, not a prediction concern. Replaced by portfolio-level risk caps in `risk_limits`. |
| `Cerberus` | Dynamic head multipliers based on portfolio health, execution quality, regime confidence | TRANSFORM (observation only) | **REMOVE** | Over-Architecture candidates list. Observation-only multipliers with no proven edge. The module explicitly does not create signals or override doctrine. |
| `alpha_router` | Regime-based allocation multipliers, dynamic capital allocation | TRANSFORM | **REMOVE** | Over-Architecture audit: regime multipliers (1.05, 0.90, 0.85, 0.60) are unproven TRANSFORM. Probability-First audit: move to $q_\text{risk}$ or $\text{size}(p)$ if empirically justified; otherwise remove. |
| `Minotaur` | Microstructure-aware execution: wide-spread detection, TWAP slicing, throttling | CONTROL | **KEEP_AS_CONTROL** | Protects against execution-quality degradation. Friction audit: execution quality directly affects net edge. Does not produce signal; prevents bad fills. |
| `slippage_model` | Expected vs realized slippage EWMA, spread tracking | CONTROL (measurement) | **KEEP_AS_CONTROL** | Friction audit: slippage is a friction source. Measurement feeds cost estimate for edge calculation. Never gates; pure metrics. |
| `NAV` | Sole NAV authority: `nav_health_snapshot()` | CONTROL (truth) | **KEEP_AS_CONTROL** | Constitutional. Stale NAV triggers veto. Cannot be removed without losing risk truth. |
| `position_ledger` | Canonical position state + TP/SL merge | CONTROL (state) | **KEEP_AS_CONTROL** | State authority. Episode observability depends on accurate position tracking. |
| `DLE shadow` | Shadow observation: logs doctrine verdicts, never blocks | CONTROL (audit) | **KEEP_AS_CONTROL** | DLE Doctrine §5.4: shadow mode is the current operational mode. Audit surface for decision traceability. |
| `binary_lab_s2_model` | Dual prediction tracks: naive baseline + isotonic calibration | SIGNAL (reference pattern) | **KEEP_AS_SIGNAL** | Demonstrates the calibration discipline the minimal core requires. Isotonic calibration pattern is the recommended fix from the Calibration audit (§8). |
| `hydra_monotonicity` | Score quality measurement: Spearman $\rho$, quintile analysis | CONTROL (measurement) | **KEEP_AS_CONTROL** | Provides the signal–outcome causality test infrastructure. Without it, the system cannot detect signal degradation. |

### Doctrine kernel — vetoes retained vs removed

The doctrine kernel is the only component that requires a split classification. The following table enumerates which veto categories survive in the minimal core.

| Veto category | Current behavior | Verdict | Rationale |
|---------------|-----------------|---------|-----------|
| **Stale NAV** | Veto if NAV age > 90s | **KEEP** | Fail-closed on stale state. Constitutional. Maps to $q_\text{live} \geq \tau_\text{live}$. |
| **Crisis mode** | Veto all entries during CRISIS regime | **KEEP** | Catastrophe protection. Equivalent to $q_\text{risk} < \tau_\text{crisis}$. |
| **Execution crunch** | Veto entries during acute execution-quality degradation | **KEEP** | Maps to $q_\text{exec} < \tau_\text{exec}$. Prevents orders in degraded conditions. |
| **Portfolio DD halt** | Veto all entries when drawdown exceeds threshold | **KEEP** | Hard risk boundary. Already in `risk_limits`; doctrine enforcement is belt-and-suspenders. |
| **Kill switch** | Blocks risk-increasing orders; never blocks doctrine exits | **KEEP** | Emergency stop. Constitutional (Law 7). |
| **Regime-direction match** | TREND\_UP → long only, TREND\_DOWN → short only | **REMOVE** | Narrative authority. Direction must derive from $\arg\max(p_\text{long}, p_\text{short})$ per Probability-First audit §2.3. |
| **Confidence floor** | Veto if regime confidence below threshold | **REMOVE** | Hard label confidence has no calibration evaluation. Convert to $q_\text{live}$ if needed. |
| **Blocked execution regimes** | CHOPPY / CRISIS → no entries | **PARTIALLY KEEP** | Crisis block survives as catastrophe protection. CHOPPY block is removed — it is narrative authority unless proven to improve net edge. |
| **Alpha survival floor** | Entry blocked if alpha signal decayed | **REMOVE** | Subsume into feature staleness via $q_\text{live}$. No independent edge proof. |
| **Head-budget constraint** | Entry blocked if Hydra head budget exhausted | **REMOVE** | Head budgets are REMOVE (Hydra arbitration). Portfolio-level caps in `risk_limits` replace this. |

---

## §5 — Minimal System Specification

### Architecture

The minimal futures core has five layers. No layer may be omitted. No additional layer is permitted unless it can independently justify itself under §2 rules.

```
Observation → Prediction → Decision → Execution → Ledger
```

### Layer 1: Observation

**Purpose:** Capture market state and execution outcomes for every candidate signal, whether acted on or not.

| Artifact | Content | Write frequency |
|----------|---------|-----------------|
| Passive observation log | All candidate signals with features at emission time, regardless of whether an entry follows | Every screening cycle |
| Episode record | Entry price, exit price, side, duration, fees, notional, net PnL for every completed trade | On episode close |
| Feature snapshot | Full $x_t$ vector at decision time | Every decision point |

**Critical requirement:** Passive observations must include vetoed candidates to eliminate the selection-bias problem identified in the Causality audit (§7). Without this, the signal cannot be globally validated.

### Layer 2: Prediction

**Purpose:** Convert numeric features into a calibrated probability of a net-positive outcome.

**Feature vector** $x_t$ (minimal candidate set):

| Feature | Source | Type |
|---------|--------|------|
| $f_\text{trend}$ | Trend strength (from hybrid score components) | Continuous |
| $f_\text{carry}$ | Carry estimate | Continuous |
| $f_\text{expectancy}$ | Expectancy alpha | Continuous |
| $f_\text{rq}$ | Router quality score | Continuous |
| $f_\text{rv}$ | Realized volatility momentum | Continuous |
| $f_\text{cost}$ | Estimated round-trip cost in BPS (from slippage model + fee schedule) | Continuous |
| $f_\text{regime\_probs}[j]$ | Regime probability vector from Sentinel-X (optional — include only if improves calibration) | Continuous vector |
| $f_\text{head\_k}$ | Hydra head score for head $k$ (optional — include only if improves calibration) | Continuous |

**Probability model:**

$$p_\text{long} = \text{Cal}(\text{model}_\text{long}(x_t))$$
$$p_\text{short} = \text{Cal}(\text{model}_\text{short}(x_t))$$

where Cal is isotonic or Platt calibration, following the `binary_lab_s2_model` pattern: preserve baseline output, then apply explicit calibration once enough observations exist, and report calibration state rather than hiding it.

**Calibration state must be exposed:**

| Field | Meaning |
|-------|---------|
| `brier` | Current Brier score |
| `bss` | Brier Skill Score relative to base-rate baseline |
| `ece` | Expected Calibration Error |
| `calibration_verdict` | CALIBRATED / OVERCONFIDENT / COLLAPSED / MISCALIBRATED / INSUFFICIENT\_DATA |
| `n_episodes` | Sample count supporting the current calibration |

### Layer 3: Decision

**Purpose:** Convert calibrated probability into a binary enter/no-enter decision and a continuous position size.

**Decision rule:**

$$\text{enter\_long} = \mathbf{1}\{p_\text{long} \geq \tau_p \;\wedge\; \text{edge}_\text{long} > \tau_\text{edge} \;\wedge\; q_\text{exec} \geq \tau_\text{exec} \;\wedge\; q_\text{risk} \geq \tau_\text{risk} \;\wedge\; q_\text{live} \geq \tau_\text{live}\}$$

$$\text{enter\_short} = \mathbf{1}\{p_\text{short} \geq \tau_p \;\wedge\; \text{edge}_\text{short} > \tau_\text{edge} \;\wedge\; q_\text{exec} \geq \tau_\text{exec} \;\wedge\; q_\text{risk} \geq \tau_\text{risk} \;\wedge\; q_\text{live} \geq \tau_\text{live}\}$$

where:

| Symbol | Meaning | Source |
|--------|---------|--------|
| $p_\text{long}, p_\text{short}$ | Calibrated probability of net-positive outcome | Prediction layer |
| $\text{edge}$ | $p \cdot G - (1-p) \cdot L$ — expected payoff after cost | Prediction layer + friction estimate |
| $q_\text{exec}$ | Execution quality probability ($1 - P(\text{execution failure})$) | Minotaur / router health |
| $q_\text{risk}$ | Risk-state probability (not in DD halt, not in crisis) | Risk limits / NAV state |
| $q_\text{live}$ | Data liveness ($1 - P(\text{stale state})$) | NAV age, feature freshness |
| $\tau_p, \tau_\text{edge}, \tau_\text{exec}, \tau_\text{risk}, \tau_\text{live}$ | Thresholds (tuned by calibration, not by narrative) | Config |

**Sizing rule:**

$$\text{size\_frac} = \text{clip}\left(k \cdot \frac{p - \tau_p}{1 - \tau_p},\; 0,\; \text{size\_max}\right)$$

This replaces discrete conviction bands with a continuous monotone function of calibrated probability.

**Retained control checks (from §4):**

| Check | Maps to | Fail behavior |
|-------|---------|---------------|
| Stale NAV | $q_\text{live} < \tau_\text{live}$ | Deny entry |
| Crisis mode | $q_\text{risk} < \tau_\text{crisis}$ | Deny entry |
| Execution crunch | $q_\text{exec} < \tau_\text{exec}$ | Deny entry |
| Portfolio DD halt | $q_\text{risk} < \tau_\text{dd}$ | Deny entry |
| Kill switch | Exogenous emergency flag | Deny risk-increasing; never block exits |
| Per-symbol / correlation caps | Post-decision sizing cap | Reduce size or deny |

### Layer 4: Execution

**Purpose:** Route the sized order to the exchange with measurable friction.

| Component | Role | Intelligence claims |
|-----------|------|---------------------|
| `order_router` | Maker-first POST\_ONLY with taker fallback | **None** — execution efficiency, not prediction |
| `Minotaur` | Wide-spread detection, TWAP slicing, throttling | **None** — microstructure safety, not prediction |
| `slippage_model` | EWMA expected vs realized slippage | **None** — friction measurement |

The execution layer must produce per-order slippage, fee, and fill-type records for the friction audit pipeline.

### Layer 5: Ledger

**Purpose:** Maintain the append-only audit trail required by DLE Doctrine.

| Artifact | Schema concept | Write rule |
|----------|---------------|------------|
| **Signal observation record** | DLE InputSnapshot | Append on every screening cycle (including vetoed candidates) |
| **Probability / calibration record** | Prediction state | Append on recalibration; expose current Brier/BSS/ECE |
| **Decision / denial record** | DLE DecisionEnvelope + Denial | Append on every decision point; denied entries are first-class (Gate Invariant #10) |
| **Episode record** | DLE LedgerEntry | Append on episode close; links entry decision to realized outcome |

### Explicit non-goals

The minimal system does **not** attempt to:

* classify market regimes as authorization primitives
* score conviction as a separate stage
* arbitrate between multiple strategy heads for intent priority
* dynamically allocate capital across heads or regime states
* provide microstructure-aware intelligence beyond safety guards
* optimize routing beyond maker-first with fallback

---

## §6 — Removal Table

| Component | Current role | Verdict | Why removable | What replaces it |
|-----------|------------|---------|---------------|-----------------|
| `conviction_engine` | Score → band → size multiplier | **REMOVE** | TRANSFORM (Over-Arch). No Brier/BSS evaluation. Conviction score ≠ probability (Calibration §12). | Continuous $\text{size}(p)$ from calibrated probability |
| `Sentinel-X` hard labels | Categorical regime → direction gate | **REMOVE** | TRANSFORM (Over-Arch). Narrative authority. No ablation vs simpler gating. | Regime probability vector as numeric features |
| `Sentinel-X` cycles stable | Integer counter → stability gate | **REMOVE** | No isolated evaluation. Possible numeric feature only. | Include in $x_t$ if predictive; otherwise drop |
| Doctrine regime-direction gates | TREND\_UP → long only, etc. | **REMOVE** | Narrative authority. Direction from $\arg\max(p_\text{long}, p_\text{short})$. | Probability-derived direction |
| Doctrine confidence floor | Regime confidence < threshold → veto | **REMOVE** | Hard label confidence uncalibrated. | $q_\text{live} \geq \tau_\text{live}$ for data quality |
| Doctrine alpha survival gate | Alpha decay → veto | **REMOVE** | No independent edge proof. | Feature staleness via $q_\text{live}$ |
| Doctrine CHOPPY block | CHOPPY regime → no entries | **REMOVE** | Narrative authority. No net-edge proof for label-based blocking. | Handled by $p_t < \tau_p$ (low probability = no entry) |
| Doctrine head-budget constraint | Head budget exhausted → veto | **REMOVE** | Hydra arbitration is REMOVE. | Portfolio-level caps in `risk_limits` |
| `Hydra` priority / arbitration | Head conflict resolution, priority ordering | **REMOVE** | TRANSFORM (Over-Arch). No measured probability lift. | Head scores as features; model does the weighting |
| `Hydra` per-head budgets | NAV allocation per strategy head | **REMOVE** | Sizing concern, not prediction. | Portfolio-level caps in `risk_limits` |
| `Cerberus` | Dynamic head multipliers | **REMOVE** | Observation-only multipliers. No edge proof. Module self-describes as not creating signals. | Nothing — multipliers were observation-only |
| `alpha_router` | Regime-based allocation multipliers | **REMOVE** | TRANSFORM (Over-Arch). Regime multipliers (1.05, 0.90, 0.85, 0.60) unproven. | $\text{size}(p)$ or $q_\text{risk}$ threshold if empirically justified |

---

## §7 — Contract Surfaces

The minimal system declares four required artifact types, mapped to DLE concepts.

### Artifact 1: Signal observation record

| Field | Type | DLE concept |
|-------|------|-------------|
| `observation_id` | UUID | InputSnapshot reference |
| `symbol` | String | Scope |
| `timestamp` | Unix seconds | Decision time |
| `features` | Object (full $x_t$) | InputSnapshot data |
| `p_long`, `p_short` | Float | Prediction output |
| `edge_long`, `edge_short` | Float | Expected payoff |
| `acted_on` | Boolean | Whether entry followed |
| `veto_reason` | String or null | If denied, why |

**Write rule:** Append every screening cycle. Includes vetoed candidates.

### Artifact 2: Probability / calibration record

| Field | Type | DLE concept |
|-------|------|-------------|
| `calibration_ts` | Unix seconds | Recalibration time |
| `brier` | Float | Brier score |
| `bss` | Float | Brier Skill Score |
| `ece` | Float | Expected Calibration Error |
| `n_episodes` | Integer | Sample count |
| `calibration_verdict` | Enum | CALIBRATED / OVERCONFIDENT / COLLAPSED / MISCALIBRATED / INSUFFICIENT\_DATA |
| `calibration_method` | String | "isotonic" / "platt" / "none" |

**Write rule:** Append on every recalibration event. Expose current state in `logs/state/`.

### Artifact 3: Decision / denial record

| Field | Type | DLE concept |
|-------|------|-------------|
| `decision_id` | UUID | DecisionEnvelope reference |
| `observation_id` | UUID | Links to signal observation |
| `symbol` | String | Scope |
| `direction` | LONG / SHORT | Derived from $\arg\max(p)$ |
| `permitted` | Boolean | Whether entry was allowed |
| `deny_reason` | Canonical code or null | DLE denial taxonomy |
| `p_trade` | Float | Calibrated probability used |
| `edge_trade` | Float | Expected edge used |
| `size_frac` | Float | Position size fraction |
| `control_checks` | Object | $q_\text{exec}$, $q_\text{risk}$, $q_\text{live}$ values |

**Write rule:** Append on every decision point. Denials are first-class (Gate Invariant #10).

### Artifact 4: Episode record

| Field | Type | DLE concept |
|-------|------|-------------|
| `episode_id` | UUID | LedgerEntry reference |
| `decision_id` | UUID | Links to decision |
| `symbol` | String | Scope |
| `side` | LONG / SHORT | As executed |
| `entry_price`, `exit_price` | Float | Fill prices |
| `entry_ts`, `exit_ts` | Unix seconds | Timestamps |
| `duration_s` | Integer | Hold time |
| `fees` | Float | Total commissions |
| `gross_pnl`, `net_pnl` | Float | Before / after fees |
| `realized_return` | Float | Direction-adjusted fractional return |
| `p_at_entry` | Float | Calibrated probability at decision time |
| `edge_at_entry` | Float | Expected edge at decision time |

**Write rule:** Append on episode close. Immutable (Doctrine §5.1).

---

## §8 — Validation Plan

The minimal system is only valid if it can be tested in isolation. Five test categories are required. Each specifies what can be tested today versus what requires the probability-first migration.

### Test 1: Causality

| Test | Method | Runnable today? |
|------|--------|-----------------|
| Monotonicity on full passive universe | Spearman $\rho$ on passive observations (including vetoed candidates) | **No** — requires passive observation logging (not yet deployed) |
| Monotonicity on traded region | Spearman $\rho$ on completed episodes | **Yes** — `hydra_monotonicity.py` already runs in executor |
| Selection-bias delta | Compare $\rho_\text{traded}$ vs $\rho_\text{passive}$ | **No** — requires passive observation data |
| Temporal stability | $\rho$ computed in non-overlapping time slices | **Yes** — framework defined in Causality audit §3.4 |
| Q5–Q1 spread | Return spread between top and bottom quintiles | **Yes** — already computed in `hydra_monotonicity.py` |

### Test 2: Calibration

| Test | Method | Runnable today? |
|------|--------|-----------------|
| Brier score | $\frac{1}{N}\sum(p_i - o_i)^2$ on completed episodes | **No** — requires calibrated $p$ output (not yet produced) |
| BSS | Brier relative to base-rate baseline | **No** — same dependency |
| ECE | Weighted mean |gap| across reliability-diagram buckets | **No** — same dependency |
| Murphy decomposition | Reliability, resolution, uncertainty components | **No** — same dependency |
| Model vs naive baseline | Brier(model) vs Brier(always predict base rate) | **No** — same dependency |

### Test 3: Tradability

| Test | Method | Runnable today? |
|------|--------|-----------------|
| Raw vs net edge | Decompose per-episode return into raw edge and fee drag | **Yes** — episode data + fee data available |
| Friction-kill rate | Count trades where raw edge > 0 but net edge ≤ 0 | **Yes** — `compute_friction_decomposition()` defined |
| Break-even hurdle | Mean fee drag as minimum required raw edge | **Yes** — same pipeline |
| Fee-to-edge ratio | $\Sigma \text{fee\_drag} / \Sigma \text{raw\_edge}$ | **Yes** — same pipeline |
| Kill rate by score bucket | Quintile kill-rate analysis | **Yes** — defined in Friction audit §4.1 |

### Test 4: Control sufficiency

| Test | Method | Runnable today? |
|------|--------|-----------------|
| Stale state denies | Inject stale NAV; confirm veto fires | **Yes** — doctrine kernel handles this now |
| Fail-closed ambiguity denies | Inject malformed request; confirm DENY | **Yes** — can test against doctrine kernel |
| Catastrophic risk denies | Inject DD-halt state; confirm all entries blocked | **Yes** — risk\_limits handles this now |
| Kill switch exempts exits | Activate kill switch; confirm doctrine exits proceed | **Yes** — Law 7 enforcement exists |
| Crisis mode blocks entries | Inject CRISIS regime; confirm entries blocked | **Yes** — retained crisis veto |

### Test 5: Ablation

| Test | Method | Runnable today? |
|------|--------|-----------------|
| Minimal system vs full stack | Compare calibration metrics (Brier, $\rho$) between reduced and full system on same episode set | **No** — requires calibrated probability model to exist |
| No hidden dependence on removed layers | Verify reduced system produces same or better metrics without conviction, regime labels, head arbitration | **No** — same dependency |
| Regime-label ablation | Compare net edge with vs without regime-direction gating | **Partially** — can compare episode outcomes by regime label retrospectively |

---

## §9 — Pass / Fail Criteria

### PASS

The extraction passes only if the reduced system:

| Criterion | Measurement | Threshold |
|-----------|------------|-----------|
| Calibration preserved or improved | Brier(minimal) ≤ Brier(full stack) | Must not degrade |
| Monotonicity preserved in traded region | $\rho_\text{minimal} \geq \rho_\text{full}$ | Must not degrade |
| Positive net edge after friction, if any exists | Mean net edge BPS > 0 | Same sign as full stack |
| Fully auditable | All four artifact types (§7) populated | Complete ledger trail |
| Fail-closed on ambiguity | All control tests (§8 Test 4) pass | Zero tolerance |
| Explainable by one competent human | No hidden layer interactions required to explain behavior | DLE Constitution §12.2 |

### FAIL

The extraction fails if:

* Removing a layer causes a **major** drop in calibration or net edge **and** that layer cannot be replaced by a simpler numeric feature.
* The remaining system still depends on hard regime labels, conviction bands, or qualitative priority logic for core behavior.
* The system cannot be expressed as $\text{features} \to \text{probability} \to \text{edge} \to \text{decision}$.
* Explanation still requires architectural storytelling — "the regime model detects X, which feeds conviction, which gates the head budget, which..." — instead of "the model says $p = 0.62$, edge = 4.1 BPS, thresholds met, enter."

---

## §10 — What This Audit Cannot See

| Limitation | Impact | Resolution path |
|-----------|--------|-----------------|
| **Selection bias** | Monotonicity measured on traded region only. Vetoed candidates have no outcome data. | Deploy passive observation logging; compare $\rho_\text{traded}$ vs $\rho_\text{passive}$. |
| **Variable horizon** | System uses thesis-driven exits, not fixed holding periods. Duration confounds return dispersion. | Log hold duration per episode; control for duration in bucketed analysis. Consider fixed-horizon marking for calibration training. |
| **Funding rate** | Perpetual futures funding is not tracked per episode. Holding periods crossing 8-hour intervals incur invisible costs. | Add funding snapshot at entry/exit to episode record. |
| **Calibrated model does not exist yet** | Tests 2 and 5 (Calibration, Ablation) require a probability model that has not been built. | Build using `binary_lab_s2_model` pattern; calibrate on accumulated episode data. |
| **Passive observation data is not yet collected** | Global validation (non-selection-biased) requires observations on vetoed candidates. | Deploy observation logging; accumulate sufficient data before claiming PASS. |
| **`enforcement_gate.py` does not exist** | DLE Phase C/D enforcement gate is specified but not implemented. The minimal system operates in SHADOW\_MODE. | Acknowledged as future work. Shadow-mode audit surface is sufficient for current extraction. |

---

## §11 — Relationship to Other Audits

| Predecessor | What this audit consumes | What this audit adds |
|-------------|-------------------------|---------------------|
| **Signal Causality** | `hybrid_score` as sole testable composite. Monotonicity / $p$-value framework. Selection-bias identification. | Identifies which numeric components of `hybrid_score` form the minimal $x_t$. Determines that `hybrid_score` itself (or its decomposition) is the only KEEP\_AS\_SIGNAL survivor. |
| **Probability-First** | Target architecture `x_t → p → edge → decision`. Mapping table for every component. Migration sequence. | Executes the mapping: classifies every component as KEEP, DEMOTE, or REMOVE. Produces the canonical minimal spec that the mapping audit defined as the target. |
| **Over-Architecture** | SIGNAL / TRANSFORM / CONTROL classification for four major components. Evidence standard ("no component clears the bar for proven SIGNAL"). | Extends classification to all ~20 components. Applies the evidence standard uniformly. Uses TRANSFORM = removable as the binding extraction rule. |
| **Friction-Aware Edge** | Kill rate, fee-to-edge ratio, break-even hurdle frameworks. | Integrates friction measurement into the decision rule ($\text{edge}_t > \tau_\text{edge}$). Retains execution-quality controls as essential CONTROL. |
| **Calibration** | Brier, BSS, ECE metrics. Verdict classes. "Conviction score was never designed as a probability." | Replaces conviction with calibrated probability as the core prediction output. Mandates calibration state exposure. |

---

## §12 — Implementation Status

### Completed

| Item | Status |
|------|--------|
| Component classification (§4) | **Done** — all major components classified |
| Minimal system specification (§5) | **Done** — five-layer architecture defined |
| Removal table (§6) | **Done** — 12 components removed or demoted |
| Contract surfaces (§7) | **Done** — four artifact types specified |
| Validation plan (§8) | **Done** — five test categories with runnable-today assessment |
| Monotonicity framework | **Exists** — `execution/hydra_monotonicity.py` runs in live executor |
| Friction decomposition framework | **Exists** — functions defined in Friction audit, ready for implementation |
| Calibration reference pattern | **Exists** — `execution/binary_lab_s2_model.py` demonstrates isotonic calibration |
| Control tests (stale NAV, crisis, DD, kill switch) | **Runnable today** — existing doctrine/risk infrastructure |

### Pending

| Item | Dependency | Blocks |
|------|-----------|--------|
| Passive observation logging | Deploy feature in executor | Causality Test 1 (global monotonicity), selection-bias delta |
| Calibrated probability model | Accumulate episode data; fit using `binary_lab_s2_model` pattern | Calibration Tests (Brier, BSS, ECE), Decision layer, Sizing rule |
| Feature vector finalization | Ablation on optional features (regime probs, head scores) | Final $x_t$ specification |
| Continuous $\text{size}(p)$ implementation | Calibrated model exists | Conviction engine replacement |
| Doctrine kernel refactor | Remove regime-direction gates, confidence floor, alpha survival, head-budget veto; retain catastrophe controls | Decision layer simplification |
| Ablation testing (minimal vs full) | Calibrated model + sufficient episodes | PASS/FAIL determination |
| Funding rate per episode | Add funding snapshot at entry/exit | Full friction accounting |

---

## §13 — The Honest Position

1. **Most layers are reshaping, not producing.** The Over-Architecture audit found zero components that clear the bar for proven SIGNAL. The edge, if it exists, lives in the numeric features — not in the conviction bands, regime labels, or head-arbitration logic layered on top of them. The minimal core is dramatically smaller than the current stack.

2. **The minimal core is necessary but not sufficient.** Defining the smallest system does not prove that system works. The signal has never been globally validated (Causality audit §12). The calibrated probability model does not exist yet (Calibration audit §9). This audit defines the **target**, not the **current state**.

3. **Removal is the easy part.** The hard part is the five tests in §8 — particularly Causality Test 1 (passive-universe monotonicity) and Calibration Tests (Brier, BSS, ECE on the reduced system). Until those tests produce numbers, the extraction is a structural improvement, not a performance improvement.

4. **The doctrine kernel split is the most sensitive change.** Removing regime-direction gates means the system trades based on probability, not on regime narrative. That is the correct posture if the probability model is calibrated. If the model is miscalibrated, removing directional guardrails could be worse than keeping them. The validation plan (§8 Test 5) must resolve this before implementation.

5. **The fix is available.** The `binary_lab_s2_model` already demonstrates isotonic calibration in this codebase. The monotonicity framework already runs in the executor. The friction decomposition framework is defined and ready for implementation. The minimal system does not require new theory — it requires connecting the existing measurement tools to a simpler architecture, then running the tests honestly.

---

## §14 — Final Verdict

**`MINIMAL_CORE_FOUND`**

The smallest executable futures system that can honestly answer the four audit questions is:

1. **Does a signal exist?** — Test via monotonicity on `hybrid_score` components (framework exists, passive data pending).
2. **Does it calibrate?** — Test via Brier/BSS/ECE on calibrated probability output (model pending, pattern exists in `binary_lab_s2_model`).
3. **Does it survive friction?** — Test via net edge, kill rate, fee-to-edge ratio (framework exists, partially runnable today).
4. **Can it be traded without hidden dependence on architectural complexity?** — Yes: the minimal spec (§5) is expressible as $x_t \to p \to \text{edge} \to \text{decision}$ with five explicit control checks. No conviction bands. No regime authorization. No head arbitration.

The verdict is `MINIMAL_CORE_FOUND` because the reduced system is structurally sound and expressible in the required form. The verdict is **conditional** on the pending tests (§8, §12) — the structure is proven minimal, but the performance claim requires data that does not yet exist.

The stack is **not** still dependent on removed layers for its core operation: every removed component is either a TRANSFORM (reshaping existing signal) or a narrative authority (making decisions without calibrated justification). The retained controls are independently justified as fail-closed safety or constitutional requirements.

---

## Appendix A — Recommended Operational Prompt

This prompt is preserved as a binding artifact for future audit execution.

> You are conducting the **Minimal System Extraction Audit** for the GPT-Hedge futures engine.
>
> Your job is to reduce the system to its **smallest falsifiable core** while preserving explicit authority, auditability, and measurable post-friction edge.
>
> You must use the findings of:
>
> * the Signal → Outcome Causality Audit
> * the Probability-First Mapping Audit
> * the Over-Architecture Audit
> * the Friction-Aware Edge Audit
> * the Calibration Audit
>
> Tasks:
>
> 1. Identify the smallest numeric feature set that can still produce a tradable prediction.
> 2. Express the system in the form:
>    `features -> calibrated probability -> expected edge after cost -> decision`
> 3. Classify every major component as:
>    `KEEP_AS_SIGNAL`, `KEEP_AS_CONTROL`, `DEMOTE_TO_FEATURE`, or `REMOVE`
> 4. Remove any component that does not:
>
>    * improve calibration
>    * improve monotonicity
>    * improve post-friction net edge
>    * or enforce a hard fail-closed safety / authority boundary
> 5. Produce one canonical minimal system specification, not multiple alternatives.
> 6. Produce a removal table with justification for every deleted or demoted layer.
> 7. Produce a validation plan proving the reduced system can be tested in isolation.
>
> Constraints:
>
> * No narrative justifications.
> * No component survives on architectural sophistication alone.
> * Hard authority boundaries must remain explicit.
> * Denials, episodes, and audit artifacts remain first-class.
> * If a signal does not survive no-filter, no-transform, post-friction testing, treat it as nonexistent.
>
> Required outputs:
>
> * Minimal system spec
> * Removal / demotion table
> * Retained control boundaries
> * Isolation test plan
> * Final verdict:
>   `MINIMAL_CORE_FOUND` or `STACK_STILL_DEPENDENT`
