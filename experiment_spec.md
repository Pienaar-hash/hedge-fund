# Binary Lab — Experiment Specification

**Report date:** 2026-03-30
**Status:** Experimental. No claims of profitability. S1 completed (inconclusive). S2 active with ablation gates.

---

## Overview

Binary Lab is a dual-sleeve 30-day controlled experiment on binary prediction markets. The experiment operates under strict freeze discipline — no parameter changes, no tuning, deterministic rule enforcement.

Two sleeves:
- **S1 (Synthetic Binary):** Validates directional signal quality over 15-minute horizons
- **S2 (Polymarket CLOB):** Validates a calibrated probability model on binary outcomes

---

## S1: Synthetic Binary Sleeve

### Hypothesis

High-conviction directional signals from the futures pipeline exhibit band separation: trades tagged `very_high` conviction should produce higher expected value than `high`, which should exceed `medium`. This relationship should be monotonic.

### Entry Rules

| Parameter | Value |
|-----------|-------|
| Round size | $20 per round (fixed) |
| Horizon | 15 minutes |
| Min conviction band | medium |
| Regime confidence floor | ≥ 0.60 |
| Blocked regimes | CHOPPY |
| Max concurrent | 3 |
| Stacking | Prohibited |
| Martingale | Prohibited |

Entry is a single submission attempt per round. Position is held to resolution (no early exit).

### Sizing

| Parameter | Value |
|-----------|-------|
| Capital allocation | $2,000 |
| Per-round size | $20 (exact, not variable) |
| Max deployed | $1,200 |

### Constraints

- Freeze discipline: all parameters immutable for the 30-day window
- Round duration: 900 s (15 min)
- Entry offset: 30 s after round start
- Deterministic reducer: `state + event + limits → next_state + action`
- No mid-run restarts
- Config hash verification enforced at each activation

### Failure Criteria

| Condition | Threshold | Action |
|-----------|-----------|--------|
| NAV breach | NAV < $1,700 | Hard termination |
| Drawdown breach | DD > $300 (15%) | Hard termination |
| Band separation failure | Monotonic ordering absent at n ≥ 100 | Hypothesis rejected |
| Rule violation | Any concurrent cap / stacking / size mismatch | Hard termination |
| Freeze hash mismatch | Config hash ≠ frozen baseline | Hard termination |

### Success Rubric

- Band separation monotonic: `very_high > high > medium` EV
- Positive EV in at least one conviction band
- Drawdown stays below $300 throughout
- Zero rule violations

---

## S2: Polymarket CLOB Sleeve

### Hypothesis

A calibrated probability model can exploit payoff asymmetry at extreme CLOB prices. Edge concentrates in tail regions (price < 30% or > 70%) where market mis-pricing is structurally larger.

### Model Architecture

Dual-track design:

| Track | Method | Authority |
|-------|--------|-----------|
| Baseline | Always predicts `p_yes_mid` (CLOB mid-price) | Default |
| Calibrated | Isotonic regression fitted once ≥ 50 observations | Active at 50–150 obs; confident at ≥ 150 obs |

Both tracks are logged for lift measurement. The calibrated track does not activate until minimum observation threshold is met.

### Entry Rules

| Parameter | Value |
|-----------|-------|
| Min edge threshold | 0.03 (3 percentage points, mid-to-mid) |
| Executable edge gate | `|p_model − ask| ≥ 3pp` (friction check) |
| Max spread | 0.04 |
| Min time remaining | 120 s |
| Max quote age | 75 s |
| Ablation gate | `min_edge_abs = 0.10`, YES_ONLY (when enabled) |

### Sizing

| Parameter | Value |
|-----------|-------|
| Capital allocation | $900 |
| Per-round size | $30 |

### Friction Handling

- Quote reconstruction: `mid ± spread/2` (NO-side derived as complement)
- Entry cost = ask-side (not mid) — friction is baked into edge calculation
- Polymarket fee: `2% × min(p, 1−p) × notional`

### Constraints

- Freeze discipline: identical to S1
- Config hash verification required
- Polymarket datasets scoped to `BINARY_LAB_ONLY` — forbidden for `futures_execution`, `doctrine_kernel`, `risk_limits`, `router`
- Containment is enforced: S2 data does not leak into the primary execution pipeline

### Failure Criteria

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Calibration failure | BSS ≤ 0 at n ≥ 100 | Hypothesis rejected |
| No tail profit | Tail regions (< 30%, > 70%) not profitable at n ≥ 100 | Hypothesis rejected |
| Paper PnL shortfall | Paper PnL < 70% of backtest PnL | Hypothesis weakened |
| Config hash mismatch | Frozen config changed | Hard termination |

### Success Rubric

- Brier Skill Score > 0
- Payoff asymmetry: > 50% of PnL from tail regions
- Paper PnL ≥ 70% of backtest
- Tail regions remain profitable

### Measurement

| Metric | Method |
|--------|--------|
| Win rate | Outcomes / total rounds |
| Expected value | Mean net PnL per round |
| BSS (Brier Skill Score) | `1 − (Brier_model / Brier_baseline)` |
| Band separation | EV by edge bucket |
| Calibration | Isotonic residuals by decile |

---

## State Machine

Both sleeves use a deterministic state reducer:

```
state + event + limits → next_state + action
```

**Status flow:**
```
DISABLED → NOT_DEPLOYED → ACTIVE → TERMINATED / COMPLETED
```

Hard termination triggers: kill line breach, rule violation, freeze hash mismatch.

---

## Activation Requirements

For live activation (per P2 decision document):

1. Prediction phase = `P2_PRODUCTION`
2. Config hash proof passes
3. Polymarket datasets marked `PRODUCTION_ELIGIBLE`
4. Dual-key acknowledgment: `ACTIVATION_WINDOW_ACK=1` (env) + config flag
5. Manifest audit returns `MANIFEST_OK`

---

## Code Structure

| Component | File | Role |
|-----------|------|------|
| State machine | `execution/binary_lab_executor.py` | Deterministic reducer, freeze checks |
| S2 model | `execution/binary_lab_s2_model.py` | Dual baseline/calibrated tracks, isotonic refit |
| S2 signals | `execution/binary_lab_s2_signals.py` | CLOB quotes → features → edge calculation |
| S2 shadow | `execution/binary_lab_s2_shadow.py` | 15 min round orchestration, binary resolution |
| S2 runtime | `execution/binary_lab_runtime.py` | State surface writer |
| S2 evaluation | `scripts/binary_lab_s2_eval.py` | Post-hoc: edge buckets, calibration, friction |
| S1 limits | `config/binary_lab_limits.json` | S1 frozen contract |
| S2 limits | `config/binary_lab_limits_s2.json` | S2 frozen contract |

---

*Source files: `config/binary_lab_limits.json`, `config/binary_lab_limits_s2.json`, `execution/binary_lab_executor.py`, `execution/binary_lab_s2_model.py`, `docs/notes/BINARY_LAB_S1.md`, `ops/BINARY_LAB_EXECUTOR_STATE_MACHINE.md`, `docs/decisions/P2_BINARY_LAB_ACTIVATION_DECISION_v1.md`*
