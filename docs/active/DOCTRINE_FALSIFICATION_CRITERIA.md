# Doctrine Falsification Criteria

## GPT-Hedge v7.X — Live-Fire Evaluation Framework

**Purpose**: Define the conditions under which the Trading Doctrine Engine would be proven *wrong* — not unprofitable, but architecturally incorrect.

**Philosophy**: Falsification comes before storytelling. These criteria must be defined *before* regime cycles complete, so post-mortems are forensic, not narrative.

---

## The One Question That Matters

After each regime cycle, ask:

> **Did the system behave like it understood *when not to exist*?**

If yes → doctrine is sound, regardless of PnL.  
If no → no amount of alpha mining will save it.

---

## I. HARD RED FLAGS — Doctrine Failure

Any one of these is sufficient to declare the doctrine **incorrect or incomplete**.  
Do not tune. Admit structural failure.

---

### 🔴 RF-1: Regime Alignment Without Participation

**Condition**
- Sentinel-X enters `TREND_UP` or `TREND_DOWN`
- Confidence ≥ threshold (default: 0.55)
- `cycles_stable` ≥ 2
- Screener produces direction-aligned signals
- **No entries for extended window (>3 cycles)**

**Log Pattern**
```
[sentinel_x] regime=TREND_UP confidence=0.7+ cycles_stable=3+
[screener] signal=BUY emitted for N symbols
[doctrine] VETO reason=??? (unexpected)
```

**Interpretation**
- Authority stack is over-constraining
- Capital withheld when market explicitly permits

**Diagnosis Targets**
- Screener hygiene too aggressive
- Doctrine gate too brittle
- Signal semantics mismatched to regime semantics

**Severity**: CRITICAL — This is the most important falsification signal.

---

### 🔴 RF-2: Seatbelt-Dominated Exits

**Condition**
- Majority of exits (>50%) triggered by `SEATBELT` / stop-loss
- `REGIME_FLIP` and `STRUCTURAL_FAILURE` exits are rare

**Log Pattern**
```
[exit_scanner] reason=SEATBELT (repeated)
[exit_scanner] reason=REGIME_FLIP (rare or absent)
```

**Interpretation**
- Thesis invalidation is lagging reality
- System reacting to price, not regime
- Rebuilt ATR stops with better vocabulary

**Severity**: CRITICAL — Violates core doctrine principle.

---

### 🔴 RF-3: Churn During CHOPPY

**Condition**
- Sentinel-X = `CHOPPY`
- Repeated partial entries or re-entries
- Capital oscillates instead of going quiet

**Log Pattern**
```
[sentinel_x] regime=CHOPPY
[executor] SEND_ORDER ... (entry, not exit)
[executor] SEND_ORDER ... (another entry)
```

**Interpretation**
- Regime authority is leaking
- Some module bypassing doctrine gate
- "Cleverness" overriding discipline

**Severity**: CRITICAL — Direct constitutional violation.

---

### 🔴 RF-4: Crisis ≠ Immediate Contraction

**Condition**
- Sentinel-X enters `CRISIS`
- Exposure reduction is slow, inconsistent, or incomplete
- Positions linger beyond 1-2 cycles

**Log Pattern**
```
[sentinel_x] regime=CRISIS crisis=True
[positions] count > 0 (persisting across cycles)
```

**Interpretation**
- Survival principle violated
- Any profit after this is luck, not design

**Severity**: CRITICAL — Existential failure mode.

---

## II. YELLOW FLAGS — Doctrine Stress

These don't invalidate the system, but demand investigation.

---

### 🟡 YF-1: Excessive Flat Time vs Opportunity

**Condition**
- Long stretches of `TREND_UP` or `TREND_DOWN` regimes
- Few or no trades executed
- Post-analysis shows missed large directional moves

**Interpretation**
Could be:
- ✅ Correct discipline (signals didn't align)
- ❌ Over-refusal masquerading as prudence

**Resolution**: Only regime-cycle post-mortem can determine which.

**Action**: Review screener emission logs during trend periods.

---

### 🟡 YF-2: Exit Too Gentle

**Condition**
- Thesis clearly invalidated (regime flip confirmed)
- Exit urgency remains `STEPPED` for extended period
- Capital lingers while edge decays
- Unrealized PnL erodes significantly during exit

**Interpretation**
- Urgency calibration may be too conservative
- Not a logic failure, but parameter stress

**Action**: Review `STEPPED` → `IMMEDIATE` escalation thresholds.

---

### 🟡 YF-3: Asymmetric Exit Speed

**Condition**
- Winning positions exit slowly
- Losing positions exit quickly
- Pattern suggests price-reactive behavior, not thesis-reactive

**Interpretation**
- Possible emotional leak into exit logic
- May indicate hidden stop-loss dependency

**Action**: Audit exit reasons by PnL sign.

---

## III. GREEN FLAGS — Doctrine Working

These confirm the engine is sound, regardless of PnL.

---

### 🟢 GF-1: Flatness Is Stable

**Condition**
- Regime = `CHOPPY`
- Position count = 0
- No probing entries
- No "just one small trade"
- Idle state persists across multiple cycles

**Log Pattern**
```
[sentinel_x] regime=CHOPPY
[positions] count=0
[screener] emitted=0 (or vetoed by doctrine)
```

**Interpretation**: System knows when not to exist. Rare and valuable.

**Status**: ✅ Observed 2025-12-18

---

### 🟢 GF-2: Exits Are Explainable in One Sentence

**Condition**
Every exit can be summarized as:

> "The reason we entered no longer exists."

NOT:

> "Price went against us."

**Log Pattern**
```
[exit_scanner] reason=REGIME_FLIP explanation="Regime flipped from X to Y"
[exit_scanner] reason=STRUCTURAL_FAILURE explanation="Alpha survival < threshold"
```

**Interpretation**: Thesis-driven, not reactive.

**Status**: ✅ Observed 2025-12-18

---

### 🟢 GF-3: Capital Reacts Before Pain

**Condition**
- Exposure reduces *before* drawdown accelerates
- Unrealized PnL may still be positive at exit
- Exit triggered by regime, not by loss

**Interpretation**: Regime intelligence, not trading skill.

**Status**: ✅ Observed 2025-12-18 (exited with +$33 unrealized)

---

### 🟢 GF-4: Exit Gradient Matches Asset Character

**Condition**
- Higher-volatility assets exit faster
- Structural assets (BTC) exit slower
- No correlated panic exits

**Interpretation**: STEPPED urgency is contextual, not blunt.

**Status**: ✅ Observed 2025-12-18 (SOL fastest, BTC slowest)

---

## IV. Monitoring Checklist

### Per-Cycle Checks
- [ ] Regime state logged with confidence
- [ ] Entry vetoes have explicit reasons
- [ ] Exit triggers have explicit reasons
- [ ] Position count matches expected behavior

### Per-Regime Checks
- [ ] No RF-1 through RF-4 triggered
- [ ] Yellow flags investigated if present
- [ ] At least one green flag confirmed

### Post-Cycle Checks
- [ ] Exit reasons auditable
- [ ] No unexplained exposure changes
- [ ] Capital behavior matches regime narrative

---

## V. Falsification Log

Track occurrences of red/yellow flags for pattern detection.

| Date | Flag | Description | Resolution |
|------|------|-------------|------------|
| 2025-12-18 | GF-1,2,3,4 | First thesis-driven wind-down observed | Baseline established |

---

## VI. Amendment Protocol

This document may only be amended:
1. After a full regime cycle completes
2. With explicit falsification evidence
3. Never during active positions

**Reason**: Criteria defined under calm are honest. Criteria defined under stress are rationalizations.

---

*Document Version: 1.0*  
*Established: 2025-12-18*  
*First Validation: Wind-down under CHOPPY regime*
