# GPT-HEDGE v7.x

## REGIME-CYCLE POST-MORTEM (FORENSIC TEMPLATE)

**Purpose**  
To analyze one *complete regime cycle* (entry → hold → exit → flat) and determine whether the **doctrine behaved correctly**, independent of PnL.

This document is not a performance review.  
It is a **constitutional audit**.

---

## I. Cycle Identification

| Field | Value |
|-------|-------|
| **Cycle ID** | |
| **Dates** | `YYYY-MM-DD → YYYY-MM-DD` |
| **Primary Asset(s)** | |
| **Timeframe(s)** | |
| **Strategy Heads Active** | |
| **Executor Version / Git SHA** | |

---

## II. Regime Timeline (Sentinel-X Authority)

| Timestamp | Regime | Confidence | Cycles Stable | Crisis |
|-----------|--------|------------|---------------|--------|
| | | | | |

**Narrative (1–2 lines only):**

> Example: "Market transitioned from CHOPPY → TREND_DOWN with confidence rising above 0.55 and remained stable for 4 cycles."

---

## III. Entry Authorization Review

### A. Regime Permission

- [ ] Regime allowed participation (`TREND_UP`, `TREND_DOWN`, etc.)
- [ ] Confidence ≥ threshold
- [ ] Regime persistence ≥ N cycles
- [ ] No ambiguity flags present

**If NO ENTRY occurred during an allowed regime:**  
→ Explicitly evaluate **RF-1 (Regime Alignment Without Participation)**

---

### B. Screener Hygiene

| Symbol | Signal | Trend | Screener Decision | Reason |
|--------|--------|-------|-------------------|--------|
| | | | | |

Confirm:
- Signals were directional, not authoritative
- Counter-trend signals were blocked at screener level
- No screener bypass observed

---

### C. Doctrine Gate

| Symbol | Doctrine Verdict | Reason |
|--------|------------------|--------|
| | | |

Confirm:
- Doctrine vetoes were explicit and logged
- No silent refusals
- No config-based overrides

---

## IV. Position Lifecycle Review

For each position that **did** open:

### A. Entry Context

| Field | Value |
|-------|-------|
| Entry Regime | |
| Regime Confidence | |
| Entry Price | |
| Initial Size | |
| Head / Strategy | |
| Alpha Survival | |

**Question:**

> Did we enter because the market allowed us — not because a signal fired?

---

### B. Hold Phase Behavior

- [ ] No churn
- [ ] No size oscillation without regime change
- [ ] No stop-loss pressure during valid regime
- [ ] Exposure tracked regime confidence

**Notes:**  
(Only structural observations. No storytelling.)

---

## V. Exit Analysis (CRITICAL)

### A. Exit Trigger Classification

| Symbol | Exit Reason | Urgency | Price at Exit |
|--------|-------------|---------|---------------|
| | | | |

Allowed exit reasons only:
- `REGIME_FLIP`
- `STRUCTURAL_FAILURE`
- `TIME_STOP`
- `CRISIS`
- `SEATBELT` (secondary only)

---

### B. Exit Sequence Quality

Confirm:
- [ ] Exit reason preceded price damage
- [ ] Exits occurred before RF-2 (seatbelt dominance)
- [ ] Exit urgency matched asset volatility
- [ ] No panic correlation across assets

---

### C. Flat State Verification

After exits:
- Position count = 0
- No re-entries under same regime
- System stayed quiet

This is required to validate **GF-1 (Flatness Stable)**.

---

## VI. Capital Behavior Summary

| Metric | Value |
|--------|-------|
| Peak Exposure | |
| Exit Exposure | |
| Unrealized PnL at First Exit | |
| Realized PnL | |
| Max Adverse Excursion | |

**Interpretation Rule:**  
PnL is **descriptive**, never justificatory.

---

## VII. Falsification Checklist

Reference: [DOCTRINE_FALSIFICATION_CRITERIA.md](DOCTRINE_FALSIFICATION_CRITERIA.md)

### 🔴 Red Flags (Any = Doctrine Failure)

- [ ] RF-1: Regime alignment without participation
- [ ] RF-2: Seatbelt-dominated exits
- [ ] RF-3: Churn during CHOPPY
- [ ] RF-4: Crisis without contraction

If **any checked** → Doctrine falsified. Stop analysis.

---

### 🟡 Yellow Flags (Investigate)

- [ ] YF-1: Excessive flat time
- [ ] YF-2: Exit too gentle
- [ ] YF-3: Asymmetric exit speed

Document findings, no tuning yet.

---

### 🟢 Green Flags (Confirmation)

- [ ] GF-1: Flatness stable
- [ ] GF-2: Exits explainable in one sentence
- [ ] GF-3: Capital reacted before pain
- [ ] GF-4: Exit gradient matched asset character

---

## VIII. One-Sentence Verdict

Complete **exactly one sentence**:

> "This cycle demonstrates that the system **[did / did not]** understand when it was allowed to exist in the market."

No adjectives. No PnL references.

---

## IX. Amendment Consideration

- [ ] No amendment required
- [ ] Amendment proposed (attach falsification evidence)

Reminder:

> Amendments are allowed **only after cycles**, never during them.

---

## X. Archival

| Field | Value |
|-------|-------|
| Logged to | `logs/post_mortems/CYCLE_ID.md` |
| Linked doctrine version | |
| Linked falsification criteria version | |

---

## Final Note

If this document feels boring to write, the system is healthy.  
If it feels emotional, something leaked.

---

*Template Version: 1.0*  
*Established: 2025-12-18*  
*Reference: DOCTRINE_FALSIFICATION_CRITERIA.md v1.0*
