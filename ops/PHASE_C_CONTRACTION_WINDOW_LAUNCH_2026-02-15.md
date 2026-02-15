# Phase C — Contraction Window Launch Memo

**Date:** 15 February 2026  
**System:** GPT Hedge v7.9-S1  
**Status:** 14-day scored validation window — active  
**Classification:** Fund-Ops Governance

---

## 1. Executive Summary

The trading system has been structurally contracted to enforce measurement discipline. Scoring is now live. Risk caps are tightened. The universe is reduced to three liquid majors. A 14-day validation window has begun with no configuration changes permitted.

---

## 2. What Was Corrected

The prior architecture had a measurement gap that prevented edge attribution:

| Issue | Impact |
|-------|--------|
| Confidence defaulted to 1.0 for all trades | Every trade appeared equally convicted — no differentiation possible |
| Conviction engine ran in shadow mode only | Scoring was computed but never influenced capital deployment |
| Hybrid scoring was not enabled | Score decomposition logs were never populated |
| Episode records carried no scoring fields | Completed trades could not be joined back to the signals that created them |
| Risk limits were too permissive | 10% daily loss cap, 15 concurrent positions — inconsistent with capital preservation |

**Consequence:** The prior 660-trade sample is structurally incapable of proving or disproving edge. It cannot be used to evaluate strategy quality.

---

## 3. Changes Made

All changes are committed and auditable.

| Change | Detail |
|--------|--------|
| Scoring pipeline activated | Every trade now carries conviction score, hybrid score, confidence, and strategy attribution |
| Conviction mode | Shadow → **Live** (minimum entry band: "low") |
| Hybrid scoring | **Enabled** — score decomposition now populates |
| Confidence default | 1.0 → **0.25** (unscored trades are penalized, not promoted) |
| Strategy attribution gate | Entries without strategy attribution are rejected |
| Universe | 8 symbols → **3** (BTC, ETH, SOL) |
| Daily loss cap | 10% → **1%** |
| Weekly loss cap | **3%** (new) |
| Max concurrent positions | 15 → **2** |

**Commits:** `6733c08a`, `9c069755`

---

## 4. Day 0 Baseline

Recorded 15 February 2026 10:19 UTC.

| Metric | Value |
|--------|-------|
| NAV | $9,688.30 |
| Market regime | CHOPPY (78.8%) |
| Open positions | 0 |
| Post-contraction episodes | 0 |
| Hybrid score dispersion (std dev) | 0.0011 |
| Conviction distribution | 100% very_low |
| Capital at risk | $0.00 |

**Configuration hashes (locked):**

| File | SHA-256 (prefix) |
|------|-----------------|
| strategy_config.json | `267dd161` |
| risk_limits.json | `9f8d0b37` |
| pairs_universe.json | `885539d6` |

---

## 5. Success Criteria

Success at Day 14 is defined by structural differentiation, not returns.

| Criterion | Threshold |
|-----------|-----------|
| Conviction band spread | Trades distributed across multiple bands (not collapsed to one) |
| Scoring–outcome relationship | Higher conviction bands produce better PnL than lower bands |
| Hybrid score dispersion | Materially wider than baseline 0.0011 |
| Risk compliance | No daily or weekly loss cap breaches |
| Configuration stability | All three config hashes unchanged |

**Returns may be negative.** A small controlled loss in a hostile regime with clear scoring differentiation is progress. A gain without differentiation is not.

---

## 6. Commitments During This Window

For 14 days (15 February – 1 March 2026):

- No configuration changes
- No scoring weight adjustments
- No conviction threshold modifications
- No universe expansion
- No risk cap increases
- No binary sleeve deployment without dispersion proof

The system will be observed, not optimized.

---

## 7. Next Communication

Day 14 performance brief will be issued on or after 1 March 2026, containing:

- Conviction band distribution and PnL by band
- Hybrid score dispersion analysis
- Risk compliance confirmation
- Configuration hash verification
- Forward recommendation

---

*Prepared by: GPT Hedge Fund-Ops*  
*Window: 15 Feb – 1 Mar 2026*  
*Governance ref: Phase C contraction, v7.9-S1*
