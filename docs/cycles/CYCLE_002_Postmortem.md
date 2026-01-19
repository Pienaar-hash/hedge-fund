# CYCLE_002 Postmortem — Observation Phase Complete

**Period:** December 18, 2025 – January 19, 2026  
**Duration:** 33 days  
**Status:** LOCKED (no doctrine changes)  
**Version:** v7.8 Doctrine Kernel

---

## Executive Summary

CYCLE_002 was a month-long observation of the regime-gated execution system under challenging market conditions. Initial reporting indicated the system was "flat and refusing to trade" — this was **incorrect**. Post-cycle forensic analysis revealed:

- **270 completed trade episodes** occurred within the observation window
- **Trading happened on 15 of 26 active days**
- **Net PnL: -$54.54** (gross: -$11.25, fees: $43.28)
- System behaved exactly as designed: cautious entry, rapid exit, bounded losses

The reporting failure was caused by conflating position snapshots with historical activity. An **Episode Ledger** has been added to prevent future blind spots.

---

## Corrected Metrics

| Metric | Initially Reported | Actual |
|--------|-------------------|--------|
| Trading Days | 0 | **15** |
| Blocked Days | 32+ | 9 |
| Completed Episodes | 0 | **270** |
| Entry Fills | 0 | **431** |
| Exit Fills | 0 | **1,666** |
| Doctrine Allows | 0 | **977** |
| Veto Rate | ~100% | **96.9%** |
| Net PnL | Unknown | **-$54.54** |

---

## What Actually Happened

### Behavioral Pattern

The system exhibited **structural skepticism under false regime escape**:

1. **Permission windows opened frequently** — Regime conditions occasionally satisfied entry criteria
2. **Execution attempted cautiously** — Small position sizes, limited exposure
3. **Follow-through failed repeatedly** — Thesis broke down quickly
4. **Positions unwound rapidly** — Exit:Entry ratio of 3.9:1
5. **Exposure never accumulated** — Positions closed within trading windows
6. **Losses remained bounded** — Fee-dominated, not trend losses

### Daily Activity

```
Phase 1: Active Trading (Dec 18-25)
  - 8 trading days
  - 251 entries, 1,199 exits
  - High activity, rapid cycling

Phase 2: Blocked (Dec 26-31)
  - 4 blocked days, 2 trading days
  - System correctly refused most entries
  - Minor unwinding continued

Phase 3: Extended Pause (Jan 2-10)
  - 8 consecutive blocked days
  - Zero fills
  - Regime unstable, doctrine working as intended

Phase 4: Final Activity (Jan 11-19)
  - 6 days with activity
  - Declining entry rate
  - Final unwinding on Jan 19
```

---

## Why Reporting Failed

| Data Source | What It Shows | What It Does NOT Show |
|-------------|--------------|----------------------|
| `positions_state.json` | Current exposure | Historical participation |
| `regime_pressure.json` | Permission dynamics | Execution reality |
| `cycle_statistics/` | Gate behavior | Completed trade cycles |
| `orders_executed.jsonl` | **Ground truth** | Nothing |

**Root cause:** Treating a snapshot (`positions_state == []`) as evidence of historical inactivity. Positions opened and closed within windows, leaving the snapshot empty — but trading occurred.

---

## PnL Attribution

| Component | Amount |
|-----------|--------|
| Gross Trading PnL | -$11.25 |
| Total Fees | -$43.28 |
| **Net PnL** | **-$54.54** |

**Interpretation:** 
- Gross losses were minimal (-$11)
- Fee drag dominated at small trade sizes
- No catastrophic drawdown events
- Losses explainable and bounded

---

## Doctrine Validation

### What Worked Correctly ✓

- **Regime gating** — No entries during CHOPPY/CRISIS
- **Stability requirements** — Waited for regime confirmation
- **Exit logic** — Thesis-based exits triggered appropriately
- **Risk limits** — No overexposure accumulated
- **Veto persistence** — 96.9% rejection rate appropriate for conditions

### Falsification Results

| Rule | Criterion | Result |
|------|-----------|--------|
| RF-2 | Seatbelt ratio < 5% | ✓ Pass (0.4%) |
| RF-3 | Zero CHOPPY entries | ✓ Pass |
| RF-4 | All exits logged | ✓ Pass |

---

## Remediation: Episode Ledger

To prevent future analytical blind spots, an **Episode Ledger** has been added:

**Location:** `logs/state/episode_ledger.json`  
**Owner:** `episode_ledger` (derived, read-only)  
**Update:** On-demand rebuild from execution logs

### Episode Schema

```json
{
  "episode_id": "EP_0001",
  "symbol": "BTCUSDT",
  "side": "LONG",
  "entry_ts": "2025-12-18T06:19:58+00:00",
  "exit_ts": "2025-12-18T11:01:03+00:00",
  "duration_hours": 4.68,
  "entry_fills": 6,
  "exit_fills": 1,
  "entry_notional": 953.94,
  "exit_notional": 87.22,
  "gross_pnl": 0.50,
  "fees": 0.33,
  "net_pnl": 0.17,
  "exit_reason": "tp",
  "strategy": "vol_target"
}
```

### Aggregate Stats

The ledger now tracks:
- Total episodes completed
- Gross/net PnL with fee breakdown
- Win rate and average duration
- Exit reason distribution

---

## Conclusions

### CYCLE_002 was successful — not profitable, but informative

- **Did not reveal edge** — Market offered permission, not continuation
- **Did reveal protection** — Bounded losses under deception
- **Validated discipline** — System did not overtrade false structure
- **Exposed fee drag** — Small scale trading is fee-dominated
- **Proved survivability** — Can operate in hostile conditions without damage

### What must NOT change

- Doctrine thresholds
- Regime definitions  
- Hostility handling
- Stability requirements
- Exit logic

All performed correctly.

### What changed (observability only)

- Episode Ledger added for completed trade cycle visibility
- Manifest updated with `episode_ledger` state file
- Tests added for ledger functionality

---

## CYCLE_003 Entry Criteria

CYCLE_002 is locked. CYCLE_003 begins with:

1. **Same doctrine** — No parameter changes
2. **Better visibility** — Episode ledger active
3. **Clear success criteria:**
   - Sustained confidence escape (>0.65 for 48h+)
   - Lower exit:entry ratio (<2:1)
   - Positive net of fees

If CYCLE_003 shows edge, we act.  
If not, we keep listening.

---

*Document generated: January 19, 2026*  
*Doctrine version: v7.8*  
*Episode ledger version: 1.0*
