# MHD Audit Report Template

> Use this template for all MHD audit reports. Fill in results, attach evidence.

---

## Report Metadata

| Field | Value |
|-------|-------|
| **Report ID** | `MHD-YYYY-MM-DD-NNN` |
| **Cycle** | CYCLE_NNN Phase X |
| **Audit Date** | YYYY-MM-DD |
| **Auditor** | (agent or human) |
| **System Version** | (from VERSION file) |

---

## Executive Summary

| Audit | Verdict | P0 | P1 | P2 |
|-------|---------|----|----|----| 
| 0: Evidence Admission | PASS/FAIL | 0 | 0 | 0 |
| 1A: Replay Determinism | PASS/FAIL | 0 | 0 | 0 |
| 1B: Scoring Decomposability | PASS/FAIL | 0 | 0 | 0 |
| 2A: Intervention Frequency | PASS/FAIL | 0 | 0 | 0 |
| 2B: Freeze Integrity | PASS/FAIL | 0 | 0 | 0 |
| 3A: Degradation Capability | PASS/FAIL | 0 | 0 | 0 |
| 3B: Data Fragility | PASS/FAIL | 0 | 0 | 0 |
| 4A: Temporal Integrity | PASS/FAIL | 0 | 0 | 0 |
| 4B: Chain Completeness | PASS/FAIL | 0 | 0 | 0 |
| 4C: Deny Reason Closure | PASS/FAIL | 0 | 0 | 0 |

**Overall Verdict:** PASS / FAIL / CONDITIONAL

---

## Audit 0: Evidence Admission Gate

### Evidence Coverage

| Evidence | Present | Fresh | Coverage |
|----------|---------|-------|----------|
| DLE Events | ✓/✗ | ✓/✗ | NN% |
| Doctrine Events | ✓/✗ | ✓/✗ | NN% |
| Orders Attempted | ✓/✗ | ✓/✗ | NN% |
| Orders Executed | ✓/✗ | ✓/✗ | NN% |
| Risk Vetoes | ✓/✗ | ✓/✗ | NN% |
| Episode Ledger | ✓/✗ | ✓/✗ | NN% |
| Sentinel-X | ✓/✗ | ✓/✗ | NN% |

### Chain Linkage

```
Total chains:      NNNN
Complete chains:   NNNN (NN.N%)
Partial chains:    NNNN (NN.N%)
```

### Verdict: PASS / FAIL

### Violations

| ID | Severity | Description | Fix Plan |
|----|----------|-------------|----------|
| V001 | P0/P1/P2 | Description | Plumbing/Logic/Governance |

---

## Audit 1A: Replay Determinism

### Sample Episodes

| Episode ID | Symbol | Side | Replayable | Missing |
|------------|--------|------|------------|---------|
| EP_NNNN | XXXUSDT | LONG/SHORT | ✓/✗ | (fields) |
| EP_NNNN | XXXUSDT | LONG/SHORT | ✓/✗ | (fields) |
| ... | ... | ... | ... | ... |

### Opaqueness Registry

| Field | Count Missing | Impact |
|-------|---------------|--------|
| sizing_factors | N | P1/P2 |
| entry_signal | N | P1/P2 |

### Verdict: PASS / FAIL

---

## Audit 1B: Scoring Decomposability

### Scores Audited

| Score | Factors | Weights | Bounds | Complete |
|-------|---------|---------|--------|----------|
| symbol_score_v6 | NN | ✓/✗ | ✓/✗ | ✓/✗ |
| expectancy_v6 | NN | ✓/✗ | ✓/✗ | ✓/✗ |

### Verdict: PASS / FAIL

---

## Audit 2A: Intervention Frequency

### Window: (24h / 7d)

| Metric | Count | Threshold | Status |
|--------|-------|-----------|--------|
| Supervisor restarts | N | ≤2/day | ✓/✗ |
| NAV stale events | N/N (N%) | ≤5% | ✓/✗ |
| Sentinel stale | N/N (N%) | ≤5% | ✓/✗ |
| Error rate | N% | ≤1% | ✓/✗ |

### Ops Burden Estimate: ~N minutes/day

### Verdict: PASS / FAIL

---

## Audit 2B: Freeze Integrity

| Check | Status |
|-------|--------|
| Config changes in cycle | N |
| Manifest drift | ✓ None / ✗ Drift detected |
| Threshold tweaks | N |

### Change Pressure Index: LOW / MEDIUM / HIGH

### Verdict: PASS / FAIL

---

## Audit 3A: Degradation Capability

| Capability | Available | Tested |
|------------|-----------|--------|
| Testnet mode | ✓/✗ | ✓/✗ |
| Offline episode ledger | ✓/✗ | ✓/✗ |
| DLE replay | ✓/✗ | ✓/✗ |
| Backtest pipeline | ✓/✗ | ✓/✗ |

### Verdict: PASS / FAIL

---

## Audit 3B: Data Source Fragility

### Data Sources

| Source | Type | Fallback | Status |
|--------|------|----------|--------|
| Binance Futures API | Public | Testnet | ✓/✗ |
| Price feeds | WebSocket | REST poll | ✓/✗ |
| Regime data | Derived | Cache | ✓/✗ |

### Single Points of Failure

| SPOF | Mitigation | Risk |
|------|------------|------|
| (description) | (mitigation) | LOW/MED/HIGH |

### Verdict: PASS / FAIL

---

## Audit 4A: Temporal Integrity

```
Chains audited:    NNNN
Inversions found:  N
```

### Inversions (if any)

| Chain ID | Inversion | Timestamps |
|----------|-----------|------------|
| CHAIN_XXX | request > decision | ts1, ts2 |

### Verdict: PASS / FAIL

---

## Audit 4B: Chain Completeness

| Link | Complete | Orphaned | Rate |
|------|----------|----------|------|
| REQUEST → DECISION | NNNN | N | NN.N% |
| DECISION → PERMIT | NNNN | N | NN.N% |
| ENTRY → EXIT | NNNN | N | NN.N% |

### Total Orphan Count: N (N.N%)

### Verdict: PASS / FAIL

---

## Audit 4C: Deny Reason Closure

### Reason Histogram

| Reason | Count | Mapped |
|--------|-------|--------|
| VETO_DIRECTION_MISMATCH | NNN | ✓ |
| VETO_REGIME_UNSTABLE | NNN | ✓ |
| min_notional | NNN | ✓ |
| ... | ... | ... |

### Unmapped Reasons

| Reason | Count | Action Required |
|--------|-------|-----------------|
| (none) | — | — |

### Verdict: PASS / FAIL

---

## Canonical Replay Pack

### Replay 1: ALLOW → Filled

```json
{
  "chain_id": "CHAIN_XXXX",
  "type": "ENTRY_ALLOW",
  "symbol": "XXXUSDT",
  "side": "LONG",
  "signal": { ... },
  "regime": "TREND_UP",
  "outcome": "FILLED",
  "fill_qty": 0.XXX,
  "fill_price": XXXXX.XX
}
```

### Replay 2: ALLOW → Downstream Veto

```json
{
  "chain_id": "CHAIN_XXXX",
  "type": "ENTRY_ALLOW",
  "symbol": "XXXUSDT",
  "outcome": "VETOED",
  "veto_stage": "risk_limits",
  "veto_reason": "min_notional"
}
```

### Replay 3: DENY with Explicit Reason

```json
{
  "chain_id": "CHAIN_XXXX",
  "type": "ENTRY_DENY",
  "symbol": "XXXUSDT",
  "deny_reason": "VETO_DIRECTION_MISMATCH",
  "context": {
    "signal_direction": "LONG",
    "regime": "TREND_DOWN"
  }
}
```

---

## Violations Summary

| ID | Audit | Severity | Description | Status |
|----|-------|----------|-------------|--------|
| V001 | 1A | P2 | Missing sizing factors in 2 episodes | Open |
| V002 | 3A | P2 | Backtest pipeline not implemented | Backlog |

---

## Sign-Off

| Role | Name | Date |
|------|------|------|
| Audit Runner | (agent/human) | YYYY-MM-DD |
| Reviewer | (if applicable) | YYYY-MM-DD |

---

## Attachments

- [ ] DLE event sample (10 chains)
- [ ] Episode ledger snapshot
- [ ] Risk snapshot at audit time
- [ ] Sentinel-X state at audit time
