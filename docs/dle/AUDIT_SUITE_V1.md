# GPT Hedge MHD Audit Suite v1

> **Messy · Heavy · Dependent** — The three failure modes that kill quant systems before strategy does.

## Purpose

This audit suite provides **falsifiable, repeatable assessments** of system health across dimensions that matter for investor defensibility. Every audit produces:

- **PASS / FAIL** verdict
- **Violations** ranked P0 / P1 / P2
- **Required evidence** (files + fields)
- **Minimal fix plan** (plumbing vs logic vs governance)

---

## Audit Structure

```
MHD Audit Suite
├── Audit 0: Evidence Admission Gate
├── Audit 1: Messy Audit (Meaning Drift & Opacity)
│   ├── 1A: Replay Determinism
│   └── 1B: Scoring Decomposability
├── Audit 2: Heavy Audit (Operational Burden)
│   ├── 2A: Intervention Frequency
│   └── 2B: Stability Under Freeze
├── Audit 3: Dependent Audit (External Fragility)
│   ├── 3A: Degradation Capability
│   └── 3B: Data Source Fragility
└── Audit 4: Cross-Cutting Integrity
    ├── 4A: Temporal Integrity
    ├── 4B: Chain Completeness
    └── 4C: Deny Reason Closure
```

---

## Authoritative Data Sources

### Decision Logs (append-only JSONL)

| Log | Path | Purpose |
|-----|------|---------|
| **DLE Events** | `logs/dle/dle_events_v1.jsonl` | REQUEST → DECISION → PERMIT chains |
| **Doctrine Events** | `logs/doctrine_events.jsonl` | Entry/exit gate verdicts |
| **Orders Attempted** | `logs/execution/orders_attempted.jsonl` | All order submissions |
| **Orders Executed** | `logs/execution/orders_executed.jsonl` | All fills |
| **Risk Vetoes** | `logs/execution/risk_vetoes.jsonl` | All risk-layer denials |
| **Execution Health** | `logs/execution/execution_health.jsonl` | System health snapshots |

### State Files (point-in-time JSON)

| File | Path | Purpose |
|------|------|---------|
| **Episode Ledger** | `logs/state/episode_ledger.json` | Completed trade cycles |
| **Positions State** | `logs/state/positions_state.json` | Current open positions |
| **Positions Ledger** | `logs/state/positions_ledger.json` | Position TP/SL registry |
| **Sentinel-X** | `logs/state/sentinel_x.json` | Regime state |
| **Risk Snapshot** | `logs/state/risk_snapshot.json` | Risk engine state |
| **NAV State** | `logs/state/nav.json` | NAV and equity |
| **Router Health** | `logs/state/router_health.json` | Execution quality |

### Strategy Identifiers

Current strategy set in logs:
- `vol_target` — Primary volatility-targeting strategy
- `unknown` — Legacy/unmapped episodes

---

## Audit 0: Evidence Admission Gate

**Purpose:** Prevent arguing from vibes. If evidence is missing, the audit FAILS by default.

### Required Evidence (Minimum)

| Evidence | Source | Required Fields |
|----------|--------|-----------------|
| DLE shadow log | `logs/dle/dle_events_v1.jsonl` | `event_type`, `payload.chain_id`, `ts` |
| Doctrine events | `logs/doctrine_events.jsonl` | `verdict`, `reason`, `ts` |
| Orders attempted | `logs/execution/orders_attempted.jsonl` | `symbol`, `side`, `qty`, `ts` |
| Orders executed | `logs/execution/orders_executed.jsonl` | `symbol`, `side`, `qty`, `ts`, `orderId` |
| Risk vetoes | `logs/execution/risk_vetoes.jsonl` | `veto_reason`, `symbol`, `ts` |
| Positions state | `logs/state/positions_state.json` | `positions[]`, `updated_at` |
| Sentinel-X | `logs/state/sentinel_x.json` | `regime`, `confidence`, `ts` |
| Episode ledger | `logs/state/episode_ledger.json` | `episodes[]`, `last_rebuild_ts` |

### PASS Criteria

Every trade attempt is reconstructible as:

```
REQUEST → DECISION → PERMIT → (ATTEMPTED) → (EXECUTED or VETOED) → EPISODE
```

- Coverage threshold: **≥95%** of chains complete
- Episode linkage: **≥90%** of fills map to episodes

### FAIL Modes

- DLE log missing or empty
- Episode ledger stale (>30 minutes)
- >5% of fills have no chain linkage

### Output

```
Evidence Coverage Report
========================
DLE chains:        1,234 total
  - Complete:      1,198 (97.1%) ✓
  - Partial:          36 (2.9%)
Episode linkage:   3,780 fills
  - Mapped:        3,502 (92.6%) ✓
  - Orphaned:        278 (7.4%)

VERDICT: PASS
```

---

## Audit 1: Messy Audit (Meaning Drift & Opacity)

**Question:** Can we replay each trade without interpretation?

### 1A: Replay Determinism Audit

**PASS Criteria:**

For a sampled set of episodes (N=10):
- We can reconstruct **why** (signal + gates)
- We identify **which gate mattered** (explicit reason)
- We know **what size** and why (factors logged)
- No "composite" without decomposed reasons

**Evidence Required:**

| Field | Source | Purpose |
|-------|--------|---------|
| `payload.context.signal` | DLE | Entry signal |
| `payload.context.regime` | DLE | Regime at decision |
| `payload.deny_reason` | DLE | Gate that blocked |
| `payload.context.sizing` | DLE | Size factors |

**FAIL Modes:**

- Composite score with no factor breakdown
- AI outputs used without abstain bounds
- >2 interacting filters with no "primary cause" logged

**Output:**

```
Replay Determinism Report
=========================
Episodes sampled: 10
  - Fully replayable:     8 (80%)
  - Partial (missing):    2 (20%)

Missing surfaces:
  - EP_0234: sizing factors not logged
  - EP_0456: entry signal not captured

Opaqueness registry: 2 fields missing schema
VERDICT: PASS (with P2 violations)
```

### 1B: Scoring Decomposability Audit

**PASS Criteria:**

Any multiplier/score has:
- Factor list
- Weights
- Clamp bounds
- Effective contribution shown in logs

**Evidence Required:**

- `logs/state/symbol_scores_v6.json`
- `logs/state/factor_diagnostics.json`
- `logs/state/expectancy_v6.json`

**Output:**

```
Scoring Decomposability Report
==============================
Scores audited: symbol_score_v6, expectancy_v6
  - Factors exposed:     12/12 (100%) ✓
  - Weights logged:      Yes ✓
  - Clamp bounds:        Yes ✓

VERDICT: PASS
```

---

## Audit 2: Heavy Audit (Operational Burden)

**Question:** Can we run unattended for meaningful stretches?

### 2A: Intervention Frequency Audit

**PASS Criteria:**

Over window (24h / 7d):
- Manual interventions: **≤2 per day**
- Staleness events: **≤5% of cycles**
- Error rate: **≤1% of log lines**

**Evidence Required:**

| Metric | Source | Threshold |
|--------|--------|-----------|
| Supervisor restarts | `supervisorctl status` history | ≤2/day |
| NAV stale events | `risk_snapshot.json` | ≤5% cycles |
| Sentinel stale events | `sentinel_x.json` | ≤5% cycles |
| Executor errors | `hedge-executor.err.log` | ≤1% |

**Output:**

```
Intervention Frequency Report (24h)
===================================
Supervisor restarts:   1
NAV stale events:      3/144 cycles (2.1%) ✓
Sentinel stale:        0/144 cycles (0%) ✓
Error rate:            0.3% ✓

Ops burden: ~5 minutes/day
VERDICT: PASS
```

### 2B: Stability Under Freeze Audit

**PASS Criteria:**

- No config changes within cycle
- VERSION manifest matches deployed
- No ad-hoc threshold tweaks

**Evidence Required:**

- `VERSION` file
- `v7_manifest.json`
- Git commit history for cycle period

**Output:**

```
Freeze Integrity Report
=======================
Cycle: CYCLE_004 Phase A
Config changes:      0 ✓
Manifest drift:      None ✓
Threshold tweaks:    0 ✓

Change pressure index: LOW
VERDICT: PASS
```

---

## Audit 3: Dependent Audit (External Fragility)

**Question:** Can external failures stop learning?

### 3A: Degradation Capability Audit

**PASS Criteria:**

If exchange/feed fails, we can still:
- Observe signals (logs exist)
- Produce research artifacts
- Run episode ledger rebuilds
- Execute DLE replays

**Evidence Required:**

- Testnet mode available (`BINANCE_TESTNET=1`)
- Episode ledger CLI works offline
- DLE replay capability documented

**Output:**

```
Degradation Capability Report
=============================
Testnet mode:        Available ✓
Offline ledger:      Works ✓
DLE replay:          Available ✓
Backtest pipeline:   Not implemented (P2)

VERDICT: PASS (with P2 gap)
```

### 3B: Data Source Fragility Audit

**PASS Criteria:**

- Edge does not require private endpoints
- Regime derived from public market data
- Dependency failures logged explicitly

**Evidence Required:**

- Data source map (inputs → producers)
- Timeout configurations
- Fallback behaviors documented

**Output:**

```
Data Source Fragility Report
============================
Data sources:
  - Binance Futures API: public ✓
  - Price feeds: exchange WebSocket ✓
  - Regime: derived from OHLCV ✓

Single points of failure:
  - Binance API availability (mitigated: testnet fallback)

VERDICT: PASS
```

---

## Audit 4: Cross-Cutting Integrity Audits

**Always-on audits that prevent silent failure.**

### 4A: Temporal Integrity Audit

**PASS Criteria:**

No temporal inversions:
```
request_ts ≤ decision_ts ≤ permit_ts ≤ attempted_ts ≤ executed_ts
```

**Evidence Required:**

- DLE events with timestamps
- Orders attempted/executed with timestamps

**FAIL = P0:**

Any temporal inversion is a critical violation.

**Output:**

```
Temporal Integrity Report
=========================
Chains audited:    1,234
Inversions found:  0 ✓

VERDICT: PASS
```

### 4B: Chain Completeness Audit

**PASS Criteria:**

No orphaned records above tolerance (5%):
- REQUEST without DECISION
- DECISION without PERMIT
- EXIT without prior ENTRY

**Output:**

```
Chain Completeness Report
=========================
Orphan analysis:
  - REQUEST → DECISION:  98.2% complete ✓
  - DECISION → PERMIT:   99.1% complete ✓
  - ENTRY → EXIT:        94.8% complete ✓

Orphan count: 23 (1.9%)
VERDICT: PASS
```

### 4C: Deny Reason Closure Audit

**PASS Criteria:**

Every deny reason maps to canonical list:

```
VETO_DIRECTION_MISMATCH
VETO_REGIME_UNSTABLE
VETO_REGIME_CONFIDENCE
SENTINEL_STALE
NAV_STALE
RISK_MODE_HALTED
min_notional
per_symbol_cap
portfolio_dd
correlation_cap
```

Unmapped reasons are **P0**.

**Output:**

```
Deny Reason Closure Report
==========================
Unique reasons found: 8
Mapped to canonical:  8/8 (100%) ✓
Unmapped:             0

Histogram:
  VETO_DIRECTION_MISMATCH:  439 (52%)
  VETO_REGIME_UNSTABLE:      99 (12%)
  min_notional:             187 (22%)
  ...

VERDICT: PASS
```

---

## Audit Cadence

### Daily

| Audit | Purpose |
|-------|---------|
| 4A: Temporal Integrity | Catch inversions early |
| 4B: Chain Completeness | Detect orphans |
| 4C: Deny Reason Closure | Catch unmapped reasons |
| 2A: Intervention Frequency | Track ops burden |

### Weekly

| Audit | Purpose |
|-------|---------|
| 1A: Replay Determinism | Sample 10 episodes |
| 3A: Degradation Capability | Simulate one outage |
| 2B: Freeze Integrity | Verify no drift |

### Per Cycle Close (Investor Grade)

Full MHD scorecard with:
- All audits run
- Top violations documented
- Evidence pack with 3 canonical replays:
  1. ALLOW → filled
  2. ALLOW → downstream veto
  3. DENY with explicit reason

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial MHD audit suite |
