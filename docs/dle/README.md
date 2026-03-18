# Decision Ledger Engine (DLE) — GPT-Hedge Integration

> **Version:** v2.0  
> **Status:** ACTIVE — Phase A complete, Phase B.1–B.4 complete, Phase C validation in progress  
> **Date:** 2026-01-28 (spec), updated 2026-03-11  
> **Current Phase:** Phase C (Activation Window v8.0 — 14-day structural certification)

## Overview

The Decision Ledger Engine (DLE) is a governance framework that makes **authority explicit, traceable, and immutable**. It prevents silent drift by ensuring every execution can be traced back to a formal decision.

**Current operational mode: SHADOW_MODE** — DLE observes and logs all doctrine verdicts as shadow events. It does **not** gate or block execution. Phase B enforcement is structurally complete but not yet activated (requires `DLE_ENFORCED=1`).

**Implementation:** `execution/dle_shadow.py` (427 lines, 5 classes, fail-open).  
**Feature flags:** `SHADOW_DLE_ENABLED`, `SHADOW_DLE_WRITE_LOGS`, `DLE_ENFORCE_ENTRY_ONLY` (see `execution/v6_flags.py`).  
**Shadow log:** `logs/execution/dle_shadow_events.jsonl` (append-only, schema v2).  
**Exit reason normalization:** `config/exit_reason_map.yaml` (10 canonical values, binding).

---

## Phase Progression

| Phase | Status | Milestone |
|-------|--------|-----------|
| **Phase A** — Shadow gate (observation only) | **COMPLETE** | `execution/dle_shadow.py` active in production |
| **Phase B.1** — Exit reason normalization | **COMPLETE** | `config/exit_reason_map.yaml` + `execution/exit_reason_normalizer.py` |
| **Phase B.2** — Decision enrichment | **COMPLETE** | Shadow events include context_snapshot + provenance |
| **Phase B.3** — Permit enrichment | **COMPLETE** | Permit suppression on DENY enforced |
| **Phase B.4** — Episode binding | **COMPLETE** | Episode UID determinism (`EP_<sha256_12>`) |
| **Phase B.5** — Enforcement rehearsal | **ACTIVE** | Counterfactual logging: `logs/execution/dle_enforcement_rehearsal.jsonl` |
| **Phase C** — Activation Window v8.0 | **ACTIVE** | 14-day structural certification (see `docs/ACTIVATION_WINDOW_v8.md`) |
| **Phase D** — Full enforcement | **FUTURE** | Requires `DLE_ENFORCED=1`, `enforcement_gate.py` (not yet created) |

---

## Documents

### Foundational

| Document | Purpose | Status |
|----------|---------|--------|
| [DLE_CONSTITUTION_V1.md](DLE_CONSTITUTION_V1.md) | 13 foundational doctrines — immutable principles | Foundational |
| [DLE_DOCTRINE.md](DLE_DOCTRINE.md) | Binding operational doctrine (mode-gated) | Binding |

### Core Schemas

| Document | Purpose | Status |
|----------|---------|--------|
| [DLE_DECISION_SCHEMA.md](DLE_DECISION_SCHEMA.md) | Authority objects that permit/forbid actions | Specification |
| [DLE_PERMIT_SCHEMA.md](DLE_PERMIT_SCHEMA.md) | Single-use execution tokens | Specification |
| [DLE_EXECUTION_REQUEST_SCHEMA.md](DLE_EXECUTION_REQUEST_SCHEMA.md) | Input format for Gate requests | Specification |
| [DLE_DENY_REASONS.md](DLE_DENY_REASONS.md) | Canonical denial codes (26 codes) | Partially implemented (C.1) |
| [DLE_EPISODE_SCHEMA.md](DLE_EPISODE_SCHEMA.md) | Complete trade lifecycle binding | Specification |
| [DLE_GATE_INVARIANTS.md](DLE_GATE_INVARIANTS.md) | 10 constitutional invariants (fail-closed, mode-gated) | Partially implemented |
| [LEDGER_ENTRY_SCHEMA.md](LEDGER_ENTRY_SCHEMA.md) | Universal decision entry schema (domain-agnostic) | Binding v1.0 |

### Audit, Migration & Planning

| Document | Purpose | Status |
|----------|---------|--------|
| [DLE_SPEC_CONSISTENCY_CHECK.md](DLE_SPEC_CONSISTENCY_CHECK.md) | Internal consistency audit (100% pass, updated 2026-02-12) | Audit complete |
| [DLE_CROSSWALK_GPT_HEDGE.md](DLE_CROSSWALK_GPT_HEDGE.md) | Current logs → DLE migration mapping | Migration guide |
| [DLE_PREDICTION_LAYER_SPEC.md](DLE_PREDICTION_LAYER_SPEC.md) | DLE authority for prediction subsystem (P1 advisory) | Design spec |
| [AUDIT_SUITE_V1.md](AUDIT_SUITE_V1.md) | MHD audit suite — 5 audit bundles + continuous monitoring | Active |
| [CYCLE_004_PHASE_A_PLAN.md](CYCLE_004_PHASE_A_PLAN.md) | Phase A implementation plan (approved, executed) | Sealed |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  Hydra  │  │  Exit   │  │  Risk   │  │     Manual      │ │
│  │  Heads  │  │ Scanner │  │ Engine  │  │   Intervention  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       │            │            │                 │          │
│       └────────────┴─────┬──────┴─────────────────┘          │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │   ExecutionRequest    │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    DLE SHADOW GATE                            │
│                  (execution/dle_shadow.py)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SHADOW_MODE (current v7.9):                         │   │
│  │  1. Build request → decision → permit chain          │   │
│  │  2. Log shadow verdict (never blocks)                │   │
│  │  3. Compare with doctrine verdict (mismatch detect)  │   │
│  │                                                       │   │
│  │  ENFORCED_MODE (Phase D, not yet active):            │   │
│  │  1–8. Full gate processing per DLE_GATE_INVARIANTS   │   │
│  │  No permit = no order (Gate Invariant #9 binding)    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│            ┌─────────────┴─────────────┐                    │
│            ▼                           ▼                    │
│    ┌──────────────┐           ┌──────────────┐             │
│    │    Permit    │           │    Denial    │             │
│    │   (logged,   │           │   (logged,   │             │
│    │   advisory)  │           │   auditable) │             │
│    └──────┬───────┘           └──────────────┘             │
│           │                                                 │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                      ORDER ROUTER                             │
│  (Currently: permit advisory only — not validated by router)  │
│  (Phase D: permit_id required, validated before execution)    │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                     LEDGER LAYER                              │
│                                                               │
│  dle_shadow_events.jsonl   ←  Shadow chain (Phase A+, active) │
│  dle_enforcement_rehearsal ←  Counterfactual (B.5, active)    │
│  dle_entry_denials.jsonl   ←  Entry denials (C.1, active)     │
│  doctrine_events.jsonl     ←  Doctrine verdicts (existing)    │
│  orders_executed.jsonl     ←  Fill evidence (existing)        │
│  episode_ledger.json       ←  Trade lifecycle (existing)      │
│                                                               │
│  (Phase D — not yet active):                                  │
│  decision_ledger.jsonl     ←  Authority statements            │
│  permits_issued.jsonl      ←  Execution tokens                │
│  denials.jsonl             ←  Refused requests                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Implementation Status

### Active in Production

| Component | File | Status |
|-----------|------|--------|
| Shadow gate | `execution/dle_shadow.py` | 427 lines, 5 classes, fail-open |
| Feature flags | `execution/v6_flags.py` | 3 DLE flags defined |
| Exit normalization | `execution/exit_reason_normalizer.py` | 10 canonical reasons |
| Exit reason map | `config/exit_reason_map.yaml` | Binding, startup-verified |
| Shadow log | `logs/execution/dle_shadow_events.jsonl` | Schema v2, append-only |
| Rehearsal log | `logs/execution/dle_enforcement_rehearsal.jsonl` | B.5 counterfactual |
| Entry denials | (via manifest) | C.1 subset |
| Unit tests | `tests/unit/test_dle_shadow.py` | 6 test classes |
| Integration tests | `tests/integration/test_dle_shadow_integration.py` | File I/O + toggle |

### Not Yet Implemented (Phase D)

| Component | Blocked By | Notes |
|-----------|-----------|-------|
| `enforcement_gate.py` | Phase D activation | Full DLE gate with permit requirement |
| Decision object emission | Phase D | Currently shadow-only |
| Permit lifecycle (hard) | Phase D | Currently advisory; hard enforcement requires `DLE_ENFORCED=1` |
| Legacy log deprecation | After full migration | Existing logs remain authoritative |

### ID Format (Implementation)

| ID Type | Format | Source |
|---------|--------|--------|
| `decision_id` | `DEC_<sha256_hex16>` | `dle_shadow.py:derive_decision_id()` |
| `permit_id` | `PERM_<sha256_hex16>` | `dle_shadow.py:derive_permit_id()` |
| `episode_id` | `EP_<sha256_hex12>` | Episode binding (Phase B.4) |

> **Note:** Schema specification documents reference UUID v4 format. Implementation uses deterministic SHA256-based IDs for reproducibility. This is an intentional divergence — implementation format is authoritative for Phase A/B/C artifacts.

---

## Manifest Entries (`v7_manifest.json`)

| Key | Path | Description |
|-----|------|-------------|
| `dle_shadow_events` | `logs/execution/dle_shadow_events.jsonl` | REQUEST → DECISION → PERMIT → LINK chain |
| `dle_prediction_events_log` | `logs/prediction/dle_prediction_events.jsonl` | DLE authority for prediction layer |
| `dle_enforcement_rehearsal` | `logs/execution/dle_enforcement_rehearsal.jsonl` | B.5 counterfactual rehearsal |
| `dle_entry_denials` | (via manifest) | Entry denial tracking |

---

## Relationship to Current System

| Current Component | DLE Relationship | Status |
|-------------------|------------------|--------|
| `doctrine_kernel.py` | DLE observes verdicts via shadow gate | Active (shadow) |
| `_doctrine_gate()` | Shadow chain built after each verdict | Active |
| `doctrine_events.jsonl` | DLE enriches events (decision_id, permit_id) | Active |
| `episode_ledger.json` | Episodes bound with `EP_<sha256_12>` UIDs | Active (Phase B.4) |
| `exit_reason_normalizer.py` | Maps 3 vocabularies → 10 canonical reasons | Active (Phase B.1) |
| `risk_limits.py` | Provides constraint context for decisions | Existing |
| `sentinel_x.py` | Provides regime context for constraint checking | Existing |

---

## Related Documents (Outside DLE Corpus)

| Document | Location | Relevance |
|----------|----------|-----------|
| Phase B completion report | `docs/PHASE_B_SHADOW_AUTHORITY_COMPLETE.md` | B.1–B.4 milestone |
| Activation Window v8.0 | `docs/ACTIVATION_WINDOW_v8.md` | Phase C certification protocol |
| System baseline | `docs/SYSTEM_BASELINE_v7.9.md` | v7.9 known-good anchor |
| Phase C amendment | `docs/amendments/PHASE_C_ACTIVATION_WINDOW_AMENDMENT_v1.md` | Structural certification layer |
| Copilot instructions | `.github/copilot-instructions.md` | DLE section for agent awareness |

---

## Constraints

- **SHADOW_MODE (current):** Observation only; no execution changes
- **Backward Compatible:** Works alongside existing system
- **Fail-Closed (spec):** Ambiguity = denial; no fallback to implicit authority
- **Fail-Open (shadow):** Shadow gate failures are silent; execution continues
- **Append-Only:** All ledgers and logs are immutable

---

## Open Questions (For Phase D)

1. **Enforcement activation:** When to set `DLE_ENFORCED=1`? Requires Phase C pass.
2. **Decision TTL:** Default expiration for auto-generated decisions?
3. **Conflict resolution:** Escalation path when decisions conflict?
4. **Performance:** Permit issuance latency requirements under enforcement?
5. **Recovery:** Orphaned permit handling on crash?
6. **ID reconciliation:** UUID (spec) vs SHA256-hex (implementation) — formalize choice?
