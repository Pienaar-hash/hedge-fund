# DLE Documentation Hygiene Review

**GPT-HEDGE v7.9 — Documentation Audit**  
**Date:** 2026-02-09  
**Scope:** DLE corpus (14 docs), governance docs (9), MHD audit (3), cycles (3), incidents (1), top-level doctrine addendums (7)  
**Status:** COMPLETE  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Contradictions & Inconsistencies](#2-contradictions--inconsistencies)
3. [Governance Model Mapping](#3-governance-model-mapping)
4. [Missing, Redundant & Outdated Documents](#4-missing-redundant--outdated-documents)
5. [Engineering Alignment Assessment](#5-engineering-alignment-assessment)
6. [Proposed Normalized Structure](#6-proposed-normalized-structure)
7. [Prioritized Remediation Recommendations](#7-prioritized-remediation-recommendations)
8. [Appendix — Full Document Inventory](#8-appendix--full-document-inventory)

---

## 1. Executive Summary

The DLE corpus is **architecturally sound** — schemas are well-specified, the constitution is coherent, and the Phase A shadow implementation aligns well with the spec. However, the documentation ecosystem suffers from **four systemic problems**:

| Problem Class | Severity | Count |
|---------------|----------|-------|
| **Contradictions** — mutually exclusive statements across docs | CRITICAL | 3 |
| **Vocabulary fragmentation** — same concept, different names | HIGH | 4 |
| **Governance gaps** — no ownership, no RACI, no versioning policy | HIGH | 6 |
| **Staleness/rot** — outdated, empty, or duplicated files | MEDIUM | 5 |

**Key contradiction:** `DLE_DOCTRINE` §5.4 declares *"DLE may never block decisions"* while `DLE_GATE_INVARIANTS` Invariant #9 mandates *"No permit = no order."* These are architecturally irreconcilable — one must yield — and no document resolves the conflict.

**Key gap:** The DLE is active in production (61K+ shadow events), yet `copilot-instructions.md` contains zero DLE references, `v7_manifest.json` does not register the shadow log, and no authoritative "current phase" declaration exists.

---

## 2. Contradictions & Inconsistencies

### 2.1 CRITICAL — DLE Non-Criticality vs Gate Enforcement

| Document | Statement | Implication |
|----------|-----------|-------------|
| `DLE_DOCTRINE` §5.4 | *"If DLE goes offline, work continues. DLE may explain decisions — it may never block them."* | DLE is advisory; fail-open |
| `DLE_GATE_INVARIANTS` Inv. #9 | *"No permit = no order. Executor cannot place orders without a valid permit."* | DLE is mandatory; fail-closed |

**Impact:** These define opposite system behaviors. Phase A (shadow) sidesteps this by being observation-only, but Phase B+ **must resolve it** before enforcement begins. No resolution protocol is documented.

**Resolution needed:** Amend `DLE_DOCTRINE` §5.4 to scope the non-criticality guarantee to *specific phases* (e.g., "During shadow phases, DLE may never block..."), or amend `DLE_GATE_INVARIANTS` to add an explicit bypass mode.

### 2.2 CRITICAL — DLE Lifecycle Management Denial vs Episode State Machine

| Document | Statement |
|----------|-----------|
| `DLE_DOCTRINE` §5.2 | *"DLE is not a lifecycle management tool."* |
| `DLE_EPISODE_SCHEMA` | Defines full lifecycle: `PENDING → OPEN → CLOSING → CLOSED` with state transitions, validity rules, and timing constraints |

**Impact:** The Episode Schema IS lifecycle management by any reasonable definition. The doctrine denial creates a semantic contradiction.

**Resolution needed:** Reframe §5.2 to clarify intent (e.g., "DLE does not *replace* operational lifecycle tooling; the Episode schema records lifecycle *observations*").

### 2.3 HIGH — Self-Referential Consistency Check Error

`DLE_SPEC_CONSISTENCY_CHECK.md` reports 94% consistency and flags 4 permit-related denial codes as **"MISSING"** from `DLE_DENY_REASONS.md`:

- `DENY_PERMIT_CONSUMED`
- `DENY_PERMIT_EXPIRED`
- `DENY_PERMIT_REVOKED`
- `DENY_NO_PERMIT`

**Actual state:** All four codes **exist** in `DLE_DENY_REASONS.md` (Category 6: "Permit Lifecycle"). The consistency check is stale — it was written before those codes were added and never updated.

**Impact:** The audit tool designed to catch inconsistencies is itself inconsistent. Undermines trust in the self-audit layer.

### 2.4 HIGH — Three Incompatible Exit Reason Vocabularies

| Source | Examples | Format |
|--------|----------|--------|
| `DLE_EPISODE_SCHEMA` exit_reason enum | `TAKE_PROFIT`, `STOP_LOSS`, `REGIME_CHANGE`, `CONVICTION_DECAY` | UPPER_SNAKE |
| `doctrine_kernel.py` ExitReason enum | `REGIME_FLIP`, `TREND_DECAY`, `CRISIS_OVERRIDE`, `MANUAL` | UPPER_SNAKE (different values) |
| `episode_ledger.py` exit strings | `"tp"`, `"sl"`, `"regime_flip"`, `"manual"` | lowercase abbreviations |

**Impact:** No normalization map exists. When DLE Episode Schema goes live, exit reasons from doctrine_kernel and the ledger won't map cleanly. Dashboard consumers will face ambiguity.

### 2.5 MEDIUM — ID Format Mismatch (Spec vs Implementation)

| Artifact | Format |
|----------|--------|
| DLE Schema specs | `DEC_<hex12>`, `PRM_<hex12>` (12 hex chars) |
| `dle_shadow.py` implementation | `DEC_<hex16>`, `PERM_<hex16>` (16 hex chars, different prefix) |

**Impact:** Phase A only (shadow). Must be reconciled before Phase B to avoid data migration issues.

### 2.6 MEDIUM — `request_id` Field Contradiction

`DLE_SPEC_CONSISTENCY_CHECK.md` flags `request_id` as missing from Permit Schema. The field **is present** in `DLE_PERMIT_SCHEMA.md` (§ Fields table). Same stale-check problem as §2.3.

---

## 3. Governance Model Mapping

### 3.1 Current State: No Formal Governance

The documentation corpus has **no RACI matrix**, **no document ownership registry**, and **no version control policy**. The following governance primitives are absent:

| Governance Element | Status | Location if exists |
|-------------------|--------|-------------------|
| Document ownership (who maintains each doc) | **MISSING** | — |
| RACI matrix for doctrine changes | **MISSING** | — |
| Version control policy (when to bump, how to deprecate) | **MISSING** | — |
| Review/approval workflow | **PARTIAL** | Constitution Amendment Protocol (Art. 14) |
| Escalation path for contradictions | **MISSING** | — |
| Change log per document | **MISSING** | Only CHANGELOG_SUMMARY.md (system-level) |

### 3.2 Decision Rights (Inferred)

From reading all docs, decision authority is *implied* but never explicitly codified:

| Decision Domain | Inferred Authority | Source |
|----------------|-------------------|--------|
| Doctrine changes | Engineering + full regime cycle validation | `DOCTRINE_FALSIFICATION_CRITERIA.md` |
| DLE Constitution amendments | Art. 14 protocol (propose → review → test → ratify) | `DLE_CONSTITUTION_V1.md` |
| Phase promotion (P0→P1→P2) | Gated by measurable criteria | `CYCLE_004_PHASE_A_PLAN.md` |
| Schema changes | Backward-compatible only ("never remove, only add") | `DLE_DECISION_SCHEMA.md` |
| Runtime config changes | Unconstrained (no approval gate) | `runtime.yaml` |
| Risk limit changes | Unconstrained (no approval gate) | `risk_limits.json` |

**Gap:** Runtime and risk config changes have **no documented review process** despite being execution-critical.

### 3.3 Amendment Protocols (Two Exist, Unreconciled)

1. **DLE Constitution Art. 14:** Propose → Review → Test → Ratify. Defines quorum and testing requirements. Applies to the 14 DLE doctrines.

2. **Doctrine Falsification Criteria:** Changes only allowed "after a full regime cycle with statistical evidence." Red/Yellow/Green flag system. Applies to trading doctrine.

These two protocols **never reference each other**. If a DLE doctrine change also affects trading doctrine (e.g., changing gate invariants), which protocol applies? No escalation path is defined.

---

## 4. Missing, Redundant & Outdated Documents

### 4.1 Missing Documents

| What's Missing | Why It Matters | Priority |
|---------------|----------------|----------|
| **Exit reason normalization map** | Three vocabularies exist; no mapping between them | HIGH |
| **DLE integration in `copilot-instructions.md`** | Agent-facing instructions have zero DLE awareness despite active production usage | HIGH |
| **DLE shadow log in `v7_manifest.json`** | Primary observability artifact (61K+ events) not registered in canonical state registry | HIGH |
| **"Current phase" authoritative declaration** | Docs reference both CYCLE_003 and CYCLE_004; no single source says "we are HERE" | MEDIUM |
| **RACI / ownership matrix** | No one owns any document; no review cadence | MEDIUM |
| **PREDICTION_P2_GATE.md** | Referenced in P1 postmortem as deliverable #12 but not found in workspace | LOW |
| **Unified glossary** | Same concepts named differently across DLE, Doctrine, and execution code | LOW |

### 4.2 Redundant Documents

| Document A | Document B | Issue |
|-----------|-----------|-------|
| `docs/dle/AUDIT_SUITE_V1.md` (463 lines) | `docs/mhd/AUDIT_SUITE_V1.md` (same) | **Exact duplicate.** Two files, same content. Which is canonical? |
| `docs/active/OPERATIONS.md` | `docs/active/Runbook.md` | **Overlapping scope.** Both describe production operations. OPERATIONS.md is v5.6 (stale); Runbook.md is current. |

### 4.3 Outdated Documents

| Document | Issue | Severity |
|----------|-------|----------|
| `docs/active/OPERATIONS.md` | Labeled **v5.6 Runbook**. References wrong supervisor process names, wrong paths. System is v7.9. | HIGH |
| `docs/TESTNET_RESET_PROTOCOL.md` | **Completely empty** — 0 bytes of content. File exists as a placeholder. | MEDIUM |
| `docs/dle/DLE_SPEC_CONSISTENCY_CHECK.md` | Claims 94% pass rate but flags codes that actually exist. Stale since codes were added. | MEDIUM |
| `docs/EXCHANGE_UNAVAILABLE_DOCTRINE.md` | Implementation gap table lists 3 items as "Not yet implemented" — status unknown. | LOW |

### 4.4 Orphaned/Unlinked Documents

These docs exist but are not referenced from any index, README, or other document:

- `docs/CODEX_PROMPT_D_LOOP_AUDIT.md` — Loop cadence audit, no inbound links
- `docs/PHASE_A3_VETO_ATTRIBUTION.md` — Exists alongside `PHASE_A3_DOCTRINE_ATTRIBUTION.md`; relationship unclear
- `docs/DATASET_ROLLBACK_CLAUSE.md` — Well-written but not linked from DLE corpus or admission gate
- `docs/active/EXTENDED_OBSERVATION_REPORT.md` — Not referenced from any index
- `docs/active/CHANGELOG_SUMMARY.md` — Not referenced from any navigation doc

---

## 5. Engineering Alignment Assessment

Cross-reference of documentation claims vs actual implementation:

### 5.1 Alignment Scorecard

| Component | Doc → Code Alignment | Notes |
|-----------|---------------------|-------|
| **DLE Shadow Gate** (`dle_shadow.py`) | ✅ WELL ALIGNED | 325 lines. Logging, feature flags, shadow verdicts all match spec. Minor ID format gap (§2.5). |
| **Feature Flags** (`v6_flags.py`) | ✅ FULLY ALIGNED | `SHADOW_DLE_ENABLED`, `SHADOW_DLE_LOG_MISMATCHES` present and documented. |
| **Doctrine Kernel** (`doctrine_kernel.py`) | ⚠️ CONCEPTUALLY ALIGNED | 776 lines, 11 veto codes, fail-closed. But output format (flat dict) is schema-incompatible with DLE canonical Decision/Permit objects. Migration work needed for Phase B. |
| **Episode Ledger** (`episode_ledger.py`) | ❌ PRE-DLE FORMAT | No `decision_id`, `permit_id`, or `authority_chain` fields. Crosswalk doc describes migration but it's not started. |
| **Exit Scanner** (`exit_scanner.py`) | ⚠️ PARTIAL | Uses `ExitReason` enum from doctrine_kernel, not DLE Episode Schema vocabulary. |
| **v7_manifest.json** | ❌ GAP | DLE shadow log (`logs/execution/dle_shadow_events.jsonl`) not registered despite 61K+ events in production. |
| **copilot-instructions.md** | ❌ GAP | Zero DLE references. No mention of `dle_shadow.py`, feature flags, shadow logs, or DLE folder. |
| **Test Coverage** (`test_dle_shadow.py`) | ✅ GOOD | 308 lines, 6 test classes covering shadow verdicts, feature flag gating, mismatch logging, edge cases. |

### 5.2 Schema Migration Gap

The `DLE_CROSSWALK_GPT_HEDGE.md` defines a 6-step migration plan:

1. ✅ Shadow gate (Phase A) — **DONE**
2. ❌ Decision object emission — NOT STARTED
3. ❌ Permit lifecycle — NOT STARTED
4. ❌ Episode binding — NOT STARTED
5. ❌ Enforcement gate — NOT STARTED
6. ❌ Legacy log deprecation — NOT STARTED

Steps 2–6 are documented but have **no timeline, no ownership, and no dependency graph** beyond ordering.

### 5.3 Testability Assessment

| Spec Document | Test Coverage | Verdict |
|--------------|---------------|---------|
| `DLE_GATE_INVARIANTS` (10 invariants) | Test stubs defined in doc but only shadow tests exist | ⚠️ Stubs only |
| `DLE_DECISION_SCHEMA` | No JSON Schema validation tests | ❌ Missing |
| `DLE_PERMIT_SCHEMA` | No lifecycle tests | ❌ Missing |
| `DLE_EPISODE_SCHEMA` | No episode state machine tests | ❌ Missing |
| `DLE_DENY_REASONS` (26 codes) | Shadow tests cover ~6 codes | ⚠️ Partial |
| `AUDIT_SUITE_V1` (10 audits) | Audit commands defined but no automated test harness | ❌ Missing |

---

## 6. Proposed Normalized Structure

### 6.1 Current State (Flat, Untyped)

```
docs/
  ├── CODEX_PROMPT_D_LOOP_AUDIT.md       # orphaned
  ├── DATASET_ADMISSION_GATE.md           # doctrine addendum
  ├── DATASET_ROLLBACK_CLAUSE.md          # doctrine addendum
  ├── EXCHANGE_UNAVAILABLE_DOCTRINE.md    # doctrine addendum
  ├── GPT-HEDGE_Operating_State_v7.9.md   # operating state
  ├── P1_PREDICTION_POSTMORTEM.md         # postmortem
  ├── PHASE_A3_DOCTRINE_ATTRIBUTION.md    # report
  ├── PHASE_A3_VETO_ATTRIBUTION.md        # report
  ├── PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md  # doctrine addendum
  ├── TESTNET_RESET_PROTOCOL.md           # empty!
  ├── active/         # 9 files, mixed types
  ├── archive/        # legacy docs
  ├── cycles/         # 3 files
  ├── dle/            # 14 files, the DLE corpus
  ├── incidents/      # 1 file
  └── mhd/            # 3 files (1 duplicate)
```

**Problems:** Top-level is a dumping ground. `active/` mixes runbooks with doctrine specs. No naming convention. No lifecycle metadata. Documents are not classified by type.

### 6.2 Proposed Target State

```
docs/
  ├── INDEX.md                          # Master navigation doc (NEW)
  ├── GOVERNANCE.md                     # RACI matrix, ownership, review cadence (NEW)
  │
  ├── doctrine/                         # Binding system laws
  │   ├── TRADING_DOCTRINE_ENGINE_SPEC.md
  │   ├── DOCTRINE_FALSIFICATION_CRITERIA.md
  │   ├── DATASET_ADMISSION_GATE.md
  │   ├── DATASET_ROLLBACK_CLAUSE.md
  │   ├── EXCHANGE_UNAVAILABLE_DOCTRINE.md
  │   └── PREDICTION_ADVISORY_DOCTRINE.md
  │
  ├── dle/                              # DLE corpus (keep as-is, well-organized)
  │   ├── README.md                     # Add "Current Phase" declaration
  │   ├── DLE_CONSTITUTION_V1.md
  │   ├── DLE_DOCTRINE.md
  │   ├── DLE_DECISION_SCHEMA.md
  │   ├── DLE_PERMIT_SCHEMA.md
  │   ├── DLE_EXECUTION_REQUEST_SCHEMA.md
  │   ├── DLE_DENY_REASONS.md
  │   ├── DLE_GATE_INVARIANTS.md
  │   ├── DLE_EPISODE_SCHEMA.md
  │   ├── DLE_CROSSWALK_GPT_HEDGE.md
  │   ├── DLE_PREDICTION_LAYER_SPEC.md
  │   ├── DLE_SPEC_CONSISTENCY_CHECK.md # Fix or archive
  │   └── EXIT_REASON_NORMALIZATION.md  # (NEW — maps 3 vocabularies)
  │
  ├── operations/                       # Runbooks, protocols
  │   ├── Runbook.md                    # Current production runbook
  │   ├── TESTING.md
  │   ├── TESTNET_RESET_PROTOCOL.md     # Write content or delete
  │   └── REGIME_AWARE_SYSTEM_EXPLAINER.md
  │
  ├── audit/                            # MHD + DLE audits (deduplicated)
  │   ├── AUDIT_SUITE_V1.md             # Single canonical copy
  │   ├── AUDIT_REPORT_TEMPLATE.md
  │   ├── AUDIT_RUNBOOK.md
  │   └── APPENDIX_A_DOCTRINE_VALIDATION.md
  │
  ├── cycles/                           # Cycle planning, postmortems
  │   ├── CYCLE_002_Postmortem.md
  │   ├── CYCLE_003_Goals.md
  │   ├── CYCLE_004_PHASE_A_PLAN.md     # Move from dle/
  │   ├── P1_PREDICTION_POSTMORTEM.md   # Move from top-level
  │   └── REGIME_CYCLE_POST_MORTEM_TEMPLATE.md
  │
  ├── incidents/                        # Keep as-is
  │   └── 2025-12-Position-Ledger-Incident.md
  │
  ├── reports/                          # Point-in-time analysis
  │   ├── PHASE_A3_DOCTRINE_ATTRIBUTION.md
  │   ├── PHASE_A3_VETO_ATTRIBUTION.md
  │   ├── CODEX_PROMPT_D_LOOP_AUDIT.md
  │   └── EXTENDED_OBSERVATION_REPORT.md
  │
  └── archive/                          # Deprecated/superseded
      ├── OPERATIONS_v5.6.md            # Rename from active/OPERATIONS.md
      └── ... (existing archive)
```

### 6.3 Naming Conventions

| Rule | Convention | Example |
|------|-----------|---------|
| **Doctrine docs** | `UPPER_SNAKE.md` | `DATASET_ADMISSION_GATE.md` |
| **Schemas** | `DLE_<OBJECT>_SCHEMA.md` | `DLE_PERMIT_SCHEMA.md` |
| **Runbooks** | Title case `.md` | `Runbook.md` |
| **Postmortems** | `<TOPIC>_Postmortem.md` or `CYCLE_NNN_Postmortem.md` | `CYCLE_002_Postmortem.md` |
| **Reports** | `PHASE_<ID>_<TOPIC>.md` | `PHASE_A3_DOCTRINE_ATTRIBUTION.md` |

### 6.4 Document Lifecycle Metadata

Every document should carry a frontmatter block:

```markdown
**Status:** DRAFT | ACTIVE | BINDING | SEALED | DEPRECATED
**Owner:** <team or individual>
**Last reviewed:** <date>
**Supersedes:** <doc if applicable>
**Superseded by:** <doc if applicable>
```

Currently, status headers exist on ~60% of docs, but `Owner` and `Last reviewed` exist on **zero** documents.

---

## 7. Prioritized Remediation Recommendations

### P0 — Critical (Blocks Phase B, undermines trust)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| **R1** | **Resolve DLE_DOCTRINE §5.4 vs GATE_INVARIANTS #9 contradiction.** Add phase-scoped non-criticality clause: "During shadow phases (A), DLE is non-blocking. During enforcement phases (B+), gate invariants are authoritative." | 1 hour | Removes architectural ambiguity |
| **R2** | **Create exit reason normalization map.** Single document mapping DLE Episode enum ↔ doctrine_kernel ExitReason ↔ episode_ledger strings. Add to `docs/dle/EXIT_REASON_NORMALIZATION.md`. | 2 hours | Unblocks Episode Schema adoption |
| **R3** | **Update `DLE_SPEC_CONSISTENCY_CHECK.md`.** Fix the 4 false-negative findings (codes that exist). Update score to actual. Or archive the file and replace with automated test. | 30 min | Restores trust in self-audit |

### P1 — High (Production gaps, developer experience)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| **R4** | **Add DLE section to `copilot-instructions.md`.** Include: `dle_shadow.py` role, feature flags (`SHADOW_DLE_ENABLED`, `SHADOW_DLE_LOG_MISMATCHES`), shadow log path, Phase A status, and "DLE does not gate execution yet" note. | 30 min | All agents gain DLE awareness |
| **R5** | **Register DLE shadow log in `v7_manifest.json`.** Add `logs/execution/dle_shadow_events.jsonl` entry with schema, writer, consumers. | 15 min | Shadow data becomes first-class |
| **R6** | **Archive `docs/active/OPERATIONS.md`.** Rename to `docs/archive/OPERATIONS_v5.6.md`. The v7.9 Runbook.md is the current authority. | 5 min | Removes misleading doc |
| **R7** | **Deduplicate `AUDIT_SUITE_V1.md`.** Delete one copy (recommend keeping `docs/dle/` version, symlinking or removing `docs/mhd/` copy). | 5 min | Eliminates drift risk |
| **R8** | **Reframe DLE_DOCTRINE §5.2.** Change from "DLE is not a lifecycle management tool" to "DLE records lifecycle observations; it does not replace operational lifecycle tooling." | 15 min | Removes semantic contradiction with Episode Schema |

### P2 — Medium (Structural improvements)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| **R9** | **Create `docs/INDEX.md`.** Master navigation document linking all active docs by category. | 1 hour | Discoverability |
| **R10** | **Create `docs/GOVERNANCE.md`.** RACI matrix for doctrine changes, document ownership, review cadence. | 2 hours | Accountability |
| **R11** | **Add "Current Phase" to `docs/dle/README.md`.** Single authoritative line: "Current phase: PHASE_A (Shadow). Next: CYCLE_004 Phase A enforcement." | 10 min | Eliminates CYCLE_003 vs 004 ambiguity |
| **R12** | **Write or delete `docs/TESTNET_RESET_PROTOCOL.md`.** File is empty. Either populate with actual protocol or remove. | 30 min | Removes dead placeholder |
| **R13** | **Reconcile DLE ID formats.** Decide on `DEC_<hex12>` vs `DEC_<hex16>` and `PRM_` vs `PERM_`. Update spec or code. | 30 min | Prevents Phase B migration friction |
| **R14** | **Reconcile amendment protocols.** Add cross-reference between Constitution Art. 14 and Falsification Criteria. Define which applies when scopes overlap. | 1 hour | Clear change management |

### P3 — Low (Polish, future-proofing)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| **R15** | **Add lifecycle metadata to all docs.** Status/Owner/Last-reviewed frontmatter. | 2 hours | Document hygiene |
| **R16** | **Reorganize `docs/` per §6.2 taxonomy.** Move files into doctrine/, operations/, audit/, reports/ folders. | 1 hour | Structural clarity |
| **R17** | **Create unified glossary.** Map terminology: "veto" vs "deny" vs "refuse", "permit" vs "approval", "episode" vs "position lifecycle". | 2 hours | Terminology alignment |
| **R18** | **Add DLE schema validation tests.** Automated tests that validate Decision, Permit, Episode JSON against schemas. | 4 hours | Spec-as-test |
| **R19** | **Link orphaned docs.** Connect DATASET_ROLLBACK_CLAUSE, CODEX_PROMPT_D_LOOP_AUDIT, etc. to relevant index/parent docs. | 30 min | Discoverability |

---

## 8. Appendix — Full Document Inventory

### 8.1 DLE Corpus (`docs/dle/`) — 14 documents

| Document | Lines | Status | Type |
|----------|-------|--------|------|
| `README.md` | 195 | v1.0-SPEC | Index/Overview |
| `DLE_CONSTITUTION_V1.md` | 272 | Foundational | Constitution |
| `DLE_DOCTRINE.md` | 399 | Binding | Doctrine |
| `DLE_DECISION_SCHEMA.md` | 340 | v1 | Schema |
| `DLE_PERMIT_SCHEMA.md` | 304 | v1 | Schema |
| `DLE_EXECUTION_REQUEST_SCHEMA.md` | 365 | v1 | Schema |
| `DLE_DENY_REASONS.md` | 287 | v1 | Reference |
| `DLE_GATE_INVARIANTS.md` | 423 | Binding | Invariants |
| `DLE_EPISODE_SCHEMA.md` | 458 | v1 | Schema |
| `DLE_SPEC_CONSISTENCY_CHECK.md` | 483 | ⚠️ STALE | Audit |
| `DLE_CROSSWALK_GPT_HEDGE.md` | 483 | v1 | Migration |
| `DLE_PREDICTION_LAYER_SPEC.md` | 349 | v1 | Spec |
| `AUDIT_SUITE_V1.md` | 463 | v1 | Audit |
| `CYCLE_004_PHASE_A_PLAN.md` | 1161 | APPROVED | Plan |

### 8.2 Governance & Doctrine (`docs/active/`) — 9 documents

| Document | Lines | Status | Issue |
|----------|-------|--------|-------|
| `GPT-HEDGE_v7.x_TRADING_DOCTRINE_ENGINE_SPEC.md` | 269 | BINDING | OK |
| `DOCTRINE_FALSIFICATION_CRITERIA.md` | 299 | ACTIVE | OK |
| `APPENDIX_A_DOCTRINE_VALIDATION.md` | ~200 | ACTIVE | OK |
| `Runbook.md` | ~250 | ACTIVE | OK |
| `TESTING.md` | ~300 | ACTIVE | OK |
| `REGIME_AWARE_SYSTEM_EXPLAINER.md` | ~150 | ACTIVE | OK |
| `OPERATIONS.md` | ~200 | ❌ v5.6 | OUTDATED — archive |
| `EXTENDED_OBSERVATION_REPORT.md` | — | Unknown | Orphaned |
| `CHANGELOG_SUMMARY.md` | — | Unknown | Orphaned |

### 8.3 Top-Level Doctrine Addendums (`docs/`) — 7 documents

| Document | Status | Issue |
|----------|--------|-------|
| `GPT-HEDGE_Operating_State_v7.9.md` | ACTIVE | OK |
| `DATASET_ADMISSION_GATE.md` | ACTIVE | OK |
| `DATASET_ROLLBACK_CLAUSE.md` | ACTIVE | Orphaned (not linked) |
| `EXCHANGE_UNAVAILABLE_DOCTRINE.md` | ACTIVE | 3 items "not yet implemented" |
| `PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md` | ACTIVE | OK |
| `TESTNET_RESET_PROTOCOL.md` | ❌ EMPTY | Write content or delete |
| `CODEX_PROMPT_D_LOOP_AUDIT.md` | Unknown | Orphaned |

### 8.4 Reports & Postmortems

| Document | Location | Status |
|----------|----------|--------|
| `PHASE_A3_DOCTRINE_ATTRIBUTION.md` | `docs/` | Report |
| `PHASE_A3_VETO_ATTRIBUTION.md` | `docs/` | Report |
| `P1_PREDICTION_POSTMORTEM.md` | `docs/` | SEALED |
| `CYCLE_002_Postmortem.md` | `docs/cycles/` | SEALED |
| `CYCLE_003_Goals.md` | `docs/cycles/` | ACTIVE |
| `2025-12-Position-Ledger-Incident.md` | `docs/incidents/` | SEALED |

### 8.5 MHD Audit (`docs/mhd/`) — 3 documents

| Document | Issue |
|----------|-------|
| `AUDIT_SUITE_V1.md` | ❌ DUPLICATE of `docs/dle/AUDIT_SUITE_V1.md` |
| `AUDIT_REPORT_TEMPLATE.md` | OK |
| `AUDIT_RUNBOOK.md` | OK |

---

*End of review. Recommendations R1–R3 should be addressed before any Phase B planning begins.*
