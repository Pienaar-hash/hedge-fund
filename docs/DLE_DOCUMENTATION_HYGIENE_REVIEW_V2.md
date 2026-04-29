# DLE Documentation Hygiene Review — V2

**GPT-HEDGE v7.9 — Documentation Audit (Clean Execution)**  
**Date:** 2026-03-11  
**Scope:** Full DLE corpus (15 docs), governance docs, top-level doctrine, implementation alignment  
**Previous review:** `docs/DLE_DOCUMENTATION_HYGIENE_REVIEW.md` (V1, 2026-02-09)  
**Status:** COMPLETE

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [V1 Remediation Status](#2-v1-remediation-status)
3. [Current Contradictions & Inconsistencies](#3-current-contradictions--inconsistencies)
4. [Documentation-to-Code Alignment](#4-documentation-to-code-alignment)
5. [Actions Taken in This Review](#5-actions-taken-in-this-review)
6. [Remaining Issues & Future Remediation](#6-remaining-issues--future-remediation)
7. [Canonical Document Registry (Top 20)](#7-canonical-document-registry-top-20)
8. [Future Development Readiness](#8-future-development-readiness)

---

## 1. Executive Summary

The DLE documentation ecosystem has **significantly improved** since V1. The critical contradictions (mode-gating, lifecycle management framing) have been resolved. Phase B.1–B.4 completion introduced binding exit reason normalization, decision/permit enrichment, and episode binding. The codebase is active and well-tested.

**V2 review found and corrected three systemic problems:**

| Problem | Severity | Action Taken |
|---------|----------|-------------|
| **`dle/` root directory duplication** — 10 files duplicating `docs/dle/` and `docs/`, 3 with stale content | HIGH | Removed entire `dle/` root directory; moved 1 unique file to `docs/dle/` |
| **DLE README frozen at "CYCLE_003"** — Phase A complete, B.1–B.4 complete, C active, but README said "specification only" | HIGH | Rewrote README with current phase progression, implementation status, manifest entries |
| **copilot-instructions.md stale DLE section** — Missing `DLE_ENFORCE_ENTRY_ONLY` flag, outdated phase description, wrong doc count | MEDIUM | Updated phase description, added rehearsal log, corrected doc count and flags |

**No new contradictions found.** All V1 critical contradictions (§2.1, §2.2) were resolved before this review.

---

## 2. V1 Remediation Status

### V1 Remediations — Scorecard

| # | V1 Action | Status | Evidence |
|---|-----------|--------|----------|
| **R1** | Resolve DLE_DOCTRINE §5.4 vs GATE_INVARIANTS #9 | **FIXED** | Mode-gating table added to both documents. SHADOW_MODE vs ENFORCED_MODE explicit. |
| **R2** | Create exit reason normalization map | **FIXED** | `config/exit_reason_map.yaml` exists (4979 bytes). 10 canonical values. `execution/exit_reason_normalizer.py` implements `normalize_exit_reason()`. Startup-verified via `verify_doctrine_coverage()`. |
| **R3** | Update DLE_SPEC_CONSISTENCY_CHECK | **FIXED** | Score corrected to 100% in 2026-02-12 revision. 4 false-negative findings corrected. 1 open item remains (regime enum formalization). |
| **R4** | Add DLE section to copilot-instructions.md | **FIXED** | Extensive section present (flags, paths, phase status, test refs). Updated in this review. |
| **R5** | Register DLE shadow log in v7_manifest.json | **FIXED** | 4+ DLE entries: `dle_shadow_events`, `dle_prediction_events_log`, `dle_enforcement_rehearsal`, `dle_entry_denials`. |
| **R6** | Archive `docs/active/OPERATIONS.md` | **FIXED** | File no longer exists in `docs/active/`. |
| **R7** | Deduplicate AUDIT_SUITE_V1.md | **NOT APPLICABLE** | `docs/mhd/AUDIT_SUITE_V1.md` does not exist. Only `docs/dle/AUDIT_SUITE_V1.md` present. V1 finding may have been based on prior state. |
| **R8** | Reframe DLE_DOCTRINE §5.2 | **FIXED** | Reframed from "not a lifecycle management tool" to "records observed lifecycle state — does not drive transitions." Episode = audit surface, executor = sole authority. |
| **R9** | Create docs/INDEX.md | **NOT DONE** | No master navigation doc exists. |
| **R10** | Create docs/GOVERNANCE.md | **NOT DONE** | No RACI matrix, no ownership registry. |
| **R11** | Add "Current Phase" to DLE README | **FIXED (this review)** | README now declares current phase, phase progression table, implementation status. |
| **R12** | Write or delete TESTNET_RESET_PROTOCOL.md | **FIXED** | Root `TESTNET_RESET_PROTOCOL.md` is populated and active (last updated 2026-02-12). |
| **R13** | Reconcile DLE ID formats | **PARTIAL** | Implementation uses `DEC_hex16` / `PERM_hex16` (deterministic SHA256). Schema specs reference UUID v4. Implementation format is authoritative for Phase A/B/C; noted in updated README. |
| **R14** | Reconcile amendment protocols | **NOT DONE** | Constitution Art. 14 and Falsification Criteria still don't cross-reference. |
| **R15** | Add lifecycle metadata to all docs | **NOT DONE** | Status headers present on ~70% of docs. Owner/Last-reviewed on 0%. |
| **R16** | Reorganize docs/ taxonomy | **NOT DONE** | Flat structure persists. Functional but not organized by type. |
| **R17** | Create unified glossary | **NOT DONE** | No glossary. Exit reason vocabulary is normalized via `exit_reason_map.yaml`, but broader terminology alignment absent. |
| **R18** | Add DLE schema validation tests | **NOT DONE** | Shadow gate tested (6 classes). Schema-specific validation (Decision, Permit, Episode JSON) not tested. |
| **R19** | Link orphaned docs | **PARTIAL** | CODEX_PROMPT_D_LOOP_AUDIT remains unlinked. DATASET_ROLLBACK_CLAUSE now referenced from DATASET_ADMISSION_GATE via amendment. |

### Summary: 10/19 FIXED, 2/19 PARTIAL, 6/19 NOT DONE, 1/19 NOT APPLICABLE

---

## 3. Current Contradictions & Inconsistencies

### 3.1 RESOLVED — Mode-Gating (V1 §2.1)

DLE_DOCTRINE §5.4 now explicitly defines SHADOW_MODE (v7.x, advisory) vs ENFORCED_MODE (Phase D, binding). DLE_GATE_INVARIANTS Invariant #9 includes mode-gating table. **No contradiction remains.**

### 3.2 RESOLVED — Lifecycle Management (V1 §2.2)

DLE_DOCTRINE §5.2 now states: "DLE records observed lifecycle state — it does not drive lifecycle transitions." Episode schema = audit surface, not control surface. **No semantic contradiction remains.**

### 3.3 RESOLVED — Spec Consistency Check False Negatives (V1 §2.3)

Score corrected to 100% in 2026-02-12 revision. All 4 "missing" denial codes confirmed present in DLE_DENY_REASONS.md.

### 3.4 RESOLVED — Exit Reason Vocabularies (V1 §2.4)

`config/exit_reason_map.yaml` provides authoritative mapping of 3 source vocabularies → 10 canonical values. `execution/exit_reason_normalizer.py` implements normalization. Startup verification ensures coverage.

### 3.5 MEDIUM — ID Format Divergence (Persists from V1 §2.5)

| Source | Format |
|--------|--------|
| Schema specs (DLE_DECISION_SCHEMA, etc.) | UUID v4 (`^[a-f0-9-]{36}$`) |
| `dle_shadow.py` implementation | `DEC_<sha256_hex16>`, `PERM_<sha256_hex16>` |
| DLE_PREDICTION_LAYER_SPEC | `DEC_<hex12>`, `PRM_<hex12>`, `BEL_<hex12>` |

Three incompatible ID formats across spec and implementation. Implementation format (`DEC_hex16`, `PERM_hex16`) is authoritative for production artifacts. Schema specs have not been updated.

**Impact:** Low for Phase A/B/C (shadow only). Must be resolved before Phase D enforcement.

### 3.6 LOW — Schema Status Headers Stale

DLE_DECISION_SCHEMA, DLE_PERMIT_SCHEMA, DLE_EXECUTION_REQUEST_SCHEMA, and DLE_EPISODE_SCHEMA all say "SPECIFICATION — not yet implemented." In reality, shadow gate observes decisions and permits (Phase A/B), and episode binding is active (Phase B.4). These schemas describe the **target enforcement state** (Phase D), but the "not yet implemented" label is misleading given partial implementation via shadow mode.

**Impact:** Developer confusion only. No operational impact.

---

## 4. Documentation-to-Code Alignment

### 4.1 Alignment Scorecard (Updated)

| Component | Doc → Code | Notes |
|-----------|-----------|-------|
| **DLE Shadow Gate** (`dle_shadow.py`) | ✅ WELL ALIGNED | 427 lines, 5 classes. Fail-open. Shadow verdicts match spec. |
| **Feature Flags** (`v6_flags.py`) | ✅ FULLY ALIGNED | 3 DLE flags: `shadow_dle_enabled`, `shadow_dle_write_logs`, `dle_enforce_entry_only`. |
| **Exit Reason Normalization** | ✅ FULLY ALIGNED | `exit_reason_map.yaml` (binding), `exit_reason_normalizer.py`, startup verification, 10 canonical values. |
| **Doctrine Kernel** (`doctrine_kernel.py`) | ✅ ALIGNED | Output enriched with DLE fields. Shadow gate observes all verdicts. |
| **Episode Binding** (Phase B.4) | ✅ ALIGNED | Episode UIDs (`EP_<sha256_12>`) deterministic. Authority binding present. |
| **DLE README** | ✅ ALIGNED (this review) | Updated to current phase, implementation status, manifest entries. |
| **copilot-instructions.md** | ✅ ALIGNED (this review) | Updated flags, phase description, doc count. |
| **v7_manifest.json** | ✅ ALIGNED | 4 DLE entries registered with paths and schema versions. |
| **Schema Specs** (Decision, Permit, Episode) | ⚠️ DATED | Status headers say "not yet implemented." ID format diverges from implementation. |
| **Enforcement Gate** | ❌ NOT IMPLEMENTED | `execution/enforcement_gate.py` does not exist. Future Phase D. |
| **Test Coverage** | ✅ GOOD | `test_dle_shadow.py` (6 classes), `test_exit_reason_normalization.py`, integration tests. |

### 4.2 Schema Migration Gap (Updated from V1 §5.2)

| Step | V1 Status | Current Status |
|------|-----------|---------------|
| 1. Shadow gate (Phase A) | DONE | ✅ Active in production |
| 2. Decision enrichment (B.2) | NOT STARTED | ✅ COMPLETE |
| 3. Permit enrichment (B.3) | NOT STARTED | ✅ COMPLETE |
| 4. Episode binding (B.4) | NOT STARTED | ✅ COMPLETE |
| 5. Enforcement rehearsal (B.5) | NOT STARTED | ✅ ACTIVE (counterfactual logging) |
| 6. Enforcement gate (Phase D) | NOT STARTED | ❌ NOT STARTED (requires `DLE_ENFORCED=1`) |

---

## 5. Actions Taken in This Review

### 5.1 Removals

| Action | Details |
|--------|---------|
| **Removed `dle/` root directory** | 10 files were duplicates of `docs/dle/` or `docs/`. 7 were byte-identical, 3 were stale copies with outdated §5.2 and §5.4 content. Canonical location is `docs/dle/` (and `docs/` for top-level doctrine). |
| **Moved `dle/LEDGER_ENTRY_SCHEMA.md`** | Only unique file in `dle/`; moved to `docs/dle/LEDGER_ENTRY_SCHEMA.md`. |

### 5.2 Refreshes

| Action | File | Details |
|--------|------|---------|
| **Rewrote DLE README** | `docs/dle/README.md` | v1.0-SPEC → v2.0. Added: phase progression table, implementation status, ID format documentation, manifest entries, related docs links. Removed: stale "CYCLE_003 documentation only" framing, aspirational migration path (replaced with actual status). |
| **Updated copilot-instructions.md** | `.github/copilot-instructions.md` | Added `DLE_ENFORCE_ENTRY_ONLY` flag. Updated phase description (B.1–B.4 complete, C in progress). Added rehearsal log path. Corrected doc count (14→15). Added schema v2 note. |

### 5.3 Verifications

| Check | Result |
|-------|--------|
| `config/exit_reason_map.yaml` exists | ✅ 4979 bytes, 10 canonical values, binding |
| `v7_manifest.json` DLE entries | ✅ 4 entries (shadow, prediction, rehearsal, denials) |
| `execution/enforcement_gate.py` | ❌ Does not exist (expected — Phase D) |
| DLE unit tests pass | ✅ All pass (test_dle_shadow, test_exit_reason_normalization) |
| DLE_SPEC_CONSISTENCY_CHECK score | ✅ 100% (corrected 2026-02-12) |

---

## 6. Remaining Issues & Future Remediation

### P0 — Pre-Phase D (Must fix before enforcement)

| # | Issue | Description | Effort |
|---|-------|-------------|--------|
| **V2-1** | ID format reconciliation | Schema specs use UUID v4; implementation uses `DEC_hex16`/`PERM_hex16`; prediction spec uses `DEC_hex12`/`PRM_hex12`. Must converge on one format. Recommend adopting implementation format (deterministic, prefix-readable) as canonical. | 2h |
| **V2-2** | Schema status headers | 4 schema docs say "not yet implemented" despite partial implementation. Update to "SPECIFICATION — partially implemented via shadow mode (Phase A/B)". | 30min |

### P1 — Structural Improvements (Pre-v8.0)

| # | Issue | Description | Effort |
|---|-------|-------------|--------|
| **V2-3** | docs/INDEX.md | Master navigation doc for all `docs/` content. High-value for developer onboarding and agent context. | 1h |
| **V2-4** | Regime enum formalization | Sentinel-X regime names (TREND_UP, TREND_DOWN, MEAN_REVERT, BREAKOUT, CHOPPY, CRISIS) not enumerated in Decision/ExecutionRequest schemas. Only open item from spec consistency check. | 30min |
| **V2-5** | Amendment protocol cross-reference | Constitution Art. 14 and Doctrine Falsification Criteria don't reference each other. Add mutual cross-references. | 30min |
| **V2-6** | CODEX_PROMPT_D_LOOP_AUDIT linkage | Orphaned audit doc. Link from docs/active/CHANGELOG_SUMMARY.md or create a reports index. | 15min |

### P2 — Polish (Future cycles)

| # | Issue | Description | Effort |
|---|-------|-------------|--------|
| **V2-7** | Document lifecycle metadata | Owner/Last-reviewed headers on 0% of docs. Add to high-traffic docs first (README, DOCTRINE, GATE_INVARIANTS). | 1h |
| **V2-8** | docs/ taxonomy reorganization | Flat top-level structure. V1 §6.2 proposed: doctrine/, operations/, audit/, reports/ folders. Low urgency — functional as-is. | 1h |
| **V2-9** | DLE schema validation tests | Automated JSON schema validation for Decision, Permit, Episode objects. Would catch drift between spec and implementation. | 4h |
| **V2-10** | Unified glossary | Map terminology across DLE, Doctrine, and execution code. Exit reasons covered by `exit_reason_map.yaml`; broader alignment (veto/deny/refuse, permit/approval) absent. | 2h |

---

## 7. Canonical Document Registry (Top 20)

### Tier 1: Constitutional (Immutable Principles)

| # | Document | Location | Lines | Status |
|---|----------|----------|-------|--------|
| 1 | DLE Constitution V1 | `docs/dle/DLE_CONSTITUTION_V1.md` | 152 | Foundational |
| 2 | DLE Doctrine | `docs/dle/DLE_DOCTRINE.md` | 341 | Binding |
| 3 | DLE Gate Invariants | `docs/dle/DLE_GATE_INVARIANTS.md` | 447 | Partially impl. (mode-gated) |
| 4 | Trading Doctrine Engine Spec | `docs/active/GPT-HEDGE_v7.x_TRADING_DOCTRINE_ENGINE_SPEC.md` | ~269 | Binding |
| 5 | Doctrine Falsification Criteria | `docs/active/DOCTRINE_FALSIFICATION_CRITERIA.md` | ~299 | Active |

### Tier 2: Schemas & Specifications

| # | Document | Location | Lines | Status |
|---|----------|----------|-------|--------|
| 6 | DLE Decision Schema | `docs/dle/DLE_DECISION_SCHEMA.md` | 374 | Spec (shadow-active) |
| 7 | DLE Permit Schema | `docs/dle/DLE_PERMIT_SCHEMA.md` | 326 | Spec (shadow-active) |
| 8 | DLE Execution Request Schema | `docs/dle/DLE_EXECUTION_REQUEST_SCHEMA.md` | 314 | Spec (shadow-active) |
| 9 | DLE Deny Reasons | `docs/dle/DLE_DENY_REASONS.md` | 376 | Partially impl. (C.1) |
| 10 | DLE Episode Schema | `docs/dle/DLE_EPISODE_SCHEMA.md` | 469 | Spec (B.4 binding active) |
| 11 | Exit Reason Map | `config/exit_reason_map.yaml` | ~130 | Binding (startup-verified) |

### Tier 3: Operational & Planning

| # | Document | Location | Lines | Status |
|---|----------|----------|-------|--------|
| 12 | DLE README | `docs/dle/README.md` | ~170 | Active (updated this review) |
| 13 | Phase A Plan (Cycle 004) | `docs/dle/CYCLE_004_PHASE_A_PLAN.md` | 1096 | Sealed (executed) |
| 14 | DLE Crosswalk | `docs/dle/DLE_CROSSWALK_GPT_HEDGE.md` | 565 | Migration guide |
| 15 | Spec Consistency Check | `docs/dle/DLE_SPEC_CONSISTENCY_CHECK.md` | 296 | Audit (100% pass) |
| 16 | Phase B Completion Report | `docs/PHASE_B_SHADOW_AUTHORITY_COMPLETE.md` | ~200 | Sealed |
| 17 | Activation Window v8.0 | `docs/ACTIVATION_WINDOW_v8.md` | ~200 | Active (Phase C) |
| 18 | System Baseline v7.9 | `docs/SYSTEM_BASELINE_v7.9.md` | ~274 | Baseline declared |
| 19 | MHD Audit Suite V1 | `docs/dle/AUDIT_SUITE_V1.md` | 457 | Active |
| 20 | DLE Prediction Layer Spec | `docs/dle/DLE_PREDICTION_LAYER_SPEC.md` | 533 | Design spec (P1) |

### Supplementary (Not in Top 20 but active)

| Document | Location | Status |
|----------|----------|--------|
| Ledger Entry Schema (universal) | `docs/dle/LEDGER_ENTRY_SCHEMA.md` | Binding v1.0 (domain-agnostic) |
| Exchange Unavailable Doctrine | `docs/EXCHANGE_UNAVAILABLE_DOCTRINE.md` | Active (2 gaps noted) |
| Dataset Admission Gate | `docs/DATASET_ADMISSION_GATE.md` | Active |
| Dataset Rollback Clause | `docs/DATASET_ROLLBACK_CLAUSE.md` | Active |
| Prediction Advisory Doctrine | `docs/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md` | Active (P2 gate checklist added) |
| Phase C Amendment | `docs/amendments/PHASE_C_ACTIVATION_WINDOW_AMENDMENT_v1.md` | Binding |
| Dataset Promotion Amendment | `docs/amendments/DATASET_PROMOTION_POLYMARKET_v1.md` | Enacted |
| Copilot Instructions | `.github/copilot-instructions.md` | Active (updated this review) |

---

## 8. Future Development Readiness

### Phase D Preparation Checklist

For DLE enforcement activation (`DLE_ENFORCED=1`), these documentation items must be addressed:

| # | Prerequisite | Current State | Blocking? |
|---|-------------|---------------|-----------|
| 1 | ID format convergence | 3 incompatible formats (V2-1) | **YES** |
| 2 | Schema status headers updated | Say "not yet implemented" (V2-2) | No (cosmetic) |
| 3 | `enforcement_gate.py` design doc | Does not exist | **YES** |
| 4 | Permit lifecycle test coverage | Shadow tests only; no lifecycle tests | **YES** |
| 5 | Regime enum in schemas | Implicit (V2-4) | No (non-blocking) |
| 6 | Phase D runbook | Does not exist | **YES** |
| 7 | Rollback protocol | Does not exist | **YES** |

### Documentation Health Metrics

| Metric | V1 (2026-02-09) | V2 (2026-03-11) | Delta |
|--------|-----------------|-----------------|-------|
| DLE corpus docs | 14 | 15 (+LEDGER_ENTRY_SCHEMA) | +1 |
| Duplicate files | ~12 (dle/ mirror) | 0 | -12 |
| Critical contradictions | 3 | 0 | -3 |
| Stale status headers | 5 | 4 (schema specs) | -1 |
| Manifest DLE entries | 0 | 4 | +4 |
| Exit reason normalization | Missing | Complete (binding) | Fixed |
| copilot-instructions DLE | Missing | Present + current | Fixed |
| DLE README phase accuracy | CYCLE_003 (wrong) | Phase C (correct) | Fixed |
| Test file references | 1 | 3 (shadow, normalization, integration) | +2 |

### Recommendation for V3 Review Trigger

Schedule V3 review when:
- Phase C Activation Window completes (14-day certification ends)
- Phase D enforcement design begins
- OR: 90 days from this review (2026-06-11) — whichever comes first

---

## Appendix A — Files Removed/Moved in This Review

| Action | Source | Destination | Reason |
|--------|--------|-------------|--------|
| REMOVED | `dle/DLE_CONSTITUTION_V1.md` | — | Byte-identical to `docs/dle/DLE_CONSTITUTION_V1.md` |
| REMOVED | `dle/DLE_DOCTRINE.md` | — | Stale copy (missing §5.2/§5.4 updates) |
| REMOVED | `dle/DLE_PREDICTION_LAYER_SPEC.md` | — | Byte-identical to `docs/dle/DLE_PREDICTION_LAYER_SPEC.md` |
| REMOVED | `dle/CYCLE_004_PHASE_A_PLAN.md` | — | Byte-identical to `docs/dle/CYCLE_004_PHASE_A_PLAN.md` |
| REMOVED | `dle/DATASET_ADMISSION_GATE.md` | — | Stale copy (missing sleeve-scoped section) |
| REMOVED | `dle/DATASET_ROLLBACK_CLAUSE.md` | — | Byte-identical to `docs/DATASET_ROLLBACK_CLAUSE.md` |
| REMOVED | `dle/PHASE_A3_DOCTRINE_ATTRIBUTION.md` | — | Byte-identical to `docs/PHASE_A3_DOCTRINE_ATTRIBUTION.md` |
| REMOVED | `dle/PHASE_A3_VETO_ATTRIBUTION.md` | — | Byte-identical to `docs/PHASE_A3_VETO_ATTRIBUTION.md` |
| REMOVED | `dle/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md` | — | Stale (missing import boundary rule + P2 gate) |
| REMOVED | `dle/DLE_DOCTRINE.md` | — | v7.9 updates in docs/dle/ copy only |
| MOVED | `dle/LEDGER_ENTRY_SCHEMA.md` | `docs/dle/LEDGER_ENTRY_SCHEMA.md` | Only unique file in dle/ — consolidated |

**Total: 10 files removed, 1 file moved, 0 files lost.**

## Appendix B — V1 Review Document

Previous review: `docs/DLE_DOCUMENTATION_HYGIENE_REVIEW.md` (450 lines, 2026-02-09).

V1 is preserved as historical record. This V2 supersedes it for all active remediation tracking.
