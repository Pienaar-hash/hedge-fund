# Dataset Promotion Amendment — Polymarket Feeds (v1)

**Amendment ID:** `AMEND_DATASET_PROMOTION_POLYMARKET_V1`  
**Date:** 2026-02-19  
**Status:** ENACTED  
**Applies to:** `polymarket_snapshot`, `prediction_polymarket_feed`  
**Authority class:** Dataset execution-authority expansion (bounded sleeve only)
**Enacted on:** 2026-02-19

---

## 1. Purpose

Define a formal promotion path for Polymarket datasets that:

- preserves replay determinism,
- preserves Phase C freeze discipline for futures,
- preserves P1/P2 authority boundaries,
- limits execution influence to Binary Lab sleeve only.

This amendment does **not** activate live trading by itself.

---

## 2. Baseline At Proposal Time (Verified)

### 2.1 Current enum states in `config/dataset_admission.json`

Exact state names already in use:

- `REJECTED`
- `OBSERVE_ONLY`
- `RESEARCH_ONLY`
- `PRODUCTION_ELIGIBLE`

No new dataset state enum is introduced by this draft.

This amendment uses operational stages without adding new enum values.

### 2.2 Dataset states at proposal time

- `polymarket_snapshot`: `OBSERVE_ONLY`
- `prediction_polymarket_feed`: `OBSERVE_ONLY`

### 2.3 Current enforcement

- Binary Lab runtime/reducer is deterministic and fail-closed.
- LIVE activation remains blocked unless:
  - prediction phase is `P2_PRODUCTION`, and
  - both datasets are `PRODUCTION_ELIGIBLE`.

---

## 3. Authority Change (Bounded)

If promoted, these datasets may influence **Binary Lab sleeve only**.
They must not influence core futures execution surfaces.

### Hard containment rule

Even in promoted state:

- allowed: Binary Lab reducer/runtime activation path
- forbidden: doctrine kernel, futures risk limits, futures router/sizing/signal paths

Any non-binary consumer attempt is deny-logged and ignored.

Recommendation in this amendment:

- `PRODUCTION_ELIGIBLE` for these two datasets is interpreted as
  `PRODUCTION_ELIGIBLE + binary_sleeve_scope_only`.
- It is not a blanket grant to all prediction consumers.

---

## 4. Promotion Ladder (Operational)

Because enum values are fixed, ladder stages are expressed operationally:

1. `OBSERVE_ONLY` + **Shadow-only authority**
   - Dataset can drive Binary Lab shadow decisions (no orders).
2. `PRODUCTION_ELIGIBLE` + **Binary live authority**
   - Dataset may drive Binary Lab LIVE mode only.
3. Core futures authority remains forbidden (out of scope).

Interpretation note:

- "`SHADOW_EXECUTION_ELIGIBLE`" maps to stage (1) operationally and does **not**
  require adding a new enum in `dataset_admission.json`.

---

## 5. Preconditions (Evidence-Gated)

Promotion cannot proceed unless all are true:

1. Determinism replay proof passes (byte-identical + SHA-256 step hashes).
2. `binary_lab_state.json` surface is stable and manifest-aligned.
3. Runtime writer has zero order path coupling.
4. Phase C frozen futures configs remain untouched.
5. P-phase authority permits execution influence (`P2_PRODUCTION` required for LIVE).

---

## 6. Mechanical Invariants (Permanent)

These are mandatory regression invariants:

1. If dataset state is not `PRODUCTION_ELIGIBLE` -> reducer cannot enter LIVE.
2. If phase is not `P2_PRODUCTION` -> reducer cannot enter LIVE.
3. Limits hash mismatch/missing proof -> `status=DISABLED`.
4. Promotion cannot bypass freeze/hash lock.
5. Duplicate same-day checkpoints are reducer no-ops.

---

## 7. Patch Scope (Exact)

This amendment authorizes the following bounded patch scope only.

### 7.1 Allowed files

- `config/dataset_admission.json`
  - update states for the two target datasets only (when enacted)
  - add explicit bounded authority metadata for these datasets:
    - `execution_scope: "BINARY_LAB_ONLY"`
    - `allowed_consumers: ["binary_lab"]`
    - `denied_consumers: ["futures_execution", "doctrine_kernel", "risk_limits", "router"]`
- `docs/DATASET_ADMISSION_GATE.md`
  - add bounded-consumer rule for sleeve-scoped promotion
- `docs/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md`
  - clarify that any execution influence requires `P2_PRODUCTION`
- `docs/amendments/DATASET_PROMOTION_POLYMARKET_v1.md`
  - this decision artifact
- `tests/unit/*`, `tests/integration/*`
  - invariant and containment tests

### 7.2 Forbidden files (for this amendment)

- `config/strategy_config.json`
- `config/risk_limits.json`
- `config/pairs_universe.json`
- any futures execution logic that could alter order flow

### 7.3 Explicit non-goals

- No change to core futures decisioning.
- No change to doctrine kernel authority.
- No expansion to other datasets.

---

## 8. Rollback Clause (Amendment-Specific)

On any of the following, rollback is immediate:

- replay non-determinism detected,
- phase mismatch (`P1` while LIVE requested),
- cross-sleeve influence leak detected,
- limits hash/freeze breach.

Rollback action:

1. set target datasets back to `OBSERVE_ONLY`,
2. keep Binary Lab runtime in `DISABLED` or `NOT_DEPLOYED`,
3. preserve logs (no history rewrite).

---

## 9. Verification Plan

Minimum required verification after promotion patch:

1. `pytest -q tests/unit tests/integration`
2. manifest audit returns `MANIFEST_OK`
3. binary invariants pass:
   - OBSERVE_ONLY cannot activate LIVE
   - non-`P2_PRODUCTION` cannot activate LIVE
   - restart-invariant checkpoint behavior
4. replay determinism harness remains green

5. containment tests pass:
   - binary-only consumers allowed
   - non-binary consumers denied even after promotion

---

## 10. Decision Record Template

```text
Decision: APPROVE | DENY
Amendment ID: AMEND_DATASET_PROMOTION_POLYMARKET_V1
Datasets: polymarket_snapshot, prediction_polymarket_feed
Scope: Binary Lab sleeve only
Effective phase: P2_PRODUCTION only
Rollback owner: Fund-Ops
Evidence bundle:
  - replay test run id
  - manifest audit log
  - invariant test report
Sign-off:
  - Doctrine authority
  - Operator
  - Engineering
```

---

## 11. Current Disposition

This amendment is enacted.

Post-enactment constraints:

- dataset states are promoted with bounded consumer scope,
- Binary Lab remains non-invasive,
- LIVE activation remains blocked unless `P2_PRODUCTION` and all reducer gates pass.

---

## Appendix A: Patch-Scope Checklist (Execution)

Use this checklist when enacting this amendment:

1. Confirm replay harness and hash-step determinism proofs are green.
2. Confirm Binary Lab runtime writer is active and manifest-aligned.
3. Patch only allowed files in Section 7.
4. Do not modify futures configuration or futures execution logic.
5. Run verification in Section 9 and archive evidence bundle.
6. Record explicit sign-off using Section 10 template.
