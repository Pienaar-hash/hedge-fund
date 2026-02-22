# P2 Activation Decision — Binary Lab Sleeve Only (v1)

**Decision ID:** `DEC_P2_BINARY_LAB_SLEEVE_ACTIVATION_V1`  
**Date:** 2026-02-19  
**Status:** DRAFT (awaiting sign-off)  
**Applies to:** Binary Lab sleeve only (Polymarket BTC Up/Down 15m)  
**Does not apply to:** Futures execution, doctrine kernel, risk limits, router

---

## 1. Purpose

Authorize transition of Binary Lab sleeve authority from **P1 advisory** to
**P2 production** for the sleeve only, enabling existing LIVE activation gates
in `execution/binary_lab_executor.py`.

This decision does not itself place orders and does not modify futures
execution authority.

---

## 2. Authority And Scope

### 2.1 Allowed

Binary Lab reducer/runtime may enter LIVE only if all are true:

1. prediction phase is `P2_PRODUCTION`
2. `config/binary_lab_limits.json` hash proof passes
3. `polymarket_snapshot` and `prediction_polymarket_feed` are `PRODUCTION_ELIGIBLE`
4. dataset scope permits consumer `binary_lab`
5. all Binary Lab reducer invariants pass (horizon, freeze, rule checks)

### 2.2 Forbidden

Polymarket datasets are forbidden for:

- `futures_execution`
- `doctrine_kernel`
- `risk_limits`
- `router`

No parameter changes to Phase C frozen futures configs are authorized by this
decision.

### 2.3 Containment Statement

This decision is sleeve-local authority only. It is not a grant for global
prediction consumers and is not a futures execution authorization.

---

## 3. Preconditions (Evidence Bundle)

Sign-off requires attached evidence that all of the following are true:

1. manifest audit returns `MANIFEST_OK`
2. replay determinism proof is byte-identical with stepwise SHA-256 checks
3. containment tests prove:
   - `binary_lab` allowed under scoped promotion
   - non-binary consumers denied with deny-log evidence
4. `logs/state/binary_lab_state.json` is emitted and restart-invariant

Evidence references (2026-02-19 validation run):

- `python3 scripts/manifest_audit.py enforce` -> `MANIFEST_OK`
- `pytest -q tests/unit tests/integration` -> passed
- `tests/unit/test_prediction_firewall.py` scope allow/deny coverage present
- `tests/integration/test_binary_lab_replay_determinism.py` proves replay
  determinism

---

## 4. Activation Window Constraints

- Horizon: `15m` only
- Limits: locked to `config/binary_lab_limits.json` (hash enforced)
- Concurrency: Binary Lab sleeve-local only (no futures coupling)
- Kill line: enforced by reducer/runtime fail-closed path
- Freeze discipline: no parameter changes during window

---

## 5. Rollback Triggers And Actions

### 5.1 Immediate rollback triggers

- cross-sleeve influence leak detected
- replay non-determinism detected
- limits hash mismatch
- LIVE requested outside `P2_PRODUCTION`
- kill line breach or hard-rule violation (reducer termination conditions)

### 5.2 Rollback actions

1. set prediction phase back to P1 advisory posture for Binary Lab consumers
2. keep dataset promotions in place (authority remains phase-gated and scoped)
3. preserve all logs append-only (no history rewrite)

Rollback owner: Fund-Ops.

---

## 6. Governance Mapping

This decision satisfies Constitution phase-boundary requirements by explicitly
declaring:

- model boundary (`P1` -> `P2`)
- scope boundary (Binary Lab only)
- risk acknowledgement and rollback plan

Primary references:

- `docs/dle/DLE_CONSTITUTION_V1.md`
- `docs/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md`
- `docs/amendments/DATASET_PROMOTION_POLYMARKET_v1.md`
- `ops/BINARY_LAB_EXECUTOR_STATE_MACHINE.md`

---

## 7. Sign-off

- Doctrine authority: ____________________
- Operator (Fund-Ops): ____________________
- Engineering: ____________________

Decision outcome:

- [ ] APPROVED
- [ ] DENIED

