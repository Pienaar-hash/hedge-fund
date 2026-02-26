# Phase C Amendment — Activation Window v8.0 as Structural Certification Layer

**Amendment ID:** PHASE_C_AW_v1  
**Status:** Binding  
**Effective:** 2026-02-26  
**Phase:** C (Contraction Window)  
**Authority:** DLE Constitution v1, Doctrine §5 (Boundaries), §7 (Contracts), §8 (Risk)  

---

## 1. Declaration

Activation Window v8.0 (`execution/activation_window.py`) is hereby declared a
**STRUCTURAL_CERTIFICATION_LAYER** within Phase C governance.

It is not a feature. It is not a guard rail. It is a **constitutional enforcement layer**
that must complete with a formal 7/7 GO verdict before any production capital scaling
is authorized.

> All execution is subordinate to explicit decisions.  
> — DLE Constitution §1 (Prime Directive)

Activation Window v8.0 is a decision boundary. Crossing it without formal
verification is a governance violation.

---

## 2. Constitutional Binding

### 2.1 Mandatory Pre-Scale Gate

No production sizing increase (e.g. 0.5% → 1% per-trade NAV) may occur unless:

| Condition | Verification |
|-----------|-------------|
| Activation window completed | `activation_window_state.json → window_expired: true` |
| Verification score 7/7 | `activation_verification_verdict.json → verdict: "GO", passed: 7` |
| Manifest hash intact | Verdict `manifest_hash` matches current `v7_manifest.json` hash |

If any condition is unmet, the certification sizing cap (`per_trade_nav_pct`)
remains in force, regardless of whether the activation window is enabled or disabled
in `runtime.yaml`.

### 2.2 No Silent Override

The scale gate (`require_go_for_scale: true` in `runtime.yaml`) cannot be bypassed
without a Decision Ledger entry. Disabling it:

- Requires explicit config change (auditable in git)
- Must be documented in `ops/phase_c_window.log`
- Is logged as a `STRUCTURAL_GUARD` event in `dle_shadow_events.jsonl`

### 2.3 Doctrine Supremacy Preserved

Activation Window v8.0 operates **alongside** Doctrine Kernel, not above it.

- Doctrine gates individual trades (entry/exit permission)
- Activation Window gates system-level scaling (structural certification)
- Neither can override the other
- Both must permit for an order to execute at full production sizing

---

## 3. DLE Event Contract

Activation Window v8.0 emits **STRUCTURAL_GUARD** events to the DLE shadow log
(`logs/execution/dle_shadow_events.jsonl`). These events are governance artifacts,
not trade-gating events. They are emitted regardless of `SHADOW_DLE_ENABLED` state.

### 3.1 Lifecycle Events

| Event | Action | Trigger |
|-------|--------|---------|
| **STARTED** | Window activated at executor boot | `log_activation_boot_status()` when dual-key confirmed |
| **HALTED** | Structural kill condition triggered | `check_activation_window()` on drawdown/drift/freeze violation |
| **COMPLETED** | 14-day window elapsed without structural kill | `check_activation_window()` when `window_expired` |
| **VERIFIED** | Day-14 formal verification executed | `scripts/activation_verify.py` records verdict |

### 3.2 Event Schema

```json
{
  "schema_version": "dle_shadow_v2",
  "event_type": "STRUCTURAL_GUARD",
  "ts": "ISO-8601",
  "payload": {
    "guard_type": "ACTIVATION_WINDOW",
    "action": "STARTED|HALTED|COMPLETED|VERIFIED",
    "phase_id": "PHASE_C",
    "window_start_ts": "ISO-8601",
    "duration_days": 14,
    "elapsed_days": 3.5,
    "manifest_hash": "abc123def456...",
    "config_hash": "789abc012def...",
    "halt_reason": null,
    "verification_score": null,
    "verdict": null,
    "provenance": {
      "source": "activation_window",
      "version": "v8.0"
    }
  }
}
```

### 3.3 Observability Contract

- All lifecycle events are append-only to `dle_shadow_events.jsonl`
- Events use `schema_version: "dle_shadow_v2"` (backward-compatible extension)
- Consumers that do not recognize `STRUCTURAL_GUARD` event_type skip them safely
- Episode binding (`LINK` events) is not emitted for structural guards — they are
  system-level, not trade-level

---

## 4. Production Scale Authorization Protocol

### Step 1: Complete Activation Window

```yaml
# config/runtime.yaml
activation_window:
  enabled: true
  duration_days: 14
  start_ts: "2026-02-15T00:00:00Z"
  drawdown_kill_pct: 0.05
  per_trade_nav_pct: 0.005
  require_go_for_scale: true
```

### Step 2: Survive 14 Days

Window must reach `window_expired: true` without any structural kill.
If halted, investigate root cause, reset, and restart the window.

### Step 3: Run Formal Verification

```bash
PYTHONPATH=. python scripts/activation_verify.py
```

Must output: `VERDICT: GO (7/7)`.
Verdict is persisted to `logs/state/activation_verification_verdict.json`.

### Step 4: Scale Sizing

Only after GO verdict with matching manifest hash, the operator may increase
`per_trade_nav_pct` in `strategy_config.json` above the certification cap.

The `require_go_for_scale` flag enforces this automatically — sizing is
capped at the certification level until a valid verdict exists.

---

## 5. Invariants

1. **No scale without GO** — Production sizing above certification cap requires 7/7 GO verdict.
2. **Manifest must match** — Verdict manifest hash must match current manifest. Any manifest
   change after verification invalidates the verdict.
3. **Gate survives disable** — Disabling `activation_window.enabled` does NOT remove the
   sizing cap. Only a valid GO verdict clears it.
4. **Events are governance** — STRUCTURAL_GUARD events are emitted unconditionally (not gated
   by `SHADOW_DLE_ENABLED`). They are constitutional artifacts.
5. **Fail-closed for scale** — If verification verdict is missing, corrupted, or stale →
   sizing cap remains active.
6. **Fail-open for execution** — DLE lifecycle event emission failures never crash the executor.
   Governance observability is best-effort; execution continues.

---

## 6. State Surfaces

| File | Purpose | Owner |
|------|---------|-------|
| `logs/state/activation_window_state.json` | Per-loop window state | `activation_window` (existing) |
| `logs/state/activation_verification_verdict.json` | One-time verification verdict | `activation_verify` (new) |
| `logs/execution/dle_shadow_events.jsonl` | DLE shadow log (STRUCTURAL_GUARD events) | `dle_shadow` (extended) |

---

## 7. Amendment Process

This amendment was adopted under DLE Constitution §13 (Amendment Process):

- **Proposal:** Phase C / DLE governance mapping for Activation Window v8.0
- **Rationale:** Activation Window performs structural certification — it must be
  declared as a constitutional boundary, not treated as a runtime guard.
- **Decision:** Recorded in this document and in `dle_shadow_events.jsonl` as a
  `STRUCTURAL_GUARD` event.

Future amendments to this layer require the same process: explicit proposal,
documented rationale, recorded decision.

---

## 8. References

| Document | Relevance |
|----------|-----------|
| [DLE Constitution v1](../dle/DLE_CONSTITUTION_V1.md) | Foundational authority |
| [DLE Doctrine](../dle/DLE_DOCTRINE.md) | Observer-not-actor posture |
| [DLE Gate Invariants](../dle/DLE_GATE_INVARIANTS.md) | Constitutional invariants |
| [Activation Window v8.0 Runbook](../ACTIVATION_WINDOW_v8.md) | Operational procedures |
| [C.1 Ops Protocol](../../ops/C1_OPS_PROTOCOL.md) | Entry-only enforcement ladder |
| [System Baseline v7.9](../SYSTEM_BASELINE_v7.9.md) | Phase status table |
