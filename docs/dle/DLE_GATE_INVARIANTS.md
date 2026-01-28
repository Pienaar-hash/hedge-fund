# DLE Gate Invariants v1

> **Status:** SPECIFICATION — not yet implemented  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28

## Purpose

The **DLE Gate** is the single entry point for all execution authority. This document defines its invariants—the rules that cannot be violated under any circumstances.

These invariants are **constitutional**: they define what the system *is*, not what it *does*.

---

## Core Invariants

### 1. No Implicit Authority

```
IF no Decision explicitly permits an action
THEN the action is FORBIDDEN
```

There is no "default allow." Silence is denial.

**Implementation:** Gate returns `DENY_NO_DECISION` if no matching decision found.

---

### 2. Fail Closed on Ambiguity

```
IF the Gate cannot determine with certainty whether to permit
THEN DENY
```

Uncertainty is not resolved by heuristics. Uncertainty is denial.

**Implementation:** Gate returns `DENY_AMBIGUOUS` on any parsing error, missing field, or logic exception.

---

### 3. Forbidden Overrides Permitted

```
IF Decision.forbidden_actions includes action
AND Decision.permitted_actions includes action
THEN action is FORBIDDEN
```

Explicit prohibition always wins.

**Implementation:** Check `forbidden_actions` before `permitted_actions`.

---

### 4. Single Use Permits

```
IF Permit.state != ISSUED
THEN Permit cannot be consumed
```

A permit can only be used once. Replay is impossible.

**Implementation:** Atomic state transition `ISSUED → CONSUMED` with CAS semantics.

---

### 5. Phase Boundary Enforcement

```
IF Request.context.phase_id != Decision.phase_id
THEN DENY
```

Decisions do not cross phase boundaries. A decision in CYCLE_003 cannot authorize execution in CYCLE_004.

**Implementation:** Gate returns `DENY_PHASE_MISMATCH`.

---

### 6. Conflict Halts Execution

```
IF multiple Decisions could authorize the same action
AND they have conflicting constraints
THEN DENY
```

The Gate does not "merge" or "pick best." Conflict requires human resolution.

**Implementation:** Gate returns `DENY_CONFLICT` with list of conflicting decision IDs.

---

### 7. State Drift Invalidates Permits

```
IF state at execution time differs from state at permit issuance
THEN permit is invalid
```

The permit's `scope_snapshot` must match current state within tolerance.

**Implementation:** Executor verifies snapshot before placing order; rejects on drift.

---

### 8. Ledger Is Append-Only

```
Decisions, Permits, Denials, Episodes CANNOT be edited or deleted
```

History is immutable. Corrections are made by superseding, not editing.

**Implementation:** All ledger writes are append-only. No UPDATE or DELETE operations.

---

### 9. Every Execution Has a Permit

```
IF order is placed
THEN order.permit_id must reference a valid, consumed permit
```

No permit = no execution. This is the fundamental traceability guarantee.

**Implementation:** Order router rejects orders without `permit_id`.

---

### 10. Every Denial Is Logged

```
IF Gate returns DENY
THEN a Denial record is written to denials.jsonl
```

Refusals are first-class artifacts. They prove the system is working.

**Implementation:** Gate logs denial before returning response.

---

## Invariant Verification

### At Startup

Gate verifies:
- [ ] Decision ledger is readable and valid JSON
- [ ] No duplicate decision IDs
- [ ] All decisions have `fail_closed: true`
- [ ] Phase boundaries are consistent

### At Each Request

Gate verifies:
- [ ] Request has all required fields
- [ ] Context snapshot is fresh (< 5s old)
- [ ] Phase ID matches current operational phase
- [ ] Requester is a known component type

### At Each Permit Issuance

Gate verifies:
- [ ] Decision is not expired
- [ ] Decision is not exhausted
- [ ] Action is in permitted_actions
- [ ] Action is not in forbidden_actions
- [ ] All constraints are satisfied
- [ ] No conflicting decisions

---

## Failure Modes

| Failure | Gate Response | System Behavior |
|---------|---------------|-----------------|
| Decision ledger unreadable | `DENY_INTERNAL_ERROR` | All execution halts |
| Multiple matching decisions with conflicts | `DENY_CONFLICT` | Request denied, escalation required |
| Permit expired before use | `DENY_STATE_DRIFT` | Re-request required |
| State drifted during execution | Order rejected | Episode logged as interrupted |
| Unknown deny reason | `DENY_AMBIGUOUS` | Fail closed |

---

## Non-Invariants (What the Gate Does NOT Do)

The Gate does **not**:
- Make trading decisions (that's Hydra/strategy layer)
- Assess market conditions (that's Sentinel-X)
- Calculate position sizes (that's risk layer)
- Route orders (that's order router)
- Manage positions (that's position manager)

The Gate **only** answers: "Is this action permitted right now?"

---

## Relationship to Current Doctrine Kernel

The current `doctrine_kernel.py` is the proto-Gate. Migration path:

| Current | DLE |
|---------|-----|
| `_doctrine_gate()` in executor | `dle_gate.request_permit()` |
| Inline permission logic | Decision lookup |
| Veto logging | Denial ledger |
| No permit artifact | Explicit permit issuance |
| Mixed authority + telemetry | Separated ledgers |

The Gate **wraps** the current doctrine logic initially, then progressively replaces implicit authority with explicit decisions.

---

## Pseudocode

```python
def request_permit(request: ExecutionRequest) -> Union[Permit, Denial]:
    """
    DLE Gate: the single authority checkpoint.
    
    Invariants:
    - No implicit authority
    - Fail closed on ambiguity
    - Forbidden overrides permitted
    - Single use permits
    - Phase boundary enforcement
    - Conflict halts execution
    """
    
    # 1. Validate request
    if not validate_request(request):
        return Denial(reason="DENY_AMBIGUOUS", details="Invalid request format")
    
    # 2. Check phase
    if request.context.phase_id != get_current_phase():
        return Denial(reason="DENY_PHASE_MISMATCH")
    
    # 3. Find matching decisions
    decisions = find_matching_decisions(
        action=request.action.type,
        symbol=request.action.symbol,
        phase=request.context.phase_id
    )
    
    if not decisions:
        return Denial(reason="DENY_NO_DECISION")
    
    # 4. Check for conflicts
    if len(decisions) > 1 and decisions_conflict(decisions):
        return Denial(reason="DENY_CONFLICT", details=f"Conflicting: {[d.id for d in decisions]}")
    
    decision = decisions[0]
    
    # 5. Check expiration
    if decision.is_expired():
        return Denial(reason="DENY_DECISION_EXPIRED")
    
    # 6. Check forbidden (BEFORE permitted)
    if request.action.type in decision.forbidden_actions:
        return Denial(reason="DENY_FORBIDDEN_ACTION")
    
    # 7. Check permitted
    if request.action.type not in decision.permitted_actions:
        return Denial(reason="DENY_NO_DECISION")
    
    # 8. Check constraints
    constraint_check = check_constraints(decision, request.context)
    if not constraint_check.ok:
        return Denial(reason=constraint_check.deny_reason)
    
    # 9. Issue permit
    permit = Permit(
        permit_id=generate_uuid(),
        decision_id=decision.decision_id,
        action=request.action,
        scope_snapshot=snapshot_current_state(),
        expires_ts=now() + PERMIT_TTL,
        state="ISSUED"
    )
    
    log_permit(permit)
    return permit
```

---

## Testing Invariants

Each invariant must have:
1. Unit test proving the invariant holds
2. Integration test proving the invariant holds under load
3. Chaos test proving the invariant holds under failure

Test file: `tests/unit/test_dle_gate_invariants.py`

```python
def test_no_implicit_authority():
    """Gate denies when no decision exists."""
    
def test_fail_closed_on_ambiguity():
    """Gate denies on malformed requests."""
    
def test_forbidden_overrides_permitted():
    """Explicit forbid wins over permit."""
    
def test_single_use_permits():
    """Permit cannot be consumed twice."""
    
def test_phase_boundary_enforcement():
    """Cross-phase execution denied."""
    
def test_conflict_halts_execution():
    """Multiple conflicting decisions = deny."""
```
