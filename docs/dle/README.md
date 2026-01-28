# Decision Ledger Engine (DLE) — GPT-Hedge Integration

> **Version:** v1.0-SPEC  
> **Status:** SPECIFICATION — approved for future implementation  
> **Date:** 2026-01-28  
> **Phase:** CYCLE_003 (documentation only)

## Overview

The Decision Ledger Engine (DLE) is a governance framework that makes **authority explicit, traceable, and immutable**. It prevents silent drift by ensuring every execution can be traced back to a formal decision.

This specification defines the target state for GPT-Hedge governance. Implementation is deferred until CYCLE_004 or later.

---

## Documents

| Document | Purpose |
|----------|---------|
| [DLE_DECISION_SCHEMA.md](DLE_DECISION_SCHEMA.md) | Authority objects that permit/forbid actions |
| [DLE_PERMIT_SCHEMA.md](DLE_PERMIT_SCHEMA.md) | Single-use execution tokens |
| [DLE_EXECUTION_REQUEST_SCHEMA.md](DLE_EXECUTION_REQUEST_SCHEMA.md) | Input format for Gate requests |
| [DLE_DENY_REASONS.md](DLE_DENY_REASONS.md) | Canonical denial codes and meanings |
| [DLE_EPISODE_SCHEMA.md](DLE_EPISODE_SCHEMA.md) | Complete trade lifecycle binding |
| [DLE_GATE_INVARIANTS.md](DLE_GATE_INVARIANTS.md) | Constitutional rules (fail-closed) |

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
│                       DLE GATE                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                                                       │   │
│  │  1. Validate request format                          │   │
│  │  2. Check phase boundary                             │   │
│  │  3. Find matching decisions                          │   │
│  │  4. Detect conflicts (halt if found)                 │   │
│  │  5. Check forbidden_actions (deny if matched)        │   │
│  │  6. Check permitted_actions                          │   │
│  │  7. Verify all constraints                           │   │
│  │  8. Issue permit OR deny with reason                 │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│            ┌─────────────┴─────────────┐                    │
│            │                           │                    │
│            ▼                           ▼                    │
│    ┌──────────────┐           ┌──────────────┐             │
│    │    Permit    │           │    Denial    │             │
│    │   (single    │           │   (logged,   │             │
│    │    use)      │           │   auditable) │             │
│    └──────┬───────┘           └──────────────┘             │
│           │                                                 │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                      ORDER ROUTER                             │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Verify permit.state == ISSUED                        │   │
│  │  Verify permit not expired                            │   │
│  │  Verify state matches scope_snapshot                  │   │
│  │  Execute order                                        │   │
│  │  Mark permit CONSUMED                                 │   │
│  │  Attach permit_id to order record                     │   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                     LEDGER LAYER                              │
│                                                               │
│  decision_ledger.jsonl  ←  Authority statements               │
│  permits_issued.jsonl   ←  Execution tokens                   │
│  denials.jsonl          ←  Refused requests                   │
│  episode_ledger.jsonl   ←  Complete trade lifecycle           │
│  orders_executed.jsonl  ←  Fill evidence (existing)           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Migration Path

### Phase 1: Ledger Separation (Non-Breaking)

1. Create `logs/dle/` directory
2. Split `doctrine_events.jsonl`:
   - Authority → `decision_ledger.jsonl`
   - Telemetry → remains in `doctrine_events.jsonl`
3. No execution changes

### Phase 2: Gate Wrapper (Shadow Mode)

1. Implement `dle_gate.py` as wrapper around existing `_doctrine_gate()`
2. Log what DLE *would* decide alongside actual decision
3. Compare for divergence
4. No execution changes

### Phase 3: Permit Requirement (Breaking)

1. Executor requires permit for order placement
2. Order router validates permit before execution
3. Episode ledger binds decision_id + permit_id
4. Full traceability active

### Phase 4: Decision-First Authority

1. Remove implicit doctrine authority
2. All trading requires explicit Decision objects
3. Doctrine becomes decision *generator*, not authority *source*

---

## Relationship to Current System

| Current Component | DLE Relationship |
|-------------------|------------------|
| `doctrine_kernel.py` | Becomes decision generator + constraint checker |
| `_doctrine_gate()` | Wrapped by `dle_gate.request_permit()` |
| `doctrine_events.jsonl` | Split into decision_ledger + telemetry |
| `episode_ledger.json` | Extended with authority binding |
| `risk_limits.py` | Becomes constraint input to decisions |
| `sentinel_x.py` | Provides regime context for constraint checking |

---

## Benefits

1. **No Implicit Authority** — Every action traceable to explicit decision
2. **No Double Execution** — Single-use permits prevent replay
3. **Perfect Audit Trail** — Episode binds Decision → Permit → Order → Outcome
4. **Regeneration-Safe** — Code can change; ledger remains truth
5. **Conflict Detection** — Multiple decisions = halt, not heuristic
6. **Investor Transparency** — Single artifact (episode_ledger) proves everything

---

## Constraints

- **CYCLE_003:** Documentation only; no execution changes
- **Backward Compatible:** Must work alongside existing system during migration
- **Fail-Closed:** Ambiguity = denial; no fallback to implicit authority
- **Append-Only:** All ledgers are immutable

---

## Open Questions (For CYCLE_004+)

1. **Decision Generation:** How are decisions created? Manual? Auto from Doctrine?
2. **Decision TTL:** Default expiration for auto-generated decisions?
3. **Conflict Resolution:** Escalation path when decisions conflict?
4. **Performance:** Permit issuance latency requirements?
5. **Recovery:** What happens to orphaned permits on crash?

---

## Approval

This specification is approved as the target governance model for GPT-Hedge.

Implementation deferred until:
- CYCLE_003 observation period complete
- Regime stability achieved
- Explicit decision to enter CYCLE_004
