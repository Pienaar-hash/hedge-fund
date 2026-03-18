# LEDGER ENTRY SCHEMA

**Status:** Binding Schema Specification  
**Version:** 1.0  
**Scope:** Universal schema for DLE decision entries across all domains

---

## Core Principle

A LedgerEntry is an **immutable record of one decision event**. It answers:

> "What was decided, why, under what constraints, and with what expected outcome?"

It does NOT answer:

> "Was this a good decision?" (That's hindsight)  
> "What should be decided next?" (That's advice)

---

## Schema Definition

```typescript
interface LedgerEntry {
  // === IDENTITY ===
  id: UUID;                          // Immutable unique identifier
  created_at: ISO8601Timestamp;      // When entry was created
  created_by: ActorID;               // Who/what created this entry
  domain: DomainID;                  // Which system domain (PDE, Yellow-Ops, etc.)
  
  // === THE DECISION ===
  decision_type: DecisionTypeEnum;   // Categorization from fixed vocabulary
  decision_statement: string;        // Plain-language description of what was decided
  action_taken: ActionEnum | null;   // What action followed (if any)
  
  // === INPUTS ===
  inputs: {
    documents?: DocumentRef[];       // Policy files, specs, contracts referenced
    observations?: ObservationRef[]; // Data points observed
    prior_entries?: EntryRef[];      // Previous ledger entries this builds on
    external_sources?: SourceRef[];  // URLs, APIs, manual inputs
  };
  
  // === CONTEXT ===
  constraints: {
    time_constraint?: ISO8601Duration;  // Decision deadline if any
    authority_constraint?: string;      // Who had final say
    resource_constraint?: string;       // Budget, capacity limits
    regulatory_constraint?: string;     // Compliance requirements
  };
  
  assumptions: Assumption[];         // Explicit assumptions made
  alternatives_considered: Alternative[]; // What else was on the table
  
  // === OUTPUTS ===
  expected_outcome: {
    description: string;             // What we expect to happen
    success_criteria?: string;       // How we'd know it worked
    time_horizon?: ISO8601Duration;  // When we'd know
  };
  
  // === PROVENANCE ===
  provenance: {
    model_version?: string;          // If ML/AI involved, which version
    doctrine_ref?: DoctrineRef;      // Which doctrine permitted this
    audit_trail: AuditEvent[];       // System events leading here
  };
  
  // === STATE ===
  status: EntryStatus;               // Current state of this entry
}
```

---

## Field-by-Field Specification

### Identity Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `id` | UUID | Yes | Immutable reference. Never changes, never reused. |
| `created_at` | ISO8601 | Yes | Wall-clock time of creation. Used for ordering. |
| `created_by` | ActorID | Yes | Human user ID, API key ID, or system process ID. |
| `domain` | DomainID | Yes | Namespace: `pde`, `yellow-ops`, `gpt-hedge`, etc. |

### Decision Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `decision_type` | Enum | Yes | Fixed vocabulary per domain. E.g., `POLICY_COMPARISON`, `EQUIPMENT_VALUATION`, `TRADE_EXECUTION`. |
| `decision_statement` | String | Yes | Human-readable: "Chose Policy B over Policy A because deductibles were lower." Max 500 chars. |
| `action_taken` | Enum/null | No | If decision translated to action: `ACCEPTED`, `REJECTED`, `DEFERRED`, `ESCALATED`, or null if observational. |

### Input Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `inputs.documents` | DocumentRef[] | No | File hashes, policy IDs, contract refs that were read. |
| `inputs.observations` | ObservationRef[] | No | Data points: prices observed, metrics read, signals received. |
| `inputs.prior_entries` | EntryRef[] | No | Other ledger entries this decision referenced. Creates audit chains. |
| `inputs.external_sources` | SourceRef[] | No | URLs, API endpoints, manual data entry sources. |

### Context Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `constraints.time_constraint` | Duration | No | "Had to decide within 24 hours." ISO8601 duration. |
| `constraints.authority_constraint` | String | No | "Required client sign-off." Free text. |
| `constraints.resource_constraint` | String | No | "Budget limited to $X." Free text. |
| `constraints.regulatory_constraint` | String | No | "Must comply with X regulation." Free text. |
| `assumptions` | Assumption[] | Yes | At least one. Format: `{ statement: string, confidence: 'high'|'medium'|'low' }` |
| `alternatives_considered` | Alternative[] | No | Format: `{ option: string, reason_rejected?: string }` |

### Output Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `expected_outcome.description` | String | Yes | "We expect X to happen." |
| `expected_outcome.success_criteria` | String | No | "We'll know it worked if Y." |
| `expected_outcome.time_horizon` | Duration | No | "We'll know within Z days." |

### Provenance Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `provenance.model_version` | String | Conditional | Required if any ML/AI was involved. Semantic version. |
| `provenance.doctrine_ref` | DoctrineRef | No | Which doctrine document permitted this entry type. |
| `provenance.audit_trail` | AuditEvent[] | Yes | System events: entry created, modified, exported. Append-only. |

### Status Field

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `status` | Enum | Yes | `ACTIVE` (current), `SUPERSEDED` (replaced by another entry), `VOIDED` (administratively cancelled) |

---

## Forbidden Fields

**These fields may NEVER appear in a LedgerEntry:**

| Forbidden Field | Why |
|-----------------|-----|
| `recommendation` | Ledger records decisions, doesn't make them |
| `score` | Implies judgment; we record, not rate |
| `risk_level` | Advisory language; violates observation boundary |
| `priority` | Implies what to do next; outside scope |
| `action_required` | Commands action; ledger is passive |
| `suggested_next_step` | Advisory; forbidden |
| `confidence_in_decision` | Meta-judgment on the decision itself; out of scope |
| `quality_rating` | Retrospective judgment; not permitted |
| `should_have` | Counterfactual advice; forbidden |
| `better_option` | Post-hoc advisory; forbidden |

---

## Type Definitions

```typescript
type UUID = string;  // UUIDv4 format
type ISO8601Timestamp = string;  // e.g., "2024-01-15T14:30:00Z"
type ISO8601Duration = string;   // e.g., "P7D" (7 days), "PT24H" (24 hours)
type ActorID = string;           // User ID, API key, or system process
type DomainID = 'pde' | 'yellow-ops' | 'gpt-hedge' | 'dle-core';

type DecisionTypeEnum = string;  // Domain-specific vocabulary

type EntryStatus = 'ACTIVE' | 'SUPERSEDED' | 'VOIDED';

type ActionEnum = 'ACCEPTED' | 'REJECTED' | 'DEFERRED' | 'ESCALATED' | null;

interface DocumentRef {
  type: 'policy' | 'contract' | 'spec' | 'file';
  id: string;
  hash?: string;        // SHA-256 of content if available
  version?: string;
}

interface ObservationRef {
  type: 'price' | 'metric' | 'signal' | 'state';
  source: string;        // Where observed
  value: string | number;
  observed_at: ISO8601Timestamp;
}

interface EntryRef {
  entry_id: UUID;
  relationship: 'builds_on' | 'supersedes' | 'references';
}

interface SourceRef {
  type: 'url' | 'api' | 'manual';
  identifier: string;
  accessed_at: ISO8601Timestamp;
}

interface Assumption {
  statement: string;
  confidence: 'high' | 'medium' | 'low';
}

interface Alternative {
  option: string;
  reason_rejected?: string;
}

interface DoctrineRef {
  document: string;      // e.g., "DLE_DOCTRINE.md"
  section?: string;      // e.g., "Section 3.2"
  version?: string;
}

interface AuditEvent {
  event_type: 'CREATED' | 'EXPORTED' | 'SUPERSEDED' | 'VOIDED' | 'REFERENCED';
  timestamp: ISO8601Timestamp;
  actor: ActorID;
  details?: string;
}
```

---

## Validation Rules

1. **Non-empty assumptions.** Every entry must have at least one explicit assumption.
2. **Non-empty decision_statement.** Cannot record a decision without stating it.
3. **Valid references.** All EntryRefs must point to existing entries.
4. **Append-only audit_trail.** Events can be added, never removed or modified.
5. **Immutable core fields.** Once created: `id`, `created_at`, `created_by`, `decision_statement`, `inputs` never change.
6. **Status transitions.** `ACTIVE` → `SUPERSEDED` or `VOIDED`. No other transitions. Never back to `ACTIVE`.

---

## Example Entry

```json
{
  "id": "7f3a8b29-4c5d-4e6f-8a9b-0c1d2e3f4a5b",
  "created_at": "2024-01-15T14:30:00Z",
  "created_by": "user:pienaar",
  "domain": "pde",
  "decision_type": "POLICY_COMPARISON",
  "decision_statement": "Selected Commercial Auto policy from Carrier B over Carrier A based on lower deductibles and equivalent coverage limits.",
  "action_taken": "ACCEPTED",
  "inputs": {
    "documents": [
      { "type": "policy", "id": "pol-2024-001", "hash": "abc123..." },
      { "type": "policy", "id": "pol-2024-002", "hash": "def456..." }
    ],
    "observations": [
      { "type": "price", "source": "carrier-quote", "value": 12500, "observed_at": "2024-01-14T10:00:00Z" }
    ]
  },
  "constraints": {
    "time_constraint": "P7D",
    "authority_constraint": "Client approval required"
  },
  "assumptions": [
    { "statement": "Current fleet size remains stable for policy period", "confidence": "high" },
    { "statement": "No major claims expected based on historical data", "confidence": "medium" }
  ],
  "alternatives_considered": [
    { "option": "Carrier A policy", "reason_rejected": "Higher deductibles for equivalent premium" },
    { "option": "Split coverage across carriers", "reason_rejected": "Administrative complexity" }
  ],
  "expected_outcome": {
    "description": "Coverage in place for 12-month term with reduced out-of-pocket on claims",
    "success_criteria": "No coverage gaps discovered during term; claims processed without dispute",
    "time_horizon": "P365D"
  },
  "provenance": {
    "doctrine_ref": { "document": "PDE_DOCTRINE.md", "section": "Policy Comparison" },
    "audit_trail": [
      { "event_type": "CREATED", "timestamp": "2024-01-15T14:30:00Z", "actor": "user:pienaar" }
    ]
  },
  "status": "ACTIVE"
}
```

---

## One Sentence

> **A LedgerEntry is the immutable fact of a decision: what, why, under what constraints, expecting what — never what should have been.**
