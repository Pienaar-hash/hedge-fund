# DECISION-LEDGER ENGINE (DLE) — DOCTRINE

**Status:** Binding

## Authority Level

This doctrine has **higher authority than features, UX, integrations, adoption pressure, or revenue ideas**.
Any violation invalidates the system.

---

## 1. Prime Definition (Immutable)

> **The Decision-Ledger Engine exists to freeze decisions as immutable ledger entries by recording what was known, what was assumed, what constrained the choice, and what options were valid — without managing work, recommending outcomes, or owning responsibility.**

If DLE begins to **decide**, **coordinate**, or **optimize**, it ceases to be DLE.

---

## 2. Core Posture — Observer, Not Actor

### 2.1 What DLE Is

DLE is:

* A **memory system**
* An **audit surface**
* A **decision explanation engine**
* A **constraint recorder**

### 2.2 What DLE Is Not

DLE is **not**:

* A workflow engine
* A task manager
* A CRM
* A committee tool
* An approval system
* A recommendation engine
* An optimizer
* A simulator of reality

DLE observes decisions.
Humans remain accountable.

---

## 3. Canonical Objects (Fixed)

Every DLE ledger entry consists of **only** the following objects:

1. **InputSnapshot**

   * Observed facts
   * Source references
   * Timestamps
   * Hashes

2. **AssumptionSet**

   * Explicit assumptions
   * Declared uncertainties
   * "Unknown" markers
   * Confidence qualifiers

3. **ConstraintSet**

   * Policies
   * Rules
   * Risk limits
   * Doctrines
   * Prohibitions

4. **DecisionEnvelope**

   * Allowed options
   * Bounded ranges
   * Excluded options (with reasons)
   * Silence where evidence is insufficient

5. **LedgerEntry**

   * Immutable binding of all above
   * Append-only
   * Replayable
   * Hash-addressed

6. **Provenance**

   * Who recorded
   * When
   * Under which version of rules
   * With which tools

No additional object types are permitted.

---

## 4. MESSY DOCTRINE

*(How DLE handles ambiguity without absorbing it)*

### 4.1 Ambiguity Rule

Ambiguity must be:

* **Declared**
* **Isolated**
* **Visible**

Ambiguity must never be:

* Resolved implicitly
* Smoothed over
* Converted into narrative certainty

---

### 4.2 Assumption Discipline

Every assumption must be:

* Explicit
* Bounded
* Separately identifiable from facts

Assumptions are **first-class citizens**, not footnotes.

---

### 4.3 Forbidden Meaning Inflation

DLE must never claim:

* Correctness
* Suitability
* Fairness
* Optimality
* Compliance beyond encoded constraints

If meaning cannot be justified by:

> input + assumption + constraint

it must not appear.

---

### 4.4 Schema Lock

* No per-user schema changes
* No per-domain field drift
* No freeform narrative as primary truth

If something cannot fit the schema, it does not belong in DLE.

---

## 5. HEAVY DOCTRINE

*(Preventing operational gravity)*

### 5.1 Append-Only Rule

Ledger entries:

* Are written once
* Are never edited
* Are never "completed"
* Are never reopened

Correction happens only via **new entries**.

---

### 5.2 No Lifecycle Management

DLE records **observed lifecycle state** — it does not drive lifecycle transitions.

DLE must not:

* Own or orchestrate status transitions (Episode schema records what execution does)
* Manage ownership, due dates, reminders, escalations, notifications, or SLAs
* Replace the position state machine or execution orchestrator

The Episode schema (see `DLE_EPISODE_SCHEMA.md`) captures lifecycle observations
(PENDING → OPEN → CLOSING → CLOSED) as **audit surface**, not as **control surface**.
The executor remains the sole lifecycle authority.

Any lifecycle *management* implies responsibility.
Responsibility is forbidden — observation is required.

---

### 5.3 Convergence Requirement

DLE must converge to:

> capture → freeze → export → observe

If DLE requires:

* Continuous tuning
* Daily attention
* Operational support

…it violates doctrine.

---

### 5.4 Non-Criticality Rule

DLE operates in one of two modes:

| Mode | Flag | Authority | Failure Behavior |
|------|------|-----------|------------------|
| **SHADOW_MODE** | `SHADOW_DLE_ENABLED=1` | Observation only | DLE offline → work continues, no system halts |
| **ENFORCED_MODE** | `DLE_ENFORCED=1` (Phase B+) | Permit-authoritative | Permit absence blocks execution per Gate Invariant #9 |

**In SHADOW_MODE (default, v7.x):**

* DLE may explain decisions — it may never block them.
* Permits are advisory; missing permits log a shadow violation.
* If DLE goes offline, work continues and decisions can still be made.

**In ENFORCED_MODE (Phase B+, not yet active):**

* Gate Invariant #9 applies: no permit = no order.
* DLE transitions from observer to control surface.
* Offline DLE triggers fail-closed behavior (no new orders until restored).

This scoping resolves the relationship between this section and `DLE_GATE_INVARIANTS.md` Invariant #9.
Invariant #9 is **constitutionally valid** but **mode-gated**: it is binding only in ENFORCED_MODE.

---

## 6. DEPENDENCY DOCTRINE

*(Avoiding veto power and adoption coupling)*

### 6.1 Local-First Principle

DLE must be usable by:

* A single individual
* Without organizational permission
* Without integrations
* Without migration

Value must appear immediately.

---

### 6.2 Export-First Mandate

Every ledger entry must be:

* Exportable
* Portable
* Replayable elsewhere

DLE does not trap data.
It **releases** it.

---

### 6.3 No Adoption Dependency

DLE must never require:

* Company-wide rollout
* Process change
* Tool replacement
* Central authority buy-in

Partial, silent usage is valid.

---

## 7. AI & Automation Constraints

### 7.1 AI Role (Strictly Limited)

AI may:

* Normalize inputs
* Surface inconsistencies
* Flag missing assumptions
* Enforce schema validity

AI may **not**:

* Recommend decisions
* Rank options
* Optimize outcomes
* Infer intent
* Fill gaps silently

Silence is preferable to hallucination.

---

### 7.2 Explanation Boundary

AI explanations must:

* Be traceable to ledger objects
* Cite inputs and assumptions explicitly
* Avoid normative language

If AI cannot explain deterministically, it must abstain.

---

## 8. UI / UX Doctrine

### 8.1 Subordinate Presentation

DLE UI must:

* Emphasize structure over narrative
* Make assumptions visually explicit
* Highlight uncertainty
* Avoid persuasive design

This is a **record**, not a pitch.

---

### 8.2 Forbidden Language

UI must not use:

* "Recommended"
* "Best"
* "Should"
* "Approved"
* "Pending"
* "Resolved"

Allowed language:

* "Observed"
* "Assumed"
* "Constrained"
* "Permitted"
* "Excluded"

---

## 9. Revenue & Product Discipline

### 9.1 Revenue Constraint

Revenue must never depend on:

* Decision frequency
* Outcome success
* Workflow centrality
* Organizational lock-in

If revenue pressures push DLE toward responsibility, revenue is invalid.

---

### 9.2 Expansion Test (Mandatory)

Before adding any feature, ask:

1. Does this create mutable state?
2. Does this imply ownership of outcomes?
3. Does this pull work *into* the system?
4. Does this require adoption to function?

If **yes** to any → feature is forbidden.

---

## 10. Kill Conditions (Explicit)

DLE must be frozen or terminated if:

* Users treat it as a decision-maker
* It becomes a required step in workflows
* Custom schemas proliferate
* Tasks, owners, or deadlines appear
* It is marketed as "decision automation"
* Accountability shifts away from humans

These are **structural failures**, not bugs.

---

## 11. Final Classification

Under this doctrine, DLE is:

* 🟢 **MHD-compliant**
* 🟢 **Observer-grade**
* 🟢 **Low-gravity**
* 🟡 **Adjunct, never core**
* 🔴 **Never operational**

DLE's value lies in **memory, clarity, and restraint**.

---

## Closing Statement

> **DLE does not make better decisions.
> It makes decisions legible — to humans, regulators, and time itself.**

That constraint is what keeps it clean.
