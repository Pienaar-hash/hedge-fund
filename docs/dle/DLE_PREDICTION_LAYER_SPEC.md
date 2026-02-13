# DLE NATIVE PREDICTION LAYER — SPECIFICATION

**Status:** Design Specification

## One-Sentence Definition

> **A DLE native prediction layer is a belief ledger that produces Decision and Permit objects for "belief updates" the same way we already do for trades, with logical consistency enforced by the engine, not by traders.**

This means: **no belief affects anything unless it is admitted (Dataset Admission Gate) and permitted (Decision Permit).**

---

## 1. Core Objects

### 1.1 Belief Event

A single user or model update: "probability of outcome X is now p, with confidence c, at time t".

It's *not* a market order. It's a **statement**.

### 1.2 Constraint Set

A set of logical rules binding beliefs together:

* Mutually exclusive outcomes sum to 1
* Conditional dominance (A implies B)
* Bounds (0 ≤ p ≤ 1)
* Optional "budget" constraints (max change per step)

These are enforced **centrally** (engine layer logic), so we don't rely on arbitrage.

### 1.3 Belief Aggregate

The engine's current canonical probabilities after applying:

* All admitted belief events
* Weighting rules
* Constraints
* Replay determinism

### 1.4 DLE Authority Objects

We reuse DLE primitives:

* **Decision**: authorizes a class of belief updates (who can write, which questions, max delta, TTL)
* **Permit**: single-use token that authorizes *one* belief event (write)
* **Episode**: end-to-end record linking belief updates → downstream use → outcome scoring

---

## 2. System Architecture

### Dataflow

```
1. INGRESS
   └── humans, internal models, external feeds

2. DATASET ADMISSION GATE
   └── every source is a dataset with state:
       REJECTED → OBSERVE_ONLY → RESEARCH_ONLY → PRODUCTION_ELIGIBLE

3. DLE GATE (PREDICTION)
   └── converts belief update request into Decision check
   └── issues Permit if allowed

4. CONSTRAINT SOLVER
   └── applies belief update
   └── enforces logical consistency
   └── emits canonical probabilities

5. BELIEF LEDGER
   └── append-only logs of events, decisions, permits, aggregates

6. DOWNSTREAM CONSUMERS
   └── research
   └── routing hints
   └── risk overlays
   └── never "authoritative" unless explicitly promoted
```

### Why This Passes MHD

| Gate | How It Passes |
|------|---------------|
| Messy | Beliefs are normalized into one event type + constraints |
| Heavy | All authority is explicit, logged, replayable |
| Dependent | No reliance on traders/bots or hidden liquidity; engine enforces logic and rollbacks |

---

## 3. Schemas

### 3.1 BeliefEvent Schema

```json
{
  "belief_event_id": "BEL_<hex12>",
  "ts": "ISO8601",
  "dataset_id": "polymarket|human|model_x",
  "question_id": "Q_<id>",
  "outcome_id": "O_<id>",
  "p": 0.63,
  "confidence": 0.72,
  "evidence_hash": "sha256:<8>",
  "dle": {
    "decision_id": "DEC_<hex12>",
    "permit_id": "PRM_<hex12>"
  },
  "state_snapshot_hashes": {
    "constraints": "<sha256_8>",
    "prior_aggregate": "<sha256_8>",
    "config": "<sha256_8>"
  }
}
```

### 3.2 QuestionGraph Schema

Defines logical constraints:

```json
{
  "question_id": "Q_election_2026",
  "outcomes": ["O_a", "O_b", "O_c"],
  "constraints": [
    {
      "type": "SUM_TO_ONE",
      "outcomes": ["O_a", "O_b", "O_c"]
    },
    {
      "type": "IMPLIES",
      "if": {"Q": "Q_x", "O": "O_yes"},
      "then": {"Q": "Q_y", "O": "O_yes"}
    },
    {
      "type": "BOUNDS",
      "outcomes": ["O_a", "O_b", "O_c"],
      "min": 0,
      "max": 1
    }
  ]
}
```

### 3.3 AggregateState Schema

```json
{
  "ts": "ISO8601",
  "question_id": "Q_<id>",
  "probs": {
    "O_a": 0.2,
    "O_b": 0.5,
    "O_c": 0.3
  },
  "aggregate_hash": "sha256:<8>",
  "inputs_window": {
    "from": "ISO8601",
    "to": "ISO8601"
  },
  "solver": {
    "name": "proj_simplex_v1",
    "status": "OK",
    "residual": 1e-6
  }
}
```

Decision, Permit, Episode schemas remain as existing spec anchors.

---

## 4. Constraint Engine

The core MHD move: simple and deterministic.

### 4.1 SUM_TO_ONE Groups

1. Take proposed vector p
2. Apply weights per source (confidence × dataset trust)
3. Compute unconstrained aggregate
4. Project onto simplex (sum to 1, each in [0,1]) using deterministic algorithm

### 4.2 IMPLIES Constraints

If A implies B, enforce: `p(A) ≤ p(B)`

If violated after aggregation:

* Minimally adjust using deterministic repair
* Raise p(B) to p(A), then renormalize the affected set

### 4.3 Key Invariants

* **Deterministic**: same inputs produce bit-identical output
* **Monotonic logging**: no edits, only new entries
* **Fail closed on ambiguity**: if constraints can't be satisfied within tolerance, aggregate is marked INVALID and downstream consumers ignore it

---

## 5. Authority Rules

### 5.1 Decision Types for Prediction

Parallel action set:

* `WRITE_BELIEF`
* `UPDATE_CONSTRAINTS` (rare, admin only)
* `PROMOTE_DATASET_STATE` (cycle boundary only)

### 5.2 Decision Specification

A Decision specifies:

* Which dataset_ids or actors may write
* Which question_ids
* Max delta per update (prevents "pump")
* TTL and max uses

### 5.3 Permit Binding

A Permit binds a single BeliefEvent:

* Single use
* Short TTL
* Frozen snapshots

This makes belief updates **auditable and governable**, not vibes.

---

## 6. Dataset Admission Gate Integration

Treat every prediction source as a dataset:

* Polymarket feed
* Crowd input form
* Model output
* News classifier

**Default: REJECTED.**

Promote only when it proves survivability.

### Rollback Triggers

* If the feed restates history → revoke
* If latency explodes → downgrade
* If it corrupts regime authority → revoke

---

## 7. Episode Concept for Predictions

A Prediction Episode binds:

```
Decision → Permit → BeliefEvent(s) → Aggregate changes → Outcome resolution
```

After event resolves:

* Score sources via proper scoring rule (log score or Brier)
* Update dataset trust weights (but only as ADVISORY tier unless promoted)

This creates investor-grade audit artifacts.

---

## 8. Phased Rollout

### Phase P0 — Observe Only

* Ingest sources
* Log belief events
* Do not influence anything

### Phase P1 — Research Only

* Compute aggregates
* Show in dashboard as "advisory belief layer"
* No execution impact

### Phase P2 — Production Eligible (Cycle Boundary)

* Allow belief layer to influence *only* non-authoritative modules:
  * Router health hints
  * Symbol prioritization
  * Alert ranking
* Never override Sentinel-X regimes until explicitly amended

This mirrors Dataset Admission Gate states.

---

## 9. MHD Compliance Summary

### Messy Reduced

* One input primitive: BeliefEvent
* One logic layer: QuestionGraph constraints
* One output: AggregateState

### Heavy Contained

* Every write is permitted, logged, replayable
* Constraints enforced centrally
* Episodes create investor audit artifacts

### Dependent Removed

* No reliance on arbitrage or participant sophistication
* No "shadow liquidity" requirement
* External feeds are gated and revocable

---

## 10. Deliverable List

### Files to Create

1. `prediction/question_graph.json`
2. `prediction/constraints.py` (simplex projection + implies repair)
3. `prediction/belief_ingest.py` (normalize inputs)
4. `prediction/dle_prediction_gate.py` (Decision/Permit issuance for belief writes)
5. `logs/prediction/belief_events.jsonl`
6. `logs/prediction/aggregate_state.jsonl`
7. `logs/prediction/prediction_episodes.jsonl`
8. `config/dataset_admission.json` entries for each source

### Implementation Strategy

Start with **shadow permits** for belief events first, then enforce later — exactly like Phase A of main DLE rollout.

---

## 11. Forbidden Extensions

This layer must never:

* Trigger trades directly
* Override Sentinel-X vetoes
* Create incentives or payouts
* Become required infrastructure
* Allow unvetted sources to write

These are structural prohibitions, not feature deferrals.
