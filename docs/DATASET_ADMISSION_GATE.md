# DATASET ADMISSION GATE

**GPT-HEDGE v7.x — Doctrine Addendum**  
**Established:** January 24, 2026  
**Version:** 1.0

---

## I. PURPOSE

This gate defines **when a new dataset is allowed to influence the system**.

Not when it is interesting.  
Not when it backtests well.  
Only when it is **operationally survivable**.

This exists because:

* Simons explicitly incorporates new data continuously
* Most trading systems die from **data contamination**, not bad signals

---

## II. DEFINITIONS

### Dataset

Any external or derived data stream that:

* Was not present at Doctrine inception
* Can influence signals, regimes, sizing, or exits
* Includes price variants, order flow, alt data, predictions, indices, or synthetic features

### Dataset States

A dataset may exist in **only one** of the following states:

| State                 | Meaning                                        |
| --------------------- | ---------------------------------------------- |
| `REJECTED`            | Never enters system                            |
| `OBSERVE_ONLY`        | Logged, never influences decisions             |
| `RESEARCH_ONLY`       | May influence research signals only            |
| `PRODUCTION_ELIGIBLE` | Allowed to influence live system (cycle-bound) |

Default state is **REJECTED**.

---

## III. ADMISSION PRINCIPLE (NON-NEGOTIABLE)

> **A dataset must prove it will not destabilize the system *before* it is allowed to help it.**

Predictive value is **secondary** to:

* Consistency
* Latency stability
* Semantic determinism
* Replay integrity

---

## IV. ADMISSION CRITERIA (ALL REQUIRED)

A dataset may advance one state **only if all criteria below are satisfied**.

---

### 1. Temporal Integrity

The dataset must demonstrate:

* Monotonic timestamps
* No retroactive mutation
* No silent restatements
* Deterministic alignment with candle boundaries

**Failure mode:**  
If the dataset changes the past, it is banned permanently.

---

### 2. Latency Characterization

The dataset must have:

* Measured delivery latency distribution
* Stable percentile bounds (p50 / p90 / p99)
* Known update cadence

Unknown latency = unknown causality.

---

### 3. Missingness & Gaps

The dataset must specify:

* Acceptable missingness %
* Behavior during gaps (freeze, zero, last-value, null)
* Explicit "no update" signals

Silent gaps are forbidden.

---

### 4. Replay Determinism

Given:

* Identical raw inputs
* Identical system version
* Identical wall-clock offsets

The dataset must replay **bit-identical outputs**.

If we cannot replay it, we cannot audit it.

---

### 5. Regime Neutrality (Initial)

At admission:

* The dataset must **not** directly override Sentinel-X
* It may *observe* regimes, not define them
* Regime influence is only allowed after full cycle promotion

This prevents regime corruption by novelty.

---

## V. STATE TRANSITIONS

### REJECTED → OBSERVE_ONLY

Allowed when:

* Criteria 1–4 are satisfied
* Dataset is logged but ignored

No decisions may depend on it.

---

### OBSERVE_ONLY → RESEARCH_ONLY

Allowed when:

* Dataset survives ≥ 1 full regime cycle
* No operational anomalies observed
* No increase in churn, veto noise, or instability

Research signals may consume it.

---

### RESEARCH_ONLY → PRODUCTION_ELIGIBLE

Allowed only:

* At **cycle boundary**
* With explicit doctrine amendment
* With rollback plan pre-defined

No mid-cycle promotions.

---

## VI. FAILURE & REVOCATION

A dataset is **immediately revoked** if it causes:

* Regime instability
* Increased near-flip counts
* Non-deterministic behavior
* Post-hoc corrections

Revocation is silent and automatic.

No heroics.

---

## VII. RELATION TO SIMONS' PRACTICE

Simons says:

> "As new datasets become available, they are incorporated…"

What he does **not** say:

* That all datasets are equal
* That data is admitted without infrastructure
* That prediction outranks survival

This gate reconciles:

* **Continuous discovery** (research)
* **Slow permission** (doctrine)
* **Hard rollback** (survivability)

---

## VIII. FINAL CONSTRAINT

> **A dataset that improves PnL but degrades auditability is worse than useless.**

Edge that cannot be replayed is illusion.

---

## IX. LOGGING SCHEMA

All dataset state changes are logged to `logs/state/dataset_admission.json`:

```json
{
  "datasets": {
    "<dataset_id>": {
      "name": "string",
      "source": "string",
      "state": "REJECTED | OBSERVE_ONLY | RESEARCH_ONLY | PRODUCTION_ELIGIBLE",
      "state_history": [
        {
          "from": "REJECTED",
          "to": "OBSERVE_ONLY",
          "ts": "ISO8601",
          "cycle_id": "CYCLE_003",
          "criteria_met": ["temporal", "latency", "missingness", "replay"],
          "notes": "string"
        }
      ],
      "criteria": {
        "temporal_integrity": {
          "verified": true,
          "verified_at": "ISO8601",
          "notes": "Monotonic, no restatements observed over 14d"
        },
        "latency": {
          "p50_ms": 120,
          "p90_ms": 340,
          "p99_ms": 890,
          "cadence": "1m",
          "measured_at": "ISO8601"
        },
        "missingness": {
          "acceptable_pct": 0.5,
          "gap_behavior": "last_value",
          "explicit_null": true
        },
        "replay_determinism": {
          "verified": true,
          "test_window": "2026-01-01 to 2026-01-14",
          "hash": "sha256:..."
        },
        "regime_neutrality": {
          "influences_regime": false,
          "observe_only": true
        }
      },
      "revocation": {
        "revoked": false,
        "revoked_at": null,
        "reason": null,
        "automatic": null
      },
      "rollback_plan": {
        "defined": true,
        "steps": ["Disable flag X", "Revert to fallback Y"],
        "tested_at": "ISO8601"
      }
    }
  },
  "updated_at": "ISO8601"
}
```

---

## X. CURRENT DATASET REGISTRY

| Dataset | State | Admitted | Notes |
|---------|-------|----------|-------|
| Binance OHLCV | `PRODUCTION_ELIGIBLE` | Pre-doctrine | Foundational |
| Binance Order Book | `PRODUCTION_ELIGIBLE` | Pre-doctrine | Foundational |
| Binance Account/Positions | `PRODUCTION_ELIGIBLE` | Pre-doctrine | Foundational |

*Datasets added post-doctrine must go through formal admission.*

---

*Reference: DOCTRINE_FALSIFICATION_CRITERIA.md*
