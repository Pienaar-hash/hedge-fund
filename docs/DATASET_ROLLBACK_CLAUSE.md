# DATASET ROLLBACK CLAUSE

**GPT-HEDGE v7.x — Doctrine Addendum**  
**Established:** January 24, 2026  
**Version:** 1.0  
**Prerequisite:** DATASET_ADMISSION_GATE.md v1.0

---

## I. PURPOSE

This clause defines **when and how a dataset may be rolled back**.

Not when it underperforms.  
Not when a better one exists.  
Only when it **threatens system integrity**.

This exists because:

* Simons explicitly allows "changing it back"
* Most rollback decisions are made in panic, not principle
* Rollback without rules becomes arbitrary intervention

---

## II. DEFINITIONS

### Rollback

Any action that:

* Downgrades a dataset's state
* Removes a dataset from influence
* Substitutes a dataset with a fallback

### Rollback Classes

| Class | Definition | Example |
|-------|------------|---------|
| **DOWNGRADE** | State demotion (e.g., PRODUCTION → OBSERVE_ONLY) | Expectancy scoring disabled |
| **REVOKE** | Complete removal from system | Polymarket feed deleted |
| **SUBSTITUTE** | Replace with fallback source | Binance → backup exchange |

### Dataset Influence Tiers

Based on [config/dataset_admission.json](../config/dataset_admission.json):

| Tier | Datasets | Rollback Constraint |
|------|----------|---------------------|
| **EXISTENTIAL** | klines, positions, balance, fills | Cannot revoke, only substitute |
| **AUTHORITATIVE** | sentinel_x_features | Cannot revoke mid-cycle |
| **ADVISORY** | symbol_scores, expectancy, router_health | May downgrade any time |
| **OBSERVATIONAL** | regime_pressure, factor_diagnostics, coingecko, polymarket | May revoke any time |

---

## III. ROLLBACK PRINCIPLE (NON-NEGOTIABLE)

> **Rollback protects the system from the dataset, not from poor performance.**

A dataset is rolled back because it:

* Violates admission criteria it previously satisfied
* Introduces non-determinism into decisions
* Corrupts regime authority
* Cannot be audited

A dataset is **NOT** rolled back because:

* PnL declined after admission
* A "better" dataset exists
* Operator intuition suggests problems

---

## IV. ROLLBACK TRIGGERS

### 🔴 Mandatory Rollback (Automatic)

These triggers **require immediate rollback** without operator confirmation:

| Trigger | Detection | Action |
|---------|-----------|--------|
| **Temporal violation** | Retroactive data mutation detected | REVOKE |
| **Replay divergence** | Same inputs produce different outputs | DOWNGRADE to OBSERVE_ONLY |
| **Regime corruption** | Dataset directly overrides Sentinel-X | REVOKE |
| **Latency breach** | p99 latency exceeds 3x characterized bound | DOWNGRADE |
| **Silent gap** | Unannounced data absence > threshold | DOWNGRADE |

Mandatory rollback is:
* Automatic
* Logged before execution
* Not reversible within same cycle

---

### 🟡 Discretionary Rollback (Operator-Confirmed)

These triggers **allow but do not require** rollback:

| Trigger | Consideration | Default |
|---------|---------------|---------|
| **Quality degradation** | Increased noise, reduced signal | Monitor first |
| **Latency drift** | p50/p90 elevated but p99 within bounds | Log, do not act |
| **Partial outage** | Some symbols affected, others not | Isolate, do not revoke |
| **Upstream deprecation** | Source announces future EOL | Plan substitution |

Discretionary rollback:
* Requires explicit operator confirmation
* Must include written justification
* May be deferred to cycle boundary

---

### 🟢 Forbidden Rollback (Never Mid-Cycle)

These datasets **cannot be rolled back during an active cycle**:

| Dataset | Reason |
|---------|--------|
| `binance_futures_klines` | Regime authority depends on it |
| `sentinel_x_features` | Regime is defined by it |
| `binance_futures_positions` | Position truth is existential |
| `binance_futures_balance` | NAV truth is existential |

For EXISTENTIAL and AUTHORITATIVE tiers:

* Rollback is only permitted at **cycle boundary**
* Substitution requires **pre-tested fallback**
* Operator must confirm system can survive without it

---

## V. ROLLBACK SCOPE

### State Transitions (Downgrade Paths)

```
PRODUCTION_ELIGIBLE → RESEARCH_ONLY → OBSERVE_ONLY → REJECTED
                   ↘               ↘             ↘
                    ────────────────→─────────────→ REJECTED (emergency)
```

* Normal downgrade: One state at a time
* Emergency downgrade: Direct to REJECTED (mandatory triggers only)

### Timing Rules

| Tier | Immediate Allowed | Cycle-Boundary Required |
|------|-------------------|-------------------------|
| EXISTENTIAL | ❌ Never | ✅ With substitution |
| AUTHORITATIVE | ❌ Never | ✅ Required |
| ADVISORY | ✅ Yes | Preferred |
| OBSERVATIONAL | ✅ Yes | Not required |

### Historical Data Handling

On rollback:

* **Logs are never rewritten** — Historical decisions remain attributed to original dataset
* **Replay uses rollback timestamp** — Future replays exclude dataset after rollback
* **Attribution is preserved** — PnL during dataset influence remains on record

> We do not pretend the past didn't happen.

---

## VI. ROLLBACK AUTHORITY

### Automatic Rollback

Executed by system without confirmation when:

* Mandatory trigger detected
* Dataset is ADVISORY or OBSERVATIONAL tier
* Fallback behavior is pre-defined

Automatic rollback:
* Logs `rollback_event` before execution
* Sets `revocation.automatic = true`
* Sends alert (Telegram/log)

### Operator-Confirmed Rollback

Required when:

* Trigger is discretionary
* Dataset is AUTHORITATIVE tier
* No pre-defined fallback exists

Operator must provide:
* Written justification (logged)
* Confirmation of fallback readiness
* Acknowledgment of cycle-boundary constraint (if applicable)

### Rollback of Rollback (Restoration)

A rolled-back dataset may be **restored** only if:

* Original admission criteria are re-verified
* Restoration occurs at cycle boundary
* Full admission process is re-run (not grandfathered)

Restoration is treated as **new admission**, not undo.

---

## VII. ROLLBACK LOGGING SCHEMA

All rollback events logged to `logs/execution/dataset_rollback.jsonl`:

```json
{
  "ts": "ISO8601",
  "dataset_id": "string",
  "action": "DOWNGRADE | REVOKE | SUBSTITUTE",
  "from_state": "PRODUCTION_ELIGIBLE",
  "to_state": "OBSERVE_ONLY",
  "trigger": {
    "type": "mandatory | discretionary",
    "reason": "temporal_violation | replay_divergence | ...",
    "details": "string"
  },
  "authority": {
    "automatic": true,
    "operator": null,
    "justification": null
  },
  "fallback": {
    "activated": false,
    "fallback_dataset": null
  },
  "cycle_id": "CYCLE_003",
  "reversible": false
}
```

---

## VIII. FALLBACK REGISTRY

For EXISTENTIAL datasets, substitution requires pre-defined fallbacks:

| Dataset | Primary | Fallback | Fallback State |
|---------|---------|----------|----------------|
| `binance_futures_klines` | Binance FAPI | None (halt) | — |
| `binance_futures_positions` | Binance FAPI | None (halt) | — |
| `binance_futures_balance` | Binance FAPI | None (halt) | — |
| `binance_futures_fills` | Binance FAPI | None (halt) | — |

**Note:** EXISTENTIAL datasets have no fallback. Loss of primary = system halt.

For ADVISORY datasets:

| Dataset | Fallback Behavior |
|---------|-------------------|
| `symbol_scores_v6` | Default to 0.5 (neutral) |
| `expectancy_v6` | Default to 0.0 (no bias) |
| `router_health` | Default to conservative policy |

---

## IX. RELATION TO SIMONS' PRACTICE

Simons says:

> "If something doesn't work… we know how to change it back."

What he implies:

* Rollback is a **planned capability**, not emergency improvisation
* "Changing it back" requires knowing what "back" means
* Rollback authority is delegated to systems, not individuals

This clause ensures:

* Rollback is pre-defined, not ad-hoc
* Rollback is logged, not silent
* Rollback is constitutional, not emotional

---

## X. FINAL CONSTRAINTS

> **A rollback that cannot be audited is indistinguishable from manipulation.**

Every rollback must answer:

1. What triggered it?
2. Who (or what) authorized it?
3. What was the fallback?
4. Can we replay decisions before and after?

If any answer is missing, the rollback is invalid.

---

## XI. AMENDMENT HISTORY

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-01-24 | Initial establishment |

---

*Reference: DATASET_ADMISSION_GATE.md v1.0*  
*Reference: config/dataset_admission.json*
