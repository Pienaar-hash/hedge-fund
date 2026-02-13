# Phase P1 — Prediction Advisory Doctrine

**Effective:** 2026-02-06
**Status:** ACTIVE (requires `PREDICTION_PHASE=P1_ADVISORY`)
**Authority level:** Advisory only — zero execution influence

---

## Purpose

Promote the prediction layer from **P0_OBSERVE** to **P1_ADVISORY** without
granting execution authority.

P1 allows a single class of consumers — **advisory-only readers** — to consume
prediction aggregates for non-causal downstream uses (alert ranking, dashboard
overlays, research outputs). No execution path may depend on prediction outputs
during P1.

---

## Non-negotiables

Prediction layer in P1 **must not**:

1. Override or modify Sentinel-X regimes
2. Influence Doctrine verdicts, vetoes, permits, sizing, router decisions, or exits
3. Write into any execution-path logs other than prediction-scoped logs
4. Create new runtime dependencies for executor liveness
5. Cause any difference in execution events when P1 is enabled vs disabled

---

## Allowed influence surfaces

Prediction outputs may be consumed **only** by:

| Consumer | Use | Authority |
|----------|-----|-----------|
| `alert_ranking` | Reorder alerts by prediction uncertainty/shift | Advisory only — same alerts in, same alerts out |
| `dashboard` | Show prediction aggregates alongside runtime health | Read-only display |
| `research` | Offline analysis and backtesting | No execution coupling |

---

## Forbidden consumers (firewall-enforced)

Hard-blocked by `prediction/firewall.py` — no override path:

| Consumer | Reason |
|----------|--------|
| `doctrine_kernel` | Doctrine supremacy — predictions cannot gate trades |
| `sentinel_x` | Regime detection must remain signal-only |
| `executor_live` (decision path) | Execution loop must be prediction-independent |
| Any sizing or router module | Position sizing is risk-governed, not belief-governed |

---

## Dataset state requirements

During P1:

* All prediction sources remain **OBSERVE_ONLY** or **REJECTED**
* No dataset may be promoted to `RESEARCH_ONLY` or `PRODUCTION_ELIGIBLE`
* Dataset state is enforced at consumption time by the firewall

---

## Evidence requirements for P1 acceptance

P1 is accepted only if **all** of the following hold:

1. **Determinism** — aggregate hashes are stable under replay
2. **Firewall enforcement** — forbidden consumers are denied (logged)
3. **Execution invariance** — no execution events differ when P1 enabled vs disabled
4. **Rollback logging** — triggers continue logging with `enforced=False`
5. **Alert set equality** — alert ranker produces the same set of alerts (only order changes)

---

## Phase flag

```bash
# Enable P1 advisory mode
export PREDICTION_PHASE=P1_ADVISORY

# Default remains P0 unless explicitly set
# PREDICTION_PHASE=P0_OBSERVE (default)
```

The firewall reads this flag and enforces `advisory_only=True` for all consumers
in P1. No consumer can treat prediction data as authoritative until P2.

---

## Rollback to P0

If any evidence requirement fails during P1:

```bash
export PREDICTION_PHASE=P0_OBSERVE
```

This immediately reverts to observe-only. No code changes required — the firewall
enforces the phase at every `request_advisory()` call.

---

## Promotion to P2 (future)

P2 promotion requires a separate doctrine note and:

* Closed-cycle postmortem showing P1 advisory value
* All prediction datasets promoted to `PRODUCTION_ELIGIBLE`
* Rollback triggers promoted to `enforced=True`
* Explicit Doctrine approval

P2 is **not planned** — this note covers P1 only.
