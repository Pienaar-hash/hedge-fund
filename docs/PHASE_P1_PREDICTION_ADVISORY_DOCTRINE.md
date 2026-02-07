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
### Import boundary rule

Execution code may import from `prediction/` **only** via
`execution/telegram_utils._maybe_rank_alerts()`.  No other execution module
may import from `prediction/` directly.  This is the single crossing point
and it is wrapped in `try/except` (fail-open) so prediction can never
affect executor liveness.
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

## Promotion to P2_PRODUCTION — Gate Checklist

**Status:** NOT PROMOTED — checklist defined, no decision made.

P2 would allow prediction outputs to influence execution-adjacent decisions
(e.g., sizing hints, entry confidence, universe filtering). This is a
fundamentally different authority level from P1.

### Hard prerequisites (all must be TRUE)

| # | Gate | Evidence | Status |
|---|------|----------|--------|
| 1 | **P1 trial PASS** | `p1_audit_bundle.sh check` → `P1_RESULT: PASS` | ✅ Passed (29h window, 2026-02-06/07) |
| 2 | **Closed-cycle postmortem** | Written analysis of P1 advisory value — what ranking changed, what it didn't, false positive rate | ☐ Not started |
| 3 | **Dataset promotion** | All prediction datasets moved from OBSERVE_ONLY → PRODUCTION_ELIGIBLE in `config/dataset_admission.json` with justification per dataset | ☐ Not started |
| 4 | **Rollback triggers enforced** | `prediction/rollback_triggers.py` triggers promoted to `enforced=True` — constraint violations cause automatic P0 revert | ☐ Not started |
| 5 | **Firewall scope expansion** | New `_PRODUCTION_CONSUMERS` set defined — which modules may consume in P2, with what authority | ☐ Not started |
| 6 | **Sizing guard** | If prediction influences sizing: max effect capped (e.g., ±10% of base size), hard limit, logged | ☐ Not designed |
| 7 | **A/B invariance test** | Run executor for 24h with P2 enabled but decisions logged-not-applied (shadow mode), prove no crashes/hangs/regressions | ☐ Not started |
| 8 | **Doctrine approval** | Explicit sign-off that prediction may influence pre-defined surfaces | ☐ Not started |

### What P2 would allow (if promoted)

| Surface | Authority | Constraint |
|---------|-----------|------------|
| Alert ranking | Full (not just advisory) | Same as P1 — set equality preserved |
| Dashboard | Full display | Same as P1 — read-only |
| Sizing hints | Advisory → executor_live | Capped at ±N% of base size, logged |
| Universe filtering | Soft signal to screener | Cannot override doctrine/risk vetoes |
| Entry confidence | Additive factor to Hydra head score | Cannot be sole reason for entry |

### What P2 would still forbid

* Overriding Doctrine verdicts
* Replacing Sentinel-X regimes
* Bypassing risk limits
* Becoming a liveness dependency (executor must run without prediction)
* Direct position management (open/close/modify)

### Rollback from P2 → P1

```bash
export PREDICTION_PHASE=P1_ADVISORY
sudo supervisorctl restart hedge:
```

Instant. No code changes. All P2-only consumers revert to advisory-only.
Rollback triggers with `enforced=True` would auto-trigger this on constraint violation.

### Decision process

P2 promotion is **not** a configuration change — it requires:

1. All gates above marked ✅
2. This checklist reviewed by operator
3. Explicit `PREDICTION_PHASE=P2_PRODUCTION` set in supervisor config
4. Fresh `p2_audit_bundle.sh start` (to be built when P2 is approved)
