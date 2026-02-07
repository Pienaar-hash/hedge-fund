# Exchange Unavailable Doctrine — Fail-Silent Rules

**Effective:** 2026-02-07
**Status:** ACTIVE
**Scope:** All execution paths that depend on exchange connectivity

---

## Purpose

Define explicit behavior when the exchange (Binance USD-M Futures) is
unreachable, degraded, or returning errors. Prevents panic-driven complexity
and ensures the system degrades to **observe-only** rather than
**partial trading**.

This is an MHD constraint: **the system must not become Heavy or Dependent
because of exchange instability.**

---

## Fail-Silent Rules (Non-Negotiable)

### 1. No degraded trading

If the exchange is unreachable, the system must **stop attempting orders**
rather than retry indefinitely or trade with stale data.

* No "partial trading" mode
* No fallback to a different exchange
* No manual override to force trades through

### 2. No unbounded retries

Per-request retries are already bounded (5 attempts, exponential backoff).
But the system must also bound **aggregate** retry behavior:

* If N consecutive exchange calls fail within a window, the executor should
  enter a cooldown state rather than burning API rate limits
* Supervisor restart is acceptable — but the restarted process should also
  detect persistent failure and back off

### 3. Observe-only on failure

When exchange calls are failing:

* **Continue logging** — regime detection, screener signals, and doctrine
  events should still be computed and logged from cached/stale data
* **Continue state publishing** — dashboard and Firestore sync should
  reflect the error state, not go silent
* **Stop order submission** — no new entries, no exits (except manual)
* **Do not infer "no positions"** — if `get_positions()` fails, the
  previous position snapshot should be retained, not replaced with `[]`

### 4. Alert on transition, not on every failure

* Alert once when exchange becomes unreachable
* Alert once when exchange recovers
* Do not spam alerts for every failed retry

---

## Current Implementation Status

| Behavior | Status | Gap |
|----------|--------|-----|
| Per-request retry (5×, exp backoff) | ✅ Implemented | — |
| Maker→taker fallback in router | ✅ Implemented | — |
| TWAP slice failure isolation | ✅ Implemented | — |
| `_NullUMClient` stub on init failure | ✅ Implemented | — |
| Per-symbol cooldown after send failure | ✅ Implemented | — |
| Supervisor auto-restart on crash | ✅ Configured | — |
| Global circuit breaker (N failures → pause) | ☐ Not implemented | Documented here as future work |
| Top-level try/except in main loop | ☐ Not implemented | Crash → supervisor restart (acceptable) |
| Position snapshot retention on API failure | ☐ Not implemented | `get_positions()` returns `[]` on error |
| Exchange health state tracking | ☐ Not implemented | No "exchange_down" flag |

---

## Acceptable Failure Cascade

```
Exchange unreachable
  → _req() retries 5× with backoff (0.25s → 3.0s)
  → Exception propagates to caller
  → Caller-specific handling:
      get_positions()  → returns [] (KNOWN GAP — should retain prior)
      generate_intents → returns [] (screener catches, safe)
      _send_order      → logged, symbol cooldown applied
      Uncaught         → process crash → supervisor restart (30s+)
```

This cascade is **acceptable but not ideal**. The known gap
(`get_positions()` returning `[]`) should be addressed when the global
circuit breaker is implemented.

---

## What This Doctrine Prevents

1. **Adding a second exchange "for redundancy"** without full MHD audit
2. **Adding automatic failover** that increases system complexity
3. **Adding "retry harder" logic** that makes the system Heavy
4. **Ignoring exchange downtime** in postmortems

---

## Decision Log

Any change to exchange failure behavior must:

1. Pass MHD (not increase Messy, Heavy, or Dependent scores)
2. Be tested with `DRY_RUN=1` and simulated failures
3. Be documented here before deployment
