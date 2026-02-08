# GPT-HEDGE — Operating State v7.9

**Date:** 8 February 2026
**Version:** v7.9
**Status:** NOMINAL — all systems operational

---

## Portfolio Snapshot

| Metric | Value |
|--------|-------|
| **NAV** | $9,873 |
| **Open Positions** | 3 (SOL, ETH, BTC — all LONG) |
| **Gross Exposure** | ~$5,700 |
| **Cash (Stablecoins)** | ~$4,200 (42% NAV) |
| **24h PnL (NAV Δ)** | +$460 |
| **Executor Uptime** | 43+ hours continuous |

All three system processes (executor, dashboard, state sync) are
running continuously with no manual intervention required.

---

## What Changed Since Last Update

### 1. Prediction Layer (P1 Advisory)

A forward-looking conviction signal layer was built and deployed
in **advisory-only mode**. It observes the existing trading
pipeline and surfaces ranked alerts — but cannot influence
execution decisions.

- 12 components, 88 dedicated tests
- Hard firewall enforced: zero execution influence
- 29-hour production trial completed with no incidents
- Automatic rollback triggers if error thresholds are breached

The layer is currently accumulating data to evaluate signal
quality before any promotion to active use.

### 2. Dashboard Overhaul

The monitoring dashboard was audited end-to-end and corrected:

- **24h PnL** now shows true portfolio change (NAV delta),
  not just closed-trade profit
- **Risk metrics** (Sharpe, drawdown, veto counts) populated
  from authoritative sources
- **Equity curve** sparkline visible and accurate
- **Labeling** tightened to eliminate ambiguity between
  closed-trade PnL and portfolio PnL

### 3. Score Decomposition Logging

Every symbol score now logs its component weights and weighted
contributions, enabling post-hoc attribution of why specific
symbols were selected or rejected.

### 4. Exchange Unavailability Doctrine

Documented the system's behavior when Binance is temporarily
unreachable: fail-silent, preserve positions, resume on
reconnection. No manual intervention required.

---

## System Architecture (Summary)

The system operates a **regime-governed execution model**:

1. Market regime is detected continuously (trend, mean-revert,
   breakout, choppy, crisis)
2. Six strategy heads generate signals filtered by regime
3. A doctrine kernel gates all entries and exits — no trade
   occurs without regime permission
4. Risk limits provide secondary veto (position caps, drawdown
   limits, correlation caps)
5. Orders execute via maker-first routing with TWAP support

Key design principle: **the system refuses to trade when
conditions are unclear**. Refusal is a first-class outcome,
not a failure.

---

## Risk Controls

| Control | Status |
|---------|--------|
| Doctrine kernel (entry/exit gate) | Active, cannot be bypassed |
| Per-symbol position caps | Enforced |
| Portfolio drawdown limit | Enforced |
| Correlation group caps | Enforced |
| NAV staleness veto (>90s) | Active |
| Stale data auto-veto | Active |

Risk vetoes are logged and auditable. The system vetoed
approximately 1,260 potential actions in the last 24 hours —
this is normal and reflects conservative regime filtering.

---

## Test & Code Quality

| Metric | Value |
|--------|-------|
| Automated tests | 2,632 passing |
| Test failures | 0 |
| Linting | Clean (ruff) |
| Type checking | Gradual (mypy) |

All changes go through the full test suite before deployment.
No credentials are stored in the repository.

---

## Current Posture

The system is in **steady-state operation**. No new features or
strategy changes are planned for the immediate term. The
prediction advisory layer will continue to accumulate data
silently.

Next potential actions (none scheduled):

- Evaluate prediction signal quality after 7-day soak
- Review strategy performance at next monthly checkpoint
- Consider P2 promotion of prediction layer if evidence supports it

---

*This document reflects system state as of 2026-02-08. NAV and
positions are point-in-time and will change with market conditions.*
