# 🏷️ PATCHSET_V7.6_TAGPACK_RELEASE_NOTES.prompt.md

````markdown
# PATCHSET V7.6 — Tag Pack, Release Notes & Runtime Activation Docs
# Context:
# - v7.6 code + runtime + state contract + dashboard + preflight are complete (P1–P8).
# - Preflight scripts, VERSION metadata, and CI gating exist.
# - Goal: produce the *human layer* for v7.6: release notes, change log, activation runbook,
#   and risk / stability narrative.

## High-Level Deliverables

1. **v7.6 Release Notes (docs/v7.6_Release_Notes.md)**
2. **v7.6 Change Log (docs/v7.6_Change_Log.md)** — developer-facing
3. **v7.6 Runtime Activation Runbook (docs/v7.6_Runtime_Activation.md)**
4. **v7.6 Investor Stability & Risk Summary (docs/v7.6_Investor_Stability.md)** — internal / investor PDF input
5. **Tagging & Rollback Snippet** in the pre-tag audit doc (append to docs/v7.6_Pre_Tag_Audit.md)

No engine semantics changes; docs + scripts only.

---

## 1) v7.6 Release Notes — docs/v7.6_Release_Notes.md

Audience: **operators + high-level technical stakeholders**.

Structure:

```markdown
# GPT-Hedge v7.6 — Institutional Execution & Factor Governance

## Overview
Short paragraph summarising:
- Router microstructure intelligence
- Factor diagnostics & hybrid scoring governance
- Exit/ledger authority
- NAV/risk coherence
- Unified dashboard regimes
- State contract hardening + preflight

## Headline Features
- Router Microstructure & Router Health v2
- Hybrid Scoring with Vol Regime & Router Quality
- Factor Diagnostics (Orthogonalization, Covariance, IR Weights)
- Exit Pipeline & Ledger Authority
- NAV & Risk Snapshot Coherence + Anomaly Guards
- Unified Dashboard Regime Strip (Vol, DD, Router, Risk)
- State Contract & Telemetry Hardening
- Preflight & Runtime Sanity Checks (v7.6)

## Operational Impact
- What operators see differently in the dashboard
- How incident response is improved (router, exits, NAV anomalies)
- What preflight they must run before enabling v7.6

## Upgrade Notes
- Minimum steps to move from v7.5 → v7.6:
  - Pull tag / deploy container
  - Run `make pretag-v7.6`
  - Run `python scripts/runtime_sanity_check_v7_6.py`
  - Verify dashboard regimes, router health, exit coverage, factor diagnostics.

## Known Limitations / Future Work
- Short bullet list (e.g., no multi-exchange router, no full latency arb, etc.)
````

Populate each section with concise bullets built from the P1–P8 patch descriptions.

---

## 2) v7.6 Change Log — docs/v7.6_Change_Log.md

Audience: **developers / contributors**.

Format: structured, patchset-oriented:

```markdown
# v7.6 Change Log

## Runtime & Execution
- P1 RouterStats + router_health v2
- P4 Exit pipeline + ledger reconciliation
- P5 NAV & risk_snapshot coherence
- P8 VERSION + engine_metadata + preflight

## Factors & Scoring
- P2 hybrid scoring + vol_regime factor
- P3 factor_diagnostics (orthogonalization, covariance, IR weights)

## State & Diagnostics
- P4 exit diagnostics mismatch_breakdown
- P5 anomaly flags (nav_jump, var/cvar)
- P7 state contract hardening, timestamp coherence

## Dashboard
- P6 unified state loader + regime badges
- P8 system health panel + engine version

## Tooling & CI
- Preflight scripts, Make target, CI workflow
```

Each bullet should be a one-liner summarising the implemented behaviour (you can reuse the short patch summaries you’ve been writing).

---

## 3) Runtime Activation Runbook — docs/v7.6_Runtime_Activation.md

Audience: **ops / SRE / you on a Saturday night**.

Sections:

````markdown
# v7.6 Runtime Activation Runbook

## 1. Preconditions
- VERSION == v7.6
- CI green on main
- Access to runtime box, logs/state directory, dashboard URL.

## 2. Preflight Commands
```bash
make pretag-v7.6
python scripts/runtime_sanity_check_v7_6.py
````

* What success looks like:

  * All tests pass
  * Sanity check shows nav, dd_state, risk_mode, router quality, exit coverage, factor weights.

## 3. Deploying v7.6

* Pull updated code / image
* Restart supervisor services:

  * executor
  * dashboard
  * sync_state
* Verify logs:

  * executor boot logs show engine version v7.6
  * no repeated errors in state_publish

## 4. Post-Activation Checks

* Check logs/state:

  * nav.json, nav_state.json, positions_state.json, positions_ledger.json
  * kpis_v7.json, diagnostics.json, risk_snapshot.json, router_health.json
  * symbol_scores_v6.json, factor_diagnostics.json, engine_metadata.json
* Dashboard:

  * Regime strip visible (Vol, DD, Router, Risk)
  * Router health metrics populated
  * Exit coverage > X% (define threshold)
  * Factor diagnostics panel shows weights/IR

## 5. Rollback Plan

* Steps to revert to previous tag (v7.5 or prior):

  * Checkout/tag previous version
  * Redeploy services
  * Run runtime_sanity_check again
* Note: state surfaces are forward-compatible; on rollback, check diagnostics for any schema gaps.

## 6. Incident Hooks

* If NAV anomaly triggers: pointer to playbook section
* If router idle or degraded: pointer to router playbook
* If exit coverage drops: pointer to exit/ledger playbook

````

---

## 4) Investor Stability & Risk Summary — docs/v7.6_Investor_Stability.md

Audience: **internal investor deck / PDF**. Tone: factual, not marketing.

Structure:

```markdown
# GPT-Hedge v7.6 — Stability & Risk Controls Overview

## 1. Execution & Microstructure Controls
- RouterStats: slippage, latency, TWAP usage.
- Router health scoring and quality-based regimes.
- Impact: reduced execution noise, visibility into routing quality.

## 2. Risk & Drawdown Governance
- NAV anomaly guards (max NAV jump, anomaly flags).
- DD state machine (NORMAL/DRAWDOWN/RECOVERY).
- VAR/CVAR computation and breach flags.
- Impact: earlier visibility of structural risk, not just PnL.

## 3. Exit & Ledger Integrity
- Ledger–registry reconciliation with mismatch breakdown.
- Exit coverage metrics and underwater-without-TP/SL detection.
- Impact: higher exit reliability, fewer “stuck” positions.

## 4. Factor Governance & Attribution
- Orthogonalized factor set (trend, carry, rv_momentum, router_quality, vol_regime, expectancy).
- IR-based factor weighting with min/max bounds + smoothing.
- PnL attribution surface for factor-level analysis.
- Impact: disciplined factor risk, less overfitting / regime drift.

## 5. Operational Safety Nets
- Single-writer, atomic state surfaces.
- Preflight scripts (pretag-v7.6) + runtime sanity check.
- Dashboard regime and health panels for human oversight.

## 6. Summary
- One short paragraph summarising why v7.6 is a stability upgrade, not just a feature upgrade.
````

Keep it concise and directly based on v7.6 behaviours.

---

## 5) Append Tagging & Rollback Snippet — docs/v7.6_Pre_Tag_Audit.md

At the bottom of the existing pre-tag audit doc, append:

````markdown
## Tagging & Rollback

Once all sections above are green:

1. Tag v7.6 in Git:

```bash
git status    # ensure clean
git tag v7.6
git push origin v7.6
````

2. Document activation in ops log:

* Date/time
* Commit hash
* VERSION = v7.6
* Preflight commands run and results
* Any anomalies observed

3. Rollback reference:

* Previous stable tag (v7.5 or earlier)
* Pointer to Runtime Activation Runbook rollback section

```

---

## Constraints

- **No code semantics changes**: docs + scripts references only.
- Keep filenames exactly as specified.
- Existing docs must not be broken; add/append rather than delete.
- `make test-fast` must remain green (docs-only, no new tests required).

## Expected Output

A patch that:

- Adds the four new docs under `docs/`.
- Extends `docs/v7.6_Pre_Tag_Audit.md` with the tagging/rollback snippet.
- Optionally links these docs from any central docs index (if present).
- Introduces no runtime changes and keeps all tests/CI green.

```

---
