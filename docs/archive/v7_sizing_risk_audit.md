All good — let’s lock in the *audit* doc in the same voice and structure as the other v7 documents, and **crucially**, written for Codex as a *quant–infra audit coordinator*, **not** as an implementer.

This is the doc you will hand to Codex when you want a **safe, repo-aligned, non-destructive audit**, where Codex maps the code to the contract, reports inconsistencies, but *does not generate patches unless you explicitly ask for them*.

Here is the file:

---

# 📌 v7_sizing_risk_audit.md

**(Codex Quant–Infra Audit Pack)**

```markdown
# 📌 v7 Sizing & Risk Audit Pack
### Codex — Quant Infra Auditor Mode (No Patch Generation)

## Purpose

This document describes the *audit-only* workflow for inspecting the v7 sizing/risk contract.  
Codex should treat this as an **analytical pass**, not a patch session.

The goal is to verify that the repository’s codebase **conforms to the v7 design contract**:

- Screener → **single source of truth** for sizing (gross_usd, qty, min_notional, leverage).
- Executor → **pass-through**, no re-sizing, no caps, no size_model logic.
- RiskEngineV6/risk_limits → **sole owner of trade gating**, using unified NAV, unified cap normalization.
- Shadow pipeline → **no re-sizing**, mirrors screener intent through risk/router only.
- NAV sourcing unified via `nav_health_snapshot`.
- Removed components remain fully removed:
  - size_model
  - RiskGate
  - legacy archive/gpt_schema
  - executor-side size_for/inverse-vol rescale logic
  - executor-side cap merges
  - shadow size recomputation

Codex’s task: **map expectations → actual code** and produce an **Audit Report**, NOT patches.

---

## 1. Audit Objective

Codex must produce a structured audit answering:

1. **Contract Conformance**
   - Where does code match the v7 sizing contract?
   - Where does code diverge from the contract?

2. **Residual Dead Code**
   - Any leftover references to size_model, RiskGate, sizing knobs, inverse-vol helpers, executor caps?
   - Any legacy tests or docs referencing removed components?

3. **NAV Flow Integrity**
   - Is `nav_health_snapshot.nav_total` the single NAV feed?
   - Does screener/executor/risk_engine all consume it uniformly?

4. **Intent Sizing Truth Path**
   - Screener correctly sets gross_usd/qty?
   - Executor correctly passes them with no recomputation?
   - Shadow preserves sizing?

5. **Risk Layer Exclusivity**
   - All caps, veto reasons, min_notional, trade-equity % logic exist **only** in the risk engine.
   - No duplicated caps in executor or screener.

6. **Test Suite Alignment**
   - Tests reflect:
     - pass-through sizing
     - centralized risk
     - no RiskGate/size_model expectations
   - Identify any remaining tests that rely on old behavior.

---

## 2. Audit Scope (Directories)

Codex should inspect:

```

execution/
signal_generator.py
signal_screener.py
executor_live.py
risk_limits.py
risk_engine_v6.py
exchange_utils.py
nav.py
pairs_universe.json (referenced)
strategy_config.json (referenced)
risk_limits.json (referenced)

execution/pipeline_v6_shadow.py
dashboard/
tests/
config/
docs/

```

And confirm the following directories are either removed or contain no live logic:

```

execution/size_model.py        (should not exist)
archive/gpt_schema/execution/  (should be gone)

```

---

## 3. Required Audit Output Format

Codex should output:

### ✔ SECTION A — Summary
High-level compliance: *Compliant / Minor deviations / Major deviations*.

### ✔ SECTION B — File-by-File Findings
For each file inspected, include:

- **Compliant elements**
- **Deviations from contract**
- **Leftover code** (unreferenced, dead)
- **Risk of divergence**

### ✔ SECTION C — NAV Consistency Check
- All modules should use `nav_health_snapshot.nav_total`.
- Report any module using fallback or stale NAV logic inconsistently.

### ✔ SECTION D — Sizing Pipeline Verification
Verify the exact flow:

```

signals → screener sizing → executor pass-through → risk engine caps → router

```

Identify any locations where:
- sizing is recomputed
- caps are duplicated
- floors/min-notional differ
- leverage adjusted incorrectly
- shadow re-sizes anything

### ✔ SECTION E — Config Consistency
- `risk_limits.json` normalized to fractions.
- `strategy_config.json` contains no size_model knobs.
- Confirm per-symbol caps align with v7 capitalization logic.

### ✔ SECTION F — Test Suite Integrity
List:
- Tests aligned with v7 contract.
- Tests needing update.
- Tests still expecting old behavior.

### ✔ SECTION G — Recommended Cleanup (Optional)
This section allows Codex to *list* small cleanup opportunities, but **NOT generate code**.

---

## 4. Forbidden Actions During This Audit

Codex **must not**:

- Generate patches.
- Apply modifications.
- Suggest exact code-level diffs.
- Infer missing components.
- Rewrite modules.

Codex **may**:

- Highlight inconsistencies.
- Point to specific lines/sections where logic diverges.
- Suggest conceptual fixes (no code).
- Identify dead code, legacy patterns, outdated comments.

---

## 5. Example Output Snippet (for guidance only)

```

=== v7 Audit Summary ===
Compliance: Minor deviations

1. executor_live.py

   * OK: No size_model usage.
   * OK: Uses nav_health_snapshot.nav_total.
   * Issue: Line 1183 refers to 'max_gross_exposure_pct' in comment; outdated.
   * Issue: qty recompute path still references fallback_gross_usd (legacy).

2. signal_screener.py

   * OK: sizing block sets gross_usd & qty.
   * Issue: old pre-risk veto comment block remains; should be doc-only.

3. risk_limits.py

   * OK: sole owner of risk gating.
   * OK: cap normalization uses fractional caps.
   * Issue: symbol_notional_guard still imported but unused.

...

=== Recommendation ===
No functional changes required; remove stale comments and unused imports in executor & screener.

```

---

## 6. Final Instructions for Codex

Codex should operate in **audit mode**, as Quant Infra Auditor:

> “Examine, map, and report.  
> Do not patch. Do not rewrite. Do not make assumptions.  
> Stay strictly tied to the repository contents.”  

Codex should run this audit over the full repo and produce the structured output described above.