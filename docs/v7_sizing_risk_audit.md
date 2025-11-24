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

