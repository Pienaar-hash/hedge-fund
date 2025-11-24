# infra_v6.1_nav_risk_sprint.md

## Overview

This sprint formalises the next evolution of the v6 runtime: **v6.1 NAV–Risk–Universe Hardening**, together with **repo topology corrections** and updates to the existing infra audits (risk, telemetry, intel). The goal is to consolidate the current v6 pipeline, eliminate legacy bleed‑through, introduce clean testnet overrides, and lock in canonical repo surfaces.

This sprint operates across four tracks:

1. NAV Pipeline Hardening
2. Risk State & Gating Hardening
3. Universe & Symbol Configuration Refinement
4. Repo Topology & Hygiene (v6.1 upgrades)

Each track contains specific objectives, tasks, and Codex‑ready deliverables.

---

## Track 1 — NAV Pipeline Hardening

### Objectives

* Eliminate legacy peak/drawdown persistence issues.
* Separate Prod vs Testnet NAV logic cleanly.
* Add explicit NAV inspection and reset tools.
* Ensure all NAV writers and consumers are unified under v6 contract.

### Tasks

1. **NAV Pipeline Documentation**

   * Create `docs/infra_v6.1_nav_pipeline.md` describing:

     * NAV sources (exchange balances, pnl, positions)
     * Writers (`execution/nav.py`, `_persist_nav_log`, state_publish)
     * sync_state NAV commit flow
     * Consumers (risk engine, dashboard, allocator, telemetry)

2. **Implement Testnet‑Aware NAV Drawdown Override**

   * In `execution/risk_limits.py`, detect `BINANCE_TESTNET=True` and:

     * Disable or soften drawdown & daily‑loss gates
     * Prevent production gating logic from blocking testnet trading
   * Ensure override is explicit in telemetry & runtime probe.

3. **NAV Inspector CLI**

   * Add `scripts/nav_inspect.py` with:

     * Latest nav, peak, dd, sources, history count
     * `--testnet-reset` option (truncate nav series, reset peak, clear caches)

4. **NAV State Consistency Tests**

   * Add tests ensuring NAV reset produces expected baseline
   * Validate navigator against upstream exchange client in testnet mode

---

## Track 2 — Risk State & Gating Hardening

### Objectives

* Expose risk state transparently.
* Harden gating logic and improve clarity.
* Support testnet overrides cleanly.
* Unify risk config schemas.

### Tasks

1. **Risk Gate Inspector CLI**

   * Add `scripts/risk_gate_probe.py`:

     * For each symbol, show allow/veto and active gates
     * Pull from `RiskEngineV6`, `risk_snapshot.json`

2. **Risk Config Refactor**

   * Reorganise `config/risk_limits.json` into:

     * `drawdown_limits`
     * `daily_loss_limits`
     * `per_symbol_caps`
     * `router_bounds`
     * `testnet_overrides`

3. **Testnet Risk Overrides**

   * When `BINANCE_TESTNET=True`, bypass or soften drawdown & daily loss checks

4. **Risk State Purge Tools**

   * Add `scripts/risk_reset_nav.py` to safely reset session risk state

5. **Tests**

   * Extend `test_risk_engine_v6.py` to cover testnet override
   * Confirm parity with production logic when override disabled

---

## Track 3 — Universe & Symbol Configuration

### Objectives

* Establish clean symbol profiles for prod vs testnet.
* Remove degraded/unlisted symbols.
* Ensure consistent symbol visibility across all intel modules.

### Tasks

1. **Universe Profile Definitions**

   * Modify `config/pairs_universe.json` to include:

     * `testnet_minimal`: BTC, ETH, SOL
     * `prod_core`: BTC, ETH, SOL, LINK, LTC, SUI, WIF

2. **Symbol Profile Loader**

   * In `execution/universe.py`, load profile based on `ENV` + `BINANCE_TESTNET`

3. **Universe Inspector CLI**

   * Add `scripts/universe_inspect.py` to verify:

     * Risk caps
     * Leverage constraints
     * Presence in intel outputs (expectancy, symbol scores)

4. **Intel Consistency Tests**

   * Validate universe alignment in:

     * `expectancy_v6.json`
     * `symbol_scores_v6.json`
     * `router_policy_suggestions_v6.json`

---

## Track 4 — Repo Topology & Hygiene (v6.1)

### Objectives

* Remove all v5 remnants.
* Update repo layout to reflect real active modules.
* Update risk/telemetry/intel audits with new architecture.
* Ensure v6.1 surfaces are canonical and documented.

### Tasks

1. **Update `infra_v6.0_repo_topology.md` → v6.1**

   * Clean up modules not used by v6.0/6.1
   * Add canonical layout:

     * `execution/` (risk, router, intel, nav, pipeline, events)
     * `dashboard/`
     * `scripts/`
     * `config/`
     * `tests/`
     * `docs/`
     * `ops/`
     * `archive/` (legacy)

2. **Update Audits**

   * `infra_v6.0_risk_audit.md` → v6.1

     * Include testnet override logic
     * Clarify drawdown & freshness semantics
   * `infra_v6.0_telemetry_audit.md` → v6.1

     * Add nav_state/nav.json split
     * Add new inspector scripts
   * `infra_v6.0_intel_audit.md` → v6.1

     * Add consistency checks with universe profiles

3. **Repo Hygiene**

   * Build `docs/repo_hygiene_v6.1.md` listing:

     * Active modules
     * Deprecated modules moved to archive/
     * Patch prompts for Codex to enforce hygiene

4. **Unified Codex Audit Prompt**

   * Create `docs/prompt_repo_hygiene_v6.1.prompt.md`:

     * Commands for Codex to audit repo
     * How to detect v5 modules
     * How to propose deletions
     * How to validate state contracts

---

## Sprint Deliverables

1. `infra_v6.1_nav_pipeline.md`
2. `infra_v6.1_risk_gating.md`
3. `infra_v6.1_universe.md`
4. Updated:

   * `infra_v6.0_repo_topology.md`
   * `infra_v6.0_risk_audit.md`
   * `infra_v6.0_telemetry_audit.md`
   * `infra_v6.0_intel_audit.md`
5. New CLIs:

   * `nav_inspect.py`
   * `risk_gate_probe.py`
   * `universe_inspect.py`
   * `risk_reset_nav.py`
6. Repo hygiene doc + Codex prompt

---

## Acceptance Criteria

* Testnet drawdown gating behaves as intended (off/softened).
* Prod gating logic remains fully strict and unchanged.
* NAV baseline resets cleanly and predictably.
* Universe profiles load correctly across intel and risk engines.
* Repo topology reflects **final v6.1 active modules**.
* All auditors (risk, intel, telemetry) updated to match new architecture.
* No legacy v5 modules remain in active surfaces.
