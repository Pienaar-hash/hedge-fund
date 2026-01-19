# `PATCHSET_v7.6_S5_docs_refresh.prompt.md`

### (v7.6 Docs Alignment, State Contract, Runtime Flow, Agent Guides, Architecture)

---

## 🎯 **Objective**

Unify and refresh all v7.6 documentation so that:

* Docs **match the actual code**, state surfaces, modules, and telemetry flows.
* There is **no stale v5/v6 content** in active docs.
* Agents have a **single, authoritative entrypoint**.
* The full v7.6 system is documented cleanly for:

  * Humans
  * GPT agents
  * New contributors
  * Audits and debugging workflows

This is a **pure documentation patch**:
🚫 **No code changes.**
🚫 **No functional changes.**

---

## 📁 **Docs in Scope (these must be refreshed)**

We already have the uploaded v7.6 docs in your GPT project folder:

* `v7.6_Architecture.md`
* `v7.6_State_Contract.md`
* `v7.6_Runtime_Diagnostics.md`
* `v7.6_Incident_Playbooks.md`
* `v7.6_Developer_Guide.md`
* `v7.6_Testing.md`
* `v7.6 Test Topology & Suite Health A.md`
* `v7.6 State & Telemetry Contract Aud.md`
* `v7_manifest.json`
* `v7_agent_patch_entrypoints.md`

PATCHSET S5 will **refresh, unify, trim, and restructure** these.

---

## 📌 S5 DELIVERABLES

### 1. **Architecture Doc (v7.6_Architecture.md) Refresh**

Update to reflect:

* Final v7.6 module boundaries:

  * execution/
  * diagnostics/
  * state_publish/
  * router/
  * risk_limits/
  * screener/
  * exit_scanner/
  * position_ledger/
* Accurate real-time dataflow diagram:

  * Screener → Risk → Router → Executor → State Files → Dashboard
* Add new S0/S1/S2 components:

  * State atomic writers
  * Runtime diagnostics pipeline
  * Ledger unified positions

Include:

* List of canonical state surfaces (names + writers + update frequency)
* Liveness signals and router activity channel
* Data provenance: which component owns which data

---

### 2. **State Contract Doc (v7.6_State_Contract.md) Finalization**

Ensure:

* Contract includes:

  * `positions_state.json`
  * `positions_ledger.json`
  * `kpis_v7.json`
  * `runtime_diagnostics.json`
  * `risk_snapshot.json`
  * `router_state.json`
  * `symbol_scores.json`
  * `rv_momentum.json`
  * `factor_diagnostics.json`
* For each surface:

  * canonical writer
  * path
  * schema
  * invariants (updated_at, nonzero qty rules, etc.)
* Remove all outdated pre-S1/S2 descriptions

---

### 3. **Runtime Diagnostics Doc (v7.6_Runtime_Diagnostics.md)**

Refresh to include:

* S2 hardening:

  * liveness missing-timestamp semantics
  * router activity tracking
  * exit coverage
  * ledger/registry mismatch
* All runtime diagnostic fields with meaning and sample JSON payload
* Dashboard rendering semantics (how cards decide green/yellow/red)

---

### 4. **Incident Playbooks (v7.6_Incident_Playbooks.md)**

Add updated troubleshooting flows:

* “System is alive but not trading”
* “Exit pipeline not triggering”
* “positions_state.json frozen”
* “ledger mismatch detected”
* “NAV anomaly detected”
* “Router idle too long”
* “Veto storm: all signals vetoed”

Each with:

* Diagnosis
* Tools (jq, grep, timestamps)
* Invariants
* Recovery actions

---

### 5. **Developer Guide (v7.6_Developer_Guide.md)**

Update for:

* New repo hygiene structure
* Path conventions
* Test lanes:

  * unit
  * integration
  * runtime
  * legacy
* How to add new state surfaces
* How to add diagnostics counters
* How to write new factor modules
* How to attach new dashboards panels

---

### 6. **Testing Guide (v7.6_Testing.md)**

Must include:

* Unified lanes (S3):

  * `make test-fast`
  * `make test-runtime`
  * `make test-all`
* Markers (`unit`, `integration`, `runtime`, `legacy`)
* Test layout
* Golden rules for contributors:

  * No filesystem in unit tests
  * Integration tests must isolate state with tmp_path
  * Runtime tests must be manually triggered only
  * Legacy tests may fail but must not break lanes

---

### 7. **Manifest Refresh (v7_manifest.json)**

Ensure manifest fully matches refreshed docs:

* All state surfaces listed
* Each file has:

  * owner
  * update frequency
  * description
* Remove deprecated entries

Add new field:

```json
"docs_version": "v7.6"
```

---

### 8. **Agent Guide (v7_agent_patch_entrypoints.md)**

Rewrite so that:

* There is a **single authoritative entrypoint** for patchsets:

  * How agents start a new sprint
  * How agents reference state contract
  * What invariants must hold before writing any patchset
* Add lane rules:

  * Every patchset must pass `make test-fast`
  * Some require running runtime tests
* Add a “before writing code, the agent must check” checklist

---

## 🧹 Additional Cleanup in Docs (S5 only)

* Remove any:

  * v5/v6-era pseudocode
  * Old registry references
  * Shadow diagrams replaced by S0–S2 models
  * Deprecated shadow pipeline docs

Move them to:

```
docs/archive/v5-v6/
```

---

## 🧪 Tests (Documentation-Only)

No unit tests required — but add a simple Markdown linter pass target to Makefile:

```make
lint-docs:
\tnpx markdownlint docs/**/*.md
```

---

## ✅ **Acceptance Criteria**

S5 is complete when:

1. All v7.6 docs agree on:

   * State surfaces
   * Writers
   * Module boundaries
   * Runtime flow
   * Diagnostics semantics
   * Test lanes

2. There are **no contradictions** between:

   * Architecture doc
   * State contract doc
   * Runtime diagnostics doc
   * Manifest

3. v7.6 docs are clean, lean, and authoritative.

4. Agent instructions are updated and deterministic.

5. No code is modified — documentation only.

---

# When ready:

**“Proceed with S5-apply.”**

---
