# **PATCHSET_v7.6_S6 — Agent Bootstrap**

### (Unified Agent Entrypoint, Safety Rails, Patchset Protocol, Repo Context Injection)

Below is the **full Codex-ready patch prompt**, which you can paste directly into Codex/CLI to generate the files and modifications.

---

# `PATCHSET_v7.6_S6_agent_bootstrap.prompt.md`

## 🎯 **Objective**

Create a **single authoritative agent bootstrap system** so GPT/Copilot agents:

* Always understand repo layout
* Always load correct context (architecture, state contract, diagnostics, manifest)
* Always follow the patchset workflow
* Never mutate state surfaces or test lanes incorrectly
* Never touch trading logic unless instructed
* Can safely apply patchsets across v7.6 and upcoming v7.7 branches

This enables *consistent, deterministic, safe execution* from agents.

This patch is **documentation + scaffolding only**, not runtime logic.

---

## 📁 **Files to Create / Modify**

You will create or update:

```
agents/
    AGENT_BOOTSTRAP.md
    AGENT_PROTOCOL.md
    AGENT_SPRINT_TEMPLATE.md
    AGENT_PATCHSET_TEMPLATE.md
    AGENT_RAILS.md
    AGENT_CONTEXT.json

docs/
    v7.6_Agent_Guide.md   (linking all above)

repo root:
    agent-bootstrap.sh    (optional convenience script)
```

---

# 1. `agents/AGENT_BOOTSTRAP.md`

This is the *top-level agent entrypoint*.

### Content Requirements:

* Purpose: “This file is the first thing every GPT/Codex agent must read.”
* List the **required context documents**:

  * v7.6_Architecture.md
  * v7.6_State_Contract.md
  * v7.6_Runtime_Diagnostics.md
  * v7.6_Testing.md
  * v7.6_Developer_Guide.md
  * v7_manifest.json
* Describe **what an agent must do before writing a patchset**:

  1. Load architecture
  2. Load state contract
  3. Load manifest
  4. Load diagnostics semantics
  5. Verify test lanes
  6. Confirm no trading logic modifications (unless patch requires it)
* Include:

  * Repo invariants
  * Naming rules
  * Single-writer rules
  * State surface invariants
  * Patchset safety checklist

---

# 2. `agents/AGENT_PROTOCOL.md`

Formal protocol describing **how patchsets must be written and validated.**

### Must include:

* Patchset lifecycle:

  1. *Plan → Apply → Validate → Document → Commit*
* What a patchset must contain:

  * Scope
  * Files in scope
  * Code changes summary
  * Tests required
  * Acceptance criteria
* When agents must run:

  * `make test-fast`
  * `make test-runtime` (if stateful)
* When agents **must not** run tests (documentation-only patches)
* Explicit rules for:

  * No renaming state surfaces without manifest update
  * No creating new state surfaces without contract doc update
  * No modifying execution logic except in strategy patchsets (A/B/C tracks)

---

# 3. `agents/AGENT_SPRINT_TEMPLATE.md`

Template for starting any new sprint.

### Must include:

* Sprint structure (S0 → S1 → S2 → S3 → Strategy tracks)
* How to create sprint overview
* How to create patchset roadmap
* How agents should maintain continuity
* How to track test impact
* How to hand off between patchsets

---

# 4. `agents/AGENT_PATCHSET_TEMPLATE.md`

A strict patchset template with:

* Title
* Objective
* Files touched
* Before/After behavior
* Risk classification
* Test lanes required
* Acceptance criteria
* Manifest changes?
* Docs changes?

This template enforces a standardized patch format for all contributors and AI agents.

---

# 5. `agents/AGENT_RAILS.md`

**Safety rails** for GPT/Codex agents.

### Must include:

* You MUST NOT:

  * Modify trading logic unless patchset explicitly says so
  * Modify risk logic or veto rules without S-tier signoff
  * Write to any `logs/state/*.json` from tests (only through tmp_path)
  * Add new statewriters without documenting contract
  * Change diagnostics semantics casually
* You MUST:

  * Respect single-writer rules
  * Run correct test lanes
  * Maintain repo hygiene
  * Update docs when required
  * Ensure reproducibility

This is critical for safe long-term evolution of the codebase with agents in the loop.

---

# 6. `agents/AGENT_CONTEXT.json`

Machine-readable context loader for agent-aware tools.

### Fields:

```json
{
  "version": "v7.6",
  "architecture_doc": "docs/v7.6_Architecture.md",
  "state_contract": "docs/v7.6_State_Contract.md",
  "runtime_diagnostics": "docs/v7.6_Runtime_Diagnostics.md",
  "testing_guide": "docs/TESTING.md",
  "developer_guide": "docs/v7.6_Developer_Guide.md",
  "manifest": "v7_manifest.json",
  "repo_invariants": {
    "single_writer": true,
    "atomic_state": true,
    "lanes": ["unit", "integration", "runtime", "legacy"]
  }
}
```

Agents can ingest this automatically to initialize state.

---

# 7. `docs/v7.6_Agent_Guide.md`

A human-friendly guide that ties together:

* The bootstrap system
* Protocol
* Templates
* Rails
* How agents should participate in patchsets
* How to run tests as an agent or human
* How to perform pre-patch audit

This becomes part of the official v7.6 documentation set.

---

# 8. Optional: `agent-bootstrap.sh`

Simple script to print:

```
cat agents/AGENT_BOOTSTRAP.md
```

Or load context into tools that want a bootstrap command.

---

# 9. Acceptance Criteria

S6 is complete when:

1. All agent bootstrap files exist and are cross-referenced
2. A single authoritative “Agent Entrypoint” now exists
3. Patchset protocol is standardized
4. Rails are enforced
5. Sprint & Patchset templates exist
6. Manifest/dox integration is complete
7. No runtime code or trading logic changed

This finishes the bootstrap required to safely use agents in the upcoming **v7.6 Execution Layer Reinforcement** sprint.

---
