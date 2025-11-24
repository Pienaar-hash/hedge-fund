# GPT-Hedge — v7 Agent Bootstrap Prompt
# Mode: Codex IDE / Codex CLI
# Branch: v7-risk-tuning
# Purpose: Initialize the v7 Repo Agent and build fresh v7 documentation + patch entrypoints
#
# IMPORTANT:
# - Do NOT reference v5/v6/v6.1 docs.
# - Derive everything ONLY from the live repo state.
# - Treat this repo as the sole ground truth.
# - Build v7 documents that reflect actual code, actual topology, actual behaviour.
#
# OUTPUT:
# - v7 documentation set (architecture, topology, dashboard spec, risk model, router model, telemetry, developer guide)
# - v7 contract maps based on the real modules in the repo
# - a file-level manifest of every module in execution/, dashboard/, scripts/, config/, ops/
# - patch-ready entrypoints for future iterative development

---

## 1. REPO PARSING (MANDATORY)
Scan the entire repo from root:

Directories to index:
- execution/
- dashboard/
- scripts/
- config/
- ops/
- bin/
- VERSION

For each directory:
- Generate a structured index of filenames and relative paths.
- Summarize each module’s responsibility in one sentence.
- Identify active runtime surfaces.
- Ignore archived, deleted, or unused legacy directories.

DO NOT invent modules. Only describe what actually exists.

---

## 2. ARCHITECTURE EXTRACTION (v7)
Generate a fresh, clean **v7_Architecture.md** based purely on:
- executor flow (executor_live)
- risk evaluation path (risk_engine_v6, risk_limits)
- routing path (order_router)
- intel surfaces (intel/*)
- telemetry writers (state_publish, sync_state)
- dashboard readers
- config surfaces
- spot-state and futures NAV pipelines
- AUM sources
- runtime daemons (pipeline shadow heartbeat, compare service)
- environment variables, testnet flags, DRY_RUN behaviour

This must reflect the *current* repo, NO HISTORY.

Include:
- High-level architecture diagram (textual)
- Runtime loop summary
- State flow summary
- All active JSONL and JSON surfaces
- How dashboard consumes state
- Where risk tuning hooks will live in v7

---

## 3. TOPOLOGY MAP (v7)
Generate **v7_Topology.md**:

- Full directory tree (actual)
- Module groups: execution / dashboard / scripts / config / ops
- Import surfaces
- Execution→Telemetry→Dashboard pipeline (actual)
- Identify dead/unused modules if any

This must match exactly what Codex sees in the repo.

---

## 4. RISK ENGINE MODEL (v7)
Generate **v7_Risk_Model.md**:

Extract directly from:
- execution/risk_engine_v6.py
- execution/risk_limits.py
- universe_resolver
- position exposure paths
- NAV freshness behaviour in v7 runtime

Document:
- decision schema
- veto structure
- thresholds
- observations
- caps
- open positions logic
- any risk tuning scaffolding present in the code

DO NOT reference older versions (v6).  
This is a fresh v7 document.

---

## 5. ROUTER MODEL (v7)
Generate **v7_Router_Model.md** based on:
- execution/order_router.py
- maker→taker fallback logic
- policy integration
- post-only rejects
- adaptive offsets
- slippage and latency metrics
- router state published in telemetry

Document:
- current router constraints
- future v7 tuning hooks

---

## 6. TELEMETRY CONTRACT (v7)
Generate **v7_Telemetry.md**:

Extract directly from:
- execution/state_publish.py
- execution/sync_state.py
- telemetry JSON files under logs/state/*
- JSONL writers under execution/events.py

Document:
- each schema
- field types
- update cadence
- consumer paths (dashboard)
- invariants required by v7

This becomes the master telemetry spec.

---

## 7. DASHBOARD SPEC (v7)
Generate **v7_Dashboard_Spec.md**:

From:
- dashboard/*.py
- nav_helpers
- router health panel
- intel panel
- pipeline panel

Define:
- card layouts
- AUM donut spec
- KPI block (Sharpe, ATR regime, DD state, router KPIs, fee/PnL ratio)
- tables + charts
- how dashboard reads logs/state/*
- any v7-intended improvements discovered in code gaps

---

## 8. DEVELOPER GUIDE (v7)
Generate **v7_Developer_Guide.md**:

Cover:
- how executor is launched (bin/run-dashboard.sh, ops/hedge.conf)
- how to run pipeline shadow/compare
- how to run dashboard
- how to patch and test modules
- guidance for adding new KPIs, new risk logic, new telemetry fields
- explanation of config files in config/

Everything must reflect real repo behaviour.

---

## 9. PATCH ENTRYPOINT FILE
Create **v7_agent_patch_entrypoints.md**:

- List actionable patch surfaces:
  - executor_live.py
  - risk_engine_v6.py
  - risk_limits.py
  - order_router.py
  - state_publish.py
  - sync_state.py
  - dashboard panels
  - config surfaces
- For each: document where Codex should apply patches in future sprints.

This is our future sprint harness.

---

## 10. MANIFEST
Create **v7_manifest.json**:

Include:
- all recognized directories
- all active files
- summary of each module
- hash (Codex will generate JSON keys without hashing)
- used for cross-checking drift in future sprints

---

## 11. VALIDATION CHECKS (MANDATORY)
After generating all files, run:

1. Verify that every module referenced in documentation exists in the repo.
2. Verify that no documentation mentions nonexistent modules.
3. Verify that all schemas reflect actual state fields.
4. Verify that no historical references appear (v6, beta, RC, legacy).
5. Verify that documents reflect current v7-risk-tuning branch code reality.

Return a final summary:  
> “v7 agent initialized successfully.”

---

# END OF PROMPT
