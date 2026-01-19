# `PATCHSET_v7.6_S4_repo_hygiene.prompt.md`

### (Repo Hygiene, Dead-Code Purge, Structure Fixes, Naming, CI, Docs)

---

## 🎯 **Objective**

Bring the repo to a **v7.6 cleanroom standard**:

* Remove dead code, misnamed files, stale commands, broken directories
* Normalize naming across execution/, state/, diagnostics/
* Ensure CI/linting/pre-commit flows are updated
* Tighten manifest accuracy
* Ensure repo layout matches v7.6 docs and state contract
* Improve agent ergonomics (path predictability + removal of traps)

**No functional behavior changes** to risk, signals, router, or execution.
This is a **pure hygiene patch**.

---

## 📁 **Files in Scope**

You will likely touch:

```
execution/
    *.py
dashboard/
    *.py
config/
    *.json / *.yaml
tests/
    unit/
    integration/
    legacy/
docs/
    *.md
v7_manifest.json
.gitignore
Makefile / requirements.txt / pre-commit-config.yaml
```

---

## 1. Remove Stale, Deprecated, or Shadowed Code

Search for and delete modules that:

* Are never imported anywhere in v7.4+ / v7.5+ / v7.6 codepaths
* Refer to pre-v7 telemetry or pre-C3 registry logic
* Duplicate logic that is now canonicalized in diagnostics/state_publish/executor

### 1.1 Candidate files to remove (examples — let Codex detect exact list):

```
execution/router_health_*.py
execution/risk_limits_v5*.py
execution/risk_model_v5*.py
execution/order_router_v5*.py
execution/state_publish_v5*.py
execution/screener_v5*.py
execution/backtest_*.py    ← unless actively used
execution/log_compactor.py ← if unused
diagnostics/ (old dirs)
```

Remove entire legacy directories if empty after pruning.

Create a **safety summary** of what was removed in a doc:

```
docs/v7.6_Repo_Hygiene_Report.md
```

Format:

```markdown
File/Directory | Reason Removed | Notes
```

---

## 2. Normalize Naming and Cross-File Consistency

### 2.1 🟦 State files

Ensure every state file has:

* **positions_state.json**
* **positions_ledger.json**
* **kpis_v7.json**
* **runtime_diagnostics.json**
* **nav_state.json**
* **risk_snapshot.json**
* **router_state.json**
* **symbol_scores.json**
* **rv_momentum.json**
* **factor_diagnostics.json**
* **factor_weights.json** (if used)

Rename or remove shadow variants:

```
positions.json          → deprecated (keep snapshot writer for backwards compatibility)
tp_sl_registry.json     → replaced by positions_ledger.json
risk.json / risk_state  → unify to risk_snapshot.json
diagnostics.json / diag.json → unify to runtime_diagnostics.json
state.json              → remove or redirect
```

Update manifest and loaders accordingly.

### 2.2 🟪 Directory layout

Ensure repo root visually reflects architecture:

```
execution/
dashboard/
config/
docs/
tests/
utils/         ← optional (shared helpers)
scripts/       ← optional
```

Delete or merge:

```
helpers/
old/
research/
notebooks/
```

Unless explicitly maintained outside production.

---

## 3. Manifest Repair

**File:** `v7_manifest.json`

* Ensure all state files are listed and paths correct
* Remove old surfaces
* Ensure runtime diagnostics, ledger, kpis_v7 are present
* Add “owner” field per file (executor / sync_state / dashboard reader)

Example snippet:

```json
"state_files": {
  "positions_state": {
    "path": "logs/state/positions_state.json",
    "owner": "executor"
  },
  "positions_ledger": {
    "path": "logs/state/positions_ledger.json",
    "owner": "executor"
  },
  "kpis_v7": {
    "path": "logs/state/kpis_v7.json",
    "owner": "executor"
  },
  "runtime_diagnostics": {
    "path": "logs/state/runtime_diagnostics.json",
    "owner": "executor"
  }
}
```

---

## 4. CI / Linting / Requirements Hygiene

### 4.1 requirements.txt

* Remove libraries no longer used (Search imports across repo)
* Sort alphabetically
* Pin versions if not pinned

### 4.2 pre-commit (if repo uses it)

* Add black, ruff, flake8, isort
* Ensure tests run with `make test-fast` not `pytest`

### 4.3 GitHub Actions (if present)

* Replace `pytest` with lane-aware testing:

```
make test-fast
```

or

```
pytest tests/unit tests/integration -m "not runtime and not legacy"
```

---

## 5. Update `.gitignore`

Ensure it contains:

```
logs/
logs/state/
logs/execution/
.env
__pycache__/
*.pyc
*.tmp
*.swp
*.log
.notebooks/
outputs/
```

Remove obsolete entries.

---

## 6. Add Dead-Code & Repo Health Audit Commands

Add Makefile targets:

```make
lint:
\truff check .

deadcode:
\tvulture execution dashboard > deadcode.txt

format:
\tblack execution dashboard tests
```

Optional: include `pytest --collect-only` to detect hidden tests.

---

## 7. Documentation Updates

### 7.1 Update or create:

* `docs/v7.6_Repo_Hygiene_Report.md` — summary of removed files
* `docs/v7.6_Architecture.md` — update state surfaces & telemetry

### 7.2 Remove or archive deprecated docs:

* v5-era prompts
* v6-era router docs
* Redundant state descriptions superseded by v7.6 state contract

Move to:

```
docs/archive/
```

---

## 8. Tests

### 8.1 Ensure test suite remains green:

Run:

```
make test-fast
make test-runtime
make test-all
```

### 8.2 Add integration test:

`tests/integration/test_manifest_state_contract.py`

Verify:

* manifest paths exist
* owners match executor/sync_state
* no dangling state paths

---

## 9. Acceptance Criteria

S4 is complete when:

1. No stale or shadow files remain
2. Directory layout matches v7.6 architecture
3. All state surfaces follow unified naming
4. Manifest is accurate and complete
5. CI/test tooling works with lanes
6. requirements.txt & .gitignore are clean
7. Repo hygiene report created
8. Full test suite (minus legacy) is green

This is a **zero-risk patch** — diagnostics & state only, no trade logic touched.

---
