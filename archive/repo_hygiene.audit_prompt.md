# 🧹 Codex Repo Hygiene Audit Prompt
**Project:** GPT Hedge Infrastructure  
**Phase:** Sprint 5.9 — Infra & Portfolio Analytics Hardening  
**Context Source:** `Codex_Audit_Scope.md`  
**Execution Tag:** `[v5.9-prep] repo hygiene sweep`  

---

## 🎯 Objective
Perform a **controlled hygiene audit** across the repository.  
Codex’s goal is to **remove or archive legacy code**, simplify the repo tree, and keep all modules aligned with the *active Sprint 5.8–5.9 environment* and the `Codex_Audit_Scope.md` retention map.

This operation prepares the repo for clean imports, faster audits, and less legacy confusion before mainnet scaling.

---

## ⚙️ 1. Inputs
- `Codex_Audit_Scope.md` — authoritative map of what to keep vs. archive  
- `infra_v5.7_audit.md` — references the last valid production audit  
- `sprint_5_8_plan.md` — defines all active components through Nov 13, 2025  

Codex must treat these as *source of truth*.  
No file mentioned in these documents may be deleted.

---

## 🔍 2. Steps

### Step 1 — Scan & Dependency Graph
- Parse all `.py` files recursively.
- Build a dependency graph showing which modules import each other.
- Highlight any file **not imported** or **never executed** by:
  - `execution/executor_live.py`
  - `execution/sync_state.py`
  - `scripts/doctor.py`
  - `dashboard/app.py`
- Mark these as **removal candidates** unless explicitly protected by `Codex_Audit_Scope.md`.

### Step 2 — Legacy Detection
Flag files for archive if they:
- Contain deprecated functions superseded by `utils.py`
- Live under `execution/legacy_*`, `old_`, `copy_`, `_bak`, `_tmp`
- Replicate Firestore publishing or NAV logic already handled by newer modules
- Refer to symbols no longer in repo (import errors)

### Step 3 — Safe Archive Operation
For each flagged file:
1. Move to `/archive/deprecated_v5.7/<relative_path>`
2. Retain full directory structure
3. Replace original file with:
   ```python
   # Archived by Codex [v5.9-prep] repo hygiene sweep
   # Original moved to /archive/deprecated_v5.7/<path>
   # Do not reintroduce without audit approval.
Step 4 — Import Validation

Re-run import graph check after archive.

Ensure no missing imports remain in active modules.

Print a warning if any script still references archived modules.

Step 5 — Commit and Report

Stage all modified files

Commit with:

git add .
git commit -m "[v5.9-prep] repo hygiene sweep"


Emit the following markdown block:

### Repo Hygiene Summary
- Archived files:
  - execution/flatten_all.py
  - execution/nav.py
  - ...
- Remaining structure tree:
  (rendered via `tree -L 2`)
- Broken imports: none
- Notes:
  • All ML modules preserved
  • Firestore & dashboard telemetry untouched
  • Doctor CLI verified clean

🧩 3. Rules & Safeguards

NEVER delete — only archive.

NEVER modify ML modules (execution/ml/*).

NEVER touch current Firestore/NAV logic (execution/firestore_utils.py, dashboard/*, scripts/doctor.py).

NEVER rename core folders (execution/, dashboard/, utils/, tests/).

ALWAYS run pytest after archive and confirm 0 errors.

If unsure, print a ⚠️ Review Required notice and skip deletion.

Codex must produce a final Markdown report to stdout or /logs/repo_hygiene_report.md with:

Section	Content
Header	### Repo Hygiene Summary
Archived Files	List all paths moved to archive
Kept Files	Tree of remaining active repo
Issues	Import or dependency warnings
Recommendations	Follow-ups for manual cleanup
✅ 5. Completion Criteria

Repo builds and runs with no import errors.

All tests in /tests/ pass.

Supervisor restart (sudo supervisorctl restart hedge:*) logs no missing module errors.

scripts/doctor.py and dashboard/app.py load without exceptions.

🔒 Verification Command

Codex should finish by running:

pytest -q
python3 -m scripts.doctor -v | tail -n 10


and confirm:

[doctor] OK — NAV fresh, positions synced, telemetry live