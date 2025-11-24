# v6.0-rc Repo Hygiene Audit — Codex Prompt

You are auditing the `hedge-fund` repo for **v6.0-rc** readiness.

## Ground Rules

- Treat the following docs as **authoritative** and do not contradict them:
  - `docs/ARCHITECTURE_CURRENT.md`
  - `docs/v6.0_Master_Architecture_Map.md`
  - `docs/v6.0_architecture_brief.md`
  - `docs/infra_v6.0_repo_topology.md`
  - `docs/v6_runtime_telemetry_contract.md`
  - `docs/v6_state_contract.md`
  - `docs/v6_intel_contract.md`
  - `docs/v6_router_autotune_contract.md`
  - `docs/v6_risk_engine_contract.md`
  - `docs/v6_pipeline_shadow_compare_contract.md`
  - `docs/infra_v6.0_runtime_final_audit.md`
  - `docs/infra_v6.0_pipeline_audit.md`
  - `docs/infra_v6.0_risk_audit.md`
  - `docs/infra_v6.0_intel_audit.md`
  - `docs/infra_v6.0_telemetry_audit.md`
  - `docs/v6.0-rc_migration_guide.md`
  - `docs/v6.0_breaking_changes.md`
  - `docs/v6.0_known_issues.md`
  - `docs/v6_runtime_release_notes.md`

- **Do not touch v6 core logic** except where explicitly requested:
  - `execution/risk_engine_v6.py` (or equivalent RiskEngineV6 impl)
  - `execution/size_model.py`
  - `execution/position_sizing.py` (beyond already-agreed v6 clamps)
  - `execution/executor_live.py` sizing / risk enforcement blocks
  - `execution/intel/pipeline_v6_shadow.py`
  - `execution/intel/pipeline_v6_compare.py`
  - `execution/intel/router_autotune_apply_v6.py`

- Do **not** change:
  - sizing contracts tested in `tests/test_v6_sizing_contract.py`
  - pipeline compare contracts in `tests/test_pipeline_v6_compare_runtime.py`
  - the v6 runtime probe / synced-state schema

- You may:
  - delete legacy v5-only utilities, prompts, dead scripts
  - remove unused imports and dead functions
  - update docs to mark v5 content as legacy
  - simplify/modernize small bits of glue code where safe

When in doubt: prefer **no change** over speculative “cleanup”.

---

## Process Group 1 — Dashboard & Diagnostics Hygiene

**Goals**

- Ensure the v6 dashboard is the **only** UI entrypoint.
- Remove v5-only tools that break startup (`doctor.py`).
- Keep router/intel/pipeline panels aligned with v6 contracts.

**Scope**

Inspect and cleanup:

- `dashboard/app.py`
- `dashboard/main.py`
- `dashboard/live_helpers.py`
- `dashboard/nav_helpers.py`
- `dashboard/pipeline_panel.py`
- `dashboard/router_health.py`
- `dashboard/router_policy.py`
- `dashboard/intel_panel.py`
- `dashboard/dashboard_utils.py`
- `dashboard/async_cache.py`
- `dashboard/firestore_helpers.py`
- `scripts/doctor.py`

**Required changes**

1. **Remove doctor integration from dashboard (v5 diagnostic, not v6):**
   - In `dashboard/app.py`:
     - Remove any imports of `scripts.doctor` / `run_doctor_subprocess`.
     - Remove any buttons/menu items that trigger doctor.
     - Dashboard must start and run with **no dependency** on `doctor.py`.

   - In `scripts/doctor.py`:
     - Keep it as a **CLI-only tool** (optional) or mark clearly as legacy:
       - Add a header comment: “LEGACY v5 diagnostic, not part of v6 runtime”.
       - If you keep it, make imports defensive:
         ```python
         try:
             from dashboard.router_health import _load_order_events
         except Exception:
             def _load_order_events(limit: int = 500):
                 return ([], [], [])
         ```
       - It must **never** be imported by dashboard modules on startup.

2. **Confirm v6-only state/telemetry paths in dashboard:**
   - Verify dashboard uses:
     - `logs/state/nav_state.json`
     - `logs/state/positions_state.json`
     - `logs/state/risk_state_v6.json`
     - `logs/state/v6_runtime_probe.json`
     - `logs/state/router_health.json`
     - `logs/state/router_policy_suggestions_v6.json`
     - `logs/state/expectancy_v6.json`
     - `logs/state/symbol_scores_v6.json`
     - `logs/state/risk_allocation_suggestions_v6.json`
     - `logs/state/pipeline_v6_shadow_head.json`
     - `logs/state/pipeline_v6_compare_summary.json`

   - Remove any fallback to v5-era:
     - `nav.json`
     - `positions.json`
     - `risk_snapshot.json`
     - legacy router health formats

3. **Router panel hygiene**
   - `dashboard/router_health.py`:
     - Ensure it renders according to `v6_router_autotune_contract.md` and infra audits:
       - `maker_first`, `taker_bias`, `quality`, `reason`
       - allocator freshness where available
     - Any private helpers (`_load_order_events`) must be **internal only** and not required for app startup.
     - No hard failures if state files are missing: UI should degrade gracefully with “no data” placeholders.

4. **Pipeline panel hygiene**
   - `dashboard/pipeline_panel.py`:
     - Use the v6 compare summary:
       - veto parity
       - size parity
       - `sizing_diff_stats` (p50/p95/upsize_count/sample_size)
     - Color mapping:
       - **Green** when p95 ≈ 0 and upsize_count == 0
       - **Amber** when p95 small but upsize_count > 0
       - **Red** when p95 large or sustained upsize_count
     - No direct filesystem paths hard-coded beyond `logs/state/…` root.

5. **Intel panel hygiene**
   - `dashboard/intel_panel.py`:
     - Ensure values from `expectancy_v6.json`, `symbol_scores_v6.json`, `risk_allocation_suggestions_v6.json`:
       - are clamped/normalized into [0, 1] (where applicable).
       - handle missing keys gracefully without exploding the UI.

---

## Process Group 2 — Scripts / Legacy Utilities / Prompts

**Goals**

- Remove v5-only scripts that are not referenced by v6 docs.
- Keep only what is used in v6 runtime, pipeline, and telemetry.

**Scope**

Search:

- `scripts/`
- `prompts/`
- `docs/` for old infra prompts tied to pre-v6 architecture

**Required actions**

- Identify any scripts that:
  - reference old v5 state/telemetry (nav.json, positions.json, old router formats).
  - are not mentioned in any v6 doc / audit / migration guide.
- For each such script:
  - Either:
    - Move to a `scripts/legacy/` folder and add a header comment “LEGACY (v5), not part of v6 runtime”, or
    - Remove entirely if clearly redundant and unused (no references anywhere).
- For prompts/docs clearly superseded by v6 docs:
  - Add a one-line banner at the top:
    > This document describes pre-v6 behaviour and is retained for historical reference. See v6.0_* docs for current contracts.

No changes to `v6_*` docs themselves.

---

## Process Group 3 — Tests & Lint Hygiene

**Goals**

- Ensure tests target v6 behaviour and don’t rely on v5 file names.
- Remove dead tests for deleted components.

**Scope**

- `tests/`
- `pytest.ini`
- any test helpers under `execution/tests_utils` / similar

**Required actions**

- Confirm:
  - `tests/test_v6_sizing_contract.py` and `tests/test_pipeline_v6_compare_runtime.py` are **green** and aligned with the current v6 flows.
- Search for tests that:
  - reference `nav.json`, `positions.json`, `risk_snapshot.json`, or other v5 files.
  - expect v5-only router behaviour.
- For such tests:
  - Either migrate them to v6 state/telemetry paths (if behaviour is still relevant), or
  - Mark them as legacy (move to `tests/legacy/` or delete with clear commit message).

Run:

- `python -m compileall .`
- `pytest -q tests/test_v6_sizing_contract.py tests/test_pipeline_v6_compare_runtime.py`
- (Optional) `ruff check dashboard/ execution/ scripts/`

---

## Deliverables

1. A **single commit** or small set of commits that:
   - Remove dashboard dependency on `scripts/doctor.py`.
   - Guarantee v6-only dashboard paths.
   - Quarantine or delete obvious v5-only utilities.
   - Keep v6 runtime + contracts untouched.

2. A short summary in the commit message:
   - “v6.0-rc repo hygiene: remove doctor from dashboard, enforce v6 state/telemetry paths, quarantine v5 diagnostics.”

Do not change any sizing, risk, router-autotune, or pipeline-v6 logic beyond what’s necessary for imports and paths.

# Patch Scope — Remove doctor.py from v6 Dashboard Startup

Goal: The Streamlit dashboard must start and run with **no dependency** on `scripts/doctor.py`. Doctor is a v5 diagnostic tool and should not be in the v6 startup chain.

Files in scope only:

- `dashboard/app.py`
- `scripts/doctor.py`

Constraints:

- Do **not** change any other files.
- Do **not** change any sizing, risk, router, or pipeline logic.
- Do **not** touch state/telemetry schemas.
- Dashboard must still render all v6 panels as before.

Apply these changes:

1. In `dashboard/app.py`:
   - Remove:
     ```python
     from scripts.doctor import run_doctor_subprocess
     ```
   - Remove any callbacks / buttons that call `run_doctor_subprocess` or otherwise reference the doctor tool.
   - If needed, replace the panel section with a simple text placeholder:
     ```python
     st.info("Doctor diagnostics are not available in v6; use CLI tools instead.")
     ```

2. In `scripts/doctor.py`:
   - Leave the file in place but make it explicit legacy:
     - Add at the top:
       ```python
       """
       LEGACY: v5 diagnostic tool, not used by the v6 dashboard.
       Safe to ignore for normal v6 operations.
       """
       ```
   - Make sure it does **not** import anything that would break when run standalone.
     - Wrap imports in try/except or provide stubs if necessary.

After patch:

- `streamlit run dashboard/app.py` must start without ImportError.
- The v6 dashboard should load fully even if `scripts/doctor.py` is broken or missing.
