# System Baseline — v7.9

**Declared:** 2026-02-13
**Status:** Phase A structurally complete. Constitutional hardening applied.

This document records the verified invariants at the v7.9 baseline.
It is the known-good anchor for all subsequent evolution (including DLE Phase B).

---

## Version Identity

| Surface | Value | Enforcement |
|---------|-------|-------------|
| `VERSION` | `v7.9` | File on disk |
| `v7_manifest.json → docs_version` | `v7.9` | CI check: must match `VERSION` |
| Python runtime | 3.10.12 | Production, mypy.ini, CI (`setup-python`) |

---

## Architectural Authority Boundaries

### One-Way Data Flow (Hard Invariant)

```
Execution → logs/state/*.json → Dashboard
             config/*.json   ↗
```

**Dashboard is an observer.** It reads published state. It never imports from `execution/`.

| Rule | Enforcement |
|------|-------------|
| Zero `from execution` imports in `dashboard/` | CI gate: `grep -rn "^from execution\|^import execution" dashboard/` |
| All dashboard state access via `dashboard/state_client.py` | Single choke point (245 lines, stdlib only) |
| Dashboard never writes to `logs/state/` | Convention + code review |

### Doctrine Supremacy (Hard Invariant)

| Rule | Enforcement |
|------|-------------|
| All entries gated by `execution/doctrine_kernel.py` | Code structure: `_doctrine_gate()` is first check |
| No enabled flag on Doctrine | Doctrine kernel has no config — it IS the law |
| Every veto logged | `logs/doctrine_events.jsonl` (append-only) |
| DLE shadow observes but does not gate | `v6_flags.SHADOW_DLE_ENABLED` (Phase A only) |

### NAV Source of Truth (Hard Invariant)

| Rule | Enforcement |
|------|-------------|
| NAV = futures wallet only | `execution/nav.py → nav_health_snapshot()` |
| Stale NAV (>90s) triggers veto | Runtime check in executor |

---

## State Registry — `v7_manifest.json`

The manifest is an **executable contract**, not documentation.

| Section | Entries | Description |
|---------|---------|-------------|
| `state_files` | 44 | JSON state surfaces in `logs/state/` |
| `execution_logs` | 16 | Append-only JSONL in `logs/execution/` |
| `prediction_layer` | 18 | Prediction subsystem logs and state |
| **Total** | **78** | |

> Counts updated 2026-02-18 (post-Phase B + Binary Lab additions).

### Manifest Invariants

| Invariant | Enforcement |
|-----------|-------------|
| Every required manifest entry has a file on disk | `scripts/manifest_audit.py ci` (CI gate) |
| No untracked files in `logs/state/` | Same audit script |
| Optional phantoms allowed (features not yet active) | 16 at baseline |
| Rotated JSONL (`*.1.jsonl`) covered by base entry | Audit skips rotation suffixes |
| Executor startup: `preflight_check()` → `MANIFEST_OK` or fail | `scripts/manifest_audit.py enforce` |

### Current Audit State

```
MANIFEST_OK — 0 violations (16 optional phantoms)
```

---

## CI Pipeline — `.github/workflows/ci.yml`

Single authoritative workflow. No redundant pipelines.

| Step | Gate |
|------|------|
| **VERSION alignment** | `VERSION` must equal `v7_manifest.json → docs_version` |
| **Lint** | `ruff check .` clean |
| **Typecheck** | `mypy .` clean (Python 3.10 target) |
| **Dashboard boundary** | Zero `^from execution` / `^import execution` in `dashboard/` |
| **Manifest audit** | `scripts/manifest_audit.py ci` returns `MANIFEST_OK` |
| **Tests** | `pytest tests/unit tests/integration` — 3052 collected, ~47 skipped, 0 failures |

---

## Test Suite

| Metric | Value |
|--------|-------|
| Total collected | 3052 |
| Skipped | ~47 |
| Failures | 0 |
| Markers | `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.runtime` |
| Key coverage | risk_limits, state_files_schema, manifest_state_contract, manifest_audit, dashboard_intel_helpers, exit_reason_normalization, dle_shadow_b2/b3, episode_authority_b4 |

---

## DLE Status

| Phase | Status |
|-------|--------|
| **Phase A** (Shadow / Observe) | **Complete** |
| **Phase B** (Shadow Authority B.1–B.4) | **Complete** — tag `v7.9-b4-shadow-authority` |
| **Phase B.5** (Enforcement Rehearsal) | **Active** — rehearsal running, `phase_c_readiness.json` live |
| **Phase C** (Contraction Window) | **Active** — 14-day validation, Day 3/14 (2026-02-18) |
| Phase C.1 (Entry-only Enforcement) | Formalized — `ops/C1_OPS_PROTOCOL.md` — not yet enabled |

### Phase A Artefacts

- Shadow gate: `execution/dle_shadow.py` (observation only, never blocks)
- Shadow log: `logs/execution/dle_shadow_events.jsonl` (append-only)
- 14 DLE specification documents in `docs/dle/`
- Exit reason map: `config/exit_reason_map.yaml`
- Feature flags: `SHADOW_DLE_ENABLED`, `SHADOW_DLE_LOG_MISMATCHES`
- Test coverage: `tests/unit/test_dle_shadow.py`

### Phase B Prerequisites (Now Met)

- [x] Manifest enforced (state registry is executable contract)
- [x] Import boundaries clean (no authority leakage)
- [x] CI gates cover invariants (drift is expensive)
- [x] Shadow layer operational (observation baseline exists)
- [x] Python/tooling aligned (no version mismatch risk)

---

## Key Files Created/Modified in H1+H2

| File | Role |
|------|------|
| `dashboard/state_client.py` | Single choke point for dashboard → state access |
| `scripts/manifest_audit.py` | Manifest enforcement (CI + executor preflight) |
| `.github/workflows/ci.yml` | Rewritten: single authoritative pipeline |
| `tests/integration/test_manifest_audit.py` | 5 tests for manifest integrity |
| `tests/integration/test_dashboard_intel_helpers.py` | Updated for state_client layer |
| `v7_manifest.json` | Reconciled: +17 state files, +15 execution logs, new `execution_logs` section |

---

## What This Baseline Does NOT Cover

These are explicitly deferred. Do not assume they are done:

- `executor_live.py` decomposition (5,095 lines — functional, not pretty)
- `execution/` directory flattening (75 files — stable, not ideal)
- Documentation tree rewrite (README, CHANGELOG, RELEASE.md are stale)
- ~~DLE Phase B enforcement~~ — **Done** (B.1–B.4 complete, B.5 rehearsal active)
- DLE Phase C.1 enforcement activation (entry-only binding, gated on 14-day window)
- Remote state transport (state_client.py seam exists but is local-only)
- `fill_eff` in dashboard KPIs (returns `None` — requires executor publishing)

---

## Tagging

```bash
git tag -a v7.9-stable -m "Constitutional baseline: authority boundaries, manifest enforcement, CI invariants"
```

All evolution after this tag is Phase B territory.
