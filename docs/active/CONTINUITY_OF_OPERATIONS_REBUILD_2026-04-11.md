# Continuity-of-Operations Rebuild Report (Fifth-Strike)

**Date:** 2026-04-11  
**Repository:** `hedge-fund`  
**Mission Mode:** Determinism-first, fail-closed safety, minimal viable functionality.

---

## 1) Surviving Artefact Inventory

### 1.1 Surviving-artefact map (by class)

| Artefact class | Representative artefacts | Integrity | Trust | Dependency relationships |
|---|---|---|---|---|
| Core execution code | `execution/executor_live.py`, `execution/order_router.py`, `execution/risk_limits.py`, `execution/determinism_guard.py`, `execution/signal_generator.py` | **Intact (structurally)** | **Unverified** (runtime behavior not yet fully re-proven from clean state) | Depends on `config/*.json|yaml`, exchange API, local state files in `logs/state`, optional Firestore publish layers |
| Prediction/data-intent layer | `prediction/round_observer.py`, `prediction/dle_prediction_gate.py`, `prediction/rollback_triggers.py` | Intact | Unverified | Feeds decision constraints and veto path; coupled to execution intent/authority chain docs |
| Risk configs | `config/risk_limits.json`, `config/correlation_groups.json`, `config/pairs_universe.json`, `config/dataset_admission.json` | Intact | **Partially verified** (syntax and loader paths present; economic correctness unverified) | Loaded via `execution/risk_loader.py`; consumed by risk and exposure gates |
| Runtime config surface | `config/runtime.yaml`, `v7_manifest.json`, `VERSION` | Intact | Unverified | Governs system toggles, state contract paths, certification windows, and docs-version alignment |
| Environment definitions | `requirements.txt`, `Makefile`, `.github/workflows/ci.yml`, `pytest.ini`, `mypy.ini`, `ruff.toml` | **Partial** (no root Dockerfile/lockfile) | Unverified | CI uses Python 3.10 + pip install from `requirements.txt`; local reproducibility depends on host drift |
| Data snapshots (evidence/supporting) | `data_room/06_SUPPORTING/*`, `analysis/veto_attribution_full.csv`, `archive/data/*.csv` | Intact (as files) | **Unverified lineage** | Used for audit/evidence; not enough provenance metadata to treat as canonical training/live source |
| Logs / monitoring traces | Mostly historical docs and scripts (`scripts/quick_watch.sh`, `scripts/tail_exec_log.sh`); active `logs/` payload absent in repo | **Partial / missing runtime payload** | Unsafe as source of truth until regenerated | Needed for replay, forensic reconstruction, and state contract validation |
| Deployment scripts | `bin/run-executor.sh`, `bin/run-sync.sh`, `deploy/run_dashboard.sh`, `deploy/ohlcv_collector.cron` | Intact | Unverified | Expect host-level supervisor/systemd/cron assumptions; infra parity not guaranteed |
| Dashboard/observability code | `dashboard/app.py`, `dashboard/state_v7.py`, `dashboard/components/*` | Intact | Unverified | Reads state surfaces from `logs/state`; bounded by state contract and refresh cadence |
| Governance/process docs | `docs/active/Runbook.md`, `docs/active/TESTING.md`, `docs/*audit*.md`, `ops/*.md` | Intact, but some mixed-era content | **Partial** (contains contradictory vintage procedures) | Must be reconciled into single canonical ops doctrine |
| Legacy/archive artefacts | `archive/*`, `docs/archive/*` | Mixed (intact files, legacy semantics) | **Unsafe by default** | Useful only as reference material; isolate from production dependency graph |

### 1.2 Integrity/trust rules applied

- **Verified** requires: reproducible execution from pinned env + deterministic replay + test evidence + state-contract validation.
- Current repo mostly qualifies as **intact but unverified**; no clean-room proof bundle is committed for the current branch state.
- **Unsafe** status is assigned to legacy/archive components unless explicitly re-qualified.

### 1.3 Critical dependency skeleton

1. `config/runtime.yaml` + `v7_manifest.json` define runtime/state contract surface.  
2. `execution/*` enforces signal generation, risk gating, and order dispatch.  
3. `prediction/*` contributes veto/authority constraints.  
4. `dashboard/*` and ops tooling consume `logs/state/*` outputs.  
5. CI (`.github/workflows/ci.yml`) provides baseline lint/type/test discipline but not deterministic runtime replay certification.

---

## 2) Minimal Viable System Reconstruction

### 2.1 Minimal viable deterministic core (MVS)

Construct a **single-path, fail-closed core** with only these components enabled:

1. **Deterministic data loader**
   - Source: immutable local snapshot bundle (new `data/replay_seed/`) with checksum manifest.
   - No live exchange reads in MVS mode.
2. **Deterministic signal engine**
   - Keep one strategy path only (disable optional ML async/fanout modules for MVS).
   - Freeze signal-gate parameters from `config/runtime.yaml` into signed profile.
3. **Deterministic execution simulator**
   - Route intents to simulator adapter only; forbid broker API calls.
   - Fill model: fixed ruleset (timestamp, spread, slippage constants).
4. **Deterministic risk layer**
   - Use `execution/risk_limits.py` + `config/risk_limits.json` under strict fail-closed flags.
   - Enforce stale-NAV veto, max exposure, max concurrent positions, kill-switch state.

### 2.2 Component action matrix

| Component group | Status | Action |
|---|---|---|
| `execution/risk_limits.py`, `execution/risk_loader.py` | Present | **Reuse with hardening** (add deterministic replay tests + explicit config hash check) |
| `execution/signal_generator.py` | Present but includes optional dynamic/async branches | **Isolate** optional ML and multithreaded paths in MVS; run single deterministic branch |
| `execution/order_dispatch.py`, `execution/order_router.py` | Present | **Wrap/replace endpoint** in simulator mode for MVS |
| `prediction/*` DLE/rollback components | Present | **Retain only veto-critical subset**; defer nonessential adaptive behavior |
| `archive/*` infra and docs | Legacy | **Isolation required** (no production imports or operator references) |
| Live-state reliance (`logs/state`) | Missing runtime payload in repo | **Rebuild** from deterministic replay runbook |

### 2.3 Missing components (must be added)

- Canonical reproducible replay dataset + hash manifest.
- Single-command `make coop-replay` pipeline producing full state contract outputs.
- Determinism attestation artifact (config hash, dependency hash, output hash, seed).
- Explicit kill-switch integration test proving halt semantics end-to-end.

---

## 3) Environment Reconstruction Manifest

### 3.1 Rebuild baseline

- Python baseline: align to CI (`3.10`) and freeze in local runtime contract.
- Dependencies: start from `requirements.txt`, then produce fully pinned lock output (`requirements.lock.txt` with hashes).
- Build profile: set deterministic env vars (`PYTHONHASHSEED=0`, timezone UTC, locale fixed, thread caps).

### 3.2 Drift vectors and containment

| Drift vector | Current exposure | Containment |
|---|---|---|
| No root container spec | High | Add canonical Dockerfile + image digest policy |
| No hashed lockfile | High | Generate lockfile with hash pins |
| Optional network dependencies | Medium | MVS must run with network off |
| Host-level scheduler assumptions | Medium | Standardize via one supervisor profile + infra-as-code template |
| Archive infra overlap | Medium | Mark archive infra as non-authoritative |

### 3.3 Runtime validation steps (required)

1. `pip install -r requirements.lock.txt --require-hashes`.
2. Run lint/type/unit/integration in deterministic container.
3. Run replay twice; assert byte-identical canonical outputs.
4. Store attestation JSON bundle (`build_id`, `git_sha`, lock hash, config hash, output hash).

---

## 4) Data-Pipeline Restoration Plan

### 4.1 Lineage rebuild

- Promote `data_room/06_SUPPORTING/*` and `analysis/*.csv` to **reference-evidence only** until provenance manifests exist.
- Build new lineage catalog:
  - source identifier,
  - extraction timestamp,
  - schema version,
  - checksum,
  - transformation chain,
  - owner.

### 4.2 Schema and timestamp controls

- Introduce schema contracts for all ingested tables (strict required fields + types + monotonic timestamp checks).
- Enforce UTC-only timestamps at ingestion boundary.
- Reject mixed timezone and non-monotonic sequence violations (fail closed).

### 4.3 Replay capability rebuild

- Create immutable replay packs (`*.jsonl.zst` or parquet) with deterministic ordering.
- Add `replay_manifest.json` with per-file SHA256.
- Add `scripts/replay_verify.py` to validate pack before every run.

### 4.4 Contamination isolation

- Quarantine all unlabeled/legacy snapshots under `archive/data` and equivalent folders.
- Require explicit promotion checklist before any dataset becomes production-eligible.

---

## 5) Execution & Strategy Revalidation Report

### 5.1 Revalidation scope

Validate deterministic behavior for:
- signal admission (`execution/signal_generator.py`),
- risk vetoes (`execution/risk_limits.py`),
- router/dispatch path (`execution/order_router.py`, `execution/order_dispatch.py`),
- lifecycle/ledger outputs (`execution/episode_ledger.py`, `execution/position_ledger.py`).

### 5.2 Required controlled simulations

1. **Golden-path replay:** fixed inputs, fixed seeds, identical outputs over 2+ runs.
2. **Safety-invariant replay:** stale NAV, drawdown breach, max exposure, kill-switch activation.
3. **Corruption replay:** malformed config/data should halt with explicit deny reasons.
4. **No-ambiguity replay:** no dependency on wall-clock or mutable external state in MVS mode.

### 5.3 Acceptance conditions

- No nondeterministic output deltas between repeated runs.
- Every veto reason traceable to config+state evidence.
- No order emitted when risk or determinism guard indicates degraded/unsafe state.

---

## 6) Risk-Control Reinstatement Matrix

| Layer | Control | Reinstatement action | Validation proof |
|---|---|---|---|
| Strategy layer | Entry gating / expectancy / volatility filters | Freeze gate config, remove adaptive overrides in MVS | Replay compares gate decisions deterministically |
| Engine layer | Exposure caps, stale-NAV fail-close, drawdown checks | Re-derive numeric limits from policy and encode single source in `config/risk_limits.json` | Unit + integration tests for each veto path |
| Broker adapter layer | Final order guardrails and kill-switch | Add mandatory pre-dispatch policy check + emergency stop latch | Simulation showing zero dispatch post kill-switch |
| Cross-layer | Consistency of limits | Build generated risk snapshot from config hash; assert match at runtime | Contract test failing on mismatch |
| Escalation | Alert + stop semantics | Define incident escalation matrix and response timers | Game-day drill records |

---

## 7) Governance & Change-Control Reset Specification

### 7.1 Clean-slate governance controls

- Branch protection: require CI green + one code-owner review + signed commits for risk/config changes.
- Change classes:
  - **Class A (risk/execution/data contracts):** requires simulation evidence + rollback plan.
  - **Class B (ops/docs/non-critical):** standard review.
- Mandatory change packet per PR:
  - intent,
  - affected modules,
  - deterministic test evidence,
  - risk impact,
  - rollback command.

### 7.2 Audit trail rebuild

- Persist machine-readable change ledger (`docs/active/change-ledger/*.json`).
- Capture config hashes and manifest hashes at deploy time.
- Require deployment ticket linking commit SHA ↔ runtime attestation.

### 7.3 Ambiguity elimination

- Consolidate outdated runbook guidance; archive superseded procedures with explicit deprecation banner.
- Define one canonical operations runbook for current version only.

---

## 8) System Reintegration Validation

### 8.1 Reintegration sequence

1. Data replay pack validation.
2. Deterministic signal+risk simulation.
3. State-surface contract verification (`v7_manifest.json` paths).
4. Cross-module interface checks (`prediction` → `execution` → state publish → dashboard read).
5. Twin-run reproducibility proof (same input, same output hashes).

### 8.2 Readiness criteria (limited live eligibility)

- 100% passing deterministic replay suite (including safety scenarios).
- No schema/state contract violations.
- Kill-switch tested and confirmed across strategy/engine/dispatch boundaries.
- Formal sign-off from engineering + risk owner.

---

## 9) Controlled Resumption Plan

| Phase | Entry criteria | Exit criteria | Monitoring requirements | Rollback triggers |
|---|---|---|---|---|
| **Sandbox** (offline) | Environment lock + replay pack + full deterministic pass | 7 consecutive clean deterministic runs | Output hash checks, invariant dashboard, veto audit | Any nondeterminism, schema drift, missing lineage |
| **Paper trading** | Sandbox exit + simulated broker parity checks | 14 days stable policy behavior, no safety violations | Live-feed ingest audits, decision/veto trace integrity | Unexpected state divergence, stale-NAV guard failures |
| **Constrained live** | Paper exit + capped size + strict symbol subset | Predefined risk KPIs stable for 30 days | Real-time risk snapshot, kill-switch heartbeat, incident SLA | DD threshold breach, repeated control failure, exchange anomaly |
| **Monitored expansion** | Constrained-live exit + governance evidence complete | Full production profile approved | Ongoing drift detection, weekly replay certification | Any regression in determinism or policy conformance |

---

## 10) Long-Term Hardening Recommendations

1. **Deterministic-by-default architecture**
   - Mandatory replay certification before merge to protected branch.
2. **Reproducible environment as code**
   - Container image pinning + lockfile hash verification + provenance attestations.
3. **Data governance upgrade**
   - Formal lineage registry and dataset promotion gates.
4. **Risk policy compiler**
   - Generate runtime-enforced limits from policy source-of-truth to avoid manual drift.
5. **Operational resilience drills**
   - Quarterly continuity game-days with failure injection (data corruption, stale state, API outage).
6. **Documentation lifecycle control**
   - Versioned canonical runbook and automated stale-doc detection.

---

## Immediate Next 10 Actions (Execution Order)

1. Freeze current branch SHA and build baseline attestation scaffold.
2. Create deterministic lockfile + canonical container spec.
3. Build replay seed dataset + checksum manifest.
4. Implement `make coop-replay` deterministic pipeline.
5. Add twin-run hash comparison test in CI.
6. Add kill-switch end-to-end simulation test.
7. Add stale-NAV and schema-corruption fail-closed tests.
8. Isolate archive assets from production resolution paths.
9. Consolidate runbook into single current-version operator guide.
10. Gate any live resumption behind formal phase checklist in this report.
