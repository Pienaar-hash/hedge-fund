# Fourth-Strike Quant Repository Audit (Post-Mortem Forensic Reconstruction)

**Date:** 2026-04-11  
**Analyst mode:** Forensic reconstruction from repository artefacts only  
**Scope limit:** No live production log bundle was present in this checkout (`logs/` absent), so this report reconstructs probable failure sequence and causal paths using code, configs, runbooks, incident records, and commit chronology.

---

## 1) Reconstructed Timeline (with uncertainty ranges)

## T-1: Latent architecture risk exists (Dec 2025)
- Confirmed historical failure mode: restart cleared in-memory TP/SL registry; no reconstruction path; exit scanner became inert while positions remained open. Documented in incident post-mortem for Dec 4–7, 2025.
- Structural lock condition was explicitly recorded as `positions > 0`, `registry == {}`, scanner NOOP, while `max_concurrent_positions` pressure blocked new paths.

**Confidence:** High (documented incident with mechanism chain).  
**Uncertainty:** Exact wall-clock event ordering inside each loop is unavailable (raw logs not in repo).

## T0: Rapid architecture churn and migration pressure (Mar 2026)
- ECS migration runbook prescribed a multi-commit migration from fallback merge logic to candidate selection, with shadow soak and feature-flag transition.
- Git history shows dense sequential commits in March touching selector routing, fallback removal, telemetry rebasing, and diagnostics.

**Confidence:** High (runbook + git chronology).  
**Uncertainty:** Which specific commit was active at failure instant cannot be proven without deployment record + commit SHA captured in runtime artefacts.

## T+1: Cross-layer telemetry integrity concerns become explicit (Mar-Apr 2026)
- Prior post-mortem recorded multiple dashboard truth-surface mismatches (e.g., PnL and veto counters pulling wrong upstream state) later fixed by dedicated commits.
- Structural diagnostic (Apr 7, 2026) identified model-level and routing-level blind spots: BTC score-PnL inversion, zero-score routing from legacy intents, and lag-induced regime/timing regret.

**Confidence:** High (documented diagnostics and post-mortem).  
**Uncertainty:** Whether these exact defects co-occurred in one production failure or represent separate episodes.

## T+2: Repro attempt cannot be exact from repository snapshot alone (Apr 11, 2026)
- Replay subsystem exists and is deterministic in design intent, but it can fetch remote klines if local cache missing.
- Determinism guard is explicitly fail-open on /proc read failures (degrades observability certainty under stress).
- Data-room generation scripts depend on external log/state files not versioned in this repository checkout.

**Confidence:** High (code-level behavior).  
**Uncertainty:** Exact failed-day replay blocked by missing authoritative runtime artefacts.

---

## 2) Primary Root Causes

1. **State authority fragmentation across execution lifecycle (historically proven class).**  
   Registry-vs-position authority split allowed restart-time exit blindness.

2. **Selection/scoring incoherence under mixed engine outputs.**  
   Legacy intents without explicit score can collapse to `0.0`, causing blind routing behavior and degraded trade quality when Hydra candidate coverage is sparse.

3. **Regime/score lag coupling in fast markets.**  
   Sticky regime confirmation + doctrine stability + carry-direction bias can delay or invert expected score→PnL relation, especially on BTC long regimes with positive funding/basis.

4. **Evidence integrity debt (observability as single point of epistemic failure).**  
   Multiple historical truth-surface mismatches imply operational decisions could be made on misleading state during incident windows.

---

## 3) Secondary Contributing Factors

- **High architectural complexity concentration** in executor/selector/routing chains, increasing silent interaction risk.
- **Frequent config/feature-flag transitions** (ECS enablement, activation/calibration toggles, experimental sleeves).
- **Environment and infra path asymmetry** between documented production paths and repository reference configs.
- **Non-atomic lineage dependencies** (episode ledger, doctrine events, veto logs, passive observations) required for post-hoc reconstruction, but not guaranteed present together.

---

## 4) Execution-Pipeline Failures (Root-cause tree)

```text
Execution divergence
├─ A. Candidate score integrity gap
│  ├─ legacy intent lacks hybrid_score/score
│  ├─ selector fallback returns score=0.0
│  └─ route outcome depends on competing candidate presence (non-uniform behavior)
├─ B. Fill/execution timing lag
│  ├─ fill confirmation + polling windows
│  ├─ TWAP slicing delays
│  └─ queueing compounds in volatile windows
├─ C. Restart-state fragility class
│  ├─ registry reconstruction absent in older architecture
│  └─ exits stalled while position/risk pressure persisted
└─ D. Telemetry mismatch risk
   ├─ metrics sourced from stale/incorrect files (historically observed)
   └─ operator interpretation drift during live mitigation
```

---

## 5) Strategy-Logic Failures (Failure map)

- **Carry component directional penalty can anti-align with BTC long profitability** when funding and basis are persistently positive.
- **Hydra/selector score monotonicity does not imply realized monotonicity** under lagged regime confirmation and decaying alpha windows.
- **Zero-score cohorts are not purely sparse-data artefacts; documented as model-blindness path for legacy intents.**
- **Mixed maturity/feature provenance** (legacy + Hydra + overlays) raises nondeterministic outcome risk if score schema is not strictly enforced before selection.

---

## 6) Infrastructure-State Failures (Reconstruction)

- **Supervisor/runtime divergence risk:** repository reference supervisor config runs with `ENV=prod` + `BINANCE_TESTNET=1`, indicating potentially confusing env semantics if mirrored directly.
- **Runtime determinism checks are fail-open:** inability to read /proc increments counters but does not block execution, reducing certainty during degraded host conditions.
- **Replay determinism is conditional:** if kline cache absent, pipeline reaches out to remote Binance API, introducing non-local dependency at replay time.
- **Log path dependency:** several core analyses assume `logs/state/*` and `logs/execution/*` artifacts exist and are internally consistent; absent bundle prevents full infra forensics.

---

## 7) Data-Lineage Failures (Autopsy)

- **Lineage is multi-file and weakly transactional:** episode evidence, doctrine events, risk vetoes, and passive observations are generated/consumed separately.
- **Data-room exports are sample outputs, not immutable full-fidelity event stores.**
- **Generator behavior can skip missing JSONL inputs with warnings**, meaning downstream evidence packages may appear complete while silently partial.
- **Timestamp integrity and vendor-correction handling** cannot be validated post-failure without raw market snapshots and immutable ingestion manifests.

---

## 8) Risk-Control Failures (Failure matrix)

| Layer | Intended control | Observed weakness class | Failure effect |
|---|---|---|---|
| Strategy | Doctrine/veto gating | High veto complexity with limited causal avoided-loss attribution | Over/under-block ambiguity |
| Engine | Max concurrent, per-symbol caps, DD limits | Lock scenarios possible when exit path impaired | Exposure can persist while new actions blocked |
| Config | Risk limits + testnet overrides | Testnet override envelope can materially relax limits | False sense of safety parity |
| Infra | Kill-switch / activation windows | Feature flags + env ACK dual-key requires strict operational discipline | Drift between intended and actual mode |
| Monitoring | Dashboard truth surface | Historical mismatches proved | Delayed or incorrect operator response |

---

## 9) Reproducibility-Failure Ledger

1. Missing authoritative failed-day artefacts in repo snapshot (`logs/` not present).  
2. Replay depends on external market fetch when cache missing (time-varying dependency).  
3. Environment capture incomplete (package lock, container image digest, kernel/clock state not bundled with episodes).  
4. Some guardrails are fail-open (determinism checks), lowering certainty of reproduced conditions.  
5. Deployment provenance gaps: commit→runtime mapping not guaranteed in incident artefacts.

---

## 10) Governance Breakdown Map

- **Change velocity exceeded observability hardening cadence** in migration windows.
- **Architecture had historical periods where canonical source-of-truth boundaries were implicit instead of mechanically enforced.**
- **Documentation quality high, but deployment evidence binding is weaker** (runbooks and specs exist; immutable execution receipts are not consistently co-located).
- **Process anti-pattern:** fixing truth-surface after discovering mismatches indicates monitoring validation was reactive, not pre-deploy gated.

---

## 11) Systemic Weaknesses That Enabled Collapse

1. **Epistemic fragility:** operators can only act as well as telemetry truth; historical mismatches reduced trust and slowed accurate diagnosis.
2. **State model complexity:** parallel concepts (positions, registry, ledger, selector candidates, overlays) increase divergence surfaces.
3. **Mixed-mode execution stack:** legacy + Hydra + experimental sleeves complicate causal attribution and deterministic replay.
4. **Non-atomic data lineage:** post-mortems rely on assembling many files that may be missing, stale, or inconsistent.
5. **Governance gap between "documented intent" and "runtime attestation".**

---

## 12) Stabilisation & Recovery Plan (ranked by urgency)

## P0 — Immediate containment (same day)
1. **Freeze execution mode to single authoritative path** (disable legacy routing in production unless explicitly in shadow).  
2. **Enforce score schema hard-fail before selector** (`None`/missing score => reject intent, log deterministic reason).  
3. **Require immutable incident bundle capture** on every kill-switch event: state files, execution JSONL, doctrine/veto logs, runtime env dump, commit SHA, config hash.  
4. **Add startup invariant gate:** refuse live entries if positions exist but exit metadata/state authority is inconsistent.

## P1 — 72-hour hardening
5. **Make determinism guard fail-closed for entry permissions** (still allow exits).  
6. **Bind deployment receipts:** on boot, write `{git_sha, image_digest, config_hash, runtime_env}` to append-only state file and reference from every episode/intent.  
7. **Create reconciliation daemon** that continuously verifies event-chain invariants: ACK→FILL→position delta→episode close→PnL.

## P2 — 1–2 week structural repair
8. **Unify data lineage into append-only event store** with sequence IDs and monotonic timestamps; derive dashboards/reports from this source only.  
9. **Add causal risk scorecards** for each veto family (avoided-loss, false-veto, missed-opportunity) with promotion/removal thresholds.  
10. **Build deterministic replay manifest** that forbids remote fetch unless explicitly approved and records all external dependency hashes.

## P3 — governance reinforcement
11. **Pre-deploy telemetry truth tests as release blockers** (dashboard numbers must reconcile with source logs before rollout).  
12. **Mandatory post-deploy attestation checklist** signed by strategy + infra + risk owners; include rollback drill evidence.

---

## Evidence Anchors Used

- Incident and cycle post-mortems in `docs/incidents` and `docs/cycles`.
- Structural diagnostics and architecture audits in `docs/diagnostics` and `docs/*AUDIT*`.
- Runtime/risk configs in `config/`.
- Execution/replay/determinism source in `execution/` and `scripts/`.
- Deployment orchestration references in `deploy/supervisor/*.conf`.
- Commit chronology from `git log`.

