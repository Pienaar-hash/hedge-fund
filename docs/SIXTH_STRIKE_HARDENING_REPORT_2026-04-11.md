# Sixth-Strike Quant Repository Audit — Hardening & Future-Proofing

**Date:** 2026-04-11  
**Scope:** Live futures stack, prediction layer, risk/doctrine controls, state/manifest contract, operational governance.

This report assumes the system is currently functional and focuses on long-term hardening against drift, nondeterminism, corruption, and operational entropy.

---

## 1) Drift-Vector Elimination Plan

| Drift vector | Concrete components at risk | Detection mechanism | Prevention mechanism | Rollback mechanism |
|---|---|---|---|---|
| Dependency version drift | `requirements.txt`, CI runtime, optional libs (`scipy`, `scikit-learn` currently range-pinned) | Daily lock-diff check in CI (`pip freeze` snapshot hash + diff alert) and scheduled dependency manifest report | Convert all runtime deps to exact pins; split `requirements-runtime.lock` vs `requirements-dev.lock`; enforce hash-checked installs (`pip --require-hashes`) | Keep signed weekly dependency lock snapshots; one-command rollback script to previous known-good lock + artifact rebuild |
| Environment drift | `config/runtime.yaml`, env var gates (`CALIBRATION_WINDOW_ACK`, `ACTIVATION_WINDOW_ACK`), supervisor units | Startup environment fingerprint (python version, locale, TZ, env whitelist hash) emitted into immutable run header | Enforce environment allowlist; reject unknown env vars touching execution path; immutable container image digest pinning | Blue/green runtime profiles with last-good env fingerprint; automatic fallback if runtime fingerprint mismatch |
| Hardware drift | Determinism guard (`execution/determinism_guard.py`), memory/PSI sensitivity | Periodic host capability snapshot (CPU flags, kernel version, mem profile) + deterministic replay checksum comparison across hosts | Define minimum host class and disable heterogeneous schedulers for live; require replay parity before host admission | Drain and quarantine non-parity host; replay canonical sample on alternate host; fail back to certified host pool |
| Data schema drift | `v7_manifest.json`, `scripts/manifest_audit.py`, `config/dataset_admission.json` | Strict schema contract tests at ingest boundary; manifest CI + runtime preflight already present | Versioned schemas with backward-compat transforms; schema registry with breaking-change gate | Keep prior schema adapters; dual-read mode (old+new schema) until parity reaches threshold |
| Timestamp drift | event logs, doctrine/risk decision ordering, exchange timestamps | Temporal integrity monitor (ingest_ts <= decision_ts <= submit_ts <= fill_ts); monotonic clock drift alert | Normalize all internal timestamps to UTC epoch ns + source offset field; reject future timestamps outside tolerance | Reprocess from raw immutable source payloads with corrected clock map |
| Microstructure drift | router assumptions (`maker-first`, slippage/fallback), liquidity buckets | Slippage/fill-quality control charts by symbol/session + structural break detector | Periodic auto-recalibration windows with strict guardrails; fallback to conservative execution policy when drift detected | Roll back router policy bundle to last profitable stable profile; enforce reduced-risk mode |
| Config drift | `config/*.json|yaml`, strategy/risk toggles | Signed config digest at startup + every loop; diff stream to audit log | Config immutability during session (no hot mutation for critical keys); dual-key for sensitive toggles | Fast restore from signed config bundles tagged by release |
| Model-parameter drift | Sentinel-X, Hydra, conviction parameters and thresholds | Parameter vector hash captured per decision/event; weekly drift report vs baseline hash set | Parameter registry with semantic versioning and approval gate; no implicit defaults | Rehydrate prior parameter pack and replay canary episodes before restoring full traffic |
| Documentation drift | Architecture docs, runbooks, decision docs | Docs/code drift linter: unresolved file references, stale version/date checks, missing ADR links | “Docs-as-contract” for operational invariants (must be validated by CI checks, not prose only) | Keep versioned docs with release tags; restore docs to last release baseline and regenerate delta |

**Priority execution order:** dependency/config/schema/timestamp drift controls first (highest deterministic impact), then microstructure/model/docs drift.

---

## 2) Determinism-Enforcement Specification

### 2.1 Determinism risk register

| Risk | Where it can occur | Hardening control |
|---|---|---|
| Nondeterministic libraries / numeric kernels | NumPy/SciPy/sklearn operations and BLAS threading | Pin linear algebra backend, set single-thread deterministic env (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), freeze RNG seeds centrally |
| Parallelism/concurrency hazards | async collectors, order routing retries, state writes | Single-writer rule for state surfaces; append-only event streams with ordered sequence IDs |
| Hidden global state | module-level caches, process-wide mutable config | Explicit state container passed via dependency injection; forbid mutable module globals on execution path |
| Time-dependent behavior | `now()` calls in decisions, loop timing jitter | Inject deterministic clock interface for simulation/replay; all decisions consume event-time not wall-time |
| Non-idempotent operations | retries around order submission/log append | Idempotency keys for order intents and risk/doctrine verdicts; duplicate detection store |
| Implicit defaults | missing config keys silently defaulting | Strict config schema validation; startup fails on unknown/missing critical fields |
| Floating-point instability | scoring thresholds near boundaries | Quantize comparison thresholds (e.g., basis-point fixed-point ints) and standardize rounding mode |
| Hardware-specific behavior | host-level math/perf variance | Certified runtime profile + replay parity gate before deployment |

### 2.2 Cross-domain enforcement

- **Strategy logic:** deterministic seed lifecycle per cycle + per symbol, fixed feature ordering, immutable feature schema version in every signal record.
- **Execution logic:** idempotent order-intent UUID, deterministic retry policy matrix, and exactly-once transition model for `intent -> submitted -> acknowledged -> filled/canceled`.
- **Data ingestion:** deterministic parse/canonicalization pipeline with strict timezone normalization and sequence-numbered input batches.
- **Simulation:** deterministic event scheduler driven only by serialized event tape; no wall-clock dependencies.
- **Replay:** canonical replay pack must include code SHA, dependency lock hash, config hash, parameter hash, and source event hash; replay must emit a bitwise-equivalent verdict stream for certified scenarios.

### 2.3 Determinism acceptance gate (new mandatory CI/runtime gate)

A release is **blocked** if any of the following fail:
1. Replay parity suite: 100% verdict parity on golden episodes.
2. State hash parity: identical terminal state for golden scenarios.
3. Risk/doctrine parity: identical veto reasons and ordering.
4. Environment parity: runtime fingerprint equals certified profile.

---

## 3) Resilience-Hardening Blueprint

### 3.1 Single-point-of-failure analysis

| Area | Current SPOF pattern | Hardening layer |
|---|---|---|
| Exchange connectivity | Single venue execution dependency | Execution mode ladder: Live venue -> reduced mode -> simulation-safe mode |
| State publication | Local filesystem primary state store | Dual-write to append-only event store + local state projection |
| Risk/doctrine availability | Inline dependency for order permission | Precomputed minimal safety policy cache (fail-closed for entries, fail-open only for emergency exits) |
| Supervisor/process management | Process-level restarts may cause blind windows | Health-aware orchestrator with readiness/liveness gates and warm-standby worker |
| Reference/config files | Runtime file corruption risk | Signed config and manifest verification with startup refusal |

### 3.2 Failover, degradation, recovery

- **Failover behavior:** if exchange or data quality degrades, automatically step down to `entry_blocked_exit_allowed` mode.
- **Degradation modes:**
  1. Full-trade mode
  2. Reduced sizing + limited symbols
  3. No new entries, managed exits only
  4. Kill-switch hard stop
- **Recovery procedure:** deterministic restart playbook:
  1. verify manifest/config hashes,
  2. replay last N minutes of event tape,
  3. reconcile positions with exchange,
  4. re-enable modes progressively with audit checkpoint.

---

## 4) Data-Pipeline Immunisation Plan

### 4.1 Schema contracts and validation

- Introduce explicit versioned schemas for:
  - market ticks/ohlcv,
  - regime feature vectors,
  - signal payloads,
  - order intents/fills,
  - risk/doctrine events.
- Enforce hard validation at boundaries (reject or quarantine malformed records, never silently coerce).

### 4.2 Lineage and replay guarantees

- Every downstream artifact carries lineage tuple: `(source_vendor, source_msg_id, ingest_batch_id, transform_version, schema_version)`.
- Store immutable raw zone and curated zone separately; curated data is reproducible from raw + transforms.
- Vendor correction protocol: append correction events, never rewrite prior facts; build “effective view” via correction overlay.

### 4.3 Contamination isolation

- Quarantine channel for malformed or timestamp-anomalous data.
- Symbol/session-level blast radius containment: isolate impacted symbols without halting whole system unless cross-symbol contamination threshold is breached.
- Timezone policy: UTC-only internal storage with explicit source timezone metadata retained for forensics.

---

## 5) Strategy-Execution Robustness Matrix

| Surface | Fragility vector | Required invariants | Robustness tests | Long-horizon guarantee |
|---|---|---|---|---|
| Order generation | threshold edge cases, stale features | No order without fresh feature and regime snapshot | Property tests on boundary thresholds; stale-state fuzzing | Zero unauthorized entries under stale/invalid state |
| Signal pipeline | feature drift / missing fields | Fixed feature schema + deterministic ordering | Missing-field chaos tests; monotonicity regression checks | Signal score semantics remain stable release-to-release |
| Feature engineering | timezone/order anomalies | Monotonic event time and no forward leak | Temporal inversion tests; leak-detection tests | No look-ahead leakage in backtest/replay/live |
| Execution assumptions | fill optimism, spread shifts | Realized slippage bounded by policy by symbol class | Stress replay with widened spreads and partial fills | Fallback path preserves capital under adverse microstructure |
| Microstructure dependencies | maker fill collapse / latency spikes | Automatic route policy downgrade when friction spikes | Synthetic latency/slippage stress suite | Stable risk-adjusted execution quality despite regime shifts |

**Mandatory suite additions:** 30/90/180-day rolling replay stress packs and exchange outage drills.

---

## 6) Risk-Control Fortification Plan

### 6.1 Limit semantics and consistency

- Canonical, machine-readable risk dictionary defining each limit as:
  - `name`, `unit`, `scope`, `aggregation`, `reset_rule`, `owner`, `escalation`.
- Enforce cross-layer consistency check: strategy sizing, doctrine vetoes, and risk limits must share same exposure definitions.

### 6.2 Kill-switch and override hardening

- Deterministic kill-switch state machine with irreversible transition to `HALT` until explicit multi-party reset.
- Override policy must be explicit, time-bounded, and fully logged with actor + reason + expiry.
- Emergency exits remain available in all states; entries blocked in degraded/halt states.

### 6.3 Drift lock between risk and strategy

- Introduce **risk-strategy contract tests** in CI:
  - Any change to strategy sizing or signal fields requires risk contract approval tests.
  - Any change to risk limits requires replay showing no unintended veto class explosion.

---

## 7) Observability & Forensic-Readiness Specification

### 7.1 Event completeness standard

Every decisionable action must emit:
- correlation ID,
- deterministic sequence number,
- code SHA,
- dependency hash,
- config hash,
- parameter hash,
- input snapshot hash,
- verdict and reason.

### 7.2 Retention and audit guarantees

- **Hot retention:** 90 days query-optimized logs.
- **Warm retention:** 1 year compressed indexed logs.
- **Cold retention:** 7 years immutable archive for regulatory and forensic replay.
- Hash-chain log segments to detect tampering.

### 7.3 Monitoring and anomaly detection

Dashboards must include:
- determinism drift score,
- veto rate by reason,
- schema rejection rate,
- timestamp inversion count,
- slippage regime shifts,
- config hash mismatch incidents.

Anomaly detectors should trigger controlled degradation mode, not just alerts.

---

## 8) Governance-Durability Framework

### 8.1 Commit and review discipline

- Mandatory commit taxonomy: `risk:`, `strategy:`, `exec:`, `data:`, `ops:`, `docs:`.
- Protected branch requires:
  1. determinism gate pass,
  2. replay parity pass,
  3. risk contract pass,
  4. manifest/schema gate pass,
  5. ownership approval.

### 8.2 Ownership and change boundaries

- Declare explicit owners for each contract surface:
  - strategy model contract,
  - risk limit contract,
  - execution routing contract,
  - data schema contract,
  - observability contract.
- Any cross-boundary change requires paired sign-off from both owners.

### 8.3 Deployment reproducibility

- Release artifact must be reproducible from tagged source + lockfile + build recipe.
- Store release attestation (SBOM + build hash + signer).
- Deploy only via immutable artifacts; prohibit “live patching” on production nodes.

---

## 9) Architectural Simplification Plan

### 9.1 Simplification targets

1. Collapse overlapping routing overlays that are not actively contributing live value.
2. Reduce conviction taxonomy to minimal falsifiable policy.
3. Prune veto branches without measurable avoided-loss contribution.
4. Remove dead/disabled modules from runtime path to cut cognitive and failure surface.
5. Standardize all module interfaces to explicit typed contracts.

### 9.2 Simplification program

- **Phase 1 (2 weeks):** dead-code and disabled-path inventory, interface map, ownership map.
- **Phase 2 (4 weeks):** remove inactive overlays and duplicate transforms; keep single canonical route and sizing path.
- **Phase 3 (2 weeks):** regenerate architecture diagrams and contract docs directly from code metadata.

### 9.3 Non-negotiable simplification rule

If a layer cannot demonstrate independent, reproducible value in replay + live telemetry, it is downgraded or removed.

---

## 10) Long-Term Stability Guarantees (Ranked by Impact)

1. **Deterministic replay parity gate before deploy** (highest impact).
2. **Immutable, signed config + dependency + parameter bundles per release.**
3. **Schema-versioned ingest with quarantine and lineage guarantees.**
4. **Fail-safe operational modes with automatic degradation and deterministic recovery.**
5. **Risk-strategy contract coupling tests preventing semantic divergence.**
6. **Forensic-complete event model with hash-chain audit trail.**
7. **Strict ownership and cross-boundary review gates.**
8. **Periodic microstructure recalibration with conservative fallback routing.**
9. **Continuous docs-as-contract validation to prevent policy drift.**
10. **Architectural pruning of non-falsifiable layers to minimize entropy growth.**

---

## Immediate Implementation Backlog (30/60/90)

### 0-30 days
- Lock dependency versions completely and add hash-verified install.
- Add runtime environment fingerprint emission + mismatch refusal.
- Add deterministic replay parity job in CI for golden episodes.
- Add config digest verification at executor startup and each cycle.

### 31-60 days
- Ship schema registry and ingest quarantine channel.
- Implement idempotency keys across order intent/submit/fill lifecycle.
- Add risk-strategy contract tests and veto-causality dashboard.

### 61-90 days
- Implement dual-write event sourcing for state reconstruction.
- Complete simplification pass on inactive routing/overlay modules.
- Formalize release attestation workflow (SBOM + reproducible artifact proof).
