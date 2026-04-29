# Catastrophic-State Diagnostic — Third-Strike Quant Repository Audit

**Date:** 2026-04-11  
**Scope:** Full repository integrity/risk reproducibility under degraded-state assumptions  
**Method:** Static structure review + config/path consistency checks + quality gate execution (`pytest`, `ruff`)

---

## 1) Structural Integrity Failures (Integrity-Collapse Map)

1. **Manifest contract is currently broken in baseline test environment.** Integration tests expecting a clean manifest fail with 26 required artifacts missing, meaning repository-state assumptions are not self-contained and fail-closed checks trip immediately.  
   - Impact: impossible to assert structural completeness from a clean clone.
2. **Manifest audit enforces runtime-generated state artifacts as “required”.** Required entries include mutable log/state files under `logs/`, but those files are not generated at bootstrap in CI/local ephemeral runs.  
   - Impact: correctness checks conflate “source integrity” and “runtime side effects”.
3. **Hard-coded absolute paths in launch scripts (`/root/hedge-fund`) create environment coupling.** Runtime launcher scripts are host-specific and bypass repository-relative execution guarantees.  
   - Impact: portability collapse across dev/CI/containers; high chance of stale code execution from wrong checkout.
4. **Repository has active architecture drift indicators.** Release metadata claims branch/head/status values that no longer match current HEAD, branch, or lifecycle phase.  
   - Impact: operator trust degradation; runbooks can direct actions against obsolete assumptions.
5. **Large unresolved lint debt with correctness-level findings.** Static analysis flags include undefined/redefined symbols and broad import/ordering violations across execution and test code paths.  
   - Impact: latent runtime surprises and maintenance non-determinism.

---

## 2) Execution Reliability Failures (Degraded-Execution Failure Map)

1. **Executor can run live order mode by default when `DRY_RUN` is unset outside `ENV=prod`.** `DRY_RUN` resolves false by default (`truthy_env(..., default="0")`), and the explicit safety hard-stop is only enforced for `ENV=prod`.  
   - Impact: unintended live-send behavior in mis-labeled environments.
2. **Critical dependency loading can fail silently.** Optional imports / config loads contain broad exception swallowing paths, reducing observability when runtime context is incomplete.
3. **Order path carries complex fallback logic with partial guard patterns.** Guarding against missing `_load_strategy_config` is dynamic and implicit; when config loading path changes, behavior can diverge by environment.
4. **Execution quality gates are not green.** `pytest` (fast profile) currently fails on manifest integrity assertions; “degraded” is effectively baseline.

---

## 3) Strategy Reliability Risks (Uncertainty-Logic Risk Map)

1. **Strategy gating depends on external config availability with weak explicit validation at process start.** Conviction-band gating attempts fallback file reads inline during order path rather than validated immutable config snapshot.
2. **Feature/context availability is not uniformly fail-closed.** Multiple execution paths use permissive defaults and soft fallbacks, allowing strategy output to continue under partial context loss.
3. **Signal/risk interaction complexity is high and centralized in monolithic executor runtime.** A single very large orchestration module increases hidden coupling between attribution, vetoes, routing, and exits.

---

## 4) Infrastructure Degradation Risks (Infra-Degradation Coupling Map)

1. **Launcher scripts encode fixed filesystem and virtualenv assumptions.** This couples operational correctness to one host layout.
2. **Dependency policy permits drift for key scientific packages (`>=` pins).** Mixed strict pins + floating ranges across core quant stack undermine reproducibility parity between environments.
3. **Quality baseline is not enforceable with current lint posture.** Thousands of lint findings (including correctness-level classes) indicate infra checks are not an active merge gate.

---

## 5) Data Lineage Breaks (Lineage-Breakdown Report)

1. **Manifest lineage model expects persistent log/state artifacts without deterministic bootstrap contract.** Data lineage cannot be reconstructed from repo-only state when required outputs are runtime-generated.
2. **Timezone/timestamp conventions are inconsistent in tooling.** Both timezone-aware (`datetime.now(timezone.utc)`) and legacy naive UTC patterns (`datetime.utcnow()`) are present.
3. **Loader error handling can hide malformed data inputs.** Core JSON loaders return `{}` on any exception, blurring distinction between “missing”, “corrupt”, and “empty”.

---

## 6) Risk-Control Misalignments (Misalignment-Risk Matrix)

1. **Percent normalization semantics can invert operator intent.** Risk config values above `1.0` are interpreted as percentages and divided by 100; e.g., `max_gross_exposure_pct: 1.5` normalizes to `0.015`.
   - Misalignment risk: config author may intend 150% gross, engine enforces 1.5% gross.
2. **Risk policy spread across multiple files with merge-time overrides (risk + pairs + env).** Effective limits are composited at runtime, increasing audit burden and mismatch probability.
3. **Testnet override logic mutates global risk fields dynamically.** Environment switch changes critical limits at load time; provenance is not strongly versioned in a single immutable artifact.

---

## 7) Reproducibility Barriers (Reproducibility-Collapse Ledger)

1. **Release status document is stale against current repository state.** Recorded HEAD/branch/system-status claims conflict with present git metadata and declared runtime phase comments.
2. **Runtime outputs are central to “truth”, but not reproducibly generated in clean environments.** Tests indicate expected state files are absent unless prior runtime processes have populated them.
3. **No single canonical frozen environment lock for the active code path.** Top-level dependency spec is partly floating; archive freeze files exist but are not clearly authoritative.

---

## 8) Governance Fragmentation Points (Governance-Fragmentation Map)

1. **Operational documentation drift:** release/certification status text describes an active window while runtime config comments indicate termination.
2. **Change-control ambiguity:** historical references to specific synced commits in status docs are disconnected from current branch evolution.
3. **Ownership/contract ambiguity at manifest boundary:** required-vs-optional artifact policy is not aligned with test/bootstrap lifecycle.

---

## 9) Systemic Weaknesses Preventing Safe Operation

1. **Structural gates are failing at baseline** (manifest enforcement not green).  
2. **Execution can enter live-send paths under incomplete environment intent declaration.**  
3. **Risk semantics are vulnerable to unit/scale mismatch at config boundaries.**  
4. **Lineage and reproducibility depend on pre-existing runtime byproducts, not deterministic bootstrap.**  
5. **Governance documents cannot currently be treated as a reliable source of operational truth.**

---

## 10) Mandatory Stabilisation Steps (Ranked by Urgency)

### P0 — Immediate (block live trading until complete)
1. **Enforce hard startup interlock:** abort executor unless explicit `DRY_RUN` is set in *all* environments except a whitelisted production profile with signed release metadata.
2. **Fix manifest contract:** either (a) auto-bootstrap required files deterministically, or (b) reclassify runtime-generated files as optional until generated; make `test_manifest_audit` green in clean checkout.
3. **Lock risk units contract:** require explicit `%` vs fraction schema fields, reject ambiguous values at load, and add startup fatal checks on normalized ranges.
4. **Freeze deployment paths:** remove hard-coded `/root/hedge-fund` in launchers; enforce repo-root discovery and explicit env/venv validation.

### P1 — Near-term (stability + determinism)
5. **Introduce immutable runtime config snapshot at startup** (hash + persisted echo + strict schema validation).
6. **Replace silent exception swallowing in critical loaders/imports** with typed errors + structured telemetry.
7. **Standardize timestamp contract** (timezone-aware ISO-8601 UTC only) and lint for `utcnow()` usage.
8. **Reduce executor monolith risk:** isolate order formation, risk gating, and dispatch into separately testable modules with contract tests.

### P2 — Governance hardening
9. **Regenerate release status from code/state automation** (no manual status assertions).
10. **Establish mandatory CI gates:** lint (selected correctness rules), manifest integrity in clean env, config schema validation, and reproducibility smoke replay.
11. **Create ownership matrix for each manifest artifact** (producer, SLO, fallback, escalation path).

---

## Evidence Anchors

- Manifest audit tests require zero missing required files and currently fail under clean execution assumptions.  
- Manifest audit logic marks required paths as violations when missing on disk.  
- Launch scripts hard-code `/root/hedge-fund` paths.  
- Runtime safety behavior for `DRY_RUN` and prod-only explicitness check.  
- Risk normalization logic and ambiguous high-level config values.  
- Documentation drift between release status metadata and current repository metadata/runtime comments.
