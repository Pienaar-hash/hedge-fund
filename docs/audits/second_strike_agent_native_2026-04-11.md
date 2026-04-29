# Second-Strike Agent-Native Red-Team Audit (2026-04-11)

## Scope
Hostile diagnostic against execution, determinism, infra coupling, data integrity, risk controls, reproducibility, and governance.

## Method (reproducible probes)

### Baseline regression checks
- `pytest -q tests/unit/test_determinism_guard.py tests/integration/test_binary_lab_replay_determinism.py tests/unit/test_position_cache.py tests/unit/test_risk_limits.py tests/unit/test_kill_switch_doctrine_exit.py tests/integration/test_signal_pipeline.py tests/integration/test_exchange_utils_errors.py`
- `pytest -q tests/unit/test_order_dispatch.py tests/unit/test_exchange_thread_safety.py tests/unit/test_fill_tracker.py`

### Adversarial fault-injection probes
- Missing config fallback:
  - monkeypatch `execution.signal_screener._load_strategy_list` to raise `FileNotFoundError`, then call `generate_signals_from_config()`.
- Stale NAV fail-open bypass:
  - invoke `execution.risk_limits.enforce_nav_freshness_or_veto()` with stale/invalid NAV (`age=9999`, `sources_ok=False`) and `fail_closed_on_nav_stale=False`.
- Clock-jump dedupe bypass:
  - seed `execution.signal_screener._DEDUP_CACHE`, run `_dedupe_prune()` at increasing synthetic times (normal then jump).
- Out-of-order bar sensitivity:
  - run `_zscore()` with same values in different order to show signal drift when sequence is corrupted.

---

## 1) Execution Failure Matrix

| Injected condition | Injection method | Observed behaviour | Classification |
|---|---|---|---|
| Missing/invalid strategy config | `_load_strategy_list` forced to raise in `generate_signals_from_config` | Returns `IntentBatch([], 0)` and only prints error; trading signal stream goes empty (silent fail-safe degrade, no hard-stop). | **Confirmed break** |
| Exchange rejects / transient HTTP errors | Existing retry path in `dispatch_with_retry` | Retries are enabled and tests pass; no immediate crash observed in covered paths. | Covered but still high-impact if exchange semantics drift |
| Malformed market rows | `get_klines` parser drops malformed rows (`continue`) | Bad rows are silently discarded; downstream signal quality depends on remaining sample count. | **Probable break** (silent data-quality degradation) |
| Stale NAV + fail-open config | `enforce_nav_freshness_or_veto(..., fail_closed_on_nav_stale=False)` | Function returned `True` despite stale NAV and `sources_ok=False`, allowing order path to proceed. | **Confirmed break** |
| Partial fills | Existing unit coverage (`test_fill_tracker`, `test_order_dispatch`) | Tested paths pass; no immediate crash in covered scenarios. | No confirmed break in covered harness |
| Stale cache entries | `PositionCache` TTL cache | Cache is intentionally permissive and single-thread-assumed; staleness bounded only by TTL + invalidation discipline. | **Probable break** under infra lag |

---

## 2) Determinism Violation Log

| Condition | Code path | Repro evidence | Impact |
|---|---|---|---|
| Clock drift / jumps | `_DEDUP_CACHE` + `_dedupe_prune(now_ts)` in `signal_screener` | Cache entry exists at `now=1050`, disappears at `now=2000` synthetic jump. Same logical candle can re-emit after time jump. | **Confirmed determinism violation** |
| Out-of-order bar sequences | `closes=[row[4] for row in kl]` + `_zscore` | Same value set reordered produced materially different z-score (`1.647509` vs `0.086711`). | **Confirmed determinism violation** |
| Global mutable state | Module global `_DEDUP_CACHE`; global singleton `POSITION_CACHE` | Behaviour depends on process lifetime and call ordering; restart boundary changes outcomes. | **Probable violation** |
| Missing data paths | multiple `except: pass` in screener preloads | Identical intended inputs can diverge based on transient loader errors that are swallowed. | **Probable violation** |

---

## 3) Infra-Strategy Coupling Failures

1. **Wall-clock is a control input**: dedupe TTL and pruning are time-based, so scheduler jitter/clock correction changes emission behaviour.
2. **Process lifetime affects stateful gates**: in-memory dedupe/cache reset on restart can alter live decisions vs replay.
3. **Fail-open risk mode can be toggled by config/env**, directly changing execution under identical market state.

Classification: **Confirmed coupling**.

---

## 4) Data Integrity Breach Report

1. **Malformed tick/bar rows are dropped silently** in `exchange_utils.get_klines` with no quality threshold/alert.
2. **Timestamp ordering is not validated** before indicators consume close sequence; out-of-order arrival can corrupt RSI/z-score/trend inputs.
3. **Screener continues on broad exceptions**, causing selective symbol blackouts without batch-failure semantics.

Classification: 1 and 2 are **confirmed vulnerability patterns**; 3 is **confirmed silent degradation path**.

---

## 5) Risk-Control Penetration Test

| Attack | Result |
|---|---|
| Stale NAV + disabled fail-closed (`fail_closed_on_nav_stale=False`) | `enforce_nav_freshness_or_veto` returned `True` with stale NAV; risk freshness gate can be bypassed by config. (**Confirmed bypass**) |
| Kill-switch env flip | `KILL_SWITCH` checked as env string in screener; operationally effective but mutable at runtime by environment control plane. |
| Rapid limit pressure | Existing tests pass for limit logic, but many preload failures are best-effort and swallowed, creating potential mismatch between perceived and real risk context. |

---

## 6) Reproducibility Failure Ledger

1. **Environment-dependent behaviour** (env vars for stale NAV handling, kill switch, feature flags).
2. **Runtime-local mutable caches** (`_DEDUP_CACHE`, `POSITION_CACHE`) are not persisted/replayed, so replay parity can drift.
3. **Silent exception swallowing** changes effective feature set based on transient IO/import errors.

Classification: **Probable to confirmed**, depending on exact deployment controls.

---

## 7) Governance Attack Surface Map

1. **Single config bit can alter risk posture** (`fail_closed_on_nav_stale`).
2. **Broad fallback patterns (`except: pass`) reduce auditability** because control-path decisions are not always explicitly vetoed/escalated.
3. **In-memory controls without immutable audit trail** (dedupe/cache lifecycle) increase difficulty proving decision lineage.

---

## 8) Consolidated Red-Team Findings

### 1. Confirmed Breaks (exploitable failures)
- Missing config can zero out intents without hard stop (`generate_signals_from_config` returns empty batch).
- Stale NAV freshness gate can be bypassed by fail-open config.
- Dedupe behaviour is clock-jump sensitive; identical logical event can be re-admitted.

### 2. Probable Breaks (high-risk behaviours)
- Silent malformed-row dropping without batch quality guard.
- Process-lifetime/global-state coupling across dedupe/cache.
- Best-effort loader failures create implicit mode shifts.

### 3. Silent Divergences (backtest vs live)
- Live mode depends on env variables and wall-clock; replay often assumes stable deterministic ordering.
- Restart boundaries reset in-memory guards.

### 4. Determinism Violations
- `_dedupe_prune` wall-clock dependency.
- Indicator outputs diverge under bar reordering.

### 5. Data Integrity Breaches
- No strict timestamp monotonicity enforcement before signal math.
- Malformed row discard with no hard alert threshold.

### 6. Risk-Control Bypass Points
- `fail_closed_on_nav_stale=False` permits stale NAV execution path.

### 7. Infra-Strategy Coupling Failures
- Scheduler/clock/process lifecycle directly alter trading behaviour.

### 8. Governance Weak Points
- Mutable defaults/env knobs can alter live posture quickly.
- Insufficiently explicit logging/escalation on fallback paths.

### 9. Highest-Impact Attack Paths
1. **Config drift attack**: set stale-NAV gate to fail-open + induce NAV feed lag.
2. **Clock control/drift attack**: force dedupe expiry and duplicate emissions.
3. **Data feed poisoning**: inject malformed/out-of-order bars to skew indicators while avoiding hard failures.

### 10. Mandatory Remediations (ranked)
1. **Critical:** force fail-closed for stale NAV in prod; require signed override with explicit expiry.
2. **Critical:** validate timestamp monotonicity and minimum sample quality before indicator calculation.
3. **High:** replace broad `except: pass` with explicit typed exceptions + structured veto reasons.
4. **High:** migrate dedupe/cache timing to monotonic clock + persist replay-relevant state.
5. **High:** add deterministic replay harness that reuses exact arrival order and fault events.
6. **Medium:** emit governance-grade audit events for every fallback and runtime flag resolution.

