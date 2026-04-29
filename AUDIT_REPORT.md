# QUANT REPO AUDIT — FIRST-STRIKE DIAGNOSTIC
**Date:** 2026-04-11  
**Scope:** GPT Hedge v7.9 — Full-stack execution, risk, infra, data, governance  
**Codebase:** 82,633 LOC across 139 modules in `execution/`, 335 test files  

---

## EXECUTIVE SUMMARY

**Overall Risk Rating: MEDIUM-HIGH**

The system implements strong doctrine-first architecture with multi-layer risk gates. Kill-switch logic correctly exempts doctrine exits (Law 7). Dependency supply chain is fully pinned. Logging is forensically sound (append-only JSONL).

**However, 4 stop-trading-severity findings exist:**
1. Peak drawdown state persistence is non-atomic → corrupt after crash
2. No distributed lock → dual-executor race condition possible  
3. TWAP state not persisted → crash mid-slice causes position inflation
4. `FAIL_CLOSED_ON_NAV_STALE=0` can silently disable NAV safety gate

**System dimensions:**
| Metric | Value |
|--------|-------|
| Core executor | 6,413 LOC, 105 functions |
| Risk engine | 2,086 LOC, 40+ discrete gates |
| Total execution code | 82,633 LOC, 139 modules |
| Test files | 335 (137 unit, 183 integration) |
| Env vars controlling behavior | 25+ |
| State surfaces | 43+ JSON/JSONL files |
| Shadow/experimental modules | ~3,000 LOC (zero execution impact) |

---

## 1. CRITICAL FAILURES (Stop-Trading Severity)

### CRIT-1: Peak State Write Is Non-Atomic
**File:** [execution/drawdown_tracker.py](execution/drawdown_tracker.py#L178-L192)  
**Code:**
```python
def save_peak_state(state):
    with open(PEAK_STATE_PATH, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
```
**Risk:** Plain `open("w")` + `json.dump()` is NOT atomic. Crash during write corrupts `peak_state.json`. On restart, `load_peak_state()` reads partial JSON → drawdown calculation fails or produces wrong values. Every other state writer in the system uses atomic `tempfile` + `os.replace()` pattern — this is the sole exception.  
**Blast radius:** Portfolio drawdown circuit breaker becomes unreliable. Could allow trading through a drawdown breach.  
**Fix:** Use `tempfile.NamedTemporaryFile()` + `os.replace()` (pattern already exists in [execution/state_publish.py](execution/state_publish.py#L96-L104)).

### CRIT-2: No Distributed Lock — Dual-Executor Race Condition
**Evidence:** Zero `flock`/`fcntl`/PID-file logic in `execution/`. Supervisor config manages process lifecycle but has no mutual-exclusion enforcement.  
**Scenario:** Operator runs `supervisorctl restart` during a slow shutdown → two executor instances overlap → both read same positions → both send contradictory orders → position doubles or inverts.  
**Blast radius:** Position size 2× intended; or position flip that creates unintended exposure.  
**Fix:** PID lock file at executor startup, fail-closed if already held.

### CRIT-3: TWAP State Not Persisted Across Crashes
**File:** [execution/order_router.py](execution/order_router.py) (TWAP slice logic)  
**Risk:** TWAP splits a large order into N child slices submitted sequentially. If executor crashes after slice 2 of 4, restart has no memory of slices 1-2. New cycle may generate identical intent → sends all 4 slices again → position 1.5× intended.  
**Blast radius:** Oversized position proportional to TWAP progress at crash time.  
**Fix:** Persist TWAP parent order + completed slice IDs to disk; skip completed slices on restart.

### CRIT-4: NAV Fail-Open Knob Exists
**File:** [execution/risk_limits.py](execution/risk_limits.py#L116)  
**Code:**
```python
DEFAULT_FAIL_CLOSED_ON_NAV_STALE = os.environ.get("FAIL_CLOSED_ON_NAV_STALE", "1") != "0"
```
**Risk:** Setting `FAIL_CLOSED_ON_NAV_STALE=0` (via env or [config/risk_limits.json](config/risk_limits.json#L34)) allows trading with arbitrarily stale NAV. All position sizing, leverage, and drawdown calculations become unreliable. Only a log warning is emitted — no veto, no exception.  
**Config current value:** `true` (safe), but the knob exists and can be flipped without code change.  
**Fix:** Remove env var override. Make fail-closed unconditional. If relaxation is needed, restrict to testnet-only with hard guard.

---

## 2. HIGH-RISK DIVERGENCES (Backtest vs Live)

### DIV-1: No Backtest Engine Exists
The system has no formal backtesting framework. The only historical analysis is P6 Replay ([execution/p6_replay.py](execution/p6_replay.py)) which replays 815 episodes through frozen models — this constitutes **lookahead bias** because the expectancy model was trained on all episodes before replay begins.

**Impact:** Cannot validate strategy changes against historical data. All strategy validation is live-only (paper or real).

### DIV-2: Testnet/Live State Contamination
**Scenario:** Switching `BINANCE_TESTNET=1 → 0` without clearing `logs/state/` causes:
- Testnet positions persist as ghost entries in `positions_state.json`
- NAV cache reflects testnet balances
- Executor resumes with false position state

**No automated state cleanup on environment switch.** Test suite manually deletes state; production has no validation.

### DIV-3: Testnet Overrides Neuter Drawdown Controls
**File:** [config/risk_limits.json](config/risk_limits.json#L95-L101)
```json
"testnet_overrides": {
    "enabled": true,
    "max_nav_drawdown_pct": 0.95,     // 95% loss allowed (vs 10% live)
    "daily_loss_limit_pct": 0.50       // 50% daily loss (vs ~5% live)
}
```
**Risk:** If `BINANCE_TESTNET=1` is accidentally set in production, drawdown controls are neutered 10×. The guard (`apply_testnet_overrides()` in [execution/risk_loader.py](execution/risk_loader.py#L69-L95)) checks env var only — no secondary confirmation.  
**Fix:** Log PROMINENT startup warning. Require secondary env var `TESTNET_OVERRIDES_CONFIRM=1`.

### DIV-4: Execution Cadence Changes Regime Detection
Sentinel-X label stickiness requires N *consecutive* matching predictions. The number of consecutive matches depends on **how frequently** the executor runs, not just market conditions. A 5s cycle reaches stability faster than a 30s cycle, producing different regime labels for identical price data.

### DIV-5: Cold Start Produces Different Signals
If `sentinel_x.json` doesn't exist (fresh deploy), regime starts as UNKNOWN/uniform. First 2+ cycles produce VETO_NO_REGIME or VETO_REGIME_UNSTABLE regardless of market conditions. Warm restarts (state file exists) skip this phase.

---

## 3. DETERMINISM BREAKS

| # | Source | Severity | Location | Impact |
|---|--------|----------|----------|--------|
| D1 | Wall-clock lookback windows | HIGH | [sentinel_x.py](execution/sentinel_x.py) `prices[-48:]` | Same market, different wall-clock → different features → different regime |
| D2 | Expectancy cutoff is `time.time()` | HIGH | [expectancy_v6.py](execution/intel/expectancy_v6.py) | Running 1 hour later shifts the lookback window → different scores |
| D3 | Module-level cycle counter | MEDIUM | [sentinel_x.py](execution/sentinel_x.py) `_SENTINEL_X_CYCLE_COUNT` | Decision logic depends on how many times function called in this process |
| D4 | EMA smoothing on stale state | MEDIUM | [sentinel_x.py](execution/sentinel_x.py) probability smoothing | Cold start → uniform prior → regime transitions slower |
| D5 | Cerberus freshness collapse | MEDIUM | [cerberus_router.py](execution/cerberus_router.py) `_is_fresh()` 300s | Stale state → multipliers reset to 1.0 (fail-open) |
| D6 | NAV async update lag | MEDIUM | [nav.py](execution/nav.py) | Risk checks use 0–60s stale NAV exposure |
| D7 | Position cache 1s TTL gap | LOW | [position_cache.py](execution/position_cache.py) | Fill confirmed but stale position used for next decision |
| D8 | Retry jitter (random) | LOW | [exchange_utils.py](execution/exchange_utils.py#L699) | `random.random() * 0.1 * sleep_for` — nondeterministic retry timing |
| D9 | `uuid.uuid4()` for RUN_ID | LOW | [executor_live.py](execution/executor_live.py) | Non-reproducible run identifiers |
| D10 | Fill polling timeout (8s) | LOW | [fill_tracker.py](execution/fill_tracker.py) | Partial fills within timeout window are nondeterministic |

**Determinism verdict:** The system is **structurally nondeterministic** by design (live trading requires real-time data). However, signal generation and doctrine gating could be made deterministic with injected timestamps and frozen state — this capability does not exist today.

---

## 4. INFRA-STRATEGY COUPLING RISKS

| Infra Component | Strategy Coupling | Failure Mode | SPOF? |
|----------------|-------------------|--------------|-------|
| **Supervisor** | All 3 services (executor, sync, dashboard) | All services die together; no independent restart | YES |
| **Binance API** | Price, position, order, NAV | Trading halts; taker-only fallback for orders | YES |
| **Local filesystem** | 43+ state files, all JSONL logs | State loss on disk failure; no Firestore backup active | YES |
| **NTP/Clock** | All timestamps, staleness checks, cooldowns | Clock skew >1s causes premature/delayed vetoes | NOT ENFORCED |
| **Python process** | GIL contention (fill polling thread + main loop) | Degraded latency under CPU pressure | MEDIUM |
| **Network latency** | Order ack (50-200ms), fill polling, price fetch | Slippage increases; TWAP slices delayed | INHERENT |

**No service mesh, health endpoint, load balancer, or automatic failover.**

**Infra-Strategy Coupling Matrix:**
```
                    Binance  Disk  Clock  CPU  Network  Supervisor
Doctrine Gate         ✗       ✓      ✓    ✗      ✗        ✗
Risk Engine           ✓       ✓      ✓    ✗      ✓        ✗
Order Router          ✓       ✗      ✗    ✗      ✓        ✗
Fill Tracking         ✓       ✓      ✗    ✓      ✓        ✗
NAV Computation       ✓       ✓      ✓    ✗      ✓        ✗
State Publishing      ✗       ✓      ✓    ✗      ✗        ✗
Dashboard             ✗       ✓      ✗    ✗      ✗        ✓

✓ = coupled (failure in infra affects strategy component)
```

---

## 5. DATA CONTAMINATION RISKS

| # | Risk | Severity | Location |
|---|------|----------|----------|
| C1 | **Testnet/live state bleed** — switching env without clearing `logs/state/` | HIGH | All state files |
| C2 | **P6 Replay lookahead bias** — model trained on all episodes before replay | MEDIUM | [p6_replay.py](execution/p6_replay.py) |
| C3 | **Dual NAV surface desync** — `nav_confirmed.json` stops updating on API failure while `nav.json` continues | MEDIUM | [nav.py](execution/nav.py) |
| C4 | **Orphaned TP/SL entries** — closed positions leave stale TP/SL in registry | LOW | [position_ledger.py](execution/position_ledger.py) |
| C5 | **EWMA slippage model never expires** — 30-day-old assumptions persist | LOW | [slippage_model.py](execution/slippage_model.py) |
| C6 | **Kline cache 600s staleness** — historical bars can be 10 minutes old | LOW | Exchange utils kline fetching |
| C7 | **Metrics EWMA accumulates across regimes** — quality metrics blend trend/range data | LOW | Execution quality tracking |
| C8 | **Episode ledger V1/V2 schema mismatch** — field name variants cause silent data loss | LOW | [position_ledger.py](execution/position_ledger.py) |

**Replay feasibility verdict: 60-70%.** Can reconstruct decision path from JSONL logs, but cannot reproduce identical decisions without: (a) per-tick regime snapshots (not archived), (b) frozen orderbook state, (c) deterministic timestamps.

---

## 6. RISK-CONTROL GAPS

### Enforced Risk Controls (40+ gates)

| Control | Hard Veto | Configurable | Tested |
|---------|-----------|-------------|--------|
| Doctrine entry (regime, direction, crisis) | YES | NO (hardcoded) | Partial |
| Kill switch (entries only) | YES | ENV | NO |
| NAV freshness (90s) | YES | YES (**see CRIT-4**) | YES |
| Portfolio drawdown circuit | YES | YES | NO |
| Correlation group caps | YES | YES | NO |
| Per-trade NAV % | YES | YES | NO |
| Per-symbol notional cap | YES | YES | NO |
| Min notional floor | YES | YES | NO |
| Symbol cooldown | YES | YES | NO |
| Fee-aware edge gate | YES | YES | NO |
| Conviction band gate | YES | YES | NO |
| Strategy attribution gate | YES | YES | NO |
| Churn guard | YES | YES | NO |
| Notional inflation guard | YES | YES | NO |

### Missing Risk Controls

| Gap | Impact | Priority |
|-----|--------|----------|
| **No max absolute position size** | Single order uncapped beyond per-trade NAV % | HIGH |
| **No leverage × notional cross-check** | Leverage not verified against gross exposure cap | HIGH |
| **No order queue depth limit** | 3 TWAP fills + new order not bounded | MEDIUM |
| **No dual-executor detection** | Race condition possible (see CRIT-2) | HIGH |
| **No position concentration by asset class** | Multiple BTC-correlated pairs not grouped | LOW |
| **VaR/CVaR fail-open on ImportError** | If scipy missing, VaR gate silently skipped | LOW |
| **No liquidation price proximity check** | Position can approach liquidation before any gate fires | MEDIUM |

### Kill Switch Audit Gap
Kill switch exemption for doctrine exits logs `LOG.warning()` but does NOT emit a veto event to `doctrine_events.jsonl`. This makes kill-switch exemptions harder to audit forensically.

### Risk Test Coverage: <10% of Gates
Only 2 end-to-end tests in `test_risk_limits.py` (both regression tests for `constraint_geometry` field). 38+ gates have zero dedicated test coverage.

---

## 7. REPRODUCIBILITY GAPS

### What's Captured (Strong)
- ✅ All order executions → `orders_executed.jsonl`
- ✅ All risk vetoes → `risk_vetoes.jsonl` (symbol, reason, thresholds, observations)
- ✅ All doctrine verdicts → `doctrine_events.jsonl`
- ✅ DLE shadow events → `dle_shadow_events.jsonl`
- ✅ Episode lifecycle → `positions_ledger.json` (entry regime, exit reason)
- ✅ Engine metadata → `engine_metadata.json` (git commit, python version)
- ✅ Sizing snapshots → `sizing_snapshot.jsonl`

### What's Missing (Gaps)
| Missing Artifact | Impact | Priority |
|-----------------|--------|----------|
| **Regime snapshot per decision** | Cannot reconstruct why regime was X at time T | HIGH |
| **Orderbook depth at order time** | Cannot calculate true slippage vs expected | MEDIUM |
| **Second-resolution NAV** | Only 60s snapshots; gap between decisions | MEDIUM |
| **Signal strength/conviction at decision** | Hydra state is ephemeral (overwritten each cycle) | MEDIUM |
| **Environment variable snapshot per run** | `engine_metadata.json` captures some; not all 25+ vars | LOW |
| **Config file checksum per cycle** | Config changes between cycles not detected | LOW |

### Reproducibility Score: 3/5
Can reconstruct *what happened* (decisions, fills, vetoes). Cannot reconstruct *why* with full fidelity (regime features, orderbook state, exact timestamps).

---

## 8. GOVERNANCE WEAKNESSES

### Deployment Pipeline (Manual)
```
Dev (DRY_RUN=1) → pytest → ruff → mypy → Code Review → SSH → git pull → supervisorctl restart
```
**No automated deployment.** Production changes require SSH access and manual commands.

### Review Gates
| Gate | Status | Gap |
|------|--------|-----|
| Full test suite (335 tests) | ✅ CI enforced | Risk gates <10% covered |
| Lint (ruff) | ✅ CI enforced | — |
| Type check (mypy) | ✅ CI enforced | Many `ignore_errors` |
| Schema version alignment | ✅ CI enforced | — |
| Import boundary (execution ↛ dashboard) | ✅ CI enforced | — |
| Manual code review | ✅ Required | — |
| Automated deployment | ❌ Missing | Manual SSH |
| Staging environment | ❌ Missing | Testnet only |
| Rollback automation | ❌ Missing | Manual git revert |
| Config change validation | ❌ Missing | No schema enforcement |

### Blast Radius of Bad Commit
**High.** A single `git pull` + `supervisorctl restart` deploys all changes atomically. No canary deployment, no feature gates for core execution logic (only for experimental modules via `v6_flags.py`). Rollback is manual `git revert` + restart.

### Config Drift Vector
All configs in `config/` are JSON/YAML files. No runtime schema validation at load time. A typo in `risk_limits.json` (e.g., `"max_nav_drawdown_pct": 10` instead of `0.10`) is caught by `normalize_percentage()` for some keys but NOT all. No checksum or diff logging on config reload.

---

## 9. NON-OBVIOUS FAILURE MODES

### F1: Dual NAV Surface Desync → Auto Kill-Switch
**Path:** Binance API degrades → `nav_confirmed.json` stops updating (conditional write) → `nav.json` continues updating (unconditional) → risk engine reads confirmed cache → age > 90s → NAV_STALE veto → all entries blocked → system self-halts even though executor is healthy.  
**Detection:** Only visible in risk veto logs. Dashboard shows executor status as "running."

### F2: Silent Gate Skip on ImportError
**Affected gates:** Fee-aware edge gate, conviction band gate, VaR/CVaR gate.  
**Pattern:**
```python
try:
    from execution.fee_edge_gate import check_edge
except ImportError:
    check_edge = None  # Gate silently skipped
```
**Risk:** If a dependency is missing or renamed, safety gates disappear without error.

### F3: Correlation Cap Single-Group Truncation
**File:** [execution/risk_limits.py](execution/risk_limits.py) correlation check loop  
**Issue:** `break` after first violated group. If 3 correlated groups are breached, only the first is reported. Operator sees "correlation_cap[BTC_GROUP]" but misses ETH_GROUP and SOL_GROUP violations.

### F4: Position Cache Invalidation Race
**Sequence:** Fill confirmed (t=0) → cache invalidated (t=0.1s) → cache TTL 1.0s → next read at t=0.5s → returns stale pre-fill position → doctrine exit check uses wrong position size.  
**Impact:** Unlikely but possible: exit order sized for wrong position.

### F5: Clock Jump → Mass Staleness Veto
If system clock jumps forward 60+ seconds (NTP correction, VM migration), all staleness checks fire simultaneously: NAV stale, regime stale, peak state stale. All trading halts until the next successful data refresh cycle.

### F6: Disk Saturation → Silent Log Loss
JSONL append fails silently on disk full. Fill records not persisted → executor believes no position exists → exchange has open position → desync. Only a disk warning is emitted (no halt).

### F7: Config Parse Failure → Stale Config Continues
```python
try:
    cfg = load_risk_config()
except Exception:
    LOG.warning("[risk] config refresh failed")
    return  # Proceeds with STALE config from previous cycle
```
A malformed `risk_limits.json` doesn't crash the executor — it silently continues with whatever config was previously loaded.

### F8: INTENT_TEST=1 in Production
Setting `INTENT_TEST=1` hard-codes a `BUY BTCUSDT` intent every cycle, bypassing the screener. If accidentally set in production, executor continuously attempts to buy BTC.

---

## 10. RECOMMENDED REMEDIATION (Ranked by Impact)

| Priority | Fix | Effort | Files |
|----------|-----|--------|-------|
| **P0** | Atomic peak state write (tempfile + os.replace) | 15 min | [drawdown_tracker.py](execution/drawdown_tracker.py#L178) |
| **P0** | PID lock file at executor startup | 30 min | [executor_live.py](execution/executor_live.py) |
| **P0** | Persist TWAP slice state to disk | 2 hr | [order_router.py](execution/order_router.py) |
| **P0** | Remove or restrict `FAIL_CLOSED_ON_NAV_STALE` to testnet-only | 30 min | [risk_limits.py](execution/risk_limits.py#L116) |
| **P1** | Add JSON schema validation for all config files at startup | 2 hr | [risk_loader.py](execution/risk_loader.py) |
| **P1** | Add testnet-override startup warning + secondary confirmation | 30 min | [risk_loader.py](execution/risk_loader.py#L69) |
| **P1** | State directory cleanup/validation on env switch | 1 hr | [executor_live.py](execution/executor_live.py) |
| **P1** | Write 20+ unit tests for uncovered risk gates | 4 hr | tests/unit/ |
| **P2** | NAV cache unconditional writes with sources_ok flag | 1 hr | [nav.py](execution/nav.py) |
| **P2** | Add NTP clock skew guard at startup (fail-closed if >5s) | 15 min | [executor_live.py](execution/executor_live.py) |
| **P2** | Staleness guards on intel surfaces (score, expectancy, router health) | 1 hr | Intel modules |
| **P2** | Kill switch doctrine-exit audit events | 30 min | [executor_live.py](execution/executor_live.py) |
| **P2** | Correlation cap — report all violated groups (remove `break`) | 15 min | [risk_limits.py](execution/risk_limits.py) |
| **P3** | Disk pressure → halt trading (not just warn) | 30 min | [executor_live.py](execution/executor_live.py) |
| **P3** | Archive regime snapshots per decision (not just current) | 2 hr | [sentinel_x.py](execution/sentinel_x.py) |
| **P3** | Extract DLE shadow + Binary Lab to separate daemons | 4 hr | Reduce executor by ~3,000 LOC |
| **P3** | Containerize with Docker for reproducible deploys | 8 hr | New Dockerfile |
| **P4** | Implement max absolute position size limit | 1 hr | [risk_limits.py](execution/risk_limits.py) |
| **P4** | Implement liquidation price proximity check | 2 hr | [risk_limits.py](execution/risk_limits.py) |
| **P4** | GitOps deployment (replace manual SSH) | 8 hr | New CI/CD pipeline |

---

## APPENDIX A: Order-Generation DAG (27 Steps)

```
Signal Generation (300s cadence)
  → Hydra Multi-Head Merge (6 strategies)
    → ECS Candidate Selection (conviction ranking)
      → DOCTRINE GATE ★ (regime, direction, crisis — SUPREME AUTHORITY)
        → DLE Shadow Hook (observation only)
          → Churn Guard (60s cooldown)
            → Strategy Attribution Gate
              → Conviction Band Gate
                → KILL SWITCH (entries only; doctrine exits exempt)
                  → NAV Fetch + Sizing ★
                    → Flip Detection (hedge-mode cleanup)
                      → RISK ENGINE ★ (40+ gates)
                        → Pipeline V6 Shadow (observation only)
                          → Price Fetch (live)
                            → Fee-Aware Edge Gate
                              → Sizing Snapshot (audit)
                                → Order Payload Build (qty normalization)
                                  → Notional Inflation Guard
                                    → ORDER ROUTER ★ (maker-first, TWAP)
                                      → Exchange Dispatch + Retry
                                        → Order ACK
                                          → Fill Polling (async, 0-8s)
                                            → Position Cache Invalidation
                                              → Execution Quality Tracking
                                                → TP/SL Registration
                                                  → Risk State Update
```

★ = primary decision gates

## APPENDIX B: Environment Variables Controlling Behavior

| Variable | Default | Risk Level | Effect |
|----------|---------|-----------|--------|
| `DRY_RUN` | (fails if unset in prod) | CRITICAL | 1=paper, 0=live orders |
| `BINANCE_TESTNET` | 0 | HIGH | 1=testnet API + relaxed limits |
| `FAIL_CLOSED_ON_NAV_STALE` | 1 | HIGH | 0=silently allows stale NAV |
| `KILL_SWITCH` | 0 | MEDIUM | 1=halt new entries |
| `FORCE_REGIME` | (unset) | MEDIUM | Override Sentinel-X regime (testnet only) |
| `INTENT_TEST` | 0 | MEDIUM | 1=hard-code BUY BTCUSDT |
| `NAV_FRESHNESS_SECONDS` | 90 | MEDIUM | Staleness threshold (seconds) |
| `ENV` | (unset) | MEDIUM | dev/prod safety guard |
| `SCREENER_INTERVAL` | 300 | LOW | Signal generation cadence (seconds) |
| `ORDER_FILL_POLL_TIMEOUT` | 8.0 | LOW | Max fill wait (seconds) |
| `EXEC_MAX_TRANSIENT_RETRIES` | 1 | LOW | Retry budget for transient errors |
| `EXTERNAL_SIGNAL` | 0 | LOW | 1=skip internal screener |

## APPENDIX C: Complexity Budget

| Module | LOC | Execution Impact | Recommendation |
|--------|-----|-----------------|----------------|
| `executor_live.py` | 6,413 | CORE — all execution | Extract shadow layers |
| `risk_limits.py` | 2,086 | CORE — all risk gates | Add test coverage |
| `order_router.py` | 2,033 | CORE — order routing | Add TWAP persistence |
| `hydra_engine.py` | 1,728 | CORE — signal generation | Stable |
| `sentinel_x.py` | 1,447 | CORE — regime detection | Archive snapshots |
| `cerberus_router.py` | 1,400 | **OBSERVATION ONLY** | Consider deferral |
| `minotaur_engine.py` | 1,112 | OBSERVATION (slippage tracking) | Consider deferral |
| `dle_shadow.py` | 425 | **OBSERVATION ONLY** | Defer to Phase D |
| `binary_lab_*.py` | ~1,500 | **EXPERIMENTAL** | Extract to daemon |
| `p6_replay.py` | ~750 | **OFFLINE ONLY** | Extract to CLI tool |

**Reducible complexity:** ~3,800 LOC of shadow/experimental code in the executor hot path adds maintenance burden and import-time latency with zero execution impact.

---

*End of audit. All findings tied to specific code paths. Remediation ranked by blast radius and implementation effort.*
