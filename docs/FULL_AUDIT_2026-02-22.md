# Full‑Stack Audit: Test Suite + Executor + VM Split

**Date:** 2026-02-22  
**Scope:** 281 test files · 5,510-line executor · two-VM architecture design  
**Commit:** `db514766` (`v7.6-dev`)

---

## Part 1 — Test Suite Audit

### 1.1 Inventory

| Category | Files | Tests | Time | Runner |
|----------|------:|------:|-----:|--------|
| `tests/unit/` | 85 | 1,906 | 41s | `pytest -m unit` |
| `tests/integration/` | 189 | 1,582 | 33s | `pytest -m integration` |
| `tests/legacy/` | 5 | 0 (all skipped) | <1s | — |
| `tests/dashboard/` | 1 | ~10 | <1s | — |
| `tests/` (root) | 1 | 5 | <1s | — |
| **Total** | **281** | **3,493+** | **~81s** | `pytest -q` |

**Conftest files:** 1 (root only — `tests/conftest.py`)  
**Cross-test imports:** 0 (clean isolation)  
**Mutable global state in tests:** 0  

### 1.2 Conftest: Root Fixtures

| Fixture | Scope | Autouse | Purpose |
|---------|-------|:-------:|---------|
| `_reset_global_state` | function | **Yes** | Resets ENV=test, telegram rate-limits, diagnostics_metrics, conviction_engine cache |
| `mock_clean_drawdown_state` | function | No | Mocks `_drawdown_snapshot` to prevent stale `peak_state.json` leaks |
| `mock_empty_nav_history` | function | No | Mocks `_nav_history_from_log` to block portfolio-DD breaker |
| `reset_telegram_state` | function | No | Clears `_send_timestamps` + `_recent_msgs` on telegram_utils |

**Missing conftest coverage:** No per-directory conftest in `unit/` or `integration/`. All fixtures live in root.

### 1.3 Classification Matrix

#### Markers in Use

| Marker | Files Decorated | Behavior |
|--------|:-:|---------|
| `@pytest.mark.unit` | 12 | Fast, pure in-process |
| `@pytest.mark.integration` | 33 | Multi-module, may touch FS |
| `@pytest.mark.runtime` | 6 | Reads live state files |
| `@pytest.mark.legacy` | 5 | v5/v6 holdovers (all skipped) |
| `@pytest.mark.skip` | 12 | Disabled — API changed |
| `@pytest.mark.xfail` | 1 | `test_firestore_publish` stub mismatch |
| `@pytest.mark.parametrize` | 18 | Data-driven tests |

#### Classification Gap

**~230 test files have NO marker at all.** They run in every `pytest` invocation regardless of `-m` filter. This is the primary obstacle to selective/parallel execution.

**Recommended reclassification:**
- Files in `tests/unit/` without `@pytest.mark.unit` → add marker (73 files)
- Files in `tests/integration/` without `@pytest.mark.integration` → add marker (156 files)
- 12 skipped files → evaluate for deletion or restoration

### 1.4 Execution Timing — Slowest Tests (Top 10)

| Test | Time | Category | Cause |
|------|-----:|----------|-------|
| `test_episode_ledger::test_build_from_actual_log` | 5.0s | unit | Parses large JSONL file |
| `test_episode_ledger_e2::test_churn_pattern_reconciliation` | 3.5s | unit | E2 ledger reconciliation |
| `test_episode_ledger_e2::test_exits_before_entries_are_orphans` | 3.2s | unit | Orphan-fill detection |
| `test_episode_ledger_e2::test_multi_fill_exit` | 2.9s | unit | Multi-fill parsing |
| `test_episode_ledger_e2::test_multi_entry_single_exit` | 2.9s | unit | Multi-entry parsing |
| `test_episode_ledger_e2::test_clean_reconciliation` | 2.8s | unit | Full reconciliation |
| `test_order_router_ack::test_route_order_ack_preserves_status` | 1.0s | integration | Router mock overhead |
| `test_router_failure_logging::test_route_order_failure_records_error` | 0.9s | integration | Router mock |
| `test_reset_guard::test_trigger_after_debounce_expires` | 0.9s | unit | `time.sleep(0.9)` debounce |
| `test_router_smoke::test_route_order_reduce_only` | 0.9s | integration | Router mock |

**Observation:** The top 6 slowest tests are ALL episode ledger tests (18.3s combined). The remaining ~3,480 tests finish in ~63s — avg 18ms each.

### 1.5 Brittleness & Flakiness Analysis

#### Timing-Dependent Tests (Risk: Medium)

| File | Issue |
|------|-------|
| `test_reset_guard.py` | `time.sleep(0.9)` for debounce expiry — wall-clock dependent |
| `test_confirm_fill_*.py` | Mock chains for `_FILL_POLL_INTERVAL` — tightly coupled to impl |

**Real `time.sleep` in tests:** Only 1 file (`test_reset_guard.py`, 0.9s). All others use mocked time.

#### Filesystem-Touching Without `tmp_path` (Risk: Low-Medium)

16 files reference `Path()` / `open()` without `tmp_path`. Breakdown:
- **6 runtime-marker tests** that read real `logs/state/` files — intended behavior, only run with `@pytest.mark.runtime`
- **5 config-reading tests** that read `config/*.json` — stable, read-only
- **5 path-reference tests** that use `Path()` for constants — no actual I/O

**Verdict:** No real filesystem leak risk. The runtime-marker tests are correctly gated.

#### External Service Coupling (Risk: Low)

29 files import exchange/telegram/firestore modules. **All are properly mocked** except:
- `test_symbol_routes.py` — imports `exchange_utils` with potentially live path (low risk, module-level mock covers it)
- `test_firestore_publish.py` — marked `xfail` (known issue)

#### Shared-State Leaks (Risk: Low)

The autouse `_reset_global_state` fixture covers known global state. No test writes to another test's state. Cross-test interference risk is minimal.

#### Parallelization Blockers

| Blocker | Affected Tests | Fix |
|---------|:---|-----|
| No markers on 230 files | All | Add `pytestmark` to every file |
| Runtime tests read live state | 6 files | Already gated by `@pytest.mark.runtime` |
| Episode ledger reads real JSONL | 6 tests | Could use synthetic fixtures |
| Single `conftest.py` for all | Everything | Split into `unit/conftest.py` + `integration/conftest.py` |

**Parallelization readiness:** The suite already runs in 81s. With `pytest-xdist` (4 workers on this 2-vCPU VM), unit tests could drop to ~12s. No shared-state blockers exist — the autouse fixture handles cleanup.

### 1.6 Dependency Graph

```
UNIT TESTS (1,906)
├── execution/doctrine_kernel.py      ← 3 test files
├── execution/hydra_engine.py         ← 2 test files
├── execution/risk_limits.py          ← 0 (covered by integration)
├── execution/sentinel_x.py           ← 1 test file
├── execution/intel/*                 ← 8 test files (scoring, expectancy)
├── execution/dle_shadow.py           ← 2 test files
├── execution/exit_*.py               ← 4 test files (exit pipeline)
├── execution/reset_guard.py          ← 1 test file (11 tests)
├── execution/episode_ledger.py       ← 2 test files (6 slow tests)
├── prediction/*                      ← 12 test files (binary lab, CLOB)
└── config/ (read-only)              ← 5 test files

INTEGRATION TESTS (1,582)
├── execution/order_router.py         ← 8 test files (routing, TWAP, ack)
├── execution/risk_limits.py          ← 6 test files (caps, correlation, DD)
├── execution/state_publish.py        ← 12 test files (state contracts)
├── execution/sync_state.py           ← 3 test files (firestore, positions)
├── execution/exchange_utils.py       ← 5 test files (all mocked)
├── execution/telegram_*.py           ← 3 test files (all mocked)
├── logs/state/*.json                 ← 6 test files (@runtime marker)
└── config/*.json                     ← 4 test files

SCENARIO/REGRESSION (implicit)
├── test_equity_curve.py              ← end-to-end dashboard rendering
├── test_risk_snapshot_invariants.py  ← parametrized invariant checks
└── test_pnl_attribution_h2.py       ← full PnL attribution pipeline

LOAD TESTS: NONE EXIST
```

### 1.7 Redundancy Inventory

| Area | Files | Overlap |
|------|:-----:|---------|
| Episode ledger | `test_episode_ledger.py` + `test_episode_ledger_e2.py` | E2 is superset; original could merge |
| NAV computation | `test_nav.py` + `test_nav_anomaly_detection.py` + `test_nav_state_contract.py` | Different layers but overlapping NAV paths |
| Risk limits | 6 integration files | Intentional granularity (caps, correlation, DD, portfolio) |
| State contracts | 12 files in `test_state_*` | Each validates one state file schema — correct pattern |
| Telegram | 3 files | `test_telegram_v7`, `test_telegram_alerts_v7`, `test_telegram_report` — different output channels |

**Verdict:** Minimal true redundancy. The `episode_ledger` pair is the only candidate for merge.

---

## Part 2 — Executor Deep-Dive

### 2.1 Vital Statistics

| Metric | Value |
|--------|-------|
| Total lines | 5,510 |
| Functions | 125 |
| Imports | 108 lines |
| `execution/` submodule dependencies | 46 distinct modules |
| Module-level mutable globals | 40+ |
| `try/except` blocks | 236 |
| Silent exception swallows | 37 |
| Binance API call sites | 28 |
| `time.sleep` sites | 4 |
| Dead functions | 6 confirmed |
| Duplicate function pairs | 3 |

### 2.2 Control-Flow Map

```
main()                                          [L5409, 101 LOC]
├── startup sequence (precision, dual-side, position check)
└── while True:
    ├── _maybe_compute_sentinel_x()             [L2641, 138 LOC] → BTC klines → regime
    ├── _loop_once(i)                           [L5083, 326 LOC]
    │   ├── _sync_dry_run()
    │   ├── _refresh_risk_config()
    │   ├── _account_snapshot()                 → get_balances() API
    │   ├── get_positions()                     → API call #1
    │   ├── check_for_testnet_reset()
    │   ├── EXIT SCAN:
    │   │   ├── scan_all_exits(positions)
    │   │   └── for exit → _send_order()
    │   ├── ENTRY PATH:
    │   │   ├── generate_intents()
    │   │   └── for intent → _send_order()
    │   ├── _pub_tick()                         [L4827, 256 LOC] → 11+ state files
    │   ├── _maybe_run_telegram_alerts()
    │   └── _maybe_run_pipeline_v6_compare()
    ├── _maybe_emit_heartbeat()
    ├── _maybe_run_internal_screener()
    └── time.sleep(60s)
```

### 2.3 The `_send_order` Monster (1,500 LOC)

**Lines 2977–4483** — the single largest function. Contains:

```
_send_order(intent)                             [L2977]
├── _doctrine_gate(intent)                      VETO → return
│   └── DLE shadow hook
├── churn_guard check
├── per_trade_nav_pct sizing
├── calibration_window cap
├── strategy attribution veto
├── conviction_band gate
├── KILL_SWITCH check
├── get_positions()                             API call #2
├── FLIP handling (opposing position):
│   ├── build_order_payload(reduce_only=True)
│   ├── send_order()                            API call (close)
│   └── _confirm_order_fill()                   BLOCKING 0–8s poll
├── _compute_nav()                              API call (account)
├── _gross_and_open_qty()
├── _evaluate_order_risk()                      risk engine gate
├── _maybe_run_pipeline_v6_shadow()
├── fee_gate check
├── build_order_payload()
├── ROUTING:
│   ├── _route_intent() or _route_order()
│   ├── _attempt_maker_first()
│   └── send_order()                            API taker fallback
├── _confirm_order_fill()                       BLOCKING 0–8s poll
├── Minotaur alpha/quality hooks
├── churn_guard fill recording
└── TP/SL registry registration
```

**Why this is problematic:**
1. 19 different responsibilities in one function
2. 4 nested closures (`_dispatch`, `_attempt_maker_first`, `_route_intent`, fill-confirm callbacks)
3. Up to 16s of synchronous blocking (two fill-confirmation polls during a flip)
4. 5-6 `get_positions()` API calls in a single loop iteration (redundant)
5. 37 silent `except: pass` blocks hiding failures
6. Only 1 `re-raise` exists in 247 `except` blocks — everything else is swallowed

### 2.4 Global Mutable State Map

**Throttle timestamps (float):** `_SENTINEL_X_LAST_RUN`, `_LAST_V6_RUNTIME_PROBE`, `_LAST_ROUTER_HEALTH_PUBLISH`, `_LAST_RISK_PUBLISH`, `_LAST_EXEC_HEALTH_PUBLISH`, `_LAST_KPI_PUBLISH`, `_LAST_HEARTBEAT`, `_LAST_SIGNAL_PULL`, `_LAST_SCREENER_RUN`, `_LAST_PIPELINE_V6_HEARTBEAT`, `_LAST_PIPELINE_V6_COMPARE`, `_LAST_ROUTER_AUTOTUNE_PUBLISH`, `_LAST_FEEDBACK_ALLOCATOR_PUBLISH`, `_LAST_CYCLE_TS`

**Cached state (dict):** `_LAST_RISK_CACHE`, `_LAST_RISK_SNAPSHOT`, `_LAST_EXECUTION_HEALTH`, `_LAST_KPIS_V7`, `_LAST_SYMBOL_SCORES_STATE`, `_LAST_NAV_STATE`, `_LAST_POSITIONS_STATE`

**Mutable registries (dict):** `_INTENT_REGISTRY`, `_SYMBOL_ERROR_COOLDOWN`, `_LAST_ALERT_TS`, `_LAST_INTEL_PUBLISH`, `_LAST_HEALTH_PUBLISH`, `_LAST_EXEC_ALERT_EVAL`

**Singletons:** `_DLE_WRITER`, `_RISK_ENGINE_V6`, `_RISK_ENGINE_V6_CFG_DIGEST`, `_BINARY_LAB_WRITER`, `_BINARY_LAB_ACTIVATION_ATTEMPTED`, `DRY_RUN`, `_RISK_CFG`, `_RISK_STATE`, `_POSITION_TRACKER`

**Total: ~40+ mutable globals.** All accessed single-threaded today. Any future async/threading would be immediately catastrophic.

### 2.5 Error Handling Profile

| Pattern | Count | % | Risk |
|---------|------:|--:|------|
| Log + continue | 119 | 48% | Low (intentional resilience) |
| Exception data captured | 54 | 22% | Low |
| Silent swallow (`pass`) | 37 | 15% | **High** — failures invisible |
| Wrapped + re-raised | 1 | <1% | — |
| Remaining (mixed) | 36 | 15% | Medium |

**The 37 silent swallows** cover: TP/SL registration, churn guard hooks, DLE shadow writes, alpha/quality hooks, telemetry emitters. Debugging production issues in these subsystems requires adding logging.

### 2.6 Dead Code

| Line | Function | Status |
|------|----------|--------|
| L1203 | `_clamp_intent_gross` | Unreferenced |
| L1183 | `_normalize_pct_value` | Duplicate of `_nav_pct_fraction` |
| L1225 | `_now_iso` | Duplicate of `events.now_utc()` |
| L4484 | `_compute_nav_snapshot` | Superseded by `_compute_nav_with_detail` |
| L2337 | `_current_bucket_gross` | No callers |
| L884 | `_record_flag_stat` | Unreachable stub |
| L1883 | `compute_final_gross_for_test` | Test-only export |

### 2.7 Redundant API Calls

A single `_loop_once` iteration can call `get_positions()` **5-6 times:**

| Call site | Purpose |
|-----------|---------|
| `_loop_once` L5148 | Baseline positions for exit scanner |
| `_send_order` L3429 | Position check before entry |
| `_send_order` L3586 | Position re-check after flip close |
| `_collect_rows` (via `_pub_tick`) | Telemetry position snapshot |
| `_position_rows_for_symbol` | Per-symbol position detail |
| `_account_snapshot` | Preflight position check |

**Impact:** At 50-200ms per Binance API call, this adds 250ms-1.2s of redundant latency per loop.

### 2.8 Blocking Call Map

| Site | Function | Duration | Impact |
|------|----------|----------|--------|
| L1848 | `_confirm_order_fill` | 0–8s per poll | Blocks main loop during fill confirmation |
| L2069 | `_startup_position_check` | 30s retries | Blocks startup indefinitely if positions exist |
| L4140 | `_dispatch` retry backoff | 1.5s × 3 | Blocks during order retry |
| L5505 | Main loop sleep | 60s default | Intentional cadence |

**Worst case per iteration:** 2 flip orders × 8s fill-poll + 3 retries × 1.5s = **20.5s blocked** (not counting API latency).

### 2.9 Refactor Map — Decomposition Targets

#### Phase 1: Extract Pure Functions (Low Risk)

| Extract To | Functions | LOC Saved |
|------------|-----------|-----------|
| `execution/sizing.py` | `_nav_pct_fraction`, `_size_from_nav`, `_normalize_pct_value` (dedup), `_clamp_intent_gross` | ~80 |
| `execution/fill_tracker.py` | `_confirm_order_fill`, `_fetch_order_status`, `_fetch_order_trades`, `_emit_order_ack`, `_should_emit_close` | ~280 |
| `execution/intent_builder.py` | `_normalize_intent`, `_build_order_intent_for_executor`, `_coerce_veto_reasons` | ~100 |
| `execution/telemetry_publisher.py` | `_pub_tick`, all `_maybe_emit_*` functions, `_persist_*` functions | ~600 |
| `execution/helpers.py` | `_to_float`, `_ms_to_iso`, `_iso_to_ts`, `mk_id`, `_now_iso`, `_git_commit`, `_truthy_env`, `_read_dry_run_flag` | ~120 |

**Subtotal: ~1,180 LOC extractable with zero behavioral change.**

#### Phase 2: Extract Subsystems (Medium Risk)

| Extract To | Functions | LOC | Notes |
|------------|-----------|-----|-------|
| `execution/order_dispatch.py` | `_dispatch` closure, `_attempt_maker_first`, routing logic | ~400 | Requires breaking `_send_order` closure scope |
| `execution/position_cache.py` | Centralized `get_positions()` cache with TTL | ~50 new | Eliminates 5-6 redundant API calls |
| `execution/nav_computer.py` | `_compute_nav`, `_compute_nav_with_detail`, merge+dedup | ~80 | Delete `_compute_nav_snapshot` dead code |

#### Phase 3: Break the `_send_order` Monolith (High Risk)

Current 1,500 LOC → target decomposition:

```
_send_order(intent)                         remaining: ~200 LOC orchestrator
├── doctrine_gate.evaluate(intent)          already exists externally
├── pre_order_checks(intent)                → NEW: churn, conviction, killswitch
├── flip_handler.handle(intent, positions)  → NEW: close opposing + confirm
├── risk_gate.evaluate(intent, nav)         already exists (risk_engine_v6)
├── order_dispatch.route_and_send(payload)  → Phase 2 extraction
├── fill_tracker.confirm(order_id)          → Phase 1 extraction
└── post_fill_hooks(fill_result)            → NEW: TP/SL, minotaur, churn
```

**Result:** `_send_order` drops from 1,500 LOC to ~200 LOC orchestrator calling 6 isolated subsystems.

#### Phase 4: State Container (Future)

Replace 40+ module-level globals with a single `ExecutorState` dataclass:

```python
@dataclass
class ExecutorState:
    dry_run: bool = False
    risk_cfg: Dict = field(default_factory=dict)
    risk_engine: Optional[RiskEngineV6] = None
    position_cache: PositionCache = field(default_factory=PositionCache)
    throttles: ThrottleRegistry = field(default_factory=ThrottleRegistry)
    intent_registry: Dict = field(default_factory=dict)
    # ... all 40+ globals become fields
```

Pass `state` as first argument to all functions. Enables testing without monkeypatching module globals.

### 2.10 Coupling Matrix

```
executor_live.py imports from 46 execution/ modules:
────────────────────────────────────────────────
CORE TRADING:    doctrine_kernel, sentinel_x, hydra_engine (via signal_generator),
                 risk_limits, risk_engine_v6, order_router, exit_scanner
NAV/RISK:        nav, risk_loader, drawdown (via state_publish)
FILL TRACKING:   exchange_utils, exchange_precision, minotaur_engine, minotaur_integration
STATE:           state_publish, position_ledger, position_tp_sl_registry, events, log_utils
OBSERVABILITY:   dle_shadow, enforcement_rehearsal, pipeline_v6_shadow, sizing_snapshot
UTILITIES:       v6_flags, runtime_config, versioning, utils.metrics, utils.execution_health,
                 calibration_window, churn_guard, fee_gate, loop_timing, cycle_statistics
SIGNALS:         signal_generator, signal_screener, universe_resolver, regime_pressure
EXTERNAL:        telegram_utils, firestore_utils
PREDICTION:      binary_lab_executor, binary_lab_runtime

Circular imports: NONE (strong property — dependency is strictly one-way)
Late/conditional imports: ~15 modules imported inside function bodies with try/except
```

---

## Part 3 — Two-VM Architecture

### 3.1 VM Roles

```
┌─────────────── VM-A (Execution) ─────────────────┐     ┌────────── VM-B (Observability) ──────────┐
│                                                    │     │                                          │
│  supervisor: hedge-executor                        │     │  supervisor: hedge-dashboard              │
│              hedge-sync_state                      │     │                                          │
│              hedge-pipeline-shadow-heartbeat       │     │  Streamlit app (dashboard/)               │
│              hedge-pipeline-compare                │     │  ops/daily_summary.py                     │
│                                                    │     │  scripts/generate_14d_brief.py            │
│  Writes:                                           │     │  test runner (pytest)                     │
│    logs/state/*.json (42 files)                    │     │                                          │
│    logs/cache/*.json (4 files)                     │     │  Reads:                                  │
│    logs/nav_log.json                               │     │    logs/state/*.json (via rsync/Firestore)│
│    logs/execution/*.jsonl (~25 logs)               │     │    logs/cache/coingecko_cache.json        │
│    logs/prediction/*.jsonl                         │     │    logs/nav_log.json                      │
│                                                    │     │    logs/execution/risk_vetoes.jsonl       │
│  External:                                         │     │    logs/execution/execution_alpha_*.jsonl │
│    Binance API (trading + data)                    │     │                                          │
│    Firestore (publish)                             │     │  External (optional):                     │
│    Telegram (alerts)                               │     │    Firestore (read-only state fetch)      │
│                                                    │     │                                          │
│  Cron:                                             │     │  No cron needed                           │
│    */15 episode_ledger rebuild                     │     │                                          │
│    0 0 daily_snapshot.py                           │     │                                          │
│                                                    │     │                                          │
└────────────────────────────────────────────────────┘     └──────────────────────────────────────────┘
                          │                                                    ▲
                          │         DATA SYNC (pick one)                       │
                          ├─── Option A: rsync over SSH (30s cron) ───────────┤
                          ├─── Option B: Firestore (already publishing) ──────┤
                          └─── Option C: NFS/shared volume ───────────────────┘
```

### 3.2 Interface Contract

**Files that must sync from VM-A → VM-B:**

| Path | Files | Size | Update Rate |
|------|------:|-----:|------------|
| `logs/state/*.json` | 42 | ~500KB total | Every 20-60s |
| `logs/cache/coingecko_cache.json` | 1 | ~2KB | Hourly |
| `logs/nav_log.json` | 1 | ~70KB (growing) | Every 30s |
| `logs/execution/risk_vetoes.jsonl` | 1 | ~10MB (current rotation) | Append-only |
| `logs/execution/execution_alpha_events.jsonl` | 1 | ~5MB | Append-only |
| **Total** | **46** | **~15MB active** | |

### 3.3 Sync Options

#### Option A: rsync over SSH (Recommended for Simplicity)

```bash
# VM-A cron (every 30s via systemd timer or watch)
rsync -az --delete logs/state/ logs/cache/coingecko_cache.json logs/nav_log.json \
  vm-b:/root/hedge-fund/logs/state/

# For JSONL files (append-only, no --delete)
rsync -az logs/execution/risk_vetoes.jsonl logs/execution/execution_alpha_events.jsonl \
  vm-b:/root/hedge-fund/logs/execution/
```

**Pros:** Dead simple, 46 files ~15MB, sub-second rsync. No code changes.  
**Cons:** 30s staleness window. Requires SSH key setup.

#### Option B: Firestore (Already Implemented)

`sync_state.py` already publishes to Firestore every 20s. The dashboard has `dashboard_utils.py` → `utils.firestore_client.get_db()` for remote state fetch.

**Changes needed:**
1. Enable Firestore reads in dashboard (`FIRESTORE_READ_ENABLED=1`)
2. Add NAV log sync to Firestore publish (currently only publishes 500 NAV points, not full 940+)
3. Add risk_vetoes summary (currently not synced — dashboard reads local JSONL)

**Pros:** Already partially built. No SSH/network setup.  
**Cons:** Firestore cost at scale. JSONL append-only files don't map well to Firestore documents. 500-point NAV limit in current sync.

#### Option C: NFS/Shared Volume

Mount a shared volume at `/root/hedge-fund/logs/` accessible to both VMs.

**Pros:** Zero latency, zero staleness, zero code changes.  
**Cons:** Cloud provider-specific. Single point of failure. IOPS may be limited.

### 3.4 What Must Be Decoupled

| Component | Currently | VM Split Action |
|-----------|-----------|----------------|
| `logs/state/` writes | Executor writes, dashboard reads locally | Sync to VM-B |
| `logs/nav_log.json` | Executor writes, dashboard reads | Must be in sync contract |
| `logs/execution/*.jsonl` | Executor appends, dashboard reads 2 files | Sync 2 specific files |
| Firestore publish | `sync_state.py` on same VM | Stays on VM-A |
| Telegram alerts | Executor sends | Stays on VM-A |
| Binance API keys | `.env` on VM-A | **Never on VM-B** |
| `config/` | Read by executor + tests | Copy to VM-B (static, version-controlled) |
| Test runner | Currently on VM-A | Move to VM-B |

### 3.5 What Can Remain Shared

| Resource | Reason |
|----------|--------|
| Git repository | Both VMs pull from same repo |
| `config/` directory | Version-controlled, identical on both |
| `v7_manifest.json` | Read-only schema definition |
| Python venv | Can be built independently on each VM |

### 3.6 Refactoring Required for Clean Split

| Change | Effort | Priority |
|--------|--------|----------|
| Dashboard: add `REMOTE_STATE=firestore` toggle | 1 day | P1 if using Option B |
| Dashboard: eliminate `risk_vetoes.jsonl` direct read | 2 hours | P1 — create `logs/state/veto_summary.json` instead |
| Dashboard: eliminate `execution_alpha_events.jsonl` direct read | 2 hours | P1 — create `logs/state/execution_alpha_summary.json` |
| Supervisor: split into `hedge-exec.conf` (VM-A) + `hedge-obs.conf` (VM-B) | 30 min | P1 |
| Tests: ensure `ENV=test` works without Binance keys | Already done | — |
| Daily snapshot: stays on VM-A only | No change | — |

### 3.7 Migration Plan

#### Phase 0: Prep (1 day, zero disruption)

1. Create `logs/state/veto_summary.json` — executor writes veto counts instead of dashboard reading raw JSONL
2. Create `logs/state/execution_alpha_summary.json` — same pattern
3. Update dashboard components to read from these new state files
4. Test: dashboard works with only `logs/state/` + `logs/nav_log.json` + `logs/cache/coingecko_cache.json`

#### Phase 1: Provision VM-B (1 day)

1. Provision VM-B (2 vCPU, 4GB RAM — same as current)
2. Clone repo, build venv, install Streamlit
3. Set up SSH key from VM-A → VM-B for rsync
4. Create `ops/hedge-obs.conf` supervisor config (dashboard only)
5. Set up rsync cron (every 30s) from VM-A
6. Verify dashboard serves correctly from synced state
7. **Do NOT cut over yet** — run in parallel

#### Phase 2: Cutover (30 minutes)

1. Update DNS/nginx to point dashboard to VM-B
2. Stop dashboard process on VM-A (`supervisorctl stop hedge:hedge-dashboard`)
3. Remove dashboard from VM-A supervisor config
4. Verify VM-B dashboard is live and data is fresh

#### Phase 3: Test Runner Migration (1 day)

1. Move `pytest` execution to VM-B
2. Install test dependencies on VM-B
3. Set `ENV=test` (no Binance keys needed)
4. Validate full suite passes: `PYTHONPATH=. pytest -q`
5. Set up CI trigger (git hook or cron)

#### Phase 4: Hardening (ongoing)

1. Monitor rsync lag — alert if state files >60s stale
2. Add health endpoint to VM-B dashboard that checks file freshness
3. Consider switching to Option B (Firestore) for lower latency
4. Set up VM-B-only log rotation policy

### 3.8 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Rsync delay > 60s | Low | Dashboard shows stale data | Alert on file age; fallback to Firestore read |
| VM-B disk fills | Low | Dashboard crashes | VM-B only needs ~50MB of state; no execution logs |
| SSH key compromised | Very Low | VM-B access | Restrict to rsync-only, read-only paths |
| Firestore quota exceeded | Medium (Option B) | Publish failures | Already degraded gracefully to noop |
| Test suite drift | Medium | Tests pass on VM-B but not VM-A | Both VMs on same git rev; CI enforces |

---

## Appendix A — Test File Inventory (281 files)

### `tests/unit/` (85 files, 1,906 tests, 41s)

```
test_alpha_decay.py               test_expectancy_prior.py
test_alpha_miner_core.py          test_factor_adaptive_weighting.py
test_alpha_router_core.py         test_factor_diagnostics_unit.py
test_binary_lab_e2e.py            test_hedge_mode_orders.py
test_binary_lab_gate.py           test_hybrid_scoring.py
test_binary_lab_replay.py         test_hybrid_variance_audit.py
test_binary_lab_round_logic.py    test_hydra_engine.py
test_binary_lab_rounds.py         test_hydra_pnl.py
test_calibration_hooks.py         test_intent_attribution.py
test_carry_pipeline.py            test_loop_timing.py
test_category_momentum.py         test_minotaur_engine.py
test_clob_market_client.py        test_nav_helpers.py
test_conviction_engine.py         test_order_router_unit.py
test_cross_pair_engine.py         test_phase_c_readiness.py
test_cycle_watermark.py           test_pipeline_v6_shadow.py
test_dataset_registry.py          test_position_ledger.py
test_diagnostics_metrics_unit.py  test_position_tracker.py
test_dle_enforcer.py              test_prediction_alert_wiring.py
test_dle_schema.py                test_regime_aware_sizing.py
test_dle_shadow.py                test_reset_guard.py
test_doctrine_kernel.py           test_risk_engine_v6.py
test_e2_hardening.py              test_round_observer.py
test_edge_scanner.py              test_rtds_oracle.py
test_enforcement_c1_entry_only.py test_signal_doctor.py
test_enforcement_split_brain.py   test_sizing_core.py
test_episode_ledger.py            test_slippage_attribution.py
test_episode_ledger_e2.py         test_state_publish_unit.py
test_error_cooldown.py            test_trend_score.py
test_exchange_utils_errors.py     test_universe_resolver.py
test_exit_dedup.py                test_vol_harvest.py
test_exit_pipeline.py             test_vol_utils_unit.py
test_exit_reason_normalization.py
```

### `tests/integration/` (189 files, 1,582 tests, 33s)

*(Top 20 by test count shown above; full list omitted for brevity)*

### `tests/legacy/` (5 files, all skipped)

```
test_infra_v5_5.py          test_nav_risk_unification.py
test_nav_modes.py           test_router_health_v2.py
test_treasury_nav.py
```

## Appendix B — Executor Function Index (125 functions)

Full function list with line numbers available via:
```bash
grep -n "^def \|^    def " execution/executor_live.py | head -125
```

## Appendix C — Pre-Existing Test Failures

| Test | Issue | Status |
|------|-------|--------|
| `test_edge_discovery_panel::test_render_full_panel` | `st.columns` mock doesn't handle list spec | Dashboard test bug |
| `test_manifest_audit::test_audit_no_violations` | Untracked file `binary_rounds_phase1_10rounds.jsonl` | Manifest update needed |
| 12 module-skipped files | Various v7 API changes | Need evaluation for removal or update |
| 1 xfail (`test_firestore_publish`) | Stub mismatch | Needs sync with v5.9 API |
