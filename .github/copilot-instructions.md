# Copilot Instructions — GPT Hedge v7.6

## Architecture Overview

**Binance futures trading system** with unified execution loop:

```
Signal Screener → Risk Engine → Order Router → State Publisher
     ↓                ↓              ↓              ↓
  intents         check_order()   POST_ONLY     logs/state/*.json
                  (veto gate)     + fallback    (dashboard reads)
```

| Component | File | Responsibility |
|-----------|------|----------------|
| Executor | `execution/executor_live.py` | Main loop (~3900 lines), orchestrates all components |
| Screener | `execution/signal_screener.py` | Generates intents, computes unlevered sizing |
| Risk | `execution/risk_limits.py` | `check_order()` is the **only** veto authority |
| Router | `execution/order_router.py` | Maker-first POST_ONLY with taker fallback, TWAP/slippage model |
| State | `execution/state_publish.py` | Writes canonical state to `logs/state/*.json` |
| NAV | `execution/nav.py` | `nav_health_snapshot()` — sole source of risk truth |
| Intel | `execution/intel/*.py` | Scoring: `expectancy_v6.py`, `symbol_score_v6.py`, `hybrid_score_engine.py` |
| Conviction | `execution/conviction_engine.py` | Conviction-weighted sizing (v7.7), uses factors + risk state |
| Conviction | `execution/conviction_engine.py` | Conviction-weighted sizing (v7.7), uses factors + risk state |

## Project Layout

```
execution/           # Core trading logic — DO NOT import from dashboard/
  intel/             # Scoring: expectancy_v6.py, symbol_score_v6.py, hybrid_score_engine.py
  utils/             # Metrics helpers, execution health, vol regime
  strategies/        # Strategy implementations (vol_target.py)
config/              # Runtime configs — all values in fractional form (0.05 = 5%)
logs/state/          # Canonical state files consumed by dashboard (read-only for dash)
dashboard/           # Streamlit app — reads from logs/state/, never writes
tests/               # ~160 pytest files — run with PYTHONPATH=.
  unit/              # Fast, pure in-process tests
  integration/       # Multi-module tests, may touch filesystem
  legacy/            # Legacy tests from v5/v6 kept for reference
v7_manifest.json     # Canonical list of state files, owners, update frequencies
```

## Critical Invariants

### NAV Source of Truth
- **NAV = futures wallet only** — never includes spot/treasury unless explicitly configured
- Use `nav_health_snapshot()` from `execution/nav.py` — the only source of risk truth
- Stale NAV (>90s default, configurable via `nav_freshness_seconds`) triggers automatic veto

### Veto Authority
- **Only `risk_limits.check_order()` can veto orders** — no exceptions
- `risk_engine_v6.py` is orchestration only, NOT enforcement
- Executor and router must never perform risk math

### Position State
- Live positions come from `exchange_utils.get_positions()` only
- Screener intents do NOT represent open positions
- Always treat exchange-side position state as authoritative

### One-Way Dependency
- `execution/` → `config/`, `logs/` (read/write)
- `dashboard/` → `logs/state/` (read-only)
- **Never import from `dashboard/` into `execution/`**

## Critical Gotchas

### Config Percentage Normalization
```python
# risk_loader.normalize_percentage() converts >1 to fractional
# Always write fractional: 0.05 (not 5 or 5%)
"max_trade_nav_pct": 0.06  # means 6% NAV
```

### JSONL Logging Contract
- All logs are **append-only** — never rewrite JSONL files
- Use `events.write_event()` or `log_utils.get_logger()` to extend logs
- Never use raw `print()` in execution code
- Log rotation handled by `log_utils.JsonlLogger` with size-based rotation

### V6 Feature Flags (`execution/v6_flags.py`)
```python
from execution.v6_flags import get_flags
flags = get_flags()
if flags.intel_v6_enabled:
    # use intel scoring
```

| Flag | Env Var | Purpose |
|------|---------|---------|
| `intel_v6_enabled` | `INTEL_V6_ENABLED` | Enable expectancy/symbol scoring |
| `risk_engine_v6_enabled` | `RISK_ENGINE_V6_ENABLED` | Use typed RiskDecision returns |
| `pipeline_v6_shadow_enabled` | `PIPELINE_V6_SHADOW_ENABLED` | Enable shadow pipeline validation |
| `router_autotune_v6_enabled` | `ROUTER_AUTOTUNE_V6_ENABLED` | Enable router auto-tuning |
| `feedback_allocator_v6_enabled` | `FEEDBACK_ALLOCATOR_V6_ENABLED` | Enable feedback-based position allocation |
| `router_autotune_v6_apply_enabled` | `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` | Actually apply router suggestions (0=safe) |

## Developer Workflow

### Required Env Flags for Local Dev
```bash
export BINANCE_TESTNET=1    # use testnet endpoints
export DRY_RUN=1            # skip actual order placement
export EXECUTOR_ONCE=1      # single iteration mode (optional)
```

### Run Tests
```bash
PYTHONPATH=. pytest -q           # ALWAYS run full suite — patches are only complete when entire suite is green
make test                         # runs tests/unit + tests/integration
make test-fast                    # skip runtime and legacy markers
make smoke                        # Firestore + doctor health check
```
**Never run individual test files unless actively debugging.** The suite includes ~160 tests; partial runs miss regressions.

Test markers (from `pytest.ini`):
- `@pytest.mark.unit` — fast, pure in-process tests
- `@pytest.mark.integration` — multi-module tests, may touch filesystem
- `@pytest.mark.runtime` — require state files
- `@pytest.mark.legacy` — kept for reference

### Test Map (examples, not exhaustive)

Agents should look up tests near the file they're editing; this list gives a fast starting point:

| Test File | Coverage |
|-----------|----------|
| `tests/integration/test_risk_limits.py` | Risk veto gates, caps, DD/risk_mode behaviour |
| `tests/integration/test_state_publish_diagnostics.py` | `diagnostics.json` schema & liveness/exit metrics |
| `tests/integration/test_state_positions_ledger_contract.py` | Positions/ledger authority & TP/SL coverage |
| `tests/integration/test_state_files_schema.py` | Core state surfaces schema checks |
| `tests/integration/test_manifest_state_contract.py` | `v7_manifest.json` alignment with state files |

### Lint & Type Check
```bash
ruff check .                      # lint (excludes dashboard/, scripts/, ops/, strategies/)
mypy .                            # type check (gradual typing — see mypy.ini ignore_errors sections)
```

**Note:** `ruff.toml` excludes `dashboard/`, `scripts/`, `strategies/`, `telegram/`, `research/`, `ops/`, `infrastructure/`, `deploy/`. `mypy.ini` has `ignore_errors = True` for many `execution/*.py` files — gradual typing adoption.

### Process Control (prod)
```bash
sudo supervisorctl restart hedge:               # all processes
sudo supervisorctl restart hedge:hedge-executor # single process

# follow logs (correct paths)
tail -f /var/log/hedge-executor.out.log         # executor stdout
tail -f /var/log/hedge-executor.err.log         # executor stderr
tail -f /var/log/hedge-dashboard.err.log        # dashboard errors
tail -f /var/log/supervisor/sync_state.out      # sync_state stdout
```

## Config Files

| File | Purpose |
|------|---------|
| `config/risk_limits.json` | Per-symbol caps, global limits, tier definitions |
| `config/runtime.yaml` | Trading window, signal gates, directional bias, TWAP/router tunables |
| `config/strategy_config.json` | Universe, per_trade_nav_pct, signal params |
| `config/pairs_universe.json` | Allowed symbols with tier metadata |
| `config/symbol_tiers.json` | Symbol → tier (CORE/SATELLITE/TACTICAL/ALT-EXT) mapping |
| `config/correlation_groups.json` | Correlated symbol groups for exposure caps |
| `config/liquidity_buckets.json` | Per-symbol liquidity bucket assignments for TWAP/slippage |

## Router & Microstructure

- Core router: `execution/order_router.py`
- Slippage / liquidity models: `execution/slippage_model.py`, `execution/liquidity_model.py`
- Writes per-symbol quality stats to `logs/state/router_health.json` (slippage, latency, TWAP use)
- Agents must **never** write to `logs/state/*.json` directly; only via executor/state_publish (see State Contract + v7_manifest)

**Example:** When adjusting router behaviour, update `slippage_model`/`liquidity_model` and verify `router_health.json` quality scores and buckets via tests and dashboard.

## Runtime Config (`config/runtime.yaml`)

Controls environment flags (prod vs. testnet, DRY_RUN), trading windows, and global signal/risk gates.

**Use for:**
- Enabling DRY_RUN / testnet runs
- Restricting trading to specific hours/windows (`trading_window.*`)
- High-level routing of which strategies are active
- TWAP/slippage tunables under `router.*`

**Do not** add strategy-specific logic here; put it in `strategy_config.json` / `risk_limits.json`, then read via the runtime loader.

**Commonly adjusted tunables (examples):**
- `runtime.dry_run` — skip order placement
- `trading_window.start_utc` / `end_utc` — active trading hours
- `router.twap.min_notional_usd` — threshold for TWAP execution
- `signal_gate.expectancy_min` — minimum rolling expectancy to allow trade

## State Files Contract (`logs/state/`)

Dashboard reads these JSON files (never writes). **Changes must be strictly additive.** See `v7_manifest.json` for canonical list with owners and update frequencies.

**Single source of truth** for state surfaces is `v7_manifest.json` + `docs/v7.6_State_Contract.md`. If an agent adds/changes a state surface, they **must**:
1. Update `v7_manifest.json`
2. Update the State Contract doc
3. Extend the schema tests in `tests/integration`

| File | Purpose |
|------|--------|
| `nav_state.json` | NAV, nav_age_s, freshness |
| `positions_state.json` | Current open positions |
| `positions_ledger.json` | **Unified ledger: positions + TP/SL (v7.4_C3)** |
| `risk_snapshot.json` | Portfolio DD, gross exposure, VaR/CVaR |
| `kpis_v7.json` | Aggregated KPIs for dashboard/investor views |
| `router_health.json` | Fill rates, latency metrics |
| `symbol_scores_v6.json` | Intel scoring state |
| `hybrid_scores.json` | Hybrid alpha scores per symbol |
| `funding_snapshot.json` | Funding rate data |
| `basis_snapshot.json` | Basis/carry data |
| `diagnostics.json` | Veto counters, exit pipeline, liveness |

### Position Ledger (v7.4_C3)

The **position ledger** (`execution/position_ledger.py`) provides a unified source of truth for positions and TP/SL levels:

```python
from execution.position_ledger import build_position_ledger, get_ledger_entry

ledger = build_position_ledger()  # Dict[str, LedgerEntry]
entry = get_ledger_entry("BTCUSDT", "LONG")
if entry:
    print(f"TP: {entry.tp_sl.tp}, SL: {entry.tp_sl.sl}")
```

| Dataclass | Fields | Purpose |
|-----------|--------|---------|
| `LedgerEntry` | symbol, side, qty, entry_price, tp_sl, strategy, metadata | Single position with TP/SL |
| `TpSlLevels` | tp, sl | Optional take-profit and stop-loss prices |
| `PositionLedger` | entries, updated_ts, source | Container for all entries |

**Key functions:**
- `build_position_ledger()` — Load ledger from state file
- `sync_ledger_with_positions()` — Merge exchange positions + registry into ledger
- `get_ledger_entry(symbol, side)` — Lookup single entry
- `check_consistency()` — Validate ledger vs exchange positions

**Exit scanner** uses ledger-first approach with registry fallback for TP/SL lookups.

## Strategy Tiering Model

| Tier | `per_symbol_nav_pct` | Behavior |
|------|---------------------|----------|
| **CORE** | 0.30 | Full risk budget, TWAP allowed |
| **SATELLITE** | 0.20 | Correlation-aware |
| **TACTICAL** | 0.10 | Size-down in high vol |
| **ALT-EXT** | 0.05 | Auto-disabled on NAV stale / DD breach |

## Common Debugging

### Order Vetoed?
```bash
tail -100 logs/execution/risk_vetoes.jsonl | jq '{reason: .veto_reason, observed, limits}'
```
Check: stale NAV? per-symbol cap? portfolio DD? correlation cap? min_notional?

### NAV Staleness
```bash
cat logs/state/nav_state.json | jq '.nav_age_s'
```
Stale NAV → `check_nav_freshness` veto → zero sizing.

### Router Fallback
```bash
tail -50 logs/execution/orders_executed.jsonl | jq 'select(.fallback==true)'
```

### Dashboard Not Updating
Check `logs/state/*.json` timestamps. Verify `state_publish` ran recently.

### Ledger Consistency Issues
```bash
cat logs/state/positions_ledger.json | jq '.entries | to_entries | map(select(.value.tp_sl.tp == null and .value.tp_sl.sl == null))'
```
Positions without TP/SL may indicate registry sync failure. Check `_sync_position_ledger()` in executor logs.

## Integration Points

Agents must understand these integration boundaries:

### Firestore State Sync (`sync_state.py`)
- **One-way publisher** — never reads from Firestore
- Writes: `nav_state.json`, `positions_state.json`, `leaderboard_state.json`
- Do not change contract without dashboard migration

### Telegram Alerts (`telegram_utils.py`)
- **Non-blocking** — must never raise exceptions that stop executor
- Only send compact technical signals (4h & daily BTC model)
- Do not extend without explicit user request

### Exchange Client (`exchange_utils.py`)
- Supports `DRY_RUN=1` for safe testing
- Must never modify or bypass `risk_limits.check_order()`
- All orders must go through router (maker-first logic, TWAP support)

### Runtime Config (`runtime.yaml`)
- **Authoritative** for execution flags and tunables
- `router_autotune_v6_apply_enabled` is a *safe mode* flag (0=safe)
- TWAP config lives under `router.twap.*`
- No hardcoded parameters — always pull from config

## Dashboard (Streamlit)

- **Entrypoint:** `dashboard/app.py`
- **State loader:** `dashboard/state_v7.py` (reads `logs/state/*.json`, no writes)
- **Panels:**
  - `equity_panel.py` — NAV / equity curve / drawdown
  - `execution_panel.py` — orders, fills, vetoes
  - `risk_panel.py` — DD state, VaR/CVaR, risk_mode (from `risk_snapshot.json`)
  - `router_health_panel.py` — execution quality (from `router_health.json`)
  - `factor_panel.py` — factor diagnostics & weights (from `factor_diagnostics.json`)
    - Factor weights support an **optional adaptive overlay** (v7.7_P2) driven by IR/PnL, controlled via `factor_diagnostics.auto_weighting.adaptive`

**Agents modifying the dashboard:**
- **Only read** from state surfaces; never write
- When adding new state fields, update `state_v7.py` loader + schema tests under `tests/integration`

## Code Conventions

- Monetary amounts in **USDT** unless suffixed (`_zar`, `_btc`)
- Timestamps are **Unix seconds** (not milliseconds) except Binance responses
- Use `typing` annotations throughout `execution/`
- Prefer `Decimal` for price/qty precision in risk calculations
- Telegram alerts are async-only, failures must not propagate to executor

## PR Acceptance

- ✅ `PYTHONPATH=. pytest -q` green (full suite, no exceptions)
- ✅ `ruff check .` clean
- ✅ `mypy .` clean (or documented suppressions)
- ✅ No credentials in commits

---

## Appendix: Legacy AGENTS.md Context

`docs/AGENTS.md` provides historical background for older v5.x agent workflows (ACK/FILL separation, Backfill Agent patterns). **Modern patches must follow all rules and patterns defined in the main sections above.** AGENTS.md conventions must not be used for new development — the v7+ engine uses Risk Engine v6, Hybrid Alpha v2, Vol Regime Model, and TWAP router which supersede those patterns.