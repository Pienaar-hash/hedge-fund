# Copilot Instructions — GPT Hedge v7.9

## Architecture Overview

**Binance futures trading system** with regime-governed execution. Doctrine Kernel is **supreme authority** — all entries/exits flow through it.

```
Signal → Hydra (multi-head) → Cerberus (multipliers) → Doctrine Gate → Risk Limits → Router → Exchange
                                                              ↓
                                                   logs/doctrine_events.jsonl
```

> *Only Doctrine and Risk can prevent market participation.*

### Core Components

| Component | File | Role |
|-----------|------|------|
| **Doctrine** | `execution/doctrine_kernel.py` | Entry/exit gating — CANNOT be bypassed |
| **Executor** | `execution/executor_live.py` | Main loop (~5690 lines), orchestrates all |
| **Sentinel-X** | `execution/sentinel_x.py` | Regime detection: TREND_UP/DOWN, MEAN_REVERT, BREAKOUT, CHOPPY, CRISIS |
| **Hydra** | `execution/hydra_engine.py` | 6 strategy heads: TREND, MEAN_REVERT, RELATIVE_VALUE, CATEGORY, VOL_HARVEST, EMERGENT_ALPHA |
| **Cerberus** | `execution/cerberus_router.py` | Dynamic head multipliers (does NOT create signals or override doctrine) |
| **Minotaur** | `execution/minotaur_engine.py` | Microstructure-aware execution, slippage tracking |
| **Risk** | `execution/risk_limits.py` | `check_order()` secondary veto (caps, DD, correlation) |
| **Router** | `execution/order_router.py` | Maker-first POST_ONLY with taker fallback, TWAP support |
| **NAV** | `execution/nav.py` | `nav_health_snapshot()` — sole source of NAV truth |
| **Helpers** | `execution/helpers.py` | Pure stateless utilities (to_float, ms_to_iso, etc.) |
| **Sizing** | `execution/sizing.py` | Position sizing (nav_pct_fraction, size_from_nav) |
| **Fill Tracker** | `execution/fill_tracker.py` | Order ack, fill polling (async core + sync wrapper), `FillTaskHandle` API |
| **Position Cache** | `execution/position_cache.py` | 1 s TTL position cache; `invalidate()` on confirmed fills |
| **Order Dispatch** | `execution/order_dispatch.py` | Exchange dispatch, maker-first logic, retry loop (extracted from `_send_order`) |

### Doctrine Laws (Hard-Coded, Not Configurable)

1. **Regime governs permission** — Signals determine direction, regimes determine permission
2. **No regime = no trade** — Sentinel-X must have stable regime before entry
3. **Direction match required** — TREND_UP→Long only, TREND_DOWN→Short only
4. **Refusal is first-class** — Every veto logged to `logs/doctrine_events.jsonl`
5. **All exits are thesis-driven** — Positions die when thesis dies, not on signals
6. **Stops are seatbelts, not strategy** — SL is catastrophe protection only
7. **Kill switch never blocks doctrine exits** — KILL_SWITCH may only block risk-increasing orders (new entries). It must NEVER block reduceOnly exits issued under doctrine authority. Enforced via two-flag guard: `doctrine_exit=True AND reduceOnly=True`.

## Project Layout

```
execution/           # Core trading logic — DO NOT import from dashboard/
  doctrine_kernel.py # SUPREME AUTHORITY - entry/exit gates
  executor_live.py   # Main loop (~5700 lines)
  helpers.py         # Pure stateless utilities (extracted from executor)
  sizing.py          # Position sizing functions (extracted from executor)
  fill_tracker.py    # Order ack, fill polling, PnL close (extracted from executor)
  sentinel_x.py      # Regime detection (6 regimes)
  hydra_engine.py    # Multi-strategy execution engine
  cerberus_router.py # Dynamic head multipliers (observation only)
  minotaur_engine.py # Microstructure/slippage tracking
  intel/             # Scoring: expectancy_v6.py, symbol_score_v6.py
config/              # All percentages in fractional form (0.05 = 5%)
logs/state/          # State files — dashboard reads, executor writes
dashboard/           # Streamlit app — READ-ONLY from logs/state/
tests/               # ~300 pytest files (unit/, integration/, legacy/)
v7_manifest.json     # Canonical state file registry
```

## Critical Invariants

### Doctrine Supremacy
- **Doctrine gate is first check** — All entries go through `_doctrine_gate()` in executor
- **No enabled flag** — Doctrine kernel has NO config; it IS the law
- **Veto = no trade** — If doctrine returns VETO, order is dropped (no fallback)

### NAV Source of Truth
- **NAV = futures wallet only** — never includes spot/treasury unless explicitly configured
- Use `nav_health_snapshot()` from `execution/nav.py` — the only source of risk truth
- Stale NAV (>90s default) triggers automatic veto

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

### V6 Feature Flags (`execution/v6_flags.py`)
```python
from execution.v6_flags import get_flags
flags = get_flags()
if flags.intel_v6_enabled:
    # use intel scoring
```

Key flags: `INTEL_V6_ENABLED`, `RISK_ENGINE_V6_ENABLED`, `PIPELINE_V6_SHADOW_ENABLED`, `ROUTER_AUTOTUNE_V6_ENABLED`, `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` (0=safe mode).

### DLE Shadow Layer (`execution/dle_shadow.py`)

The Decision Ledger Engine (DLE) runs in **shadow mode** (observation only, never blocks execution). It logs every doctrine verdict as a shadow event for future enforcement.

```python
from execution.v6_flags import get_flags
flags = get_flags()
if flags.shadow_dle_enabled:
    # DLE shadow gate logs verdicts to dle_shadow_events.jsonl
if flags.shadow_dle_log_mismatches:
    # logs cases where shadow verdict diverges from doctrine
```

Key flags: `SHADOW_DLE_ENABLED`, `SHADOW_DLE_LOG_MISMATCHES`.

**DLE does NOT gate execution in v7.x.** It is observation-only (SHADOW_MODE). Phase B enforcement is not yet active.

- Shadow log: `logs/execution/dle_shadow_events.jsonl` (append-only)
- DLE specs: `docs/dle/` (14 documents — constitution, schemas, invariants)
- Exit reason map: `config/exit_reason_map.yaml` (canonical normalization)
- Test coverage: `tests/unit/test_dle_shadow.py`

## Developer Workflow

### Local Development
```bash
export BINANCE_TESTNET=1 DRY_RUN=1    # Required for safe local dev
export EXECUTOR_ONCE=1                 # Single iteration mode (optional)
```

### Testing
```bash
PYTHONPATH=. pytest -q    # ALWAYS run full suite before PR
make test                  # tests/unit + tests/integration
make test-fast             # skip runtime and legacy markers
make smoke                 # Firestore + doctor health check
```

**⚠️ Never run individual test files** unless actively debugging. The ~300 tests catch regressions across modules.

Test markers: `@pytest.mark.unit` (fast), `@pytest.mark.integration` (filesystem), `@pytest.mark.runtime` (state files)

**Key test files:** `test_risk_limits.py`, `test_state_files_schema.py`, `test_manifest_state_contract.py`

### Lint & Type Check
```bash
ruff check .    # excludes dashboard/, scripts/, ops/
mypy .          # gradual typing (many ignore_errors)
```

### Production
```bash
sudo supervisorctl restart hedge:       # all processes
tail -f /var/log/hedge-executor.out.log # executor logs
```

## Config Files

| File | Purpose |
|------|---------|
| `risk_limits.json` | Per-symbol caps, global limits, tier definitions |
| `runtime.yaml` | Trading window, TWAP tunables, signal gates |
| `strategy_config.json` | Universe, per_trade_nav_pct, signal params |
| `symbol_tiers.json` | Symbol → tier (CORE/SATELLITE/TACTICAL/ALT-EXT) |
| `correlation_groups.json` | Correlated symbol groups for exposure caps |

**Router:** `execution/order_router.py` with `slippage_model.py` and `liquidity_model.py`. Agents must **never** write to `logs/state/*.json` directly.

**Key runtime tunables:** `trading_window.start_utc/end_utc`, `router.twap.min_notional_usd`, `signal_gate.expectancy_min`

## State Files (`logs/state/`)

Dashboard reads these JSON files (never writes). **Changes must be strictly additive.**

> *Never infer behavior from a single state file snapshot.*

**When adding/changing state surfaces:**
1. Update `v7_manifest.json`
2. Extend schema tests in `tests/integration/test_state_*.py`

**Core state:** `nav_state.json`, `positions_state.json`, `positions_ledger.json` (unified with TP/SL), `risk_snapshot.json`, `diagnostics.json`

**Engine state:** `sentinel_x.json` (regime), `hydra_state.json` (head budgets), `cerberus_state.json` (multipliers), `execution_quality.json`

**Intel state:** `symbol_scores_v6.json`, `factor_diagnostics.json`, `router_health.json`

### Position Ledger

```python
from execution.position_ledger import build_position_ledger, get_ledger_entry
ledger = build_position_ledger()
entry = get_ledger_entry("BTCUSDT", "LONG")  # Returns LedgerEntry with tp_sl
```

Exit scanner uses ledger-first approach with registry fallback for TP/SL lookups.

## Strategy Tiering Model

| Tier | `per_symbol_nav_pct` | Behavior |
|------|---------------------|----------|
| **CORE** | 0.30 | Full risk budget, TWAP allowed |
| **SATELLITE** | 0.20 | Correlation-aware |
| **TACTICAL** | 0.10 | Size-down in high vol |
| **ALT-EXT** | 0.05 | Auto-disabled on NAV stale / DD breach |

## Common Debugging

```bash
# Order vetoed? Check risk vetoes
tail -100 logs/execution/risk_vetoes.jsonl | jq '{reason: .veto_reason}'

# NAV stale? (>90s triggers veto)
cat logs/state/nav_state.json | jq '.nav_age_s'

# Hydra/Cerberus state
cat logs/state/hydra_state.json | jq '{head_budgets, head_usage}'
cat logs/state/cerberus_state.json | jq '{overall_health, head_state}'
```

**Common veto causes:** stale NAV, per-symbol cap, portfolio DD, correlation cap, min_notional

## Integration Points

| Component | Boundary Rule |
|-----------|---------------|
| `sync_state.py` | One-way publish to Firestore (never reads back) |
| `telegram_utils.py` | Non-blocking — exceptions must not stop executor |
| `exchange_utils.py` | All orders through router; respects `DRY_RUN=1` |
| `runtime.yaml` | Authoritative for execution flags — no hardcoded params |

## Dashboard (Streamlit)

- **Entrypoint:** `dashboard/app.py` → `state_v7.py` (read-only from `logs/state/`)
- **Adding state fields:** Update `state_v7.py` loader + add schema tests

## Code Conventions

- Monetary amounts in **USDT** (suffix with `_zar`, `_btc` for others)
- Timestamps: **Unix seconds** (not ms) except Binance API responses
- Use `typing` annotations in `execution/`; prefer `Decimal` for price/qty
- JSONL logs are **append-only** — use `events.write_event()`, never raw `print()`

## Common Failure Modes (Read Before Coding)

1. **Snapshot ≠ History** — `positions_state.json == []` does *not* mean "no trading occurred". Always cross-check execution logs or episode ledger.

2. **Observability Is Read-Only** — New dashboards, metrics, or ledgers must never gate execution. If it influences a decision, it's no longer observability.

3. **Doctrine Is Not Optimized** — Do not tune thresholds, confidences, or veto rates without a falsification trigger. Performance changes require a closed-cycle postmortem.

## PR Checklist

- ✅ `PYTHONPATH=. pytest -q` green (full suite)
- ✅ `ruff check .` clean
- ✅ `mypy .` clean (or documented suppressions)
- ✅ No credentials in commits
