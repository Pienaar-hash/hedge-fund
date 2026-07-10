# EXECUTION ARCHITECTURE

**System version:** v7.9  
**Last updated:** 2026-06-05  
**Canonical state contracts:** v7.6

---

## Overview

The execution system is a pipeline of independently auditable stages. Each stage logs its inputs, decision, and output. No stage bypasses the stage before it.

```
Market data (Binance API)
        │
        ▼
Signal Generator  ──────────────────────────────────────────────────────────┐
(signal_generator.py)                                              (fallback │ legacy)
        │                                                                    │
        ▼                                                                    │
Hydra Multi-Head Engine (hydra_engine.py)                                   │
  6 heads: TREND, MEAN_REVERT, RELATIVE_VALUE,                              │
           CATEGORY, VOL_HARVEST, EMERGENT_ALPHA                            │
  Conflict resolution: max 3 heads per symbol                               │
        │                                                                    │
        └──── merged_intents[] ──────────────────────────────────────────────┘
                                │
                                ▼
              Doctrine Kernel (doctrine_kernel.py)  ◄─ SUPREME AUTHORITY
                11 entry veto types / 9 exit reasons (strict priority)
                All verdicts logged to logs/doctrine_events.jsonl
                                │
                    ┌───────────┴────────────┐
                ALLOW                      VETO
                    │                       │
                    ▼                  logged to
              Risk Engine              risk_vetoes.jsonl
              (risk_engine_v6.py)
              Risk mode: OK/WARN/DEFENSIVE/HALTED
                    │
                    ▼
              Risk Limits (risk_limits.py)
              Per-symbol caps, portfolio DD,
              correlation exposure, daily loss
                    │
                    ▼
              Order Router (order_router.py)
              POST_ONLY + taker fallback
              TWAP for large orders
              Fee-aware pricing
                    │
                    ▼
              Exchange Utils (exchange_utils.py)
              Binance UM Futures client
              Position mode: ONE-WAY or HEDGE (auto-detected)
                    │
                    ▼
              Fill Tracker (fill_tracker.py)
              500ms poll, 8s timeout
              Trade summaries → position ledger
                    │
                    ▼
              Position Ledger (position_ledger.py)
              PnL attribution → episode ledger
                    │
                    ▼
              State surfaces (logs/state/)
              nav.json, positions_state.json,
              risk_snapshot.json, diagnostics.json
```

---

## Service Processes

### 1. hedge-executor (`execution/executor_live.py`)
- **Cycle:** 5-second polling loop
- **Responsibilities:** Run full pipeline, emit heartbeats, write all state surfaces
- **Single writer invariant:** Only the executor writes `logs/state/hydra_state.json`
- **Supervisor:** `autorestart=true`, `startretries=3`
- **Logs:** `/var/log/supervisor/hedge-executor.log`

### 2. hedge-sync_state (`execution/sync_state.py`)
- **Responsibilities:** Read local state surfaces, publish to Firestore (when enabled)
- **Fail-open:** Local files are always primary; Firestore is optional remote mirror
- **NAV cutoff:** Optional `CUTOFF_ISO` / `CUTOFF_SECAGO` env vars for historical replay

### 3. hedge-dashboard (`dashboard/app.py`)
- **Port:** 8501 (Streamlit, `server.address=0.0.0.0`)
- **Data source:** Reads from `logs/state/` and `logs/cache/` only; no direct exchange calls
- **Cache TTL:** 30 seconds per file read
- **Panels:** NAV, AUM, Regime, Risk, Router quality, Execution quality, PnL attribution, Hydra heads, Treasury, Diagnostics

---

## Exchange Connector: Binance UM Futures

**Module:** `execution/exchange_utils.py`  
**Client library:** `binance.um_futures.UMFutures`

### Position Mode Handling
- Auto-detected on startup via `refresh_dual_side_cache()`
- **ONE-WAY:** `positionSide` field stripped from all orders (prevents error -4061)
- **HEDGE (dual-side):** `positionSide` included (LONG / SHORT)

### Order Placement
- Default: `POST_ONLY` maker order at best-bid/ask with fee adjustment
- Fallback: Up to 4 POST_ONLY rejections before switching to `MARKET` taker
- TWAP: ≥4 slices, 10s intervals, min child order $30–50

### Precision Management
- `execution/exchange_precision.py` — lot sizes, min notional, tick sizes
- Cached at `config/exchange_precision_cache.json` (800+ symbols)
- Filter drift protection: alerts if MARKET_LOT_SIZE inflates >10×

### Testnet vs Mainnet
| Env var | Testnet | Live |
|---------|---------|------|
| `BINANCE_TESTNET` | 1 | 0 |
| API base URL | `https://testnet.binancefuture.com` | `https://fapi.binance.com` |

---

## State Surfaces

All state surfaces live in `logs/state/`. The executor is the **sole writer**. Dashboard and sync_state are readers only.

| File | Path | Content | Freshness |
|------|------|---------|-----------|
| `nav.json` | `logs/state/nav.json` | Current NAV, equity, unrealized PnL, source | Each cycle |
| `nav_state.json` | `logs/state/nav_state.json` | Mirror of nav.json; dashboard fallback | Each cycle |
| `positions_state.json` | `logs/state/positions_state.json` | Open positions, entry price, size, side | On fill |
| `risk_snapshot.json` | `logs/state/risk_snapshot.json` | DD %, daily loss, risk mode | Each cycle |
| `hydra_state.json` | `logs/state/hydra_state.json` | Head budgets, usage, merged intents | Each cycle |
| `diagnostics.json` | `logs/state/diagnostics.json` | Veto counters, pipeline liveness, alerts | Each cycle |
| `router_health.json` | `logs/state/router_health.json` | Fill ratio, slippage mean, spread mean (state surface) | Each cycle |
| `telegram_state.json` | `logs/state/telegram_state.json` | Alert dedupe state | On alert send |
| `nav_health.json` | **`logs/nav_health.json`** | NAV age, source freshness | Each cycle |

---

## Execution Logs (JSONL)

Most append-only JSONL logs are under `logs/execution/`, but two live in `logs/` directly:

| File | Path | Content |
|------|------|---------|
| `orders_executed.jsonl` | `logs/execution/` | Filled orders: fill price, qty, timestamp, fee |
| `orders_attempted.jsonl` | `logs/execution/` | All submissions: successes + rejections + vetoes |
| `risk_vetoes.jsonl` | `logs/execution/` | Risk-layer veto decisions on new entries |
| `fee_gate_events.jsonl` | `logs/execution/` | Fee gate verdicts per order intent |
| `signal_metrics.jsonl` | `logs/execution/` | Signal: symbol, side, confidence, edge, regime |
| `router_metrics.jsonl` | `logs/execution/` | Router: slippage, spread, fill ratio, TWAP stats (mirror) |
| `execution_health.jsonl` | `logs/execution/` | Uptime, error rate, ATR regime distribution |
| `sync_heartbeats.jsonl` | `logs/execution/` | Sync state service heartbeats |
| `pub_tick_heartbeat.jsonl` | `logs/execution/` | `_pub_tick()` boundary trace — proves call reached / entered / completed / failed / aborted (see below) |
| `doctrine_events.jsonl` | **`logs/`** | Doctrine entry/exit verdicts (ALLOW, VETO, exit reasons) |
| `router_health.jsonl` | **`logs/`** | Router health log (separate from the state surface) |

**NAV history:** `logs/nav_log.json` — single JSON array, entries keyed `{"t": <unix_ts>, "nav": <float>}`. The writer appends without truncation; there is no size cap in the current code. Read with Python/jq, not `tail`.

---

## Publication Heartbeat Diagnostics (`pub_tick_heartbeat.jsonl`)

**Card:** `CARD-HEDGE-PUBTICK-BOUNDARY-HEARTBEAT-INSTRUMENTATION-001`, following the freeze investigation in `docs/audits/CARD-HEDGE-LIVE-STATE-PUBLISH-FREEZE-ROOT-CAUSE-001.md`. Purely diagnostic — never read as a trading-authority input.

`_pub_tick()` (`execution/executor_live.py`) publishes `nav.json`, `positions_state.json`, `diagnostics.json`, and related state surfaces once per poll-gated loop cycle. It emits a small, bounded set of JSONL events to `logs/execution/pub_tick_heartbeat.jsonl` so that a stalled or skipped publication cycle can be diagnosed without guessing from timestamp correlation alone:

| Event | Emitted when |
|-------|--------------|
| `CALL_REACHED` | The main loop is about to call `_pub_tick()` |
| `ENTERED` | First instruction inside `_pub_tick()`, before any state computation |
| `LEDGER_REBUILD_STARTED` | The throttled episode-ledger rebuild is due and about to run |
| `LEDGER_REBUILD_COMPLETED` | The episode-ledger rebuild returned, with `duration_ms` |
| `COMPLETED` | `_pub_tick()` reached its normal terminal path (`outcome=ok`) |
| `FAILED` | An ordinary `Exception` escaped `_pub_tick()` (`outcome=failed`, with `exception_type`/`exception_message`); the exception is re-raised unchanged so existing `publish_tick_failed` call-site behavior is preserved |
| `ABORTED` | `_pub_tick()` was entered but a non-`Exception` `BaseException` (e.g. `KeyboardInterrupt`) or other unforeseen escape prevented reaching `COMPLETED`/`FAILED`; only observed, never suppressed |

Each event carries `ts`, `monotonic_s`, `loop_id`, `duration_ms`, `episode_ledger_due`, `episode_ledger_duration_ms`, `last_completed_ts`, `last_completed_duration_ms`, `outcome`, `exception_type`, `exception_message`, `engine_version`, `git_sha`. Writes go through the same fail-open `JsonlLogger` used by `execution_health.jsonl`; a telemetry-write failure is rate-limited to the executor's own log and can never block or crash `_pub_tick()`.

**Diagnostic procedure for a future abnormal publication gap:**

- `CALL_REACHED` absent → caller/scheduler path skipped before reaching `_pub_tick()`.
- `CALL_REACHED` present, `ENTERED` absent → invocation-boundary anomaly.
- `ENTERED` present, ledger `STARTED` but no `COMPLETED`, no `COMPLETED`/`FAILED`/`ABORTED` → ledger rebuild blockage or termination.
- `ENTERED` present, no ledger `STARTED`, no terminal event → failure in the pre-ledger publication section.
- Terminal `COMPLETED` present but state files unchanged → writer or filesystem/state-path defect (heartbeat cannot prove file-write success, only that the code path ran).
- All heartbeat events delayed together with everything else in the loop → broader scheduling/process-starvation defect, not specific to `_pub_tick()`.

---

## Monitoring / Liveness

`execution/sync_state.py` watches for idle processes:

| Subsystem | Max idle before alert |
|-----------|----------------------|
| Signal generation | 900s (15 min) |
| Order routing | 1800s (30 min) |
| Exit scanning | 3600s (60 min) |
| Router events | 1800s (30 min) |

Heartbeat interval: executor = 60s, sync_state = 60s.

Liveness alerts: emitted to `logs/state/diagnostics.json` and optionally Telegram.

---

## Telegram Alerting

**Module:** `execution/telegram_alerts_v7.py`  
**Config:** `config/telegram_v7.json`  
**State:** `logs/state/telegram_state.json`  
**Rate limit:** 10 messages/minute  
**Behavior:** State-driven (deduplicates repeated alerts); fails silently; never crashes execution

| Alert type | Trigger |
|-----------|---------|
| ATR regime change | Volatility tier changes |
| Drawdown state | Risk mode transition |
| Router quality | Fill ratio or slippage threshold breached |
| 4h candle summary | Every 4h, positions + PnL summary |
| Heartbeat | Every 4h if no other alerts |

**Enable:** Set `TELEGRAM_ENABLED=1` in `.env`. Provide credentials via `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (env var names come from `config/telegram_v7.json` fields `bot_token_env` / `chat_id_env`). Additional operator flags: `EXEC_TELEGRAM_4H_ONLY=1` (strict mode, only 4h summaries), `EXEC_TELEGRAM_MAX_PER_MIN=0` (emergency block).

---

## Firestore

**Module:** `execution/firestore_utils.py`  
**Status:** Disabled by default (`FIRESTORE_ENABLED=0`)  
**Collections:** execution state, diagnostics, risk snapshots  
**Safety gate:** `ALLOW_PROD_WRITE=0` prevents accidental mainnet writes  
**Credentials:** `GOOGLE_APPLICATION_CREDENTIALS` env var

Firestore is a read-replica. The authoritative state is always the local `logs/state/` files.

---

## Supervisor

**Config:** `/etc/supervisor/conf.d/hedge.conf`  
**Group:** `[group:hedge]` — all three processes

```bash
# Start all
supervisorctl start hedge:*

# Stop executor only (e.g., for emergency halt)
supervisorctl stop hedge-executor

# Tail executor logs
supervisorctl tail -f hedge-executor

# Reload config after editing
supervisorctl reread && supervisorctl update
```

---

## Dependency Graph (key modules)

```
executor_live.py
  ├── hydra_engine.py
  │     └── signal_generator.py
  ├── doctrine_kernel.py
  ├── risk_engine_v6.py
  │     └── risk_limits.py
  │           └── drawdown_tracker.py
  ├── order_router.py
  │     ├── exchange_utils.py
  │     │     └── exchange_precision.py
  │     └── fill_tracker.py
  ├── nav.py
  ├── position_ledger.py
  └── telegram_alerts_v7.py
```

---

## What KuCoin Is

Not integrated. References in `archive/` only. All live + testnet trading is Binance exclusively.
