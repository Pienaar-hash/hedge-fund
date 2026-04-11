# GPT-HEDGE v7.9 — Architecture

**Report date:** 2026-03-30
**System:** Binance USDT-margined futures trading engine with regime-governed execution

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKET DATA                                  │
│            (prices, volumes, order book, trades)                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SENTINEL-X                                                         │
│  Regime classifier: extracts features → scores 6 regimes            │
│  Output: primary regime + confidence + features                     │
│  State: logs/state/sentinel_x.json                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────────────────────┐
│  ALPHA DECAY     │ │ CERBERUS │ │  HYDRA ENGINE                    │
│  (Thanatos)      │ │          │ │  6 strategy heads generate       │
│  Survival probs  │ │ Regime + │ │  intents per symbol              │
│  per symbol      │ │ decay +  │ │  → merge conflicting intents     │
│                  │ │ meta →   │ │  → output: merged_intents[]      │
│                  │ │ head     │ │  State: logs/state/hydra_state.json│
│                  │ │ mults    │ │                                    │
│                  │ │ (0.1-3x) │ │                                    │
└──────────────────┘ └────┬─────┘ └──────────────┬───────────────────┘
                          │                       │
                          └───────────┬───────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DOCTRINE KERNEL (SUPREME AUTHORITY)                                │
│  Entry gate: 9 hard-coded checks (regime, direction, confidence,   │
│              stability, head budget, alpha survival)                 │
│  Exit gate:  7 thesis-driven triggers (crisis, regime flip,        │
│              confidence collapse, trend decay, time stop)           │
│  Verdict: ALLOW or VETO (logged to doctrine_events.jsonl)          │
│  Cannot be bypassed. No configuration. No enabled flag.            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RISK LIMITS                                                        │
│  Secondary veto: per-symbol cap, portfolio DD, correlation          │
│  exposure, min notional, NAV freshness                              │
│  Logged to: logs/execution/risk_vetoes.jsonl                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MINOTAUR ENGINE                                                    │
│  Microstructure sensing: spread, depth, book imbalance, vol         │
│  Classifies execution regime: NORMAL / THIN / WIDE_SPREAD /        │
│  SPIKE / CRUNCH                                                     │
│  Produces execution plan: INSTANT / TWAP / STEPPED                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ORDER ROUTER                                                       │
│  Maker-first: POST_ONLY with taker fallback (max 4 rejects)       │
│  Fee-aware pricing (maker rebate −1 bps, taker fee 5 bps)          │
│  TWAP splitting for large orders (min child $30)                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BINANCE FUTURES EXCHANGE                                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FILL TRACKER + POSITION CACHE                                      │
│  Order ack → fill polling (500 ms interval, 8 s timeout)           │
│  Position cache: 1 s TTL, invalidate on confirmed fill             │
│  PnL attribution → episode ledger                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Detail

### Sentinel-X — Regime Classifier

**File:** `execution/sentinel_x.py`

Classifies the market into one of six regimes using extracted statistical features and hard-coded crisis overrides.

**Regimes:**

| Regime | Meaning | Trade Permission |
|--------|---------|------------------|
| TREND_UP | Strong positive directional bias | LONG only |
| TREND_DOWN | Strong negative directional bias | SHORT only |
| MEAN_REVERT | Range-bound, mean-reverting | LONG and SHORT |
| BREAKOUT | Volatility expansion + directional intent | LONG and SHORT |
| CHOPPY | High noise, low signal | No trades |
| CRISIS | Extreme market stress | No trades |

**Extracted features:**

| Feature | Description |
|---------|-------------|
| `returns_mean` | Directional bias |
| `returns_std` | Volatility |
| `returns_skew` | Asymmetry |
| `returns_kurtosis` | Tail heaviness |
| `atr_norm` | ATR / price (normalized volatility) |
| `vol_regime_z` | Z-score vs historical volatility |
| `trend_slope` | Log-price linear regression slope |
| `trend_r2` | Trend fit strength |
| `trend_acceleration` | Second derivative of trend |

**Crisis overrides (hard rules, bypass ML):**
- Portfolio drawdown > 12% → CRISIS
- Volatility spike > 3× normal → CRISIS

**Stability hysteresis:** Entry threshold × 0.67 for exit (prevents regime flapping).

---

### Hydra Engine — Multi-Strategy Signal Generator

**File:** `execution/hydra_engine.py`

Promotes six strategy heads from intelligence weights to first-class strategy tracks with per-head budgets, conflict resolution, and a unified intent stream.

**Strategy heads:**

| Head | Function | Max NAV % |
|------|----------|-----------|
| TREND | Directional momentum / trend following | 50% |
| MEAN_REVERT | Mean reversion / reversal plays | 25% |
| RELATIVE_VALUE | Cross-pair (crossfire) trades | 30% |
| CATEGORY | Category rotation / sector tilt | 20% |
| VOL_HARVEST | Volatility harvesting via position sizing | 20% |
| EMERGENT_ALPHA | Prospector / universe expansion | 15% |

**Key data structures:**

| Structure | Purpose |
|-----------|---------|
| `HydraIntent` | Single head signal: head, symbol, side, nav_pct, score |
| `HydraMergedIntent` | Post-conflict resolution: net_side, head_contributions map |
| `HydraHeadBudget` | Budget tracking: max_nav_pct, used_nav_pct, position_count |
| `HydraState` | Full snapshot: budgets, usage, positions, merged_intents, cycle_count |

**Signal flow:** Each head generates `HydraIntent` objects → intents are scaled by Cerberus multipliers → conflicting intents on the same symbol are merged (max 3 heads per symbol) → output: `HydraMergedIntent[]`.

---

### Doctrine Kernel — Supreme Gating Authority

**File:** `execution/doctrine_kernel.py`

The Doctrine Kernel is the sole authority for trade authorization. It is hard-coded, has no configuration, no enabled flag, and cannot be bypassed. Every veto is logged.

**Entry verdict function signature:**
```
doctrine_entry_verdict(regime, intent, execution, portfolio) → DoctrineVerdict
```

**Entry gates (9 checks):**

| # | Check | Veto Condition |
|---|-------|----------------|
| 1 | Regime exists | No regime available |
| 2 | Regime freshness | Regime data > 600 s old |
| 3 | Regime stability | Not stable ≥ 2 cycles |
| 4 | Confidence floor | Confidence < 0.45 |
| 5 | Direction match | Regime does not permit trade direction |
| 6 | Crisis block | CRISIS regime active |
| 7 | Execution crunch | Minotaur regime = CRUNCH |
| 8 | Head budget | Head budget exhausted |
| 9 | Alpha survival | Alpha survival < 0.20 |

**Exit verdict (7 thesis-driven triggers):**

| Priority | Reason | Urgency |
|----------|--------|---------|
| 1 | CRISIS_OVERRIDE | IMMEDIATE |
| 2 | REGIME_FLIP | IMMEDIATE |
| 3 | REGIME_CONFIDENCE_COLLAPSE | STEPPED |
| 4 | TREND_DECAY | STEPPED |
| 5 | CARRY_DISAPPEARED | PATIENT |
| 6 | TIME_STOP (> 96 bars) | PATIENT |
| 7 | STOP_LOSS_SEATBELT | IMMEDIATE |

**Kill switch exemption:** The kill switch may block new entries but must never block `reduceOnly` exits issued under doctrine authority. Enforced via two-flag guard (`doctrine_exit=True AND reduceOnly=True`).

---

### Order Router — Maker-First Execution

**File:** `execution/order_router.py`

Places orders using a maker-first strategy with automatic fallback.

**Execution sequence:**

1. Check mid-price drift vs last trade (threshold: 5 bps)
2. If drift acceptable → POST_ONLY limit order at 2–5 bps offset from mid
3. Monitor for rejects (max 4 before fallback)
4. If fill ratio < 40% → TWAP split or taker fallback
5. Fee-aware pricing: maker rebate (−1 bps), taker cost (5 bps)

**TWAP parameters:**
- Min notional per child: $30 USDT
- Min TWAP duration: 60 s
- Max TWAP duration: 900 s (15 min)

---

### Cerberus Router — Head Multiplier Engine

**File:** `execution/cerberus_router.py`

Dynamically scales strategy head weights based on regime and market signals. Cerberus does not create signals and does not override doctrine.

**Input signals (weighted combination):**

| Signal | Weight | Source |
|--------|--------|--------|
| Regime | 0.25 | Sentinel-X confidence |
| Decay | 0.15 | Alpha survival (Thanatos) |
| Meta | 0.15 | Meta-scheduler overlays |
| Edge | 0.20 | Symbol edge scores |
| Health | 0.15 | Strategy health |
| Universe | 0.05 | Universe optimizer |
| Relative value | 0.05 | RV momentum |

**Multiplier bounds:** Clamped to [0.10, 3.00]. Per-regime baseline weights adjust heads (e.g., TREND boosted to 1.3× in TREND_UP, reduced to 0.7× in MEAN_REVERT).

---

### Minotaur Engine — Microstructure Execution

**File:** `execution/minotaur_engine.py`

Senses market microstructure and selects execution aggressiveness.

**Execution regimes:**

| Regime | Trigger | Effect |
|--------|---------|--------|
| NORMAL | Good liquidity, tight spreads | Standard execution |
| THIN | Depth < $2,000 | Slippage risk — passive |
| WIDE_SPREAD | Spread > 10 bps | Fee-aware pricing |
| SPIKE | Vol > 2.5× normal | Reduced aggressiveness |
| CRUNCH | THIN + WIDE_SPREAD | Doctrine blocks entry |

**Execution plans:**

| Mode | When | Behavior |
|------|------|----------|
| INSTANT | Small notional, good liquidity | Single order |
| TWAP | Large notional (> $500) | Time-sliced children |
| STEPPED | Moderate size, constrained liquidity | Stepped submission |

---

### Supporting Components

| Component | File | Purpose |
|-----------|------|---------|
| Fill Tracker | `execution/fill_tracker.py` | Async fill polling (500 ms interval, 8 s timeout), order ack, PnL close |
| Position Cache | `execution/position_cache.py` | 1 s TTL cache for exchange positions; invalidate on fill |
| Order Dispatch | `execution/order_dispatch.py` | Exchange dispatch, maker-first logic, retry loop |
| NAV | `execution/nav.py` | `nav_health_snapshot()` — sole source of NAV truth (futures wallet only) |
| Sizing | `execution/sizing.py` | Position sizing: `nav_pct_fraction`, `size_from_nav` |
| Helpers | `execution/helpers.py` | Pure stateless utilities: `to_float`, `ms_to_iso`, etc. |
| Position Ledger | `execution/position_ledger.py` | Unified position + TP/SL registry |
| DLE Shadow | `execution/dle_shadow.py` | Decision Ledger Engine — observation only, never blocks |

---

### Executor Main Loop

**File:** `execution/executor_live.py` (~5,700 lines)

Single-threaded event loop orchestrating all components:

```
while True:
    Phase 1 — Market intelligence
        Fetch regime (Sentinel-X)
        Fetch NAV (futures wallet)
        Fetch positions (cached, 1 s TTL)

    Phase 2 — Signal generation
        Hydra generates per-head intents
        Cerberus scales by multipliers
        Intents merged and ranked

    Phase 3 — Gating and routing
        For each merged intent:
            Doctrine entry verdict → ALLOW / VETO
            Risk limits check      → ALLOW / VETO
            Minotaur builds execution plan
            Router places order

    Phase 4 — Exit scanning
        For each open position:
            Doctrine exit verdict → HOLD / exit reason + urgency
            If exit: place reduce-only order

    Phase 5 — Fills and state persistence
        Poll pending fills
        Update state files (logs/state/*.json)

    Sleep (cycle time)
```

---

## State File Registry

The system writes 63 registered state files to `logs/state/`. The dashboard reads these files (never writes). Changes to state surfaces must be strictly additive.

**Core state files:**

| File | Owner | Frequency | Purpose |
|------|-------|-----------|---------|
| `nav_state.json` | sync_state | per_sync | NAV, AUM, drawdown |
| `positions_state.json` | executor | per_loop | Live positions |
| `positions_ledger.json` | executor | per_loop | Positions + TP/SL |
| `risk_snapshot.json` | executor | per_loop | Risk KPIs |
| `diagnostics.json` | executor | per_loop | Health, veto counters |

**Strategy state files:**

| File | Purpose |
|------|---------|
| `sentinel_x.json` | Regime probs, features, crisis flags |
| `hydra_state.json` | Head budgets, usage, merged intents |
| `hydra_pnl.json` | Per-head equity, drawdown, throttle |
| `cerberus_state.json` | Head multipliers, regime, health |

**Intel state files:**

| File | Purpose |
|------|---------|
| `symbol_scores_v6.json` | Hybrid scores with modifiers |
| `factor_diagnostics.json` | Normalized factors, covariance |
| `alpha_decay.json` | Alpha half-life, survival curves |
| `router_health.json` | Slippage, latency, fill quality |
| `execution_quality.json` | Per-symbol execution metrics |

**Observability contract:** State files are observability only. They must never gate execution or influence trading decisions.

---

## Dependency Graph

```
execution/ → config/       (reads configuration)
execution/ → logs/         (writes state + event logs)
dashboard/ → logs/state/   (reads state — NEVER writes)

execution/ ✗ dashboard/    (never imports from dashboard)
dashboard/ ✗ execution/    (no reverse dependency)
```

---

*Source files: `execution/hydra_engine.py`, `execution/sentinel_x.py`, `execution/doctrine_kernel.py`, `execution/order_router.py`, `execution/cerberus_router.py`, `execution/minotaur_engine.py`, `execution/fill_tracker.py`, `execution/position_cache.py`, `execution/executor_live.py`, `v7_manifest.json`*
