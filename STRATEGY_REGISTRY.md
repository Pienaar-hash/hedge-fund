# STRATEGY REGISTRY

**Config source:** `config/strategy_registry.json` + `config/strategy_config.json`  
**Registry version:** v6.4  
**Last updated:** 2026-06-05

---

## Active Universe

The canonical runtime universe is defined in `config/pairs_universe.json` and loaded by `execution/universe_resolver.py`. As of v6.0, **BTCUSDT**, **ETHUSDT**, and **SOLUSDT** are all `enabled: true` in that file.

Note: `config/strategy_config.json` contains a competing `"universe": ["BTCUSDT", "ETHUSDT"]` list (BTC + ETH only). This is an unresolved split in the config — `pairs_universe.json` is authoritative for the executor and risk engine. Treat SOL as active unless explicitly disabled there.

All assets are Binance UM Futures USDT-margined perpetuals.

---

## Strategy Table

| ID | Asset | Strategy registry | Universe (pairs_universe.json) | Confidence | Capacity (USD) | Timeframe |
|----|-------|-----------------|-------------------------------|------------|----------------|-----------|
| btc_micro | BTCUSDT | ✅ enabled | ✅ enabled | 0.65 | $500 | 15m |
| eth_micro | ETHUSDT | ✅ enabled | ✅ enabled | 0.55 | $350 | 15m |
| sol_micro | SOLUSDT | ✅ enabled | ✅ enabled | 0.50 | $250 | 15m |
| link_micro | LINKUSDT | ❌ disabled | ❌ disabled | — | — | — |
| ltc_micro | LTCUSDT | ❌ disabled | ❌ disabled | — | — | — |
| sui_micro | SUIUSDT | ❌ disabled | ❌ disabled | — | — | — |
| wif_micro | WIFUSDT | ❌ disabled | ❌ disabled | — | — | — |

---

## Signal Generation Architecture

Signals pass through **two parallel pipelines** before reaching the executor:

### 1. Legacy Single-Engine (fallback)
`execution/signal_generator.py`  
- Loads enabled strategies from registry
- Applies ATR-based volatility gate (lookback=50, mult=8.0)
- Applies minimum expectancy gate (expectancy_min = -0.10)
- Outputs intents per symbol

### 2. Hydra Multi-Head Engine (primary, v7.9)
`execution/hydra_engine.py`  
Six strategy heads with per-head budgets:

| Head | Focus | Conflict weight |
|------|-------|-----------------|
| TREND | Directional momentum | High |
| MEAN_REVERT | Short-term reversion | Medium |
| RELATIVE_VALUE | Cross-asset spread | Medium |
| CATEGORY | Sector rotation | Low |
| VOL_HARVEST | Volatility premium | Low |
| EMERGENT_ALPHA | Adaptive ensemble | Low |

Conflict resolution: max 3 heads per symbol. Hydra output passes to the **Doctrine Kernel** for final approval.

---

## Core Signal Parameters

### Momentum / Trend (primary signal type)
| Parameter | Value |
|-----------|-------|
| Timeframe | 15m candles |
| EMA fast | 15 bars |
| EMA slow | 50 bars |
| RSI length | 14 bars |
| RSI buy threshold | 55 |
| RSI sell threshold | 40 |
| Z-score lookback | 48 bars |
| Z-score entry | ±1.0 |
| Z-score exit | ±0.5 |

### Hybrid Scoring (Hydra)
| Parameter | Value |
|-----------|-------|
| Trend weight (w_trend) | 0.70 |
| Carry weight (w_carry) | 0.30 |
| ML scoring | Disabled |

### Signal Gates (applied before order routing)
| Gate | Threshold |
|------|-----------|
| ATR gate | lookback=50, mult=8.0 |
| Expectancy minimum | -0.10 |
| Fee gate | taker 0.04%, maker 0.02%, buffer 1.5x |
| Churn guard | 120s min hold, 300s cooldown |
| Exit dedup TTL | 300s per symbol/side/reason |

---

## Doctrine Kernel (Supreme Authority)

`execution/doctrine_kernel.py` — cannot be bypassed.

**Entry verdicts** (`DoctrineVerdict` enum — 11 veto types + ALLOW):

| Verdict | Meaning |
|---------|---------|
| `ALLOW` | Entry permitted |
| `VETO_NO_REGIME` | No regime snapshot available |
| `VETO_REGIME_STALE` | Regime data older than 600s |
| `VETO_REGIME_UNSTABLE` | Regime not stable for required cycles |
| `VETO_REGIME_CONFIDENCE` | Regime confidence < 0.45 |
| `VETO_DIRECTION_MISMATCH` | Signal direction not permitted by current regime |
| `VETO_CRISIS` | Regime is CRISIS or crisis_flag set |
| `VETO_EXECUTION_CRUNCH` | Execution environment in CRUNCH/HALT state |
| `VETO_NO_HEAD_BUDGET` | No Hydra head has remaining budget |
| `VETO_ALPHA_ROUTER_FLOOR` | Alpha router score below minimum floor |
| `VETO_ALPHA_SURVIVAL` | Alpha survival probability < 0.20 |
| `VETO_ENVIRONMENT_DEGRADED` | System environment degraded |

**Exit reasons** (`ExitReason` enum — 9 exit types, strict priority order):

| Priority | Reason | Urgency | Condition |
|----------|--------|---------|-----------|
| 0 | `HOLD` | — | No exit required; doctrine keeps position open |
| 1 | `CRISIS_OVERRIDE` | Immediate | crisis_flag set or regime == CRISIS |
| 2 | `REGIME_FLIP` | **Stepped** | Regime direction reversed against entry regime |
| 2 | `REGIME_CONFIDENCE_COLLAPSE` | Stepped | Regime confidence < 0.315 (0.45 floor × 0.7) |
| 3 | `TREND_DECAY` | **Patient** | trend_strength < 0.2 |
| 3 | `CARRY_DISAPPEARED` | — | Declared in enum; current code has `pass` — not emitted |
| 3 | `CROSSFIRE_RESOLVED` | **Patient** | crossfire_spread_remaining < 0.1 |
| 3 | `EXECUTION_ALPHA_DRAG` | **Patient** | execution_drag_bps > 20 (partial 50% exit) |
| 4 | `TIME_STOP` | Patient | bars_held ≥ 96 AND unrealized PnL < 0.5% |
| 5 | `STOP_LOSS_SEATBELT` | Immediate | sl_price breached (emergency only) |

All entry and exit verdicts logged to `logs/doctrine_events.jsonl`.

---

## Regime Detection: Sentinel-X

Detects macro regime from feature scores: momentum, mean reversion, z-score, order book imbalance, ATR volatility tier.

Output feeds Doctrine Kernel entry/exit gates. No trading is permitted in regimes that contradict a position's direction.

---

## Position Sizing

| Parameter | Value |
|-----------|-------|
| Default leverage | 3× |
| Max leverage (BTC) | 4× |
| Slippage assumption | 3 bps |
| Taker fee assumption | 5 bps |
| Volatility scalar | 0.25× (very high) → 1.0× (normal) → 0.75× (low) |
| Adaptive factors | risk_factor × atr_factor × dd_factor × perf_factor |

---

## How to Add a Strategy

1. Add entry to `config/strategy_registry.json` with id, module, enabled=false, confidence, capacity_usd.
2. Implement module under `strategies/` matching the existing interface.
3. Backtest with `research/backtest_engine_v8.py`. Log results in `BACKTEST_LEDGER.md`.
4. Enable in strategy_registry.json only after backtest passes validity audit.
5. Paper trade on testnet for ≥2 weeks before enabling at full capacity.

---

## What's NOT Here

- Discretionary / manual signal overrides: not implemented
- Options strategies: not implemented
- Spot trading: not implemented (futures only)
- Multi-exchange arbitrage: not implemented
