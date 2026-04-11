# GPT-HEDGE v7.9 — Execution Overview

**Report date:** 2026-03-30
**Disclaimer:** Factual description of execution mechanics. No optimization claims.

---

## Execution Pipeline

```
Market Data
    │
    ▼
Sentinel-X ──── Regime classification (6 regimes)
    │
    ▼
Hydra Engine ── Signal generation (6 strategy heads)
    │
    ▼
Cerberus ────── Head multiplier scaling (0.1×–3.0×)
    │
    ▼
Doctrine Gate ─ Entry/exit authorization (SUPREME — cannot be bypassed)
    │
    ▼
Risk Limits ─── Secondary veto (caps, drawdown, correlation, min notional)
    │
    ▼
Minotaur ────── Execution plan (INSTANT / TWAP / STEPPED)
    │
    ▼
Order Router ── Maker-first POST_ONLY with taker fallback
    │
    ▼
Binance Futures Exchange
    │
    ▼
Fill Tracker ── Order ack, fill polling, position cache update
```

Each stage can block execution. Only Doctrine and Risk can prevent market participation.

---

## Veto System

### Doctrine Gate (Primary Authority)

The Doctrine Kernel is the first and supreme check. It enforces hard-coded constitutional laws — there is no configuration, no enabled flag, and no fallback if it vetoes.

**Entry checks (9 gates):**

| # | Gate | Veto if |
|---|------|---------|
| 1 | Regime exists | No regime available |
| 2 | Regime freshness | Regime data > 600 s old |
| 3 | Regime stability | Regime not stable ≥ 2 cycles |
| 4 | Regime confidence | Confidence < 0.45 |
| 5 | Direction alignment | Regime does not permit trade direction |
| 6 | Crisis check | CRISIS regime active |
| 7 | Execution regime | Minotaur regime = CRUNCH or HALT |
| 8 | Head budget | Head budget exhausted |
| 9 | Alpha survival | Alpha survival < 0.20 |

**Direction permission map:**

| Regime | Permitted Directions |
|--------|---------------------|
| TREND_UP | LONG only |
| TREND_DOWN | SHORT only |
| MEAN_REVERT | LONG and SHORT |
| BREAKOUT | LONG and SHORT |
| CHOPPY | None (all trades blocked) |
| CRISIS | None (all trades blocked) |

**Exit triggers (priority-ordered):**

| Priority | Trigger | Action |
|----------|---------|--------|
| 1 | CRISIS_OVERRIDE | Immediate mandatory exit |
| 2 | REGIME_FLIP | Regime no longer permits position direction |
| 3 | REGIME_CONFIDENCE_COLLAPSE | Confidence < 0.315 (30% below floor) |
| 4 | TREND_DECAY | Trend strength < 0.2 |
| 5 | CARRY_DISAPPEARED | Carry spread < 0.1 |
| 6 | TIME_STOP | Position held > 96 bars (~24 h) |
| 7 | STOP_LOSS_SEATBELT | Emergency catastrophe protection only |

All exits are thesis-driven. Positions close when the thesis dies, not on signals.

### Risk Limits (Secondary Veto)

After doctrine approval, `check_order()` applies a second layer of veto.

| Check | Threshold | Veto reason |
|-------|-----------|-------------|
| NAV freshness | > 90 s | `stale_nav` |
| Portfolio drawdown | Exceeds max DD % | `max_dd_breach` |
| Per-symbol cap | Exposure exceeds % of 7d notional | `symbol_cap` |
| Correlation group | Group exposure exceeds max group NAV % | `correlation_breach` |
| Minimum notional | < $25 USDT | `min_notional` |

Every veto is logged to `logs/execution/risk_vetoes.jsonl`. Vetoes are append-only — never rewritten.

---

## Regime Interaction

**Current regime state (2026-03-30 10:25 UTC):**

| Field | Value |
|-------|-------|
| Primary regime | TREND_UP |
| Confidence | 0.6261 (smoothed) |
| Cycle count | 14,656 |
| Crisis flag | false |

**Regime probability distribution:**

| Regime | Probability |
|--------|------------|
| TREND_UP | 0.6261 |
| CHOPPY | 0.2006 |
| MEAN_REVERT | 0.1730 |
| TREND_DOWN | 0.0001 |
| BREAKOUT | 0.0001 |
| CRISIS | 0.0001 |

**Regime features:**

| Feature | Value |
|---------|-------|
| Trend slope | 0.000568 |
| Trend R² | 0.7921 |
| ATR normalized | 0.001735 |
| Mean reversion score | 0.5217 |
| Vol regime Z-score | 0.4912 |

The current regime is TREND_UP with confidence above the doctrine floor (0.45). Only LONG entries are permitted.

---

## Example: Trade Lifecycle

**Scenario:** ETHUSDT LONG entry attempt — 2026-03-24 09:23:30 UTC

```
STEP 1: SIGNAL GENERATION (Hydra — TREND head)
  Symbol:   ETHUSDT
  Side:     LONG
  Quantity: 0.0399
  Price:    $2,152
  Notional: $85.73 USD
  Score:    0.4673 (hybrid conviction)
  NAV:      $9,172.75 (age: 0.29 s — fresh)

STEP 2: DOCTRINE GATE
  Regime:       MEAN_REVERT (confidence 0.5475)
  Check #1:     Regime exists                     → PASS
  Check #3:     Confidence 0.5475 ≥ 0.45          → PASS
  Check #5:     MEAN_REVERT permits LONG           → PASS
  Verdict:      ALLOW
  Sizing mult:  0.6257 (doctrine internal scaling)

STEP 3: RISK LIMITS
  NAV age:      0.29 s < 90 s                     → PASS
  Drawdown:     0.0% < max                        → PASS
  Symbol cap:
    Current ETHUSDT exposure: $1,793.59
    Cap (20% of 7d notional): $1,834.45
    Requested:                $53.72
    Available budget:         $40.86
    Excess:                   $12.86 (31% overshoot)
  Verdict:      VETO — symbol_cap

STEP 4: ORDER REJECTED
  Reason:       symbol_cap
  Trade not placed. Logged to risk_vetoes.jsonl.
```

The trade passed doctrine (regime permitted the direction) but was blocked by risk limits (per-symbol exposure cap insufficient for order size).

---

## Example: Doctrine Veto

**Source:** `logs/doctrine_events.jsonl`

```json
{
  "ts": "2026-02-15T06:04:29.165824+00:00",
  "type": "ENTRY_VERDICT",
  "symbol": "SOLUSDT",
  "verdict": "VETO_DIRECTION_MISMATCH",
  "allowed": false,
  "reason": "Regime CHOPPY does not permit BUY",
  "regime": "CHOPPY",
  "confidence": 0.5553,
  "direction": "BUY",
  "multiplier": 0.0,
  "source_head": "vol_target"
}
```

**Explanation:** The vol_target head generated a BUY signal for SOLUSDT. Sentinel-X classified the regime as CHOPPY. The doctrine direction map for CHOPPY is empty — no trades are permitted. The entry was vetoed with `VETO_DIRECTION_MISMATCH`. The regime confidence (0.5553) was adequate; the direction was not.

---

## Example: Execution Log (Risk Veto)

**Source:** `logs/execution/risk_vetoes.jsonl`

```json
{
  "symbol": "SOLUSDT",
  "side": "BUY",
  "position_side": "LONG",
  "ts": "2026-03-24T09:19:51.511537+00:00",
  "veto_reason": "symbol_cap",
  "veto_detail": {
    "gate": "risk_limits",
    "limit": "symbol_cap",
    "reasons": ["symbol_cap"],
    "notional": 172.17,
    "current_symbol_exposure": 1806.66,
    "symbol_cap": 1834.57,
    "available_budget": 27.91,
    "excess_notional": 144.26,
    "budget_saturated": false,
    "nav_usd": 9171.90,
    "nav_age_s": 97.83,
    "atr_regime": "panic",
    "drawdown_pct": 0.0,
    "dd_state": "normal"
  }
}
```

**Explanation:** A $172.17 notional order for SOLUSDT was submitted. Current symbol exposure was $1,806.66 against a cap of $1,834.57, leaving only $27.91 of available budget. The order exceeded available budget by $144.26. The ATR regime was "panic" (2.7× normal volatility). The veto was logged with full constraint geometry for audit.

---

## Veto Distribution

**Most recent diagnostics (2026-03-30):**

| Metric | Value |
|--------|-------|
| Doctrine total ALLOWs | 66,629 |
| Doctrine total VETOs | 71,642 |
| Doctrine veto rate | 51.8% |
| Risk vetoes (cumulative) | 1,751 |
| Most common risk veto | min_notional (758) |

**Risk veto breakdown (top 5):**

| Reason | Count |
|--------|-------|
| min_notional | 758 |
| symbol_cap | 441 |
| portfolio_dd_circuit | 168 |
| nav_stale | 81 |
| daily_loss | 51 |

---

## Order Router Behavior

| Parameter | Value |
|-----------|-------|
| Default mode | POST_ONLY (maker-first) |
| Maker offset | 2–5 bps below/above mid |
| Mid drift threshold | 5 bps (triggers taker fallback) |
| Max POST_ONLY rejects | 4 (then fallback to taker) |
| Min fill ratio | 40% (below triggers refresh/TWAP) |
| TWAP min notional | $30 USDT per child |
| Taker fee | 5 bps |
| Maker rebate | −1 bps |

Current maker fill ratio is 5.26% — the router predominantly falls back to taker execution.

---

*Source files: `doctrine_kernel.py`, `risk_limits.py`, `sentinel_x.py`, `order_router.py`, `logs/doctrine_events.jsonl`, `logs/execution/risk_vetoes.jsonl`, `logs/state/sentinel_x.json`, `logs/state/diagnostics.json`*
