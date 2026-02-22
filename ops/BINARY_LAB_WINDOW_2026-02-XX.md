# Binary Lab — Experiment Window Launch Memo

**Date:** Pending (earliest: post-contraction Day 14 + dispersion proof)  
**System:** GPT Hedge v7.9-S1  
**Status:** FORMALIZED — not yet deployed  
**Classification:** Satellite Lab — Fund-Ops Governance  
**Sleeve ID:** `binary_lab_s1`

---

## 0. Classification

This is a **satellite lab**, not a pivot.

Binary is not a hedge against futures.  
Binary is not a replacement for futures.  
Binary is a **diagnostic mirror**.

If signal works in binary but not futures → execution friction is the problem.  
If signal fails in both → signal math is the problem.  

That clarity is worth more than profit.

---

## 1. Hypothesis

> High-conviction directional signals (top quantile) exhibit positive expectancy over short fixed-horizon binary markets.

Not "binary markets are profitable."  
Not "crypto futures don't work."  
Just that one statement. Falsifiable.

---

## 2. Deterministic Rule Set

We reuse the same signal logic from futures.

No new signal.  
No new scoring.  
No new indicators.

### Entry Condition

```
IF:
    conviction_band >= MEDIUM
    AND regime != CHOPPY (or regime_confidence >= threshold)
THEN:
    Enter binary in signal direction
```

That's it.

No spread-based tweaks.  
No microstructure guessing.  
No tape reading.

### Time Horizon

**15-minute rounds.**

Rationale:
- 5-minute is too noisy under CHOPPY regime
- 1-hour gives too few trades for 30-day measurement
- 15-minute provides ~20–40 eligible windows/day, enough for statistical power
- Balances feedback speed against variance

This is locked at deployment. No mid-experiment horizon changes.

---

## 3. Fixed Risk Envelope

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sleeve capital | $2,000 USDC | Segregated — never blended with core NAV |
| Max deployed | $1,200 | 60% of sleeve |
| Reserve (untouchable) | $800 | 40% of sleeve |
| Per-round size | $20 (1% of sleeve) | Fixed. No escalation. |
| Max concurrent | 3 | Hard cap |
| Same-direction stacking | Prohibited | No doubling down |
| Martingale | Prohibited | No loss-chasing |
| Size escalation after wins | Prohibited | No momentum betting |
| Sleeve drawdown kill | -$300 (-25% of reserve-adjusted) | Experiment ends immediately |
| Sleeve drawdown kill NAV | $1,700 | = $2,000 - $300 |

### Kill Line

If sleeve NAV ≤ $1,700, experiment ends.

Immediately.  
No reconsideration.  
No "one more trade."

---

## 4. 30-Day Parameter Freeze

**From first deployment (Day 0) through Day 30:**

| Forbidden Action | Rationale |
|------------------|-----------|
| Threshold adjustments | Contaminates measurement |
| Regime threshold changes | Contaminates measurement |
| Conviction mapping tweaks | Contaminates measurement |
| Timing adjustments (horizon) | Contaminates measurement |
| Adding filters | Contaminates measurement |
| Removing filters | Contaminates measurement |
| Size changes | Contaminates measurement |
| Kill line changes | Governance violation |

If dispersion is weak, that's data.  
You do not "fix" it mid-stream.

Any parameter change during the freeze = experiment invalidated.

---

## 5. Measurement Plan

### Per-Trade Log Fields

Every trade is logged to `logs/execution/binary_lab_trades.jsonl`:

| Field | Type | Description |
|-------|------|-------------|
| `ts` | string (ISO 8601) | Entry timestamp |
| `ts_expiry` | string (ISO 8601) | Round expiry timestamp |
| `symbol` | string | Trading pair |
| `direction` | string | `LONG` or `SHORT` |
| `conviction_band` | string | `very_high`, `high`, `medium` |
| `conviction_score` | float | Raw conviction score |
| `hybrid_score` | float | Composite hybrid score |
| `regime` | string | Sentinel-X regime at entry |
| `regime_confidence` | float | Regime confidence at entry |
| `implied_probability` | float | Market-implied win probability |
| `size_usd` | float | Position size in USD |
| `payout_ratio` | float | Binary payout ratio |
| `outcome` | string | `WIN` or `LOSS` |
| `pnl_usd` | float | Realized PnL |
| `r_multiple` | float | Return / risk |
| `concurrent_count` | int | How many positions open at entry |
| `sleeve_nav_at_entry` | float | Sleeve NAV before this trade |

### Aggregate Metrics (computed at Day 15 and Day 30)

| Metric | Formula |
|--------|---------|
| Win rate by conviction band | wins / total per band |
| EV per band | mean(pnl) per band |
| EV per regime | mean(pnl) per regime |
| Kelly fraction estimate | (bp - q) / b where b=payout, p=win_rate, q=1-p |
| Max drawdown (sleeve) | peak-to-trough in sleeve NAV |
| Autocorrelation of losses | lag-1 autocorrelation of loss streaks |
| Band separation | EV(high) - EV(low) |
| Sharpe (annualized, if applicable) | mean(daily_pnl) / std(daily_pnl) * sqrt(365) |

We are not measuring excitement.  
We are measuring structure.

---

## 6. Success Criteria

Success is **not** 50% win rate. Success is **not** 100% return.

| Criterion | Threshold |
|-----------|-----------|
| Band separation | Higher conviction bands outperform lower bands |
| Positive EV | At least one conviction band shows positive expected value |
| Risk profile | Stable — no accelerating drawdown, no clustering |
| Rule compliance | Zero violations of position rules |
| Autocorrelation | Loss streaks are not serially correlated |

Even small positive expectancy is enough — it proves signal transfers.

---

## 7. Failure Criteria

| Criterion | Description |
|-----------|-------------|
| No band separation | All conviction bands produce statistically similar outcomes |
| Random outcomes | Win/loss distribution indistinguishable from fair coin |
| Negative EV everywhere | All bands negative after 30 days |
| Rule violations | Any position rule broken |
| Emotional overrides | Any trade taken outside the rule set |
| Kill line hit | Sleeve NAV ≤ $1,700 |

**If failure occurs:**

1. Close sleeve immediately
2. Log final state
3. No postmortem drama
4. No re-entry
5. Document findings in 30-day brief

---

## 8. Deployment Prerequisites

This experiment does NOT deploy until:

| Gate | Condition |
|------|-----------|
| Contraction window | Phase C 14-day window complete |
| Dispersion proof | Hybrid score std dev materially above baseline |
| Scoring differentiation | Conviction bands are non-collapsed |
| Core NAV stable | No active drawdown breach in futures |
| Config hashes verified | Core system config unchanged |

> **"No binary sleeve deployment without dispersion proof"**  
> — Phase C Contraction Window Launch Memo, §6

---

## 9. Operational Separation

| Surface | Core Futures | Binary Lab |
|---------|-------------|------------|
| Capital | Core NAV ($9,688+) | $2,000 USDC sleeve |
| State file | `logs/state/nav_state.json` | `logs/state/binary_lab_state.json` |
| Trade log | `logs/execution/orders_attempted.jsonl` | `logs/execution/binary_lab_trades.jsonl` |
| PnL reporting | 14-day brief | Separate binary lab 30-day brief |
| Risk limits | `config/risk_limits.json` | `config/binary_lab_limits.json` |
| Dashboard section | Core portfolio | Dedicated `binary_lab` panel (if added) |
| Kill switch | Core kill switch | Independent sleeve kill |

**Never blend them. Ever.**

---

## 10. Config Hash Lock (Set at Day 0)

At deployment, compute and record:

| File | SHA-256 |
|------|---------|
| `config/binary_lab_limits.json` | _(set at Day 0)_ |
| `config/risk_limits.json` (core) | _(verify unchanged)_ |
| `config/strategy_config.json` (core) | _(verify unchanged)_ |

```bash
sha256sum config/binary_lab_limits.json config/risk_limits.json config/strategy_config.json
```

---

## 11. 14-Day Brief Integration

Add `binary_lab` section to `ops/14_DAY_PERFORMANCE_BRIEF.md` as an appendix.

The binary lab section reports:
- Sleeve NAV trajectory
- Trade count
- Win rate by band
- EV by band
- Kill line distance
- Rule compliance

This section is **additive** — it does not replace or modify any core futures reporting.

---

## 12. Governance Log Format

Append binary lab entries to `ops/binary_lab_window.log`:

```
2026-XX-XX | BINARY LAB DAY 0 | sleeve=$2000 | deployed=$0 | reserve=$800 | kill_line=$1700 | config_hash=LOCKED
2026-XX-XX | Day 1/30 | trades=3 | wins=2 | losses=1 | pnl=+$14.00 | sleeve_nav=$2014 | kill_dist=$314 | CLEAN
2026-XX-XX | Day 2/30 | trades=4 | wins=1 | losses=3 | pnl=-$38.00 | sleeve_nav=$1976 | kill_dist=$276 | CLEAN
```

Breach:
```
2026-XX-XX | Day X/30 | KILL LINE HIT | sleeve_nav=$1698 | EXPERIMENT TERMINATED
```

---

## 13. Timeline

| Phase | When | Action |
|-------|------|--------|
| Formalization | 2026-02-17 | This document (done) |
| Prerequisites | Post-contraction | Validate all gates in §8 |
| Day 0 | TBD | Fund sleeve, lock config, record baseline |
| Day 15 | Day 0 + 15 | Mid-point measurement (observation only) |
| Day 30 | Day 0 + 30 | Final measurement, decision |
| Post-experiment | Day 31+ | Close, document, decide |

---

*Prepared by: GPT Hedge Fund-Ops*  
*Satellite Lab: Binary Experiment Protocol v1.0*  
*Parent Window: Phase C Contraction (v7.9-S1)*
