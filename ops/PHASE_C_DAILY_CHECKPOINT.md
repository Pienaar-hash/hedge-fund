# Phase C — Daily Checkpoint SOP

**Duration:** ~30 seconds  
**Frequency:** Once daily (UTC morning recommended)  
**Owner:** Operator or automated cron  
**Contraction Window:** 15 Feb – 1 Mar 2026 (v7.9-S1)  
**Current Day:** Day 2/14 (as of 2026-02-17)

---

## Checklist

| # | Check | Command / Source | Pass Criteria |
|---|-------|------------------|---------------|
| 1 | Diagnostics liveness | `cat logs/state/diagnostics.json \| jq '.runtime_diagnostics.liveness'` | `idle_signals: false` |
| 2 | DLE shadow path | `ls -la logs/execution/dle_shadow_events.jsonl` | File exists, growing (77,539 events as of Day 2) |
| 3 | Rehearsal metrics | `cat logs/state/phase_c_readiness.json \| jq '.'` | See gate criteria below |
| 4 | Config hash lock | `sha256sum config/strategy_config.json config/risk_limits.json config/pairs_universe.json` | Hashes unchanged from Day 0 baseline |
| 5 | NAV delta | `cat logs/state/nav_state.json \| jq '.peak_equity'` | Track vs baseline $9,688.30 |
| 6 | Positions count | `cat logs/state/positions_state.json \| jq '.positions \| length'` | Track open positions (cap: 2) |
| 7 | Post-contraction episodes | Episode ledger entries after 2026-02-15 | Count for scoring validation |
| 8 | Ops ledger entry | Append to `ops/phase_c_window.log` | One line per day |

### Locked Config Hashes (Day 0 Baseline)

| File | SHA-256 (prefix) | Full Hash |
|------|-----------------|-----------|
| `strategy_config.json` | `267dd161` | `267dd1610f81f645a7f4b945d5e2cbffd01281538edcaf8462ee2f11b7d21fcc` |
| `risk_limits.json` | `9f8d0b37` | `9f8d0b3713f4f5c19701664997cd199de8559489313e17b4fb7c1cf26124f375` |
| `pairs_universe.json` | `885539d6` | `885539d6bf10d71d4f2190d2aaede40e384fc0a66f147ac5ec5b8c81b1f2b7ae` |

---

## Gate Criteria (from manifest)

All must be **simultaneously true** for 14 consecutive days:

| Metric | Threshold | Source field |
|--------|-----------|-------------|
| `would_block_pct` | < 0.1% | `.current_metrics.would_block_pct` |
| `expired_permit_count` | == 0 | `.current_metrics.expired_permit_count` |
| `missing_permit_count` | == 0 | `.current_metrics.missing_permit_count` |
| `rehearsal_enabled` | `true` | `.rehearsal_enabled` |
| `total_orders` | > 0 | `.current_metrics.total_orders` |

---

## Quick-Check One-Liner

```bash
jq '{
  day: .window_days_met,
  of: .window_days_required,
  criteria_met: .criteria_met,
  gate: .gate_satisfied,
  block_pct: .current_metrics.would_block_pct,
  expired: .current_metrics.expired_permit_count,
  missing: .current_metrics.missing_permit_count
}' logs/state/phase_c_readiness.json
```

Expected output when healthy:

```json
{
  "day": 7,
  "of": 14,
  "criteria_met": true,
  "gate": false,
  "block_pct": 0,
  "expired": 0,
  "missing": 0
}
```

---

## Ops Ledger Entry Format

Append one line per day to `ops/phase_c_window.log`:

```
2026-02-13 | Day 1/14 | criteria_met=true | would_block_pct=0.00 | expired=0 | missing=0 | CLEAN
2026-02-14 | Day 2/14 | criteria_met=true | would_block_pct=0.00 | expired=0 | missing=0 | CLEAN
```

If a breach occurs:

```
2026-02-15 | Day 0/14 | criteria_met=false | would_block_pct=0.15 | expired=0 | missing=0 | BREACH: would_block_pct=0.15% >= 0.1% — window reset
```

---

## Maturity Soak Tracker

After each daily checkpoint, record whether any maturity guard has flipped
from immature → mature.  This turns the soak phase into measurable
phase progression without changing the system.

```
Maturity flips observed today: slippage / expectancy / cerberus (Y/N)
```

Example entries in `ops/phase_c_window.log`:

```
2026-02-14 | maturity: slippage=N expectancy=N cerberus=N
2026-02-21 | maturity: slippage=Y expectancy=N cerberus=N  # slippage EWMA reached 20 trades
```

Sources:
- Slippage: `logs/state/execution_quality.json` → `symbols.<SYMBOL>.slippage.is_mature`
- Expectancy: `logs/state/symbol_scores_v6.json` → `symbols.<SYMBOL>.is_mature`
- Cerberus: `logs/state/cerberus_state.json` → `warmup_progress`

**Note:** All three maturity guards require post-contraction trade activity to progress. In sustained CHOPPY regime with zero orders, maturity will remain N/N/N. This is expected — maturity cannot be faked.

---

## Contraction Window — Known Structural Conditions

### `no_orders_evaluated` — Zero Orders Due to Filtration, Not Breakage

When `phase_c_readiness.json` shows `breach_reason: "no_orders_evaluated"`, this means the gate criterion `total_orders > 0` cannot be satisfied because doctrine is correctly preventing entries in a hostile regime.

**This is filtration evidence, not system failure.** In CHOPPY regime with tight conviction gating, zero orders is the intended behavior. The contraction window measures structural differentiation, not trade volume.

The 14-day window counter cannot begin accumulating clean days until the system produces at least one evaluated order. If the full 14 days pass in CHOPPY with zero trades, the contraction window still produces valid data: it proves the gates are working and the system preserves capital in adverse regimes.

**Post-window refinement (Day 15+):** Split `no_orders_evaluated` into two distinct breach reasons:
- `NO_ORDERS_FILTERED` — zero orders because doctrine/conviction correctly filtered all signals (healthy)
- `NO_ORDERS_BROKEN_PIPELINE` — zero orders because signal pipeline, permit pipeline, or upstream system is broken (unhealthy)

Same metric, different semantics. Important for ops reporting and funder communication.

---

## When Gate Satisfied

When `gate_satisfied == true` (14 consecutive clean days):

1. `phase_c_readiness.json` will show `"gate_satisfied": true`
2. Phase C becomes a **governance decision**, not a technical one
3. Recommended Phase C.1: **Entry-only enforcement** (block only ENTRY orders, never EXIT)
4. No code changes until explicit Phase C authorization

---

## Day 2 Status Snapshot (2026-02-17)

| Metric | Value |
|--------|-------|
| NAV | $9,669.08 |
| NAV delta from baseline | -$19.22 (-0.20%) |
| Regime | CHOPPY (49.1%) / MEAN_REVERT (50.9%) |
| Open positions | 0 |
| Post-contraction episodes | 0 |
| Total signals (runtime) | 10,394 |
| Total orders | 0 |
| Vetoes | 258 (all `min_notional`) |
| DLE shadow events | 77,539 |
| Phase C window | Day 0/14 (breaching on `no_orders_evaluated`) |
| Config hashes | All 3 match Day 0 baseline |
| Enforcement (C.1) | OFF |
| DLE rehearsal log | Does not exist (no orders to rehearse) |
| DLE entry denials log | Does not exist (enforcement OFF) |
| Maturity | slippage=N, expectancy=N, cerberus=N |

### Day 2 Scoring Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hybrid std dev | 0.0128 | 11.6x wider than Day 0 (0.0011) — but inter-symbol only |
| Trend component | 0.500 (frozen) | Requires closed episodes to calibrate |
| Carry component | 0.500 (frozen) | Requires position/funding data |
| Expectancy component | 0.500 (frozen) | Requires win/loss record |
| Router component | 0.45–0.66 (live) | Only varying component; per-symbol, not per-signal |
| Conviction bands | 100% unscored | Correct given frozen components |
| Confidence | 0.0 / 0.1 | Cold-start default |
| Structural finding | Self-sealing cold-start loop | Outcome-dependent scoring + tight gates + hostile regime = no calibration data |
| Post-window action | Feature-level scoring inputs | Engineer trend/carry/expectancy from observable market features (momentum z, funding rate, regime priors), not solely from closed-episode outcomes |

---

## Daily Scoring Stability Monitor

Check these each day during the contraction window. No action — observation only.

```bash
# 1. Component freeze check (expect trend=0.5, carry=0.5, expectancy=0.5)
tail -100 logs/execution/score_decomposition.jsonl | \
  jq -c '{sym: .symbol, trend: .components.trend, carry: .components.carry, exp: .components.expectancy, router: .components.router}' | \
  sort -u

# 2. Router drift check (compare to Day 2: BTC=0.657, SOL=0.641, ETH=0.452)
tail -100 logs/execution/score_decomposition.jsonl | \
  jq -r 'select(.ts >= "2026-02-17") | [.symbol, .components.router] | @tsv' | \
  sort -u

# 3. Regime mix
cat logs/state/sentinel_x.json | jq '{regime: .primary_regime, probs: .smoothed_probs}'

# 4. Confidence distribution
tail -1000 logs/execution/score_decomposition.jsonl | \
  jq -r 'select(.ts >= "2026-02-17") | .confidence' | sort | uniq -c | sort -rn
```

**What to note:**
- If any component moves off 0.5 → record in ops log (component activation)
- If router values shift >0.05 from Day 2 baseline → record (router drift)
- If regime changes from CHOPPY → record (regime transition, possible trade activity)
- If confidence distribution changes → record (scoring pipeline behavior change)
