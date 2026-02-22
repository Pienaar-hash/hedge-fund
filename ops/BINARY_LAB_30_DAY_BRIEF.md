# Binary Lab — 30-Day Performance Brief

**Period:** `____-__-__` to `____-__-__`  
**Sleeve:** binary_lab_s1  
**System:** v7.9-S1 (satellite lab)  
**Hypothesis:** High-conviction directional signals exhibit positive expectancy over 15-minute binary rounds  
**Regime Rule:** No parameter changes. No filter additions. No size escalation.

---

## 1. Capital Summary

| Metric | Value | Note |
|--------|-------|------|
| Starting sleeve NAV | $2,000.00 | Day 0 |
| Ending sleeve NAV | $ _______ | Day 30 |
| Total PnL | $ _______ | |
| Return | _____ % | |
| Max drawdown (abs) | $ _______ | Peak-to-trough |
| Max drawdown (%) | _____ % | vs starting NAV |
| Kill line distance (min) | $ _______ | Closest approach to $1,700 |
| Kill line breached | ✅ / ❌ | |

**Source:** `logs/state/binary_lab_state.json`

---

## 2. Trade Activity

| Metric | Value |
|--------|-------|
| Total trades | _____ |
| Wins | _____ |
| Losses | _____ |
| Overall win rate | _____ % |
| Avg trades/day | _____ |
| Max concurrent (observed) | _____ |
| Max concurrent (cap) | 3 |
| Days with 0 trades | _____ |

**Source:** `logs/execution/binary_lab_trades.jsonl`

---

## 3. Conviction Band Analysis (Primary Measurement)

The central question: **Do higher conviction bands outperform lower bands?**

| Band | Trades | Wins | Losses | Win Rate | Gross PnL | EV/Trade | Kelly |
|------|--------|------|--------|----------|-----------|----------|-------|
| very_high (≥0.92) | | | | | | | |
| high (≥0.80) | | | | | | | |
| medium (≥0.60) | | | | | | | |

### Band Separation Test

| Comparison | EV Difference | Direction | Significant? |
|------------|--------------|-----------|-------------|
| very_high vs medium | $ _____ | _____ | ✅ / ❌ |
| high vs medium | $ _____ | _____ | ✅ / ❌ |
| very_high vs high | $ _____ | _____ | ✅ / ❌ |

**Key signal:** If band separation is monotonic (very_high > high > medium), the hypothesis is supported. If flat or inverted, it is falsified.

---

## 4. Regime Analysis

| Regime | Trades | Win Rate | EV/Trade | Notes |
|--------|--------|----------|----------|-------|
| TREND_UP | | | | |
| TREND_DOWN | | | | |
| MEAN_REVERT | | | | |
| BREAKOUT | | | | |

_CHOPPY and CRISIS are excluded by entry gate._

---

## 5. Risk Profile

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max drawdown | $ _____ | $300 | ✅ / ❌ |
| Loss autocorrelation (lag-1) | _____ | < 0.20 | ✅ / ❌ |
| Max consecutive losses | _____ | — | — |
| Max concurrent positions | _____ | 3 | ✅ / ❌ |
| Same-direction stacking violations | _____ | 0 | ✅ / ❌ |
| Size violations | _____ | 0 | ✅ / ❌ |

---

## 6. Rule Compliance

| Rule | Compliant? |
|------|-----------|
| $20 per round (no escalation) | ✅ / ❌ |
| Max 3 concurrent | ✅ / ❌ |
| No same-direction stacking | ✅ / ❌ |
| No martingale | ✅ / ❌ |
| No parameter changes (30-day freeze) | ✅ / ❌ |
| Config hash unchanged | `sha256: ________________` |
| No emotional overrides | ✅ / ❌ |

---

## 7. Hypothesis Verdict

### Success Criteria Evaluation

| Criterion | Met? | Evidence |
|-----------|------|---------|
| Band separation (monotonic) | ✅ / ❌ | |
| Positive EV in ≥1 band | ✅ / ❌ | |
| Stable risk profile | ✅ / ❌ | |
| Zero rule violations | ✅ / ❌ | |
| Loss autocorrelation < 0.20 | ✅ / ❌ | |

### Verdict

- [ ] **SUPPORTED** — Signal transfers to binary. Band separation is real.
- [ ] **INCONCLUSIVE** — Insufficient trades or mixed signals. Need more data.
- [ ] **FALSIFIED** — No band separation, negative EV, or random outcomes.
- [ ] **TERMINATED** — Kill line hit before Day 30.

---

## 8. Diagnostic Mirror Result

| Question | Answer |
|----------|--------|
| Does signal work in binary? | _____ |
| Does signal work in futures? | _____ (from contraction window) |

### Interpretation Matrix

| Binary | Futures | Diagnosis |
|--------|---------|-----------|
| ✅ | ✅ | Signal is valid. Both venues viable. |
| ✅ | ❌ | Execution friction is the problem. Fix routing/slippage. |
| ❌ | ✅ | Binary structure doesn't suit this signal type. |
| ❌ | ❌ | Signal math is the problem. Fundamental recalibration needed. |

---

## 9. Forward Recommendation

- [ ] **Close sleeve permanently** — Hypothesis falsified
- [ ] **Extend experiment** — Inconclusive, need 30 more days (same parameters)
- [ ] **Scale sleeve** — Signal confirmed, increase capital allocation
- [ ] **Refine and re-test** — Specific parameter adjustment identified, new 30-day window
- [ ] **Integrate with core** — Binary becomes permanent satellite allocation

---

## Appendix: Data Collection Commands

```bash
# Trade summary
cat logs/execution/binary_lab_trades.jsonl | wc -l

# Win rate by band
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '.conviction_band' | sort | uniq -c | sort -rn

# PnL by band
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '[.conviction_band, .pnl_usd] | @tsv' | \
  awk '{sum[$1]+=$2; count[$1]++} END {for (b in sum) print b, "trades=" count[b], "pnl=" sum[b], "ev=" sum[b]/count[b]}'

# Win rate by regime
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '[.regime, .outcome] | @tsv' | \
  awk '{total[$1]++; if($2=="WIN") wins[$1]++} END {for (r in total) print r, "wr=" (wins[r]+0)/total[r]*100 "%", "n=" total[r]}'

# Drawdown curve
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '.sleeve_nav_at_entry' | \
  awk 'BEGIN{peak=0} {if($1>peak) peak=$1; dd=peak-$1; if(dd>maxdd) maxdd=dd; print NR, $1, dd} END {print "max_dd=" maxdd}'

# Loss autocorrelation (lag-1)
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r 'if .outcome == "LOSS" then 1 else 0 end' | \
  python3 -c "
import sys, numpy as np
x = np.array([int(l.strip()) for l in sys.stdin])
if len(x) > 2:
    ac = np.corrcoef(x[:-1], x[1:])[0,1]
    print(f'loss_autocorrelation_lag1={ac:.4f}')
else:
    print('insufficient data')
"

# Config hash verification
sha256sum config/binary_lab_limits.json

# Sleeve state
cat logs/state/binary_lab_state.json | jq '{nav: .capital.current_nav_usd, pnl: .capital.pnl_usd, kill_dist: .kill_line.distance_usd, trades: .metrics.total_trades}'
```

---

*Generated from template: `ops/BINARY_LAB_30_DAY_BRIEF.md`*  
*Satellite Lab: binary_lab_s1*  
*Parent System: GPT Hedge v7.9-S1*
