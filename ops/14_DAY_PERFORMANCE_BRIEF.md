# 14-Day Performance Brief — Contraction Window

**Period:** `____-__-__` to `____-__-__`  
**System:** v7.9-S1 (post-contraction)  
**Commit:** `6733c08a` (measurability + conviction gate)  
**Regime Rule:** No threshold changes. No weight tweaks. No universe expansion.

---

## 1. Capital Summary

| Metric | Value | Note |
|--------|-------|------|
| Starting NAV | $ _______ | Day 0 snapshot |
| Ending NAV | $ _______ | Day 14 snapshot |
| Period Return | _____ % | Net of fees |
| Max Drawdown (abs) | $ _______ | Intraday peak-to-trough |
| Max Drawdown (%) | _____ % | vs starting NAV |
| Daily Loss Cap | 1.0% | Hard limit — never breached / breached on Day __ |
| Weekly Loss Cap | 3.0% | Hard limit — never breached / breached in Week __ |

**Source:** `logs/state/nav_state.json`, `logs/state/risk_snapshot.json`

---

## 2. Trade Activity

| Metric | Value |
|--------|-------|
| Total episodes | _____ |
| Entries attempted | _____ |
| Entries executed | _____ |
| Entries vetoed | _____ |
| Acceptance rate | _____ % |
| Avg trades/day | _____ |
| Max concurrent positions (observed) | _____ |
| Max concurrent positions (cap) | 2 |

**Source:** `logs/execution/orders_attempted.jsonl`, `logs/state/episode_ledger.json`, `logs/execution/risk_vetoes.jsonl`

---

## 3. Conviction Distribution

The central question: **Is the conviction engine differentiating?**

| Band | Intents | Accepted | Vetoed | Acceptance % |
|------|---------|----------|--------|-------------|
| very_high (≥0.92) | | | | |
| high (≥0.80) | | | | |
| medium (≥0.60) | | | | |
| low (≥0.40) | | | | |
| very_low (<0.40) | | | | |
| unscored | | | | |

**Key signal:** If >80% of intents are "very_low" or "unscored", the scoring pipeline is not differentiating. That is actionable data, not a failure.

**Source:** `logs/execution/sizing_snapshots.jsonl` → `conviction_band_shadow` / `conviction_score_shadow`, `logs/execution/score_decomposition.jsonl`

---

## 4. PnL by Conviction Band

| Band | Episodes | Win Rate | Gross PnL | Net PnL | Avg Duration |
|------|----------|----------|-----------|---------|-------------|
| very_high | | | | | |
| high | | | | | |
| medium | | | | | |
| low | | | | | |
| very_low | | | | | |

**Key signal:** If higher bands don't produce better PnL, conviction scoring needs recalibration. If they do, we have measurable edge.

**Source:** `logs/state/episode_ledger.json` → `episodes_v2[].conviction_band`, `episodes_v2[].net_pnl`

---

## 5. Hybrid Score Dispersion

| Statistic | Value |
|-----------|-------|
| Mean hybrid_score | _____ |
| Median hybrid_score | _____ |
| Std dev | _____ |
| Min | _____ |
| Max | _____ |
| IQR (25th–75th) | _____ – _____ |

**Key signal:** If std dev < 0.05, all scores are clustered — the scoring function is not discriminating. If IQR spans >0.3, there is real dispersion to exploit.

**Source:** `logs/execution/score_decomposition.jsonl` → `hybrid_score`

---

## 6. Strategy Attribution

| Strategy | Episodes | Win Rate | Net PnL | Avg Confidence |
|----------|----------|----------|---------|----------------|
| vol_target | | | | |
| btc_micro | | | | |
| eth_micro | | | | |
| sol_micro | | | | |
| (unattributed) | | | | |

**Key signal:** If "unattributed" count > 0 after this patch, the attribution pipeline has a leak.

**Source:** `logs/state/episode_ledger.json` → `episodes_v2[].strategy`

---

## 7. Regime Context

| Day | Primary Regime | Confidence | Stability (cycles) |
|-----|---------------|------------|--------------------|
| 1 | | | |
| 2 | | | |
| ... | | | |
| 14 | | | |

**Regime distribution over period:**

| Regime | Days Active | % of Period |
|--------|-------------|-------------|
| TREND_UP | | |
| TREND_DOWN | | |
| MEAN_REVERT | | |
| BREAKOUT | | |
| CHOPPY | | |
| CRISIS | | |

**Source:** `logs/state/sentinel_x.json`, `logs/doctrine_events.jsonl`

---

## 8. Veto Analysis

| Veto Reason | Count | % of Total |
|-------------|-------|-----------|
| doctrine_regime_mismatch | | |
| conviction_band_below_minimum | | |
| no_strategy_attribution | | |
| nav_stale | | |
| per_symbol_cap | | |
| daily_loss_limit | | |
| correlation_cap | | |
| kill_switch | | |
| Other | | |

**Key signal:** Conviction and attribution vetoes are *desired* — they prove the gates are working. Regime mismatch vetoes are environmental (market regime ≠ signal direction).

**Source:** `logs/execution/risk_vetoes.jsonl` → `veto_reason`, `logs/doctrine_events.jsonl` → `verdict`

---

## 9. Operational Health

| Check | Status |
|-------|--------|
| NAV staleness incidents (>90s) | _____ count |
| Circuit breaker activations | _____ count |
| Manifest integrity | ✅ / ❌ |
| Executor uptime | _____ % |
| Liveness alerts | _____ count |
| Score decomposition log populating | ✅ / ❌ |
| Sizing snapshots carrying conviction | ✅ / ❌ |

**Source:** `logs/state/diagnostics.json`, `logs/execution/execution_health.jsonl`

---

## 10. Risk Compliance

| Limit | Cap | Max Observed | Breached? |
|-------|-----|-------------|-----------|
| Daily loss | 1% | _____ % | ✅ / ❌ |
| Weekly loss | 3% | _____ % | ✅ / ❌ |
| Concurrent positions | 2 | _____ | ✅ / ❌ |
| Max leverage | 4× | _____ × | ✅ / ❌ |
| Gross exposure | 150% | _____ % | ✅ / ❌ |

**Source:** `logs/state/risk_snapshot.json`

---

## 11. Discipline Compliance

Over the 14-day window, the following must all be TRUE:

| Rule | Compliant? |
|------|-----------|
| No threshold changes committed | ✅ / ❌ |
| No band redefinitions | ✅ / ❌ |
| No score weight tweaks | ✅ / ❌ |
| No universe expansion | ✅ / ❌ |
| No risk cap increases | ✅ / ❌ |
| Config hash unchanged | `sha256: ________________` |

**Verification:**
```bash
sha256sum config/strategy_config.json config/risk_limits.json config/pairs_universe.json
```

---

## 12. Forward Assessment

### What the data says:

_[ 2–3 sentences: Does conviction differentiate? Is there signal in the scoring? Are risk caps being respected? ]_

### Recommended next action:

- [ ] **Maintain current settings** — data is still accumulating
- [ ] **Widen conviction gate** — too few trades, gate is over-filtering
- [ ] **Narrow conviction gate** — low bands are losing, raise minimum
- [ ] **Recalibrate scoring weights** — hybrid scores are not dispersed
- [ ] **Expand universe** — edge confirmed in majors, add selected pairs
- [ ] **Deploy binary sleeve** — core pipeline proves measurement works

### Trade count expectation:

| Scenario | Trades/Week | Interpretation |
|----------|-------------|---------------|
| Near zero (0–1) | Gate is too tight or regime is persistent CHOPPY |
| Low (2–5) | Expected under tight risk + conviction gating |
| Moderate (5–15) | Possible if regime shifts to TREND_UP/DOWN |
| High (>15) | Unexpected — investigate if gates are bypassed |

---

## Appendix: Data Collection Commands

```bash
# Episode summary (post-contraction only)
PYTHONPATH=. python -c "
from execution.episode_ledger import build_episode_ledger
ledger = build_episode_ledger()
s = ledger.stats
print(f'Episodes: {s[\"episodes_found\"]}')
print(f'Net PnL:  \${s[\"total_net_pnl\"]:.2f}')
print(f'Win Rate: {s.get(\"win_rate_pct\", 0):.1f}%')
print(f'Winners:  {s[\"winners\"]}  Losers: {s[\"losers\"]}')
"

# Conviction band distribution from sizing snapshots
cat logs/execution/sizing_snapshots.jsonl | \
  jq -r '.conviction_band_shadow // "unscored"' | sort | uniq -c | sort -rn

# Veto reasons (last 14 days)
cat logs/execution/risk_vetoes.jsonl | \
  jq -r '.veto_reason' | sort | uniq -c | sort -rn

# Hybrid score stats from decomposition log
cat logs/execution/score_decomposition.jsonl | \
  jq '.hybrid_score' | sort -n | awk '{a[NR]=$1; s+=$1} END {
    print "count:", NR;
    print "mean:", s/NR;
    print "min:", a[1];
    print "max:", a[NR];
    print "median:", (NR%2==1) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
  }'

# Regime distribution (from doctrine events, last 14 days)
cat logs/doctrine_events.jsonl | \
  jq -r 'select(.type == "entry_verdict") | .regime' | sort | uniq -c | sort -rn

# Config hash (discipline verification)
sha256sum config/strategy_config.json config/risk_limits.json config/pairs_universe.json

# NAV snapshot
cat logs/state/nav_state.json | jq '{nav: .nav, updated: .updated_ts}'

# Risk compliance
cat logs/state/risk_snapshot.json | jq '{
  risk_mode: .risk_mode,
  dd_frac: .dd_frac,
  daily_loss_frac: .daily_loss_frac,
  portfolio_dd_pct: .portfolio_dd_pct
}'
```

---

*Generated from template: `ops/14_DAY_PERFORMANCE_BRIEF.md`*  
*System: GPT Hedge v7.9-S1*  
*Contraction commit: `6733c08a`*
