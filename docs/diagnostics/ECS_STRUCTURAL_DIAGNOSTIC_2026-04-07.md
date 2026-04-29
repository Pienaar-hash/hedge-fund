# ECS Structural Diagnostic — 2026-04-07

**Format:** Structural findings only. No narrative. No speculation.

---

## 1. ROOT CAUSE: BTC Score–PnL Inversion

### Structural Explanation

The inversion is a **carry-score direction interaction** compounded by **regime-phase misalignment**.

**Mechanism chain:**

```
BTC funding_rate persistently positive
  → _scale_funding_rate() [symbol_score_v6.py L409-430]
    → LONG direction: raw = -annual_rate (inverts sign)
    → carry_score < 0.5 for LONG when funding positive
  → hybrid_score depressed for long entries by 0.25 × carry_weight
  → Hydra's TREND head [hydra_engine.py L555-580] uses hybrid_score directly
  → score = abs(hybrid_score) biased lower for BTC longs in positive-funding regimes

BTC basis persistently positive (perp premium)
  → _scale_basis() [symbol_score_v6.py L432-460]
    → LONG direction: raw = -basis_pct (inverts sign)
    → basis_score < 0.5 for LONG when basis positive
  → carry_score = 0.6 × funding_score + 0.4 × basis_score
  → Both components directionally penalize BTC longs during bull markets
```

**Result:** High hybrid_score for BTC → short-biased carry component → enters during periods where shorts lose money. Low hybrid_score for BTC → long-biased carry deficiency → but these are the periods where BTC actually trends up.

**Quantitative test:**
```
SELECT direction, AVG(carry_score), AVG(net_pnl)
FROM score_decomposition JOIN episode_ledger ON intent_id
WHERE symbol = 'BTCUSDT'
GROUP BY direction
```

**Expected finding:** carry_score anti-correlated with BTC long PnL because funding/basis are structurally positive in uptrends (when longs profit).

### Secondary amplifier

Sentinel-X trend_slope_threshold = 0.0003/bar with hysteresis exit at 0.0002. BTC's trend regime detection is **lagged by 3 × stickiness × cycle_interval** (minimum 15 cycles at default settings). By the time TREND_UP is confirmed:
1. Carry has already penalized the hybrid_score for longs
2. The early-trend edge (where BTC longside PnL is concentrated) has decayed via alpha_decay [half-life exponential]

**Net effect:** Score monotonically increases → PnL monotonically decreases for BTC because the system scores carry-rewarded entries higher, and carry-rewarded entries are anti-correlated with BTC long profitability.

---

## 2. ZERO_SCORE TRADES: Model Blindness, Not Data Sparsity

### Classification: Model Blindness

**Evidence:**

| Property | Data Sparsity Signature | Model Blindness Signature | Observed |
|----------|------------------------|--------------------------|----------|
| Score field present | Missing/null | Present, value = 0.0 | **Present, value = 0.0** |
| Candidate path | Never enters selector | Enters selector, loses ranking | **Enters selector** |
| Regime at entry | Unknown/NONE | Known regime (valid Sentinel-X) | **Known regime** |
| Expectancy maturity | < 30 trades (prior) | ≥ 30 trades (mature) | **Both** |
| PnL distribution | Random (mean ≈ 0) | Systematically negative | **Systematically negative** |

### Production path

```
candidate_selector.py L32-40:
  _intent_score(intent) → looks for "hybrid_score" or "score"
  → Neither present on legacy intents that bypass screener
  → Returns 0.0 (fallback)

shadow_selector_v2.py L100-110:
  ZERO_SCORE_POLICY = {"action": "ROUTE_TO_LEGACY", ...}
  → Score=0 cannot be placed on profit mask axis
  → Routed to Legacy engine by default
```

**Three distinct ZERO_SCORE sources:**

| Source | Mechanism | File | Frequency |
|--------|-----------|------|-----------|
| **A. Legacy intents missing score field** | `_intent_score()` returns 0.0 fallback | candidate_selector.py L32 | HIGH |
| **B. Hydra head fallback** | Strategy head fails to produce valid signal, emits score=0.0 | hydra_engine.py L990 | RARE |
| **C. Hybrid score below emission threshold** | hybrid_score < 0.55 (LONG) or < 0.50 (SHORT) but intent exists | signal_screener.py L1250 | MODERATE |

**Source A dominates losses.** Legacy intents never compute hybrid_score → always score 0.0 in candidate selector → always lose ranking to any Hydra intent → but when NO Hydra intent exists for a symbol, Legacy wins by default.

**Testable hypothesis:**
```python
# If ZERO_SCORE = model blindness:
assert mean_pnl(score == 0) < mean_pnl(score > 0)  # True by finding
assert count(score == 0 AND regime == KNOWN) > 0.8 * count(score == 0)  # Model sees regime, can't score
# If ZERO_SCORE = data sparsity:
assert count(score == 0 AND expectancy_mature == False) > 0.8 * count(score == 0)  # Would indicate prior fallback
```

---

## 3. REGRET DECOMPOSITION: Regime Misclassification vs. Execution Timing

### Total regret = 87.8% of actual loss

**Decomposition framework:**

```
Total Regret = Regime Regret + Timing Regret + Residual

Regime Regret := PnL lost from trading in wrong regime
  = Σ (actual_pnl WHERE regime_at_entry ≠ regime_at_close)

Timing Regret := PnL lost from entry/exit lag within correct regime
  = Σ (optimal_entry_pnl - actual_entry_pnl WHERE regime_at_entry == regime_at_close)

Residual := sizing, slippage, fees
```

### Structural sources of each component

**A. Regime Misclassification Regret**

| Lag Source | Component | Measured Delay | File:Line |
|------------|-----------|----------------|-----------|
| Stickiness | 3 consecutive predictions before flip | 3 × cycle_interval | sentinel_x.py L92, L1017 |
| EMA smoothing | α = 0.3 on probabilities | ~2.3 effective lag cycles | sentinel_x.py L1070 |
| Hysteresis | Exit threshold = 0.67 × entry threshold | Asymmetric ~50% longer to exit trend | sentinel_x.py L838-842 |
| Doctrine stability | `cycles_stable >= 2` required | +2 cycles after regime confirmed | doctrine_kernel.py L237 |
| Regime staleness | 600s hard cutoff | Up to 10min stale data in fast markets | sentinel_x.py L68 |

**Combined minimum regime detection lag:** 5 cycles (stickiness=3 + stability=2) × interval

**Upper bound estimate for regime regret share:**
- From ecs_regime_switching memory: Selector regret = 108% of actual loss (BTC data)
- Regime misclassification regret can be computed as:
  ```
  regime_regret_pct = count(regime_flipped_during_position) / count(all_losing_trades)
  ```

**B. Execution Timing Regret**

| Lag Source | Component | Measured Delay | File:Line |
|------------|-----------|----------------|-----------|
| Alpha decay | Exponential half-life, min_decay_mult = 0.35 | Score × 0.35 if signal aged | symbol_score_v6.py L282-286 |
| Fill polling | 13s maximum per order (blocking) | Queue pileup on multiple orders | fill_tracker.py L540-565 |
| TWAP execution | N slices with inter-slice sleep | Minutes for large orders | order_router.py L1800+ |
| Carry score compute lag | Uses snapshot funding/basis (not live) | Stale by state_publish interval | symbol_score_v6.py L409 |

**Quantification test plan:**

```sql
-- Regime regret attribution
WITH regime_changes AS (
  SELECT intent_id,
         regime_at_entry,
         regime_at_close,
         net_pnl,
         CASE WHEN regime_at_entry != regime_at_close THEN 'REGIME_MISMATCH'
              ELSE 'TIMING' END AS regret_class
  FROM episode_ledger e
  JOIN sentinel_x_log s ON e.close_ts BETWEEN s.ts AND s.next_ts
  WHERE net_pnl < 0
)
SELECT regret_class,
       COUNT(*) AS n,
       SUM(net_pnl) AS total_loss,
       SUM(net_pnl) / (SELECT SUM(net_pnl) FROM regime_changes) AS pct_of_regret
FROM regime_changes
GROUP BY regret_class
```

**Structural prediction:** Regime misclassification accounts for 55–70% of the 87.8% regret. Reasoning: stickiness + stability = 5-cycle minimum lag; BTC trends with fast reversals will consistently have entries confirmed after the move and exits triggered after the reversal.

---

## 4. BTC PROFIT MASK HARDENING TEST PLAN

### Current state

| Property | Value | Source |
|----------|-------|--------|
| BTC profit region | (0.4197, 0.4953) | shadow_selector_v2.py L81 |
| Subsample overlap | 0.889 | ECS_REGIME_PROFIT_MASK_V2_SPEC.md |
| Stability classification | STABLE | V2 spec |
| ETH overlap | < 0.70 | DISCARDED in V2 |
| SOL | No profitable region | — |
| Sample size | 780 episodes | ecs_profit_mask.py derivation |
| Current mode | SHADOW ONLY | shadow_selector_v2.py L18 |

### Deterministic test plan

**Phase 1: Statistical validation (no code changes)**

| Test | Method | Pass Criterion | Tool |
|------|--------|----------------|------|
| T1. Boundary stability | Bootstrap (N=1000) the episode set; re-derive boundaries per resample | 95% CI of lower bound ∈ [0.40, 0.44], upper bound ∈ [0.48, 0.52] | `scripts/ecs_profit_mask.py --bootstrap` |
| T2. Temporal stability | Split episodes into 4 chronological quartiles; derive mask per quartile | Overlap ≥ 0.80 in all 4 pairs (Q1-Q2, Q2-Q3, Q3-Q4, Q1-Q4) | New: `scripts/ecs_profit_mask_temporal.py` |
| T3. Regime-conditional stability | Derive mask separately for each Sentinel-X regime | BTC mask present in TREND_UP; absent or different in MEAN_REVERT/CHOPPY | `scripts/ecs_profit_mask.py --by-regime` |
| T4. Direction stability | Derive mask separately for LONG and SHORT entries | Boundaries must not invert (LONG mask ≠ mirror of SHORT mask) | `scripts/ecs_profit_mask.py --by-direction` |
| T5. Sample size adequacy | Count episodes in profit region; require ≥ 50 | At least 50 episodes with score ∈ (0.4197, 0.4953) | Direct count |
| T6. PnL monotonicity | Within profit region, cumulative PnL curve must be monotonically non-decreasing over 20-episode windows | No drawdown > 30% of peak within region | `scripts/ecs_profit_mask.py --monotonicity` |

**Phase 2: Counterfactual simulation (shadow mode)**

| Test | Method | Pass Criterion |
|------|--------|----------------|
| T7. Shadow → counterfactual PnL | Compare: trades passing profit mask gate vs. all trades | Sharpe(mask) > Sharpe(all) + 0.10 |
| T8. Capacity impact | Count abstained trades (Candidate D verdicts) | Capacity ratio > 0.40 (≥40% of opportunities taken) |
| T9. Walk-forward | Use first 500 episodes to derive mask; validate on remaining 280+ | Out-of-sample Sharpe > 0 and region overlap ≥ 0.70 |

**Phase 3: Gating rule promotion criteria (all must pass)**

```
PROMOTE_TO_LIVE IF AND ONLY IF:
  T1.pass AND T2.pass AND T3.pass AND T4.pass
  AND T5.pass AND T6.pass
  AND T7.pass AND T8.pass AND T9.pass
  AND shadow_soak_days >= 30
  AND shadow_episode_count >= 100 (post-V2)
```

**Kill conditions for hardened mask:**

| Condition | Trigger | Action |
|-----------|---------|--------|
| Subsample overlap < 0.70 | Weekly `ecs_profit_mask.py` refresh | Revert to SHADOW; re-derive |
| Walk-forward Sharpe < 0 | Rolling 50-episode window | Disable mask gating |
| Region drift > 0.03 | Boundary shift vs. V2 reference | Flag for manual review |
| Episode count in region < 30 over 14 days | Liquidity/opportunity collapse | Suspend mask; widen to V1 |

---

## 5. RECALIBRATION DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECALIBRATION ORDER                           │
│                                                                 │
│  LAYER 0 (Foundation — fix first, zero dependencies)            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [A] Carry Score Direction Bias                          │   │
│  │      File: execution/intel/symbol_score_v6.py L409-460   │   │
│  │      Fix: Decouple carry_score from hybrid_score for     │   │
│  │           BTC, OR make carry direction-neutral with       │   │
│  │           separate directional overlay                    │   │
│  │      Test: Recompute hybrid_score with carry=0.5 (flat); │   │
│  │            check if score-PnL inversion disappears        │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  LAYER 1 (Depends on Layer 0)                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [B] ZERO_SCORE Legacy Fallback                          │   │
│  │      File: execution/candidate_selector.py L32-40        │   │
│  │      Fix: Legacy intents must carry explicit score OR     │   │
│  │           be filtered before candidate selector           │   │
│  │      Depends on: [A] — carry fix changes score surface;  │   │
│  │           zero-score boundary changes with it             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [C] Regime Detection Lag                                │   │
│  │      File: execution/sentinel_x.py L92, L838-842         │   │
│  │      Fix: Reduce stickiness from 3→2 for BTC (per-symbol │   │
│  │           config), OR add fast-path for high-confidence   │   │
│  │           transitions (prob > 0.85 = skip stickiness)     │   │
│  │      Depends on: [A] — regime regret attribution needs    │   │
│  │           corrected score surface to measure accurately   │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │         │                                             │
│         ▼         ▼                                             │
│  LAYER 2 (Depends on Layers 0+1)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [D] Profit Mask Re-derivation                           │   │
│  │      File: scripts/ecs_profit_mask.py                    │   │
│  │           execution/shadow_selector_v2.py L80-98          │   │
│  │      Fix: Re-derive PNL_POSITIVE_REGIONS after [A]+[C]   │   │
│  │           applied. Boundaries will shift.                 │   │
│  │      Depends on: [A] — score surface changed              │   │
│  │                  [C] — regime timing changed               │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  LAYER 3 (Depends on all above)                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [E] Selector V2 Promotion Decision                      │   │
│  │      File: execution/shadow_selector_v2.py               │   │
│  │      Action: Re-run Phase 1-3 test plan (Section 4)      │   │
│  │              with recalibrated components                 │   │
│  │      Depends on: [A]+[B]+[C]+[D] — all upstream must     │   │
│  │                  stabilize before mask can be validated   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  PARALLEL TRACK (independent of A-E)                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [F] Region-Filter Candidate (Sharpe improvement)        │   │
│  │      File: shadow_selector_v2.py L73 (HYDRA_PREFERENCE)  │   │
│  │      Action: Continue shadow soak; independent of carry   │   │
│  │              fix because region_filter uses raw score      │   │
│  │              boundaries, not carry-adjusted scores         │   │
│  │      Note: Score surface change from [A] will shift       │   │
│  │            region_filter boundaries. Re-derive after [A]. │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Critical path

```
[A] Carry Score Fix  →  [B] Zero-Score Gate  →  [D] Mask Re-derive  →  [E] Validation
                     →  [C] Regime Lag Fix   →  [D]
```

**Estimated minimum data requirement per layer:**
- Layer 0: Immediate (code change + historical backfill)
- Layer 1: 50+ new episodes post-fix (validate zero-score reduction + regime timing improvement)
- Layer 2: 100+ episodes post-Layer-1 (re-derive profit mask on corrected surface)
- Layer 3: 100+ episodes post-Layer-2 (shadow soak per Phase 6 prerequisites)

---

## APPENDIX: Testable Hypotheses Registry

| ID | Hypothesis | Test | Expected Outcome | Falsification |
|----|-----------|------|-------------------|---------------|
| H1 | BTC score–PnL inversion caused by carry-direction interaction | Set carry_weight=0 in hybrid; recompute score-PnL correlation | Correlation flips to positive or goes to zero | Correlation remains negative → other component driving inversion |
| H2 | ZERO_SCORE trades are legacy-only (model blindness) | Filter episodes by `intent.get("hybrid_score") is None` | >80% of ZERO_SCORE match | <50% match → Hydra also produces zeros significantly |
| H3 | Regime misclassification > 50% of regret | Tag episodes by regime_at_entry vs regime_at_close; sum PnL | Mismatch PnL > 50% of total negative PnL | Mismatch PnL < 30% → timing dominates |
| H4 | BTC profit mask boundary stable under carry fix | Re-derive mask post-[A]; compute overlap with current (0.4197, 0.4953) | Overlap ≥ 0.60 (boundaries shift but region persists) | Overlap < 0.40 → carry fix destroys the mask signal |
| H5 | Stickiness reduction (3→2) improves BTC regime timing | Shadow-compare regime history with stickiness=2 vs 3 | Mean regime-detection lag reduced by ≥ 1 cycle | Lag unchanged → EMA smoothing dominates stickiness |
| H6 | ETH/SOL mask instability is carry-independent | Re-derive ETH/SOL masks after [A] | Still unstable (overlap < 0.70) | Stabilizes → carry bias was the driver |
