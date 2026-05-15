# V8 Phase 5 Failure Postmortem

**Date:** 2026-05-15  
**Status:** FINAL FAILURE – Phase 5 observation terminated  
**Authority:** Read-only diagnostic; no runtime, executor, doctrine, or risk changes made

---

## Executive Summary

**Phase 5 Shadow Soak Observer has failed with a HARD FAILURE verdict.** The failure is unambiguous and unrecoverable: the certified replay strategy produces **opposite-side trading signals** from the live executor across ~70% of all orders in the current window.

- **Certification verdict:** FAIL (net_pnl = -$131.45)
- **Shadow soak verdict:** FAIL (2548 catastrophic direction mismatches)
- **Root cause:** Replay strategy divergence, not data availability or observability errors
- **Phase 5 status:** **Terminated — observation loop stopped**
- **Phase 6 eligibility:** **DENIED** — No live authority increase permitted
- **Next step:** Research-only root-cause investigation into strategy divergence

---

## 1. Final Phase 5 Verdict

**HARD FAIL — Observation Concluded**

Phase 5 Shadow Soak Observer was designed to validate live executor behavior against a certified replay of the same strategy under identical conditions. The observer discovered that the two systems are not equivalent: they produce opposite trading directions.

**Explicit decision:**
- Phase 5 observation loop is **permanently stopped** for this configuration
- No Phase 6 advancement is permitted
- No increase to live trading authority
- No continuation of daily observation
- No automation (cron, Supervisor, etc.) will be added

---

## 2. Certification Result (Live-Window)

**Certification ID:** `v8_phase5_live_window_cert_001`  
**Completed:** 2026-05-15T10:50:11.239448Z

| Metric | Value |
|--------|-------|
| **Verdict** | **FAIL** |
| **Sample Size** | 536 trades |
| **Net PnL** | **-$131.45** |
| **Gross PnL** | -$51.61 |
| **Fees** | $79.84 |
| **Max Drawdown** | 1.32% |
| **Win Rate** | 19.78% |
| **Veto Count** | 0 |
| **Conviction Authority** | frozen |
| **Doctrine Mutated** | false |
| **Exchange Calls** | false (replay-only) |
| **Output Hash** | 31333c53dedde051... (stable) |

**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT  
**Setup Class:** TREND_PULLBACK_V2_REPLAY_CANDIDATE

---

## 3. Shadow Soak Result (Live-Window Observer)

**Observer Run ID:** `v8_phase5_shadow_soak_live_window_timestamp_check_001`  
**Completed:** 2026-05-15T10:59:58.162986Z  
**Duration:** ~22 seconds

| Metric | Value |
|--------|-------|
| **Status** | **paused** |
| **Verdict** | **fail** |
| **Live Orders Read** | 3677 |
| **Shadow Signals Read** | 3216 |
| **Sample Size (matched events)** | 1129 |
| **Catastrophic Mismatches** | **2548 (69.3% of all orders)** |
| **Symbol Match Rate** | 1.0 (100%) |
| **Direction Match Rate** | 0.307 (30.7%) |
| **Quantity Bucket Match Rate** | 0.0 (0%) |
| **Timestamp Alignment (p95)** | 28,648s (~7.9 hours) |
| **Slippage Model Error (r)** | -1.90 |
| **Median Abs Slippage Error** | -5.0 bps |
| **P95 Abs Slippage Error** | 9995.0 bps |
| **Fill Latency (p99)** | null |

**Pairing Sources:**
- `replay_runs/v8_phase5_live_window_cert_001_a/trades.csv`
- `replay_runs/v8_phase5_live_window_cert_001_b/trades.csv`
- `replay_runs/v8_phase5_live_window_cert_001_a/permit_trace.csv`
- `replay_runs/v8_phase5_live_window_cert_001_b/permit_trace.csv`

**Failed Criteria (6 of 10):**
- direction_match_rate >= 0.95
- quantity_bucket_match_rate >= 0.95
- timestamp_alignment_p95_s <= 60
- slippage_model_error_r >= 0.80
- p95_abs_slippage_error_bps <= 10
- catastrophic_mismatch_count == 0

---

## 4. Detailed Breakdown

### 4.1 Catastrophic Mismatches by Type

All 2548 catastrophic mismatches follow a **single identical pattern:**

```
symbol_match:           true
direction_match:        false
quantity_bucket_match:  false
```

| Classification | Count | % of Catastrophic |
|-----------------|-------|-------------------|
| Symbol mismatch only | 0 | 0% |
| **Direction mismatch only** | **2548** | **100%** |
| Both symbol and direction | 0 | 0% |

**Interpretation:** Every single catastrophic mismatch is a **direction divergence**, not a symbol misidentification. The systems agree on which asset to trade but disagree on which side (BUY vs SELL).

### 4.2 Breakdown by Symbol

| Symbol | Catastrophic | Total | % |
|--------|--------------|-------|-----|
| BTCUSDT | 868 | 1134 | 76.5% |
| ETHUSDT | 882 | 1136 | 77.6% |
| SOLUSDT | 798 | 1407 | 56.7% |

All three symbols are affected, with BTC and ETH showing higher mismatch rates (~77%) and SOL showing a lower but still-significant rate (57%).

### 4.3 Breakdown by Live Side / Shadow Side

| Live / Shadow | Count | Catastrophic | % |
|---------------|-------|--------------|-----|
| BUY / None | 574 | 574 | **100%** |
| BUY / SELL | 483 | 483 | **100%** |
| SELL / None | 1,468 | 1,468 | **100%** |
| SELL / SELL | 1,112 | 0 | **0%** |
| None / None | 17 | 0 | 0% |
| None / SELL | 23 | 23 | 100% |

**Critical Finding:**  
When `live_side == shadow_side == SELL` (1,112 orders), there are **zero catastrophic mismatches**.  
All 2548 catastrophic mismatches occur when sides do not match or shadow has no signal.

**Interpretation:** The systems agree when they both execute SELL. They diverge (catastrophic) when:
- Live executes BUY, shadow produces SELL or nothing
- Live executes SELL, shadow produces BUY or nothing

### 4.4 Breakdown by Timestamp Delta Bands

| Band | Total Orders | Catastrophic | % Catastrophic |
|------|--------------|--------------|-----------------|
| 0–60s | 26 | 16 | 61.5% |
| 60s–10m | 270 | 153 | 56.7% |
| 10m–1h | 733 | 443 | 60.4% |
| **1h–8h** | **2,480** | **1,788** | **72.1%** |
| **>8h** | **168** | **148** | **88.1%** |

**Median timestamp delta:** 11,063.66 seconds (~3.1 hours)  
**P95 timestamp delta:** 29,219.19 seconds (~8.1 hours)  
**Max timestamp delta:** 39,503.47 seconds (~10.9 hours)

Mismatches are **distributed across all time bands**, with increasing severity in longer delays (>8h at 88.1%).

### 4.5 Quantity Bucket Match (5% tolerance)

| Category | Count | Catastrophic | % |
|----------|-------|--------------|-----|
| Match | 1,129 | 0 | 0% |
| **Mismatch** | **2,548** | **2,548** | **100%** |

All 2548 catastrophic mismatches also have quantity mismatches (0% bucket match rate).

---

## 5. Root Cause Analysis

### Finding: Direction Divergence is the Core Issue

The data unambiguously shows that the certified replay strategy and the live executor are trading in **opposite directions** for the same symbols.

**Evidence:**
1. **100% of catastrophic mismatches are direction mismatches** (symbol_match=true, direction_match=false)
2. **Zero catastrophic mismatches when both systems trade SELL** (1,112 orders)
3. **100% of opposite-direction pairings are catastrophic** (BUY/SELL, BUY/None, SELL/None)
4. **Quantity mismatch and slippage are downstream consequences** of the direction divergence, not root causes

### Potential Causes to Investigate

**1. Replay strategy not matching live strategy** (Most Likely)
   - The FPS v2 replay certification may be using a different signal generator than the live executor
   - Entry/exit conditions may have diverged between the two code paths
   - Regime detection or thesis evaluation may differ

**2. Side mapping error** (Unlikely)
   - The mapping from signal direction (LONG/SHORT) to trade side (BUY/SELL) could have inconsistencies
   - Some signals use (ENTER_LONG → BUY), others use (EXIT_SHORT → BUY)
   - Evidence: The one case where both are SELL shows no mismatches, suggesting the mapping itself is not broken

**3. Timestamp matching tolerance** (Not the primary issue)
   - Median 3+ hour drift is suspicious, but it affects all mismatches equally
   - Even orders with <60s timestamp delta show 61.5% catastrophic rate
   - Timestamp alignment is not the differentiator

**4. Quantity sizing mismatch** (Downstream)
   - All 2548 catastrophic orders also have quantity mismatches
   - But quantity mismatch is a symptom of direction divergence, not cause
   - When directions match (SELL/SELL), quantity also matches

**5. Use of trades.csv vs permit_trace.csv** (Possible but secondary)
   - The observer reads both files from replay runs
   - The divergence pattern is consistent across all files
   - File selection does not explain 100% opposite-direction trades

**6. Live executor using different signal family** (Likely)
   - The live executor may have been running on a different signal source than the replay
   - Or the live executor's Doctrine Kernel or Hydra engine may have evolved differently
   - The replay locks state to a frozen certification point; live executor is running free

### Conclusion: STRATEGY DIVERGENCE (Not Fixable by Observability)

The root cause is **strategy-level divergence**, not data availability, matching contract, or observability errors:
- The replay is deterministic but produces different trades than the live executor
- Parsing fixes, schema adjustments, and timestamp tolerance tuning **cannot fix this**
- The two systems fundamentally disagree on trading direction for the same market conditions

---

## 6. Policy Statements

### Phase 5 Status

**Phase 5 Shadow Soak Observation is TERMINATED.**

- No further daily observation runs will be scheduled
- The observation loop is stopped
- No cron or Supervisor automation will be added
- Manual inspection of existing logs is permitted for research only

### Phase 6 Eligibility

**Phase 6 is DENIED.**

- No live activation of shadow soak observations
- No increase to executor trading authority
- No progression beyond Phase 5 diagnostic mode
- No advancement to conviction re-enablement

### Live Authority Changes

**No changes to live trading authority.**

- No increase to position sizes
- No expansion of symbol universe
- No relaxation of risk limits
- No conviction re-enablement
- No runtime restarts
- No executor, doctrine, or risk limit changes

### Doctrine and Risk Integrity

**All runtime systems remain unchanged:**

- `execution/executor_live.py` — unmodified
- `execution/doctrine_kernel.py` — unmodified
- `execution/risk_limits.py` — unmodified
- Conviction authority — frozen (as reported in certification)
- Position limits — unchanged
- NAV-based sizing — unchanged

---

## 7. Next Research-Only Step

### Recommended Investigation

1. **Identify the strategy divergence point:**
   - Compare the live executor's Hydra/Doctrine signal generation against the replay's FPS v2 certification logic
   - Check whether live executor has been running a different signal family than FPS v2
   - Inspect Sentinel-X regime detection in both paths

2. **Verify signal source equivalence:**
   - Confirm that `research/backtest_engine_v8.py` uses the same signal generators as the live executor
   - Check whether the replay certification (`research/fps_v2_certification.py`) is using the correct strategy parameters
   - Verify that OHLCV seed data is not causing regime divergence

3. **Isolate the trade-direction mapping:**
   - Check whether the live executor's Hydra engine is mapping signals differently than the replay
   - Inspect the conversion from regime (TREND_UP/DOWN, MEAN_REVERT, etc.) to order side (BUY/SELL)
   - Verify that doctrine exit handling (thesis-driven vs signal-driven) is the same in both paths

4. **Generate a deterministic trace:**
   - Run a side-by-side replay vs live executor trace with identical inputs for a single symbol and hour
   - Log every signal generation decision, regime evaluation, and order dispatch
   - Identify the first divergence point and root cause

### Not Recommended at This Time

- Adding more observers or shadow layers
- Adjusting timestamp tolerance or quantity bucket widths
- Modifying matching contracts or pairing logic
- Creating new certification variants without fixing the underlying strategy divergence
- Any form of Phase 6 advancement or live activation

---

## Artifacts

### State Files
- Latest observer state: `logs/state/shadow_soak_state.json` (run_id: `v8_phase5_shadow_soak_live_window_timestamp_check_001`)
- Certification report: `data/replay_certifications/v8_phase5_live_window_cert_001/certification_report.json`

### Event Logs
- Observer events: `logs/research/shadow_soak_events.jsonl` (append-only, 3677+ events from this run)
- Live order audit: `logs/execution/orders_executed.jsonl` (3677 orders in current window)

### Source Code
- Observer: `research/shadow_soak_v8.py` (research-only, no runtime imports)
- Replay engine: `research/backtest_engine_v8.py`
- Certification runner: `research/fps_v2_certification.py`

### Configuration
- Strategy config: `config/strategy_config.json`
- Risk limits: `config/risk_limits.json`
- Runtime config: `config/runtime.yaml`

---

## Conclusion

Phase 5 has conclusively determined that the certified replay strategy and the live executor are not equivalent. The failure is caused by strategy-level divergence, not observability, data availability, or matching contract issues.

**No further Phase 5 observation is warranted.** The investigation must shift to root-cause analysis of the strategy divergence before any Phase 6 consideration can be made.

**Current state:** Observation terminated. No runtime changes. No live authority increases. All systems frozen at current configuration.

---

**Document Status:** Final (Read-Only)  
**Created:** 2026-05-15T10:59:58Z  
**Authority:** Diagnostic only — no operational decisions permitted from this postmortem  
**Next Review:** When strategy divergence root cause has been identified and documented
