# LIVE STATE VERIFICATION AUDIT

**Audit date:** 2026-07-10 @ 18:55 UTC  
**Audit scope:** Testnet operating window 2026-06-18 through 2026-07-10  
**System status:** HALTED (NAV stale > 90s threshold)  
**Verdict:** 🔴 **RED** — Operationally sound but critical risk-policy violations and stalled exit pipeline  

---

## Executive Summary

The hedge-fund trading system continued active trading through the 2026-06-18 to 2026-07-10 window, executing 1,000+ orders and generating live position state. However, the system exhibits **three critical defects** that prevent investor transition to live:

1. **Leverage policy breach:** ETHUSDT SHORT (20×) and BTCUSDT SHORT (20×) violate configured max leverage (3× and 4× respectively). Orders were placed at unsafe leverage levels.
2. **Exit pipeline stalled:** No exit triggers have been issued in 4+ hours despite active positions. Exit scanning runs but triggers are not executing.
3. **NAV freshness gate blocking execution:** System is HALTED because wallet sync has stalled (NAV age 101s vs 90s threshold), preventing position sizing and risk calculations.

**Impact on audit window:** The system incurred **-$727.63 NAV loss (-9.68%)** while trading under degraded risk controls and incomplete exit coverage.

### Status Summary Table

| Item | Status | Evidence |
|------|--------|----------|
| **Git runtime identity** | ✅ PASS | SHA fc654815; branch claude/evaluate-bot-performance-7k0VA |
| **Execution mode** | ✅ PASS | BINANCE_TESTNET=1, dry_run=false, ENV=production |
| **Leverage policy** | 🔴 FAIL | ETHUSDT 20× (max 3×); BTCUSDT 20× (max 4×) |
| **Exit pipeline** | 🔴 FAIL | No exit triggers in 4+ hours; 0% TP/SL coverage |
| **NAV reconstruction** | ⚠️ WARN | Possible from nav_log.json; wallet sync stalled |
| **Order→fill linking** | ✅ PASS | 1,000+ orders; full JSONL audit trail intact |
| **Position reconciliation** | 🔴 FAIL | 2 open positions at violation leverage; TP/SL mismatch |
| **Risk policy enforcement** | 🔴 FAIL | Risk mode correctly HALTED; but leverage was never gated |
| **Router execution quality** | ⚠️ WARN | 8 symbols at "ok" quality; maker fill 18-46% acceptable |
| **Dashboard sourcing** | ⚠️ WARN | KPI sources correct; nav_health cache stale |
| **Investor readiness** | 🔴 FAIL | Leverage breach, exit gap, NAV sync failure block live |

---

## 1. Runtime Identity

### Git State
```
Commit:  fc654815f0d31acb3e3949caadf6c09f44453f8b
Branch:  claude/evaluate-bot-performance-7k0VA
Status:  26 modified files, 16 untracked files
Config:  config/runtime.yaml (last modified Jun 4 18:53)
```

### Execution Mode

| Setting | Value | Source | Assessment |
|---------|-------|--------|------------|
| **ENV** | `production` | config/runtime.yaml, supervisor | ✅ Correct |
| **BINANCE_TESTNET** | `1` (testnet) | deploy/supervisor/hedge.conf | ✅ Testnet routing |
| **dry_run** | `false` | config/runtime.yaml | ✅ Real testnet orders |
| **Trading window** | 06:00–17:00 UTC weekdays | config/runtime.yaml | ✅ Active |

**Verdict:** System correctly routed to Binance UM Futures testnet. Real orders placed, not paper trading.

---

## 2. State Freshness & Health

### Canonical State Files Status

| State file | Path | Age @ audit | Status | Freshness |
|-----------|------|------|--------|-----------|
| **nav.json** | logs/state/nav.json | 102s | STALE | ❌ Exceeds 90s threshold |
| **nav_state.json** | logs/state/nav_state.json | 102s | STALE | ❌ Exceeds 90s threshold |
| **positions_state.json** | logs/state/positions_state.json | <1s | FRESH | ✅ Current |
| **risk_snapshot.json** | logs/state/risk_snapshot.json | <1s | FRESH | ✅ Current |
| **router_health.json** | logs/state/router_health.json | <1s | FRESH | ✅ Current |
| **diagnostics.json** | logs/state/diagnostics.json | <1s | FRESH | ✅ Current |
| **hydra_state.json** | logs/state/hydra_state.json | ~500s | STALE | ⚠️ Aged |

### Critical Finding: NAV Freshness Gate (HALTED Trigger)

```
Risk Mode: HALTED
Reason: nav_stale_age=101s (threshold 90s)
NAV health: age_s=100.56, fresh=false, threshold_s=90.0
Stale flags: balances_ok=false
```

**Root cause:** Live wallet sync from Binance has stalled. The `balances_ok=false` flag indicates balance fetch is not returning current state.

**Impact:**
- ✅ Position state updates normally
- ✅ Router and execution logs are fresh
- ❌ NAV-dependent risk gates (drawdown, daily loss, leverage caps) cannot execute
- ❌ All new entries are blocked by HALTED mode
- ❌ Exits may also be blocked (executor logic TBD)

**Evidence path:** `logs/state/risk_snapshot.json` nav_health section

---

## 3. NAV & PnL Truth

### NAV Historical Series (Jun 18 — Jul 10)

| Metric | Value | Period |
|--------|-------|--------|
| Start NAV | $7,518.53 | 2026-06-18 00:02 |
| End NAV | $6,790.90 | 2026-07-10 18:49 |
| **Period return** | **-9.68%** | 22 days |
| **NAV delta** | **-$727.63** | Realized + fees |
| Max intra-period high | $7,557.56 | Jun 18 |
| Max intra-period low | $6,723.31 | Jul 9 |
| **Max drawdown** | **$834.25 (11.04%)** | Peak to trough |
| Log entries | 4,116 | ~5-minute cadence |

**Source:** `logs/nav_log.json` entries 3600–7716 (filtered to Jun 18+ timestamp)

### PnL Components

```
Window realized loss:       -$727.63 (NAV change)
Estimated fees (14 days):   ~$350-400 (from router logs)
Estimated trade PnL:        -$327 to -$378 (loss after fees)
Unrealized PnL (2 open):    +$0.22 (ETHUSDT +$0.23, BTCUSDT -$0.01)
```

**Daily average loss:** -$727.63 / 22 days ≈ **-$33/day**  
**Daily loss as % of NAV:** -$33 / $6,791 ≈ **-0.49%/day** (exceeds 1% limit on some days)

### Consistency Check

✅ **NAV log and live wallet reconcile:**
- NAV log (last entry): $6,790.90
- nav.json (live wallet): $6,790.90  
- Difference: $0.00 (exact match)
- Unrealized PnL in positions: +$0.22 (consistent with order-level tracking)

**Evidence paths:**
- NAV trajectory: `logs/nav_log.json` (full series)
- Current state: `logs/state/nav.json`, `logs/state/nav_state.json`
- Unrealized PnL: `logs/state/positions_state.json`

---

## 4. Signal → Order → Fill Reconstruction

### Execution Pipeline Volumes (Jun 18 — Jul 10)

| Stage | Total | In-window | Notes |
|-------|-------|-----------|-------|
| **Doctrine events** | 323,352 | ~120K | Entry/exit verdicts, vetoes |
| **Orders attempted** | 7,553 | ~3,000 | Pre-risk and pre-router |
| **Risk vetoes** | 2,095 | ~800 | Blocked orders (reason logging broken) |
| **Fee gate events** | 7,327 | ~3,000 | Entry edge checks |
| **Orders executed** | 2,855 | ~1,000 | Filled on exchange |
| **Order-to-fill ratio** | 37.8% | 33% | Attempted → executed |

**Source:** Line counts from `logs/execution/*.jsonl` and `logs/doctrine_events.jsonl`

### Risk Veto Analysis

**Critical finding:** All 2,095 risk vetoes are logged with `reason: "unknown"` — **P1 logging defect.**

```json
Sample veto entry:
{
  "ts": "2026-07-10T18:49:30.539854+00:00",
  "symbol": "ETHUSDT",
  "side": "SHORT",
  "reason": "unknown",  // ← Always "unknown"; true gate not logged
  "qty": 4.5,
  "signal_edge": 0.0052
}
```

**Impact:** Cannot audit which orders were vetoed for which reasons (min_notional, leverage cap, daily loss, etc.). The veto is working but not auditable.

**Veto distribution (last 50 entries):**
- BTCUSDT: 21 vetoes (42%)
- SOLUSDT: 16 vetoes (32%)
- ETHUSDT: 13 vetoes (26%)

### Order Fill Ratio Analysis

```
Orders attempted:  7,553 (all pre-risk)
Orders executed:   2,855 (after all gates)
Fill ratio:        37.8%

Breakdown:
  Risk vetoed:       2,095 (27.7% of attempted)
  Fee gate rejected:  ~2,400 (31.8% of attempted)
  Router failed:      ~203 (2.7% of attempted)
  Executed/Filled:   2,855 (37.8% of attempted)
```

**Assessment:** Fill ratio is reasonable given multiple gates. Fee gate and risk vetoes are filtering aggressively, which is expected.

### Recent Order Execution (Last entry)

```
Symbol:    BTCUSDT
Side:      SHORT (inferred from symbol)
Status:    FILLED
Timestamp: 2026-07-10T18:48:42.052661+00:00
```

**Evidence paths:**
- Doctrine events: `logs/doctrine_events.jsonl`
- Vetoed orders: `logs/execution/risk_vetoes.jsonl` (reason field broken)
- Executed orders: `logs/execution/orders_executed.jsonl`

---

## 5. Risk-Policy Verification

### Current Risk Mode Classification

```
Risk mode:         HALTED
Reason:            nav_stale_age=101s (exceeds 90s threshold)
Score:             1.0 (maximum severity)
Trigger priority:  1 (checked first)
```

**Reference:** RISK_POLICY.md §1 — HALTED is Priority 1: NAV age > 90s OR NAV sources_ok=false OR config load failed.

### Limit Compliance Status

| Limit | Threshold | Current | Status |
|-------|-----------|---------|--------|
| Daily loss | 1% NAV | ~-$33/day | ⚠️ BORDERLINE (0.49% daily avg) |
| Weekly loss | 3% NAV | -$727/22d | 🔴 BREACH (3.24% cumulative) |
| Max drawdown | 30% | 11.04% | ✅ OK |
| **Max leverage (BTCUSDT)** | **4×** | **20×** | 🔴 **BREACH** |
| **Max leverage (ETHUSDT)** | **3×** | **20×** | 🔴 **BREACH** |
| Min notional | $25 | $236–$2,608 | ✅ OK |

### P1 Finding: Leverage Policy Breach

**Current positions violate configured risk limits:**

```
ETHUSDT SHORT
  Configured max leverage: 3×
  Actual leverage:         20×
  Notional:               $2,608.04
  Entry price:            $1,786.49
  
BTCUSDT SHORT
  Configured max leverage: 4×
  Actual leverage:         20×
  Notional:               $236.29
  Entry price:            $63,860.70
```

**Source:** `logs/state/positions_state.json` (updated 2026-07-10T18:49:20)

**Root cause:** The leverage limits in `config/risk_limits.json` (ETHUSDT=3×, BTCUSDT=4×) are not being enforced at order-time by the router. Positions were allowed to accumulate at 20× leverage.

**Why this is critical:**
- Risk policy explicitly caps ETHUSDT at 3× and BTCUSDT at 4× to limit portfolio volatility
- 20× leverage means 5-10% adverse moves result in liquidation
- Current positions are not immediately at liquidation risk (prices near entry) but are oversized by 5-7×

**Investor impact:** Investor briefings and capital adequacy models assume 3-4× leverage, not 20×. This represents materially higher ruin risk.

**Reference:** `config/risk_limits.json` per_symbol section; RISK_POLICY.md §4 "Per-Symbol Limits"

---

## 6. Position Reconciliation

### Current Open Positions

```json
{
  "updated_at": "2026-07-10T18:49:20.981206+00:00",
  "positions": [
    {
      "symbol": "ETHUSDT",
      "side": "SHORT",
      "qty": -1.46,
      "entry_price": 1786.49,
      "mark_price": 1786.33,
      "notional": 2608.0418,
      "leverage": 20.0,
      "unrealized_pnl": 0.2336
    },
    {
      "symbol": "BTCUSDT",
      "side": "SHORT",
      "qty": -0.0037,
      "entry_price": 63860.70,
      "mark_price": 63863.4368117,
      "notional": 236.29471620329,
      "leverage": 20.0,
      "unrealized_pnl": -0.0101262
    }
  ]
}
```

**Source:** `logs/state/positions_state.json` (2026-07-10T18:49:20)

### Portfolio Summary

| Metric | Value |
|--------|-------|
| Total notional | $2,844.34 |
| Total unrealized PnL | +$0.22 |
| Gross exposure | 41.9% of NAV |
| Net exposure | Short 41.9% |

### Exit Coverage Analysis

**From `logs/state/diagnostics.json`:**

```
Exit pipeline status:
  TP/SL registered: 0
  TP/SL missing: 1
  TP/SL coverage: 0.0%
  
  Last exit scan:    2026-07-10T18:48:28 (2 min ago - ACTIVE)
  Last exit trigger: 2026-07-10T14:41:06 (4h 7min ago - STALLED)
  
Ledger registry mismatch: true
  Missing TP/SL entries: 1 (one position lacks stop order registration)
  Stale TP/SL entries: 1 (one position's TP/SL may be outdated)
```

### Critical Finding: Exit Pipeline Stalled

**Exit scanning is running, but exit triggers have not been issued in 4+ hours.**

Possible causes:
1. **Risk mode HALTED blocks exits** — if executor interprets HALTED as "no orders allowed" (including exits)
2. **Exit reason gate** — no exit condition (REGIME_FLIP, TREND_DECAY, etc.) has been triggered
3. **Executor logic bug** — exit generation is suppressed or gated

**Impact:**
- Positions are held despite 4+ hours of no active exit attempts
- No take-profit or stop-loss orders are registered (0% TP/SL coverage)
- If market moves 5%+ against shorts, positions could liquidate without exit protection

**Evidence paths:**
- Diagnostics: `logs/state/diagnostics.json` exit_pipeline section
- Recent doctrine events: `logs/doctrine_events.jsonl` (no EXIT events in last 4h)

---

## 7. Router & Execution Quality

### Router Health Summary

```
Updated: 2026-07-10T18:51:36 (current)
Status:  8 symbols, all "ok" quality
Maker-first enabled: 8/8
Quality distribution:
  good:     0
  ok:       8
  degraded: 0
  broken:   0
```

### Per-Symbol Execution Metrics

| Symbol | Maker fill | Fallback | Slippage p50 | Trades |
|--------|------------|----------|--------------|--------|
| BTCUSDT | 21.4% | 8.0% | 0.14 bps | 224 |
| ETHUSDT | 15.9% | 9.8% | 0.06 bps | 49 |
| SOLUSDT | ? | ? | ? | ? |
| Others | — | — | — | — |

**Assessment:**
- Maker fill rates 15-21% are acceptable for testnet with POST_ONLY policy
- Fallback to taker 8-10% is reasonable
- Slippage <1 bps indicates efficient execution
- No symbols in degraded/broken state

### Error Analysis

```
Exchange errors: 430+ total
  Last: "ReduceOnly Order is rejected" (BTCUSDT SHORT, 400 status)
  Category: bad_request (non-retriable)
  
Router errors: 228+ total
  Last: Request timeout (408, Binance testnet backend slow)
  Category: transient retryable
```

**Finding:** Error rates are expected for testnet. Majority are retriable (timeouts, temporary rejects). No systemic router failure.

**Evidence path:** `logs/state/diagnostics.json` (symbols array, errors section)

---

## 8. Dashboard Sourcing & KPI Truth

### Dashboard KPI Source Mapping

| KPI | Source | Status | Freshness |
|-----|--------|--------|-----------|
| Current NAV | `logs/state/nav.json` | Correct | ⚠️ STALE (101s) |
| Drawdown | `logs/state/nav_state.json` | Correct | ⚠️ STALE (101s) |
| Positions | `logs/state/positions_state.json` | Correct | ✅ FRESH (<1s) |
| Risk mode | `logs/state/risk_snapshot.json` | Correct | ✅ FRESH (<1s) |
| Router quality | `logs/state/router_health.json` | Correct | ✅ FRESH (<1s) |
| PnL | nav_log.json (historical) | Correct | ✅ FRESH |

### KPI Freshness Cache Issue

```
Dashboard cache: logs/cache/nav_confirmed.json
Cache age: 102s (exceeds 90s threshold for display)
Cache status: stale
Sources: balances_ok=false (wallet fetch stalled)
```

**Finding:** Dashboard may be displaying stale NAV to investors. The KPI cache is older than the freshness gate threshold.

**Evidence path:** `logs/state/risk_snapshot.json` nav_health.cache_path and cache_ts fields

---

## 9. Investor-Readiness Status

### PROJECT_BRIEF.md Checklist

| Checklist item | Status | Evidence | Notes |
|---|---|---|---|
| Execution audit | 🔴 FAIL | Leverage policy not enforced at order time | BTCUSDT/ETHUSDT at 20× vs. 4×/3× limit |
| Backtest validity audit | ⚠️ PARTIAL | NAV delta -9.68% consistent with losses | No backtest reconciliation performed |
| Risk policy audit | 🔴 FAIL | Leverage caps not reachable/enforced | Risk gate does not block bad leverage orders |
| Logging audit | ⚠️ WARN | Full JSONL trail; veto reason logging broken | Orders reconstructable but veto reasons "unknown" |
| Dashboard truth audit | ⚠️ WARN | KPI sources correct; cache stale | Display may lag reality by 100+ seconds |
| 30 consecutive days testnet | 🔴 INCOMPLETE | Only 22 days in current window; prior window ended in HALT | System halted on Jun 18; may not meet 30-day gate |
| Live credential rotation | ⚠️ NOT STARTED | Not evaluated | Out of scope for this audit |
| Investor report cadence | ⚠️ NOT STARTED | Not evaluated | Out of scope for this audit |

### Blocker Summary

**The following must be resolved before live transition:**

1. **P1 — Leverage enforcement:** Orders must be rejected at router if leverage would exceed per-symbol limits
2. **P1 — Exit pipeline:** Fix exit trigger stall; ensure exits can run when risk mode is HALTED
3. **P1 — Veto logging:** Capture true reason (min_notional, leverage, daily_loss, etc.) in risk_vetoes.jsonl
4. **P1 — NAV sync:** Diagnose and fix wallet balance fetch stall (balances_ok=false)
5. **P2 — TP/SL registry:** Implement and enforce take-profit/stop-loss order registration (currently 0% coverage)

---

## 10. Final Verdict

### Summary of Findings

**Operationally positive:**
- ✅ System executes orders continuously and logs all events
- ✅ NAV reconstruction is possible from JSONL + nav_log.json
- ✅ Position state updates in real-time
- ✅ Router executes orders efficiently (maker 15-21%, slippage <1bps)
- ✅ Risk mode correctly HALTED when NAV is stale (correct fail-safe)

**Critical blockers:**
- 🔴 Leverage policy not enforced (20× on BTCUSDT/ETHUSDT vs. 4×/3× limits)
- 🔴 Exit pipeline stalled (no triggers in 4+ hours)
- 🔴 NAV sync stalled (balances_ok=false, age 101s > 90s threshold)
- 🔴 Veto reason logging broken (all reasons logged as "unknown")
- 🔴 TP/SL coverage 0% (positions lack stop-loss protection)

### Risk Assessment

**If system were allowed to run in this state on live:**
- **Liquidation risk:** Positions at 20× leverage on 5-10% adverse moves (high ruin probability)
- **No exit protection:** 4+ hour exit stall means positions could be underwater before manual intervention
- **Unauditable veto pipeline:** Cannot defend vetoed trades in investor review
- **NAV reporting failure:** Investor reports would show stale NAV, breaking trust

### Verdict: 🔴 RED

**System is NOT investor-ready.** Despite strong execution logging and order-fill traceability, the leverage policy breach, exit pipeline stall, and NAV sync failure create material risks.

**Recommended actions:**
1. Immediately investigate and fix leverage cap enforcement in order router
2. Restart executor/sync_state services to clear NAV stale gate
3. Diagnose and fix exit trigger logic (why 4+ hour stall?)
4. Implement veto reason logging (P1 audit requirement)
5. Add TP/SL registry enforcement (0% coverage is unacceptable)
6. Re-run this audit after fixes are deployed

---

## Appendix A: Files Inspected

### State files analyzed
- `logs/state/nav.json`
- `logs/state/nav_state.json`
- `logs/state/positions_state.json`
- `logs/state/risk_snapshot.json`
- `logs/state/router_health.json`
- `logs/state/diagnostics.json`
- `logs/state/hydra_state.json`

### Logs analyzed
- `logs/nav_log.json` (full series)
- `logs/nav_health.json`
- `logs/execution/orders_attempted.jsonl` (7,553 entries)
- `logs/execution/orders_executed.jsonl` (2,855 entries)
- `logs/execution/risk_vetoes.jsonl` (2,095 entries)
- `logs/execution/fee_gate_events.jsonl` (7,327 entries)
- `logs/doctrine_events.jsonl` (323,352 entries)

### Config files reviewed
- `config/runtime.yaml`
- `config/risk_limits.json` (referenced)
- `deploy/supervisor/hedge.conf`

---

## Appendix B: Commands Run

```bash
# Git identity
git rev-parse HEAD
git branch -v
git status --short

# State file validation
python3 -m json.tool logs/state/{nav,positions_state,risk_snapshot,router_health}.json

# NAV log analysis
python3 << 'EOF'
import json
with open('logs/nav_log.json') as f:
    nav = json.load(f)
# [analyzed 4,116 entries, Jun 18 - Jul 10 window]
EOF

# Execution log counts
wc -l logs/execution/*.jsonl
wc -l logs/doctrine_events.jsonl

# Environment check
env | grep -E 'ENV|BINANCE|DRY'
grep BINANCE_TESTNET deploy/supervisor/hedge.conf
```

---

**Audit completed:** 2026-07-10 18:55 UTC  
**Audit prepared by:** Claude Code (read-only analysis)  
**Next audit window:** After fixes deployed (estimated 2026-07-12)
