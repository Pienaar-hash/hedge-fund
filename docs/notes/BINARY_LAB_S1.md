## Clean 15m Directional Sleeve Under Freeze (Binary Lab S1)

This design keeps **every freeze rule intact** and removes all bundle logic. One trade per eligible round, hold to resolution, score by conviction band. 

---

## 0) Scope and Non Scope

**In scope**

* 15m only (locked)
* Directional entry only (Up *or* Down, never both)
* Hold to resolution (no mid round exits)
* Use the existing **futures pipeline** as the signal source (no new indicators) 

**Out of scope**

* Two sided bundle
* Latency arbitrage
* Tick exploitation
* Any new features, filters, thresholds, or timing tweaks during the window 

---

## 1) Frozen Contract (S1)

From `binary_lab_limits.json` (frozen):

* Sleeve total: **$2000**
* Deployed max: **$1200**
* Reserve: **$800**
* Per round: **$20** (exact)
* Max concurrent: **3**
* Blocked regimes: **CHOPPY**
* Min conviction band: **medium**
* Regime confidence fallback min: **0.60**
* No stacking, no martingale, no size escalation after wins
* Kill line: NAV **$1700** or drawdown **$300 / 15%** 

---

## 2) Entry Decision Rule (Deterministic)

For each 15m round `R`:

### Step A — Eligibility gate (binary decision)

Eligible if all true:

1. `horizon == 15m` (hard lock)
2. `sentinel.regime NOT IN blocked_regimes` (CHOPPY blocked)
3. `sentinel.confidence >= 0.60` (fallback min)
4. `conviction_band >= medium`
5. `open_positions < 3`
6. `freeze_intact == true`
7. `capital.current_nav_usd > 1700` and kill distance > 0 

If any fails → **NO_TRADE** (log reason, do nothing).

### Step B — Direction selection (single side)

Use the futures pipeline directional intent:

* If intent direction = UP → buy UP
* If intent direction = DOWN → buy DOWN

No secondary confirmation, no overrides. This preserves the “no new signals” rule. 

### Step C — Order sizing

* Always **$20 notional per round**, never deviates.
* If the venue requires size in shares, convert deterministically using the mid price at submission time.

---

## 3) Execution Mode (Deterministic, Freeze Safe)

To avoid introducing a new “timing strategy,” the sleeve uses a single fixed submission offset:

* **Submit once per eligible round at a fixed offset** (e.g., `t = round_start + X seconds`)
* X is part of the frozen configuration for the full 30 days (no adjustment allowed)

Order type policy:

* One order attempt only (no chase loops)
* If not filled within the round entry window → mark `NO_FILL` and skip the round

This stays inside the freeze envelope because it doesn’t add new timing logic during the window, it declares timing once and locks it. 

---

## 4) State Machine Integration

All lifecycle transitions go through the Binary Lab reducer:

Events:

* `ACTIVATE`
* `ROUND_CLOSED`
* `DAILY_CHECKPOINT`
* `TERMINATE` 

Hard termination triggers (immediate):

* kill line breached
* any rule violation (stacking, martingale, size mismatch, concurrent cap breach)
* freeze broken (hash mismatch)

The reducer emits:

* `TERMINATE_IMMEDIATELY`
* `CLOSE_ALL_POSITIONS` 

---

## 5) Logging Contract (Minimum Required Fields)

Append-only `binary_lab_trades.jsonl` must support replayable scoring. Minimum event fields per trade:

* `ts_ms`, `round_id`, `market_slug`, `horizon_s=900`
* `side` (UP/DOWN), `intent_direction`
* `p_fill`, `notional_usd=20`, `fee_usdc`
* `conviction_band`, `conviction_score` (if available), `regime`, `regime_confidence`
* `eligibility` outcome + `deny_reason` when no trade
* `status`: filled / no_fill / rejected
* `resolved_outcome` + `pnl_usd` when round closes

This matches the state machine requirement that all downstream metrics are derivable from explicit events. 

---

## 6) Daily Checkpoint (15 seconds)

Use the existing daily SOP fields:

* NAV above kill line
* open positions ≤ 3
* trade log growing (or 0-trade day explicitly recorded)
* config hash unchanged
* violations 0
* freeze intact true 

---

## 7) Evaluation Metrics (Directional Only)

Per day and cumulative:

* trades, wins, losses, win rate
* EV per trade (net of fees)
* EV by conviction band (medium/high/very_high)
* drawdown and kill distance
* band separation: EV(high) − EV(medium), EV(very_high) − EV(high)

No microstructure metrics are required for this sleeve. They’re out of scope.

---

## 8) Codex Handoff (Implementation Plan)

## Task: Implement 15m Directional Sleeve (Freeze Safe)

**Target:** `execution/binary_lab_executor.py`, `execution/binary_lab_signals.py` (or equivalent), `logs/state/binary_lab_state.json` writer
**Precondition:** `binary_lab_limits.json` hash is loaded and recorded at activation; horizon lock is enforced
**Change:**

* Remove any bundle logic paths entirely
* Implement eligibility gate + single-side direction selection from futures pipeline
* Enforce exact $20 sizing, 1 attempt per round, hold to resolution
* Emit `ROUND_CLOSED` events with realised PnL and resolved outcome
  **Verification:** unit tests for: horizon mismatch, CHOPPY block, confidence floor, size exactness, concurrent cap, kill-line termination, freeze-hash mismatch termination 
  **Doctrine check:** No expansion (uses existing signal source, no new indicators) 
  **Manifest impact:** If any new state surface is added, update `v7_manifest.json` + schema tests; otherwise none
  **Phase C impact:** No changes to frozen configs during the window 

---

If we want, next message can be the exact **activation gate definition** for this sleeve (what qualifies as “GO” for `ACTIVATE`) in one declarative block, matching the reducer’s required inputs. 
