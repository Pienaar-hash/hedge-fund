# PM-SLEEVE v1 — Production Spec (Binding)

**Decision ID:** `DEC_PM_SLEEVE_V1`
**Date:** 2026-04-03
**Status:** ACTIVE
**Supersedes:** S2-as-signal (ablation gate in `binary_lab_limits_s2.json`)
**Binding Authority:** This document. Implementation must not diverge.

---

## 1. Identity

**PM-SLEEVE v1: YES-only payoff asymmetry executor**

| Property | Value |
|----------|-------|
| Venue | Polymarket binary (BTC Up/Down 15m) |
| Mechanism | Payoff asymmetry at low entry prices |
| Alpha source | Price region (structural) |
| Signal role | None (primary), weak filter (secondary) |
| Side | YES only (hard lock) |

### Why This Is Correct

Data proved (n=311 OOS, walk-forward):
- Win rate ~49-53% across all regions
- Profit comes from **payoff ratio > 1.0** at low entry prices
- S2 model edge is not causal for PnL — price region is causal
- Edge magnitude serves only as a weak confidence filter

---

## 2. Entry Logic (Authoritative)

### Gate Ordering (Final)

```
Gate 1: price_region  →  Gate 2: side  →  Gate 3: confidence  →  Gate 4: friction  →  EXECUTE
```

This is a **control inversion** from the prior S2 design:
- Old: `edge → side → friction`
- New: `price → side → edge → friction`

### Gate 1 — Price Region (PRIMARY)

```
entry_cost = p_yes_ask

if entry_cost < max_entry_cost (default 0.45):
    region ∈ {extreme_low, low, mid_low} → PASS
else:
    BLOCK → skip_reason: SKIP_REGION_BLOCKED
```

**This is the alpha. Everything else is subordinate.**

Payoff ratio at entry_cost < 0.45: `(1 - 0.45) / 0.45 = 1.22x` minimum.
At entry_cost = 0.30: `(1 - 0.30) / 0.30 = 2.33x`.

### Gate 2 — Side Lock

```
side = YES only
NO trades → BLOCK → skip_reason: SKIP_SIDE_BLOCKED
```

Rationale: YES at low prices has structural payoff skew. NO at low prices
means high entry cost (complementary), which is the loss region.

### Gate 3 — Confidence Filter (Optional)

```
if confidence_filter.enabled:
    if |edge_yes| >= min_edge_abs (default 0.05):
        PASS
    else:
        BLOCK → skip_reason: SKIP_CONFIDENCE_BELOW_MIN
```

- Default: **ENABLED**
- Can be disabled via config if throughput too constrained
- Uses |edge_yes| (absolute magnitude), NOT directional prediction
- S2 model predictions are NOT used for direction — only for magnitude

### Gate 4 — Friction (Existing)

Carried forward from S2:
- `spread < max_spread_threshold` (default 0.04)
- `quote_age_s <= max_quote_age_s` (default 75s)
- `time_remaining_s > min_time_remaining_s` (default 120s)
- `freeze_intact == True`
- `open_positions < max_concurrent` (default 3)
- `current_nav_usd > kill_nav_usd` (default 650)

No changes to friction gates.

---

## 3. Exit Logic

**Unchanged from S2:**
- Hold to round close (15-min binary resolution)
- No mid-round exit logic
- Settlement: BTC price at round end vs reference price at round start
- DLE exit reason: `PHASE_END` (round close = phase boundary)

---

## 4. KPIs

### Primary KPIs (replace old edge quintile tracking)

| KPI | Definition |
|-----|-----------|
| Net expectancy per trade | `mean(pnl_usd)` over rolling window |
| Cumulative PnL | `sum(pnl_usd)` since activation |
| Max drawdown | Peak-to-trough cumulative PnL |

### Structural KPIs (new)

| KPI | Definition |
|-----|-----------|
| PnL by price region | `sum(pnl_usd)` grouped by `{extreme_low, low, mid_low, center, mid_high, high, extreme_high}` |
| Trade distribution by region | `count` per region / total count |
| Average entry cost | `mean(entry_cost)` across traded rounds |
| Payoff ratio | `mean((1 - entry_cost) / entry_cost)` across traded rounds |

### Execution KPIs (monitoring)

| KPI | Definition |
|-----|-----------|
| Spread at entry | `mean(spread)` across traded rounds |
| Skip rate (friction) | `count(SKIP_FRICTION_*)` / total eligible |
| Skip rate (region) | `count(SKIP_REGION_BLOCKED)` / total signals |
| Skip rate (confidence) | `count(SKIP_CONFIDENCE_BELOW_MIN)` / total region-passed |

---

## 5. Kill Conditions

### Hard Stops

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Rolling expectancy | mean(pnl_usd) < 0 over last N=50 trades | TERMINATE |
| Drawdown | cumulative drawdown > max_drawdown_usd (250) | TERMINATE |
| NAV kill line | current_nav_usd ≤ kill_nav_usd (650) | TERMINATE |
| Region distribution drift | < 60% of trades in {extreme_low, low, mid_low} | FLAG (warning, not kill) |

### Rationale

- Rolling expectancy kill catches regime failure (edge disappears)
- Drawdown kill is inherited seatbelt
- Region distribution drift detects market structure change (prices no
  longer available at profitable levels)

---

## 6. Known Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Regime dependence | Edge exists only if low-price opportunities persist | Region distribution drift monitoring |
| Market adaptation | Odds may reprice to eliminate asymmetry | Rolling expectancy kill over N=50 |
| Friction sensitivity | Spreads can erase structural edge | Spread gate + friction KPI tracking |
| YES bias unknown | Structural vs temporary market-making artifact | Monitor via payoff ratio stability |
| Liquidity at extremes | Extreme-low prices may have thin books | Depth score gate (existing) |

---

## 7. DLE Integration

### Entry Decision Logging

Every entry decision (trade or skip) emits a DLE shadow chain:

```
requested_action: PM_SLEEVE_ENTRY
context: {
    price_region: str,
    entry_cost: float,
    edge_magnitude: float,
    payoff_ratio: float,
    skip_reason: str | null
}
verdict: PERMIT | DENY
deny_reason: <mapped from skip_reason>
```

### Exit Logging

- Exit reason: `PHASE_END` (round close is phase boundary)
- All exits are deterministic (15-min timer, no discretionary exits)

### Contracts

- `exit_reason_map.yaml`: PM sleeve exits map to `PHASE_END`
- `decision_id`: deterministic from (phase_id, action_class, constraints, policy_version)
- Shadow ledger: `logs/execution/dle_shadow_events.jsonl` (existing, append-only)

---

## 8. Configuration Surface

### Frozen Config: `config/binary_lab_limits_s2.json`

New `pm_sleeve_v1` block (sibling to existing config):

```json
{
  "pm_sleeve_v1": {
    "enabled": true,
    "price_region": {
      "max_entry_cost": 0.45
    },
    "side_filter": "YES_ONLY",
    "confidence_filter": {
      "enabled": true,
      "min_edge_abs": 0.05
    },
    "kill_conditions": {
      "rolling_window": 50,
      "min_expectancy": 0.0,
      "max_drawdown_usd": 250,
      "min_region_hit_rate": 0.60
    }
  }
}
```

### Superseded Config

The `ablation_gate` block in `binary_lab_limits_s2.json` is superseded by
`pm_sleeve_v1`. It is retained for backward compatibility but is no longer
authoritative when `pm_sleeve_v1.enabled == true`.

---

## 9. Activation Window

- **Fresh 30-day window** upon activation
- **Phase 1: SHADOW** — observe region-first gate vs old S2 gate side-by-side
- **Phase 2: Controlled activation** — after shadow validation
- **No parameter changes** during window (freeze discipline)
- **Dual-key activation**: env var + config flag required
- **Kill conditions monitored** from day 1

### Pre-Activation Checklist

1. `pytest -q` all green
2. `v7_manifest.json` unchanged from production tag
3. PM sleeve tests pass (`test_binary_lab_s2_smoke.py`)
4. Config hash recorded in state
5. DLE shadow active
6. Region distribution baseline established

---

## 10. Implementation Boundary

### In Scope

- `extract_pm_sleeve_signal()` in `binary_lab_s2_signals.py` (new function)
- `check_pm_sleeve_eligibility()` in `binary_lab_s2_signals.py` (new function)
- `_price_region()` canonical location in `binary_lab_s2_signals.py`
- Runner wiring in `binary_lab_s2_shadow.py`
- Config update to `binary_lab_limits_s2.json`
- Test coverage in `test_binary_lab_s2_smoke.py`

### Out of Scope (Explicitly)

- No changes to `binary_lab_s2_model.py` (model untouched)
- No probability recalibration
- No new indicators
- No S1 changes
- No futures changes
- No changes to `doctrine_kernel.py`
- No changes to `risk_limits.py`

---

## 11. What This Replaces

| Before | After |
|--------|-------|
| S2 edge drives trade decision | Price region drives trade decision |
| Edge threshold = primary gate | Entry cost threshold = primary gate |
| YES/NO both tradeable | YES only |
| `ablation_gate.min_edge_abs: 0.10` | `confidence_filter.min_edge_abs: 0.05` |
| Signal = probability forecaster | Signal = weak confidence filter |
| Alpha = prediction accuracy | Alpha = payoff asymmetry |

---

**Signed:** System
**Effective:** 2026-04-03
**Freeze Duration:** 30 days from activation
