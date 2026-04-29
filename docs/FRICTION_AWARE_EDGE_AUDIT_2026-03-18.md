# Friction-Aware Edge Audit — 2026-03-18

## §1 — Purpose

This audit answers one question: **Does the system's signal edge survive execution friction?**

A signal with positive raw edge (price movement in the predicted direction) may deliver
zero or negative net edge once exchange fees, spread crossing, and slippage are deducted.
If friction routinely kills profitable signals, the system is a fee-generation machine
masquerading as a trading system.

### Key Metrics

| Metric | Definition | Location |
|--------|-----------|----------|
| **Raw edge** | `(exit_px − entry_px) / entry_px` × 10,000 (BPS, direction-adjusted) | Episode avg prices |
| **Fee drag** | `fees / entry_notional` × 10,000 (BPS) | Episode fees + notional |
| **Net edge** | `raw_edge − fee_drag` (BPS) | Computed |
| **Friction-killed** | Trade where `raw_edge > 0` but `net_edge ≤ 0` | Computed |
| **Kill rate** | `friction_killed_count / positive_raw_count` | Computed |
| **Fee-to-edge ratio** | `Σ fee_drag / Σ raw_edge` | Computed |

---

## §2 — Friction Sources

### 2.1 Exchange Fees

Binance futures commissions are the primary friction component. The system uses
maker-first routing (`POST_ONLY` with taker fallback), so the actual fee profile
depends on fill type.

| Fill Type | Rate (BPS) | Config Source |
|-----------|-----------|---------------|
| Taker | 4–5 | `config/runtime.yaml → fee_gate.taker_fee_rate` |
| Maker | 2 (or −1 rebate) | `order_router.py → MAKER_BPS` |
| Round-trip (taker/taker) | ~8–10 | Worst case |
| Round-trip (maker/maker) | ~4 | Best case |

Fee data per trade: `Episode.fees` (sum of entry + exit commissions from fills).

### 2.2 Slippage

Price impact from crossing the order book. Tracked in `execution/slippage_model.py`
via three functions:

- `estimate_expected_slippage_bps()` — VWAP vs mid (pre-trade, depth-based)
- `compute_realized_slippage_bps()` — fill price vs mid (post-trade)
- `compute_spread_bps()` — bid-ask width

EWMA state: `logs/state/slippage_metrics.json` per symbol.
Per-order: `logs/execution/router_metrics.jsonl` (field: `slippage_bps`).

**Gap:** Slippage is NOT embedded in Episode records. Raw edge computed from
`avg_entry_price` / `avg_exit_price` already reflects realized slippage (since
these are actual fill prices vs market prices). Therefore raw edge implicitly
includes slippage impact, but we cannot decompose slippage separately from
price movement without joining to `router_metrics.jsonl`.

### 2.3 Funding Rate

**Gap:** Funding rate drag is NOT tracked per episode. For perpetual futures,
holding periods crossing 8-hour funding intervals incur funding costs that are
invisible in the current episode model. This is a known measurement gap.

### 2.4 Spread Crossing

Half-spread is paid on entry and exit when taking liquidity. Maker routing
mitigates this but only when fills execute at the maker price.

---

## §3 — Decomposition Engine

### Implementation: `compute_friction_decomposition()`

For each episode with valid score, prices, and notional:

```
raw_edge_bps  = direction_adjusted_return × 10,000
fee_drag_bps  = (fees / entry_notional) × 10,000
net_edge_bps  = raw_edge_bps − fee_drag_bps
killed        = (raw_edge_bps > 0) AND (net_edge_bps ≤ 0)
```

Cross-validated against `gross_pnl` / `net_pnl` / `entry_notional` where available.

### Output: Distribution Statistics

For each of `{raw_edge_bps, fee_drag_bps, net_edge_bps}`:

| Stat | Purpose |
|------|---------|
| mean | Central tendency |
| median | Robust central tendency |
| std | Dispersion |
| p10, p25, p75, p90 | Tail behavior |
| min, max | Extremes |

### Output: Aggregate Summary

| Field | Definition |
|-------|-----------|
| `positive_raw_count` | Trades with raw edge > 0 |
| `friction_killed_count` | Of those, how many have net edge ≤ 0 |
| `friction_kill_rate` | killed / positive_raw |
| `fee_to_edge_ratio` | Σ fee_drag / Σ raw_edge (> 1.0 = fees exceed edge) |
| `total_gross_edge_bps` | Sum of all raw edges |
| `total_net_edge_bps` | Sum of all net edges |

---

## §4 — Friction-Kill Analysis

### Implementation: `compute_friction_kill_analysis()`

#### 4.1 By Score Bucket

Splits trades into quintiles by `hybrid_score` and computes kill rate per bucket.
Healthy system: kill rate decreases as score increases (high-conviction trades
should capture more edge than fees consume).

| Bucket | Kill Rate Interpretation |
|--------|------------------------|
| Q1 (lowest score) | High kill rate expected — marginal signals don't survive costs |
| Q5 (highest score) | Kill rate should be near zero — if not, edge is illusory |

**Critical test:** If Q5 kill rate > 10%, friction is erasing even the best signals.

#### 4.2 By Symbol

Computes per-symbol friction profile. Some instruments have structurally higher
friction (wider spreads, lower liquidity, higher slippage).

Output sorted by kill rate descending: worst symbols first.

**Action trigger:** Symbols with kill rate > 40% should be evaluated for universe removal
or upgraded to maker-only execution.

---

## §5 — Break-Even Edge Analysis

### Implementation: `compute_break_even_edge()`

| Output | Definition |
|--------|-----------|
| `break_even_bps` | Mean fee drag — the minimum raw edge needed to survive |
| `median_fee_bps` | Median fee drag (more robust to outliers) |
| `pct_above_hurdle` | % of trades where raw edge exceeds the hurdle |
| `implied_min_score` | Approximate score at which rolling raw edge first exceeds hurdle |
| `duration_vs_friction` | Whether longer hold periods dilute fee impact |

### Duration-vs-Friction Insight

Fees are fixed per round-trip, but raw edge scales with holding period (larger
price moves over longer holds). If short holds have higher fee drag per unit of
edge than long holds, the system should bias toward longer holding periods.

Splits trades into short-hold and long-hold halves by duration, comparing mean
fee drag. If `short_hold_mean_fee_bps >> long_hold_mean_fee_bps`, duration is
a significant friction moderator.

---

## §6 — Verdict Framework

### Master Function: `compute_friction_audit()`

Orchestrates all three sub-analyses plus the existing `compute_friction_overlay()`
and produces a single tradability verdict:

| Verdict | Conditions | Action |
|---------|-----------|--------|
| **TRADABLE** | mean_net_edge > 0 AND kill_rate < 25% | Continue trading; monitor |
| **MARGINAL** | mean_net_edge > 0 BUT kill_rate ≥ 25%, OR mean_net_edge ∈ (−2, 0] | Optimize routing; raise score threshold |
| **NOT_TRADABLE** | mean_net_edge ≤ −2 bps OR kill_rate ≥ 50% | Halt entries; re-engineer signal or reduce friction |
| **INSUFFICIENT_DATA** | < 10 scored episodes | Defer judgment |

### Severity Classification

Based on `fee_to_edge_ratio`:

| Ratio | Label | Interpretation |
|-------|-------|---------------|
| > 1.0 | `fees_exceed_edge` | System pays more in fees than it captures in edge |
| 0.5–1.0 | `fees_consume_majority` | Over half of raw edge lost to friction |
| 0.25–0.5 | `moderate_drag` | Meaningful but manageable |
| < 0.25 | `low_drag` | Friction is not the binding constraint |

---

## §7 — What This Audit Cannot See

### 7.1 Slippage Decomposition

Raw edge uses actual fill prices (`avg_entry_price`, `avg_exit_price`), which already
include realized slippage. We therefore cannot decompose:

```
raw_edge = pure_signal_edge + slippage_impact
```

To separate these, per-fill mid-price at order time would need to be stored in
episode records. Currently this data exists only in `router_metrics.jsonl` and
requires a join.

### 7.2 Funding Rate Drag

Perpetual futures incur funding every 8 hours. For trades spanning funding
intervals, this cost is invisible. A 5 BPS funding rate on a 1× leveraged
position held for 24 hours adds ~15 BPS of hidden friction.

**Recommendation:** Add `funding_cost_usd` field to Episode records.

### 7.3 Opportunity Cost of Maker Routing

Maker-first routing reduces fees but may cause missed fills (orders that timeout
and fall back to taker, or are cancelled entirely). This latent friction —
"what we didn't trade because we waited for maker fill" — is not captured.

### 7.4 Left-Truncation Bias

This audit operates on the *traded* subset (episodes). Passive observations
logged by the signal screener (§7 of causality audit) are required to
assess friction impact across the full scored universe.

---

## §8 — Integration

### State Output

Published to `logs/state/hydra_monotonicity.json` under `friction_audit` key:

```json
{
  "friction_audit": {
    "verdict": "MARGINAL",
    "severity": "moderate_drag",
    "mean_net_edge_bps": 3.14,
    "friction_kill_rate": 0.2817,
    "fee_to_edge_ratio": 0.3142,
    "n": 247,
    "decomposition": { ... },
    "kill_analysis": { ... },
    "break_even": { ... },
    "friction_overlay": { ... }
  }
}
```

### Code Location

All functions in `execution/hydra_monotonicity.py`:

| Function | Purpose |
|----------|---------|
| `compute_friction_decomposition()` | Per-trade raw/fee/net BPS + distributions |
| `compute_friction_kill_analysis()` | Kill rate by score bucket and symbol |
| `compute_break_even_edge()` | Minimum raw edge hurdle + duration insight |
| `compute_friction_audit()` | Master orchestrator + verdict |

Called from `persist_snapshot()` → `snap["friction_audit"]`.

---

## §9 — Relationship to Other Audits

| Audit | Relationship |
|-------|-------------|
| Signal–Outcome Causality (§1–§6) | This audit assumes causality exists; it asks whether causality survives costs |
| Selection Bias (§7) | Friction audit operates on traded subset; selection bias audit defines the boundary |
| Threshold Sweep (§8.1) | Optimal threshold T* should incorporate friction — raw T* may differ from net T* |
| Edge Curve (§8.2) | Shape of E[O\|S=s] should be recomputed on net returns, not just raw |
| Probability-First Mapping | Friction audit provides the "executable probability" layer |
| Over-Architecture Detection | Friction is the final arbiter — layers that add friction without edge are negative-sum |

---

## §10 — Honest Position

1. **Fee drag is the only component we can cleanly measure.** Slippage is baked into
   fill prices. Funding is untracked. Spread crossing is partially captured.

2. **Kill rate is the most important number in this audit.** A system with 40%+ kill
   rate is working against itself — generating signals that can't survive their own
   execution cost.

3. **Fee-to-edge ratio > 0.5 is a structural warning.** When more than half of
   captured edge goes to the exchange, the system's real customer is Binance.

4. **Duration is a hidden friction lever.** Short-duration trades pay the same
   round-trip fees as long-duration trades but have less time for edge to
   accumulate. This creates a systematic bias against high-frequency signals.

5. **Maker routing quality matters more than signal quality in the marginal case.**
   If fee drag drops from 10 BPS (taker/taker) to 4 BPS (maker/maker), the
   break-even hurdle halves. Router optimization may be higher-ROI than signal
   improvement.
